"""FastAPI controller and driver for the Ray RL Training loop.

Exposes endpoints to trigger Custom GRPO or veRL PPO/GRPO training on the GKE Ray cluster,
stream logs, view metrics history, and query active worker nodes for the dashboard.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import re
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Configuration (namespace-portable; defaults from env).
# --------------------------------------------------------------------------- #
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
POD_NAMESPACE = os.environ.get("POD_NAMESPACE", "default")
RAY_CLUSTER_NAME = os.environ.get("RAY_CLUSTER_NAME", "dolev-rl-ray-cluster")
DEFAULT_MODEL = os.environ.get("RL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

app = FastAPI(title="Ray RL Demo Controller")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Serve playroom static files same-origin (standalone UI)
_FRONTEND = pathlib.Path(__file__).resolve().parent / "frontend"
if _FRONTEND.is_dir():
    app.mount(
        "/static/features/ray",
        StaticFiles(directory=str(_FRONTEND)),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(_FRONTEND / "index.html"))

    @app.middleware("http")
    async def _no_cache_ui(request, call_next):
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

# --------------------------------------------------------------------------- #
# Training state
# --------------------------------------------------------------------------- #
_TRAINING_ACTIVE = False
_TRAINING_THREAD = None
_STATUS = "idle"  # idle, starting, training, error
_ERROR = None
_TRAINER_ACTOR = None

_METRICS = {
    "steps": [],
    "loss": [],
    "reward": [],
}
_LOGS = []
_STATE_LOCK = threading.Lock()

_RAY_READY = False
_RAY_LOCK = threading.Lock()


def _ensure_ray() -> None:
    """Ensure a live connection to the Ray Cluster."""
    global _RAY_READY
    import ray

    with _RAY_LOCK:
        if _RAY_READY:
            try:
                ray.cluster_resources()
                return
            except Exception:
                _RAY_READY = False

        try:
            ray.shutdown()
        except Exception:
            pass
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        ray.cluster_resources()
        _RAY_READY = True


def prepare_verl_dataset() -> str:
    """Reads local GSM8K JSON dataset and writes a parquet file for veRL ingestion."""
    import pandas as pd
    src = os.path.join(os.path.dirname(__file__), "gsm8k_train.json")
    with open(src, "r") as f:
        data = json.load(f)
        
    records = []
    for item in data:
        records.append({
            "prompt": item["question"],
            "ability": "math",
            "answer": item["answer"]
        })
        
    df = pd.DataFrame(records)
    dst = os.path.join(os.path.dirname(__file__), "verl_gsm8k_train.parquet")
    df.to_parquet(dst)
    return dst


def _run_verl(model_name: str, lr: float, batch_size: int, group_size: int) -> None:
    global _TRAINING_ACTIVE, _STATUS, _ERROR, _METRICS, _LOGS
    
    with _STATE_LOCK:
        _STATUS = "starting"
        _ERROR = None
        _LOGS.append(f"[{time.strftime('%X')}] Preparing veRL dataset parquet file...")

    try:
        parquet_file = prepare_verl_dataset()
        
        with _STATE_LOCK:
            _LOGS.append(f"[{time.strftime('%X')}] Connecting to Ray Cluster at {RAY_ADDRESS}...")
            _LOGS.append(f"[{time.strftime('%X')}] Launching veRL PPO/GRPO Trainer on GKE L4 GPUs...")
            
        cmd = [
            "python3", "-m", "verl.trainer.main_ppo",
            f"data.train_files={parquet_file}",
            f"data.val_files={parquet_file}",
            f"actor_rollout_ref.model.path={model_name}",
            f"actor_rollout_ref.actor.optim.lr={lr}",
            "actor_rollout_ref.model.use_remove_padding=True",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size=1",
            "actor_rollout_ref.rollout.generator_type=vllm", 
            f"actor_rollout_ref.rollout.n={group_size}",
            "actor_rollout_ref.ref.log_prob_micro_batch_size=1",
            "algorithm.adv_estimator=grpo",
            "algorithm.kl_ctrl.kl_coef=0.001",
            "trainer.logger=['console']",
            "trainer.project_name=dolev_rl_ray",
            "trainer.experiment_name=gsm8k",
            "trainer.n_gpus_per_node=1",
            "trainer.nnodes=1",
            "trainer.max_epochs=1",
            "trainer.total_training_steps=50"
        ]

        # Injects current Ray address environment variable
        env = dict(os.environ)
        env["RAY_ADDRESS"] = RAY_ADDRESS
        
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        with _STATE_LOCK:
            _STATUS = "training"

        # Regex matching patterns for veRL logs metrics:
        # e.g., step:1 - timing/gen:1.450 - actor/policy_loss:0.1250 - actor/total_reward:0.4500
        step_pattern = re.compile(r'step:(\d+)')
        loss_pattern = re.compile(r'actor/policy_loss:(-?\d+(?:\.\d+)?)')
        reward_pattern = re.compile(r'(?:actor/total_reward|val/test_score/openai/gsm8k):(-?\d+(?:\.\d+)?)')

        for line in iter(proc.stdout.readline, ""):
            # Check if training was stopped from UI
            with _STATE_LOCK:
                if not _TRAINING_ACTIVE:
                    proc.terminate()
                    break

            line_str = line.strip()
            if not line_str:
                continue

            with _STATE_LOCK:
                _LOGS.append(line_str)
                if len(_LOGS) > 300:
                    _LOGS = _LOGS[-300:]

            # Parse metrics
            s_match = step_pattern.search(line_str)
            l_match = loss_pattern.search(line_str)
            r_match = reward_pattern.search(line_str)

            if s_match and l_match and r_match:
                step = int(s_match.group(1))
                loss_val = float(l_match.group(1))
                reward_val = float(r_match.group(1))

                with _STATE_LOCK:
                    _METRICS["steps"].append(step)
                    _METRICS["loss"].append(loss_val)
                    _METRICS["reward"].append(reward_val)

                    if len(_METRICS["steps"]) > 200:
                        _METRICS["steps"].pop(0)
                        _METRICS["loss"].pop(0)
                        _METRICS["reward"].pop(0)

        proc.wait()

    except Exception as exc:
        with _STATE_LOCK:
            _STATUS = "error"
            _ERROR = str(exc)
            _LOGS.append(f"[{time.strftime('%X')}] veRL Error: {exc}")
            _TRAINING_ACTIVE = False
    finally:
        with _STATE_LOCK:
            if _STATUS != "error":
                _STATUS = "idle"
                _LOGS.append(f"[{time.strftime('%X')}] veRL Training loop stopped.")


def _run_custom(model_name: str, lr: float, batch_size: int, group_size: int) -> None:
    global _TRAINING_ACTIVE, _STATUS, _ERROR, _TRAINER_ACTOR, _METRICS, _LOGS
    import ray
    from tasks import RLTrainer

    with _STATE_LOCK:
        _STATUS = "starting"
        _ERROR = None
        _LOGS.append(f"[{time.strftime('%X')}] Connecting to Ray Cluster at {RAY_ADDRESS}...")

    try:
        _ensure_ray()
        with _STATE_LOCK:
            _LOGS.append(f"[{time.strftime('%X')}] Allocating GPU and initializing RLTrainer Actor...")

        # Instantiate RLTrainer as a remote Ray Actor on a GPU worker node.
        # This blocks until Ray schedules the actor (which triggers worker scale-up!)
        # and loads the model into GPU memory.
        _TRAINER_ACTOR = RLTrainer.options(num_gpus=1).remote(model_name=model_name, lr=lr)

        # Call a quick ping to verify liveness
        ray.get(_TRAINER_ACTOR.reward_fn.remote("42", "42"))

        with _STATE_LOCK:
            _STATUS = "training"
            _LOGS.append(f"[{time.strftime('%X')}] Model '{model_name}' loaded successfully on GPU!")
            _LOGS.append(f"[{time.strftime('%X')}] Starting GRPO alignment loop on GSM8K math dataset...")

        while True:
            with _STATE_LOCK:
                if not _TRAINING_ACTIVE:
                    break

            start_time = time.time()
            try:
                step_ref = _TRAINER_ACTOR.train_step.remote(batch_size=batch_size, group_size=group_size)
                result = ray.get(step_ref)
            except Exception as e:
                with _STATE_LOCK:
                    if not _TRAINING_ACTIVE:
                        break
                    _LOGS.append(f"[{time.strftime('%X')}] Spot GPU node preempted or restarted ({e}). Re-instantiating RLTrainer on replacement Spot node...")
                time.sleep(5)
                _TRAINER_ACTOR = RLTrainer.options(num_gpus=1).remote(model_name=model_name, lr=lr)
                continue
            elapsed = time.time() - start_time

            with _STATE_LOCK:
                step = result["step"]
                loss = result["loss"]
                reward = result["avg_reward"]
                pod = result["pod_name"]

                _METRICS["steps"].append(step)
                _METRICS["loss"].append(loss)
                _METRICS["reward"].append(reward)

                if len(_METRICS["steps"]) > 200:
                    _METRICS["steps"].pop(0)
                    _METRICS["loss"].pop(0)
                    _METRICS["reward"].pop(0)

                log_line = f"[{time.strftime('%X')}] Step {step} | Loss: {loss:.4f} | Avg Acc: {reward:.2%} | Time: {elapsed:.1f}s | Worker: {pod}"
                _LOGS.append(log_line)

                # Append detailed completions
                for item in result["logs"]:
                    _LOGS.append(f"  Question: {item['question']}")
                    _LOGS.append(f"  Target: {item['target'].strip()}")
                    for idx, compl in enumerate(item["completions"]):
                        clean_text = compl["text"].replace("\n", " ").strip()[:120]
                        _LOGS.append(f"    Rollout {idx+1}: Reward {compl['reward']:.1f} | Adv {compl['advantage']:.2f} | Ans: {clean_text}...")
                
                if len(_LOGS) > 300:
                    _LOGS = _LOGS[-300:]

            time.sleep(0.5)

    except Exception as exc:
        with _STATE_LOCK:
            _STATUS = "error"
            _ERROR = str(exc)
            _LOGS.append(f"[{time.strftime('%X')}] ERROR: {exc}")
            _TRAINING_ACTIVE = False
    finally:
        with _STATE_LOCK:
            if _STATUS != "error":
                _STATUS = "idle"
                _LOGS.append(f"[{time.strftime('%X')}] Training loop stopped.")
            _TRAINER_ACTOR = None


def _run_training(model_name: str, framework: str, lr: float, batch_size: int, group_size: int) -> None:
    if framework == "verl":
        _run_verl(model_name, lr, batch_size, group_size)
    else:
        _run_custom(model_name, lr, batch_size, group_size)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class TrainRequest(BaseModel):
    model_name: str = Field(default=DEFAULT_MODEL)
    framework: str = Field(default="custom")
    lr: float = Field(default=1e-5, ge=1e-6, le=1e-3)
    batch_size: int = Field(default=8, ge=1, le=32)
    group_size: int = Field(default=8, ge=2, le=16)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/start")
def start_training(req: TrainRequest) -> dict:
    global _TRAINING_ACTIVE, _TRAINING_THREAD
    with _STATE_LOCK:
        if _TRAINING_ACTIVE or _STATUS == "starting":
            return {"status": "already_running"}
        
        _TRAINING_ACTIVE = True
        _METRICS["steps"].clear()
        _METRICS["loss"].clear()
        _METRICS["reward"].clear()
        _LOGS.clear()

        _TRAINING_THREAD = threading.Thread(
            target=_run_training,
            args=(req.model_name, req.framework, req.lr, req.batch_size, req.group_size),
            daemon=True
        )
        _TRAINING_THREAD.start()
        return {"status": "started"}


@app.post("/stop")
def stop_training() -> dict:
    global _TRAINING_ACTIVE
    with _STATE_LOCK:
        if not _TRAINING_ACTIVE:
            return {"status": "not_running"}
        _TRAINING_ACTIVE = False
        _LOGS.append(f"[{time.strftime('%X')}] Stopping training loop...")
        return {"status": "stopping"}


@app.get("/status")
def get_status() -> dict:
    with _STATE_LOCK:
        return {
            "status": _STATUS,
            "active": _TRAINING_ACTIVE,
            "error": _ERROR,
            "total_steps": len(_METRICS["steps"])
        }


@app.get("/metrics")
def get_metrics() -> dict:
    with _STATE_LOCK:
        return _METRICS


@app.get("/logs/stream")
def stream_logs() -> dict:
    """Returns recent log lines."""
    with _STATE_LOCK:
        return {"logs": _LOGS}


@app.get("/workers")
def workers() -> dict:
    """List Ray pods for the cluster map (head + autoscaled workers)."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        v1 = client.CoreV1Api()
        selector = f"ray.io/cluster={RAY_CLUSTER_NAME}"
        pods = v1.list_namespaced_pod(POD_NAMESPACE, label_selector=selector)
        out = []
        for p in pods.items:
            out.append({
                "pod_name": p.metadata.name,
                "node": p.spec.node_name,
                "node_type": (p.metadata.labels or {}).get("ray.io/node-type", "unknown"),
                "status": p.status.phase,
            })
        return {"namespace": POD_NAMESPACE, "cluster": RAY_CLUSTER_NAME, "pods": out}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"cannot list pods: {exc}")


@app.get("/dashboard")
def dashboard() -> dict:
    """External URL of the Ray Dashboard LoadBalancer."""
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        v1 = client.CoreV1Api()
        svc = v1.read_namespaced_service("ray-dashboard", POD_NAMESPACE)
        ingress = (svc.status.load_balancer.ingress or []) if svc.status.load_balancer else []
        ip = ingress[0].ip if ingress else None
        return {"url": f"http://{ip}/" if ip else None}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"cannot read dashboard service: {exc}")
