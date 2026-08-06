"""Hub data-plane router for the Ray RL Training feature.

Mounted by the Hub at ``/api/features/ray`` behind the admin JWT. Kept thin:

* **LIVE** — the browser talks to the controller directly via the Gateway IP
  (CORS) for the heavy data plane (start, stop, status, metrics, logs, workers).
* **MOCK** — no cluster exists, so this router serves the *entire* surface
  (``/config``, ``/start``, ``/stop``, ``/status``, ``/metrics``, ``/logs/stream``, ``/workers``)
  with deterministic simulated data, showing an accuracy improvement curve.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid

from fastapi import APIRouter, HTTPException

# --------------------------------------------------------------------------- #
# Shared SDK
# --------------------------------------------------------------------------- #
try:  # pragma: no cover
    from showcase_admin.app import config, database, k8s_client
except Exception:
    config = None
    database = None
    k8s_client = None


def _mode() -> str:
    if config is not None:
        return getattr(config, "MODE", "MOCK")
    import os
    return os.environ.get("MODE", "MOCK").upper()


FEATURE = "ray"
GATEWAY_NAME = "ray-render-gw"

router = APIRouter()

# --------------------------------------------------------------------------- #
# MOCK state
# --------------------------------------------------------------------------- #
_MOCK_STATE = {
    "status": "idle",
    "active": False,
    "framework": "custom",
    "step": 0,
    "metrics": {
        "steps": [],
        "loss": [],
        "reward": []
    },
    "logs": []
}

_MOCK_PODS = [
    {"pod_name": "ray-render-farm-head", "node_type": "head", "status": "Running", "node": "node-default"}
]

_MOCK_TASK = None


async def _simulate_training():
    global _MOCK_STATE, _MOCK_PODS
    
    _MOCK_STATE["status"] = "starting"
    _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Connecting to Ray Cluster at ray://ray-render-farm-head-svc:10001...")
    
    if _MOCK_STATE["framework"] == "verl":
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Initializing veRL HybridFlow pipeline...")
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Configuring model placements:")
        _MOCK_STATE["logs"].append(f"  - Actor strategy: FSDP (allocated to ray-verl-actor-0)")
        _MOCK_STATE["logs"].append(f"  - Reference strategy: FSDP (allocated to ray-verl-ref-0)")
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Allocating GPUs and loading Qwen weights...")
        
        await asyncio.sleep(2.0)
        
        if len(_MOCK_PODS) == 1:
            _MOCK_PODS.append({
                "pod_name": "ray-verl-actor-0",
                "node_type": "worker",
                "status": "Running",
                "node": "gpu-spot-node-0"
            })
            _MOCK_PODS.append({
                "pod_name": "ray-verl-ref-0",
                "node_type": "worker",
                "status": "Running",
                "node": "gpu-spot-node-1"
            })
        
        _MOCK_STATE["status"] = "training"
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] veRL HybridEngine active on 2 GPUs.")
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Starting GRPO training on GSM8K dataset...")
    else:
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Allocating GPU and initializing RLTrainer Actor...")
        
        await asyncio.sleep(2.0)
        
        if len(_MOCK_PODS) == 1:
            _MOCK_PODS.append({
                "pod_name": "ray-render-farm-worker-0",
                "node_type": "worker",
                "status": "Running",
                "node": "gpu-spot-node-0"
            })
            
        _MOCK_STATE["status"] = "training"
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Model loaded successfully on GPU!")
        _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Starting GRPO alignment loop on GSM8K math dataset...")

    loss = 1.2
    acc = 0.10
    
    while _MOCK_STATE["active"]:
        await asyncio.sleep(1.5)
        _MOCK_STATE["step"] += 1
        step = _MOCK_STATE["step"]
        
        loss = max(0.05, loss - random.uniform(0.02, 0.08))
        acc = min(0.95, acc + random.uniform(0.01, 0.06))
        
        _MOCK_STATE["metrics"]["steps"].append(step)
        _MOCK_STATE["metrics"]["loss"].append(loss)
        _MOCK_STATE["metrics"]["reward"].append(acc)
        
        if _MOCK_STATE["framework"] == "verl":
            gen_time = random.uniform(1.2, 1.8)
            ref_time = random.uniform(0.4, 0.6)
            log_line = f"step:{step} - timing/gen:{gen_time:.3f} - timing/ref:{ref_time:.3f} - actor/policy_loss:{loss:.4f} - actor/total_reward:{acc:.4f} - val/test_score/openai/gsm8k:{acc:.4f}"
            _MOCK_STATE["logs"].append(log_line)
            # Tag which worker did rollout/training
            _MOCK_STATE["logs"].append(f"  [veRL] Rollouts generated on worker: ray-verl-actor-0")
        else:
            log_line = f"[{time.strftime('%X')}] Step {step} | Loss: {loss:.4f} | Avg Acc: {acc:.2%} | Time: 1.2s | Worker: ray-render-farm-worker-0"
            _MOCK_STATE["logs"].append(log_line)
            
            q = f"If Tom has {step} apples and Jerry takes {random.randint(1, 3)}, how many are left?"
            ans = step - 1
            _MOCK_STATE["logs"].append(f"  Question: {q}")
            _MOCK_STATE["logs"].append(f"  Target: Tom had {step} apples.Jerry took some. Left: {ans}\n#### {ans}")
            _MOCK_STATE["logs"].append(f"    Rollout 1: Reward 1.0 | Adv 0.50 | Ans: Tom has {ans} apples left. #### {ans}")
            _MOCK_STATE["logs"].append(f"    Rollout 2: Reward 0.0 | Adv -0.50 | Ans: Tom has 10 apples left. #### 10")
        
        if len(_MOCK_STATE["logs"]) > 200:
            _MOCK_STATE["logs"] = _MOCK_STATE["logs"][-200:]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/config")
async def config_endpoint() -> dict:
    if _mode() == "MOCK":
        return {"mode": "MOCK", "gateway_ip": None, "dashboard_url": None}

    gateway_ip = None
    if database is not None and k8s_client is not None:
        db = next(database.get_db())
        try:
            ns = database.get_feature_namespace(db, FEATURE)
            gateway_ip = await k8s_client.get_gateway_ip(ns, GATEWAY_NAME)
        except Exception:
            gateway_ip = None
        finally:
            db.close()
    return {"mode": "LIVE", "gateway_ip": gateway_ip, "dashboard_url": None}


@router.post("/start")
async def start_training(req: dict) -> dict:
    global _MOCK_STATE, _MOCK_TASK, _MOCK_PODS
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE start runs on Gateway IP.")
        
    if _MOCK_STATE["active"]:
        return {"status": "already_running"}
        
    _MOCK_STATE["active"] = True
    _MOCK_STATE["framework"] = req.get("framework", "custom")
    _MOCK_STATE["step"] = 0
    _MOCK_STATE["metrics"]["steps"].clear()
    _MOCK_STATE["metrics"]["loss"].clear()
    _MOCK_STATE["metrics"]["reward"].clear()
    _MOCK_STATE["logs"].clear()
    _MOCK_PODS[:] = [
        {"pod_name": "ray-render-farm-head", "node_type": "head", "status": "Running", "node": "node-default"}
    ]
    
    _MOCK_TASK = asyncio.create_task(_simulate_training())
    return {"status": "started"}


@router.post("/stop")
async def stop_training() -> dict:
    global _MOCK_STATE, _MOCK_TASK, _MOCK_PODS
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE stop runs on Gateway IP.")
        
    if not _MOCK_STATE["active"]:
        return {"status": "not_running"}
        
    _MOCK_STATE["active"] = False
    if _MOCK_TASK:
        _MOCK_TASK.cancel()
        _MOCK_TASK = None
        
    _MOCK_STATE["status"] = "idle"
    _MOCK_STATE["logs"].append(f"[{time.strftime('%X')}] Training loop stopped.")
    
    # Remove mock workers on stop
    _MOCK_PODS[:] = [
        {"pod_name": "ray-render-farm-head", "node_type": "head", "status": "Running", "node": "node-default"}
    ]
    return {"status": "stopping"}


@router.get("/status")
def get_status() -> dict:
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE status is on Gateway IP.")
    return {
        "status": _MOCK_STATE["status"],
        "active": _MOCK_STATE["active"],
        "error": None,
        "total_steps": len(_MOCK_STATE["metrics"]["steps"])
    }


@router.get("/metrics")
def get_metrics() -> dict:
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE metrics is on Gateway IP.")
    return _MOCK_STATE["metrics"]


@router.get("/logs/stream")
def stream_logs() -> dict:
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE logs are on Gateway IP.")
    return {"logs": _MOCK_STATE["logs"]}


@router.get("/workers")
def workers() -> dict:
    if _mode() != "MOCK":
        raise HTTPException(status_code=409, detail="LIVE workers are on Gateway IP.")
    return {"namespace": "gke-showcase-ray", "cluster": "dolev-rl-ray-cluster", "pods": _MOCK_PODS}
