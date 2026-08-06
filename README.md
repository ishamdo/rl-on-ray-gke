# RL on Ray & GKE: GRPO Alignment Loop 🌀

Distributed **GRPO (Group Relative Policy Optimization)** reinforcement learning alignment loop on **KubeRay / GKE**. Pick a model, set the training hyperparameters, hit **Start**, and watch the model learn to solve grade-school arithmetic problems from the `GSM8K` dataset in real time—while a live **metrics dashboard** shows the loss decreasing and rewards (accuracy) climbing as Ray autoscales GPU worker pods onto **Spot L4 GPU nodes**.

This is a [gke_all](https://github.com/ragoler/gke_all) showcase feature (`feature.yaml`). It runs **standalone** and as a **Hub feature**. See [DASHBOARD.md](DASHBOARD.md) for live training metrics and GRPO advantage analysis.

## How it works

```
Browser ──/start──▶ Controller (Ray driver) ──ray://head:10001──▶ RayCluster
   ▲                       │  runs GRPO step by step                    head +
   │  SSE: metrics +       │  ray.remote executes GPU rollouts     autoscaling
   └─ logs stream ◀────────┘  advantage updates model parameters   Spot workers
```

- **Distributed Rollouts:** multiple trajectories (group size $G$) are sampled in parallel on GPU worker nodes for each question.
- **Autoscaling:** KubeRay automatically requests Spot GPU nodes from GKE Node Auto-Provisioning (NAP) to handle the GPU Actor workload, scaling back to zero nodes when training goes idle.
- **GRPO Advantages:** computes relative advantages within the rollout group to reinforce correct logical paths and penalize incorrect ones without requiring a separate critic network.
- **Streaming:** the controller streams steps, completions, loss, and rewards in real time to the browser over Server-Sent Events (SSE).

## Layout

| Path | Purpose |
|---|---|
| `feature.yaml` | Hub descriptor |
| `app/` | FastAPI controller backend, training task script, requirements, and dataset builder |
| `frontend/` | playroom UI: controls, parameter selection, metrics charts, and live logging |
| `hub_router.py` | Hub data-plane router for Hub dashboard integration |
| `infra/` | per-namespace: RayCluster, controller, HTTPRoute, Gateway, and BackendPolicies |
| `cluster/` | cluster-scoped: KubeRay operator + Spot GPU ComputeClass |
| `.env.example` | standalone configuration template (`cp .env.example .env`) |
| `setup_infra.sh` | standalone: create GKE cluster + NAP configs + cluster-scoped CRDs |
| `deploy_app.sh` | standalone: build/push container image + deploy `infra/` manifests |
| `verify_setup.sh` | standalone: isolated KUBECONFIG setup + readiness smoke tests |
| `tests/` | unit tests |

## Standalone on GKE

Three steps, mirroring the showcase convention: configure, provision the cluster, deploy the app.

```bash
# 1. Configure (edit PROJECT_ID, cluster name, region, worker cap, …)
cp .env.example .env

# 2. Provision: create the GKE cluster (Gateway API + Node Auto-Provisioning)
#    and the cluster-scoped prereqs (KubeRay operator + Spot GPU ComputeClass).
./setup_infra.sh

# 3. Build & push the image, then deploy the RayCluster + controller + Gateway.
./deploy_app.sh

# 4. Validate: check readiness and trigger training via the Gateway IP.
./verify_setup.sh
```

Then open the **Gateway IP in a browser** (default serves the playroom UI standalone). The Ray Dashboard is at the `ray-dashboard` Service's external IP. Standalone uses **`PROJECT_ID`** (in `.env`); the Hub injects the equivalent as **`PROJECT_NAME`** and supplies `NAMESPACE`/`REGION`/`ARTIFACT_REGISTRY_REPO` itself.

Teardown (the cluster is only removed with `--delete-cluster`):

```bash
./setup_infra.sh --delete          # remove cluster-scoped prereqs, keep cluster
./setup_infra.sh --delete-cluster  # the above, plus delete the GKE cluster
```

## Metrics — Google Managed Prometheus (GMP)

Ray exports Prometheus metrics on each pod (`metrics` port `:8080/metrics`). The RayCluster exposes that port, and [`infra/podmonitoring.yaml`](infra/podmonitoring.yaml) tells **GMP's managed collection** to scrape every Ray pod.

`deploy_app.sh` also creates a curated **Cloud Monitoring dashboard** ("RL on Ray", from [`monitoring/ray-dashboard.json`](monitoring/ray-dashboard.json)) and stashes its URL in the `ray-links` ConfigMap, which the controller surfaces as the **"Metrics ↗"** button in the playroom.

Or query directly in **Cloud Console → Monitoring → Metrics Explorer** (PromQL):

```promql
ray_node_cpu_utilization
sum(ray_tasks{State="RUNNING"})
ray_cluster_active_nodes
```

## Tests

```bash
pip install -r app/requirements.txt pytest httpx
pytest tests/
```
