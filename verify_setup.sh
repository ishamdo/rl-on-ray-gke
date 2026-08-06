#!/usr/bin/env bash
# Post-deployment validation for the RL on Ray Training loop: waits for the controller
# and Ray head, discovers the Gateway IP, and smoke-tests the training API.
set -e

if [ -f .env ]; then
  source .env
else
  echo "Error: .env file not found."
  exit 1
fi
NAMESPACE="${NAMESPACE:-default}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KUBECONFIG="${ROOT}/.kubeconfig"
# Source of truth is the manifest, not .env.
GATEWAY_NAME=$(awk '/kind: Gateway/{f=1} f&&/^  name:/{print $2; exit}' "${ROOT}/infra/gateway.yaml")

echo "=== Targeting cluster ${CLUSTER_NAME} (${ZONE}) ==="
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}"

echo "=== Waiting for the controller and Ray head to be Ready ==="
kubectl -n "${NAMESPACE}" rollout status deployment/dolev-rl-controller --timeout=300s
kubectl -n "${NAMESPACE}" wait --for=condition=Ready pod \
  -l ray.io/cluster=dolev-rl-ray-cluster,ray.io/node-type=head --timeout=600s

echo "=== Discovering Gateway IP ==="
gateway_ip() {
  local ip
  ip=$(kubectl -n "${NAMESPACE}" get gateway "${GATEWAY_NAME}" -o jsonpath='{.status.addresses[0].value}' 2>/dev/null || true)
  [ -z "${ip}" ] && ip=$(gcloud compute forwarding-rules list --global --project="${PROJECT_ID}" \
    --filter="name~gkegw1.*-${NAMESPACE}-${GATEWAY_NAME}" --format="value(IPAddress)" 2>/dev/null | head -1)
  echo "${ip}"
}
for i in {1..30}; do
  GATEWAY_IP=$(gateway_ip)
  [ -n "${GATEWAY_IP}" ] && break
  sleep 10
done
if [ -z "${GATEWAY_IP:-}" ]; then
  echo "Error: Gateway did not receive an IP within 5 minutes."
  exit 1
fi
echo "Gateway IP: ${GATEWAY_IP}"

BASE="http://${GATEWAY_IP}"

echo "=== Waiting for the Gateway data path to be healthy ==="
HEALTHY=""
for i in $(seq 1 32); do
  if curl -fsS -m 10 "${BASE}/healthz" >/dev/null 2>&1; then
    HEALTHY=1
    echo "Gateway healthy after ~$((i * 15))s"
    break
  fi
  sleep 15
done
if [ -z "${HEALTHY}" ]; then
  echo "Error: Gateway data path not healthy yet. Re-run ./verify_setup.sh shortly."
  exit 1
fi

echo "=== Health check ==="
curl -fsS "${BASE}/healthz" && echo

echo "=== Status check ==="
curl -fsS "${BASE}/status" && echo

echo "=== Testing Training Trigger (Start) ==="
# Trigger training using a small 0.5B model to avoid huge VRAM requirements on tests
START_RESP=$(curl -fsS -X POST "${BASE}/start" \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"Qwen/Qwen2.5-0.5B-Instruct","lr":1e-4,"batch_size":1,"group_size":2}')
echo "Start response: ${START_RESP}"

sleep 5

echo "=== Status during starting/training ==="
curl -fsS "${BASE}/status" && echo

echo "=== Logs stream check ==="
curl -fsS "${BASE}/logs/stream" | head -n 20

echo "=== Metrics history check ==="
curl -fsS "${BASE}/metrics" && echo

echo "=== Testing Training Stop ==="
STOP_RESP=$(curl -fsS -X POST "${BASE}/stop")
echo "Stop response: ${STOP_RESP}"

sleep 2
echo "=== Final Status check ==="
curl -fsS "${BASE}/status" && echo

echo "=== Cluster map (workers endpoint) ==="
curl -fsS "${BASE}/workers" | python3 -c "import sys,json;d=json.load(sys.stdin);print('pods:',[p['pod_name'] for p in d['pods']])"

echo "=== Verification successful ==="
DASH_IP=$(kubectl -n "${NAMESPACE}" get svc ray-dashboard -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
[ -n "${DASH_IP}" ] && echo "Open the Ray Dashboard at: http://${DASH_IP}/" \
  || echo "Ray Dashboard LoadBalancer still provisioning"
