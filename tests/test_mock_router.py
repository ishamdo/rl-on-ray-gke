"""MODE=MOCK hub_router tests — the playroom must work fully offline."""

import json
import os
import time
import pytest

os.environ["MODE"] = "MOCK"

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import hub_router  # noqa: E402


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(hub_router.router, prefix="/api/features/ray")
    return TestClient(app)


def test_config_mock(client):
    r = client.get("/api/features/ray/config")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "MOCK"
    assert body["gateway_ip"] is None


def test_training_flow_simulated(client):
    # Check initial status is idle
    r = client.get("/api/features/ray/status")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"

    # Start training
    r = client.post("/api/features/ray/start", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    # Status should be starting or training
    r = client.get("/api/features/ray/status")
    assert r.status_code == 200
    assert r.json()["status"] in ["starting", "training"]

    # Sleep briefly to let the simulated training run a step
    # Note: since mock training uses asyncio.sleep, we have to make sure it yields.
    # In test TestClient, it runs in the same thread, so let's check logs
    time.sleep(0.1)

    r = client.get("/api/features/ray/logs/stream")
    assert r.status_code == 200
    assert "logs" in r.json()

    r = client.get("/api/features/ray/metrics")
    assert r.status_code == 200
    assert "steps" in r.json()

    # Stop training
    r = client.post("/api/features/ray/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopping"

    # Final status should go back to idle
    r = client.get("/api/features/ray/status")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"


def test_workers_mock(client):
    pods = client.get("/api/features/ray/workers").json()["pods"]
    types = {p["node_type"] for p in pods}
    assert "head" in types
