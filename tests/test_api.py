"""API contract tests in DEGRADED mode — no MLflow, no network.

We monkeypatch the champion load to fail fast, so startup exercises the
graceful-degradation path instantly. This verifies the readiness contract:
/health stays 200 (liveness) while /predict and /model-info return 503 until a
model is loaded. The happy path (real champion, /reload hot-swap) is covered by
the acceptance run against the live stack, not here.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 180, 60)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def degraded_client(monkeypatch):
    """A client whose startup finds no champion (registry 'unreachable')."""
    def _fail():
        raise RuntimeError("registry unreachable (test)")
    monkeypatch.setattr(api.state, "load", _fail)
    api.state.model = None
    api.state.version = None
    with TestClient(api.app) as client:
        yield client


def test_health_is_live_but_not_ready(degraded_client):
    r = degraded_client.get("/health")
    assert r.status_code == 200            # liveness: the process is up
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False   # readiness: no model yet
    assert body["model_version"] is None


def test_predict_returns_503_when_no_model(degraded_client):
    r = degraded_client.post(
        "/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


def test_model_info_returns_503_when_no_model(degraded_client):
    assert degraded_client.get("/model-info").status_code == 503


def test_reload_returns_503_when_registry_down(degraded_client):
    assert degraded_client.post("/reload").status_code == 503
