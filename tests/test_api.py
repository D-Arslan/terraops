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
    # "No database" must be true whatever the host runs: with the compose stack
    # up, the default URI (localhost:55433) WOULD connect, and this fixture
    # would silently test the healthy path. Port 1 is refused instantly.
    monkeypatch.setattr(api.predictions_log, "db_uri",
                        "postgresql://mlflow:mlflow@127.0.0.1:1/mlflow")
    api.predictions_log.written = 0
    api.predictions_log.ready = False
    api.predictions_log.last_error = None
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


# --- Sprint 4: the observability endpoints must survive degraded mode ----------
# A monitoring surface that only works when everything else works is useless
# precisely when it is needed, so both endpoints are asserted with NO model and
# NO database.

def test_metrics_endpoint_is_scrapable_without_a_model(degraded_client):
    r = degraded_client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    # Metric families are declared at import time, so a scraper gets a valid
    # (empty) exposition instead of a 500 while the API is still degraded.
    assert "terraops_predictions_total" in body
    assert "terraops_prediction_latency_seconds" in body
    assert "terraops_prediction_log_queue_depth" in body


def test_failed_predictions_are_counted_as_unavailable(degraded_client):
    """A 503 must land in the 'unavailable' bucket, not be silently uncounted."""
    degraded_client.post("/predict",
                         files={"file": ("t.png", _png_bytes(), "image/png")})
    body = degraded_client.get("/metrics").text
    assert 'terraops_requests_total{endpoint="/predict",outcome="unavailable"}' in body


def test_monitoring_status_reports_the_log_as_not_ready(degraded_client):
    """No Postgres in unit tests -> the log must report itself down, not lie."""
    r = degraded_client.get("/monitoring/status")
    assert r.status_code == 200
    body = r.json()
    assert body["log_ready"] is False
    assert body["last_error"]                # the reason is surfaced, not swallowed
    assert body["rows_written"] == 0
