"""Serving API for the EuroSAT classifier — FastAPI.

Two MLOps-specific choices drive this file:

1. The model is loaded FROM THE REGISTRY BY ALIAS (models:/<name>@champion), never
   from a .pth path. Changing the production model is a governance action (move the
   alias with promote.py), not a code deploy. `POST /reload` re-resolves the alias
   in-process so a freshly promoted champion is served WITHOUT restarting — that is
   the sprint's acceptance test.

2. Preprocessing comes from the SHARED module (preprocessing.py), the exact same
   code path train.py used. No transform is re-implemented here, so train/serving
   skew cannot exist.

Startup is GRACEFULLY DEGRADED: if the registry is unreachable or no champion
exists yet, the API still boots. /health reports not-ready and /predict returns
503 until a model is loaded (via startup retry or POST /reload). This survives
docker-compose start ordering, where the API may boot before MLflow is ready.

Sprint 4 adds OBSERVABILITY, under the same degraded-mode rule: every served
prediction is recorded (input statistics, predicted class, confidence, entropy,
latency, serving model version) through a non-blocking queue. A dead monitoring
database slows nothing and fails nothing — it only costs rows, and the drops are
counted rather than hidden.
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

import mlflow
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from mlflow import MlflowClient
from PIL import Image
from pydantic import BaseModel

import metrics
from dataset import EUROSAT_CLASSES
from image_features import extract_features, prediction_entropy
from prediction_log import PredictionLogger, build_row
from preprocessing import decode_image, preprocess_batch
from utils import get_device, load_params

PARAMS = load_params()
DATA_CFG = PARAMS["data"]
MODEL_NAME = PARAMS["promote"]["registry_model"]
ALIAS = "champion"
# Env override so the same code works locally (localhost) and in Docker (mlflow:5000).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(TRACKING_URI)
DEVICE = get_device()


class ChampionState:
    """Holds the in-memory champion and the registry VERSION it resolves to.

    Keeping the version lets /reload compare before paying for a full load_model,
    and lets /model-info report exactly which registered version is being served.
    A lock guards swaps so an in-flight /predict never sees a half-updated model.
    """

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.version: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self._lock = threading.Lock()

    def load(self) -> dict:
        """Re-resolve @champion and (re)load weights only if the version changed.

        Returns a small report: whether a reload happened and the version numbers.
        Raises on registry/alias errors so callers can map them to HTTP status.
        """
        client = MlflowClient()
        # Resolve the alias -> a concrete version. Raises if the registry is down
        # or no champion alias exists yet (the degraded-startup case).
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        target = mv.version

        with self._lock:
            previous = self.version
            if self.model is not None and previous == target:
                return {"reloaded": False, "version": target, "previous_version": previous}

            # Load the resolved VERSION explicitly (not @alias again): guarantees the
            # weights match the version we just reported, immune to a concurrent
            # alias move between resolve and load.
            model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{target}")
            model.to(DEVICE).eval()

            self.model = model
            self.version = target
            self.loaded_at = time.time()
            # Dashboards must follow a hot-swap without a restart, so the gauge
            # is updated where the swap happens, not at startup only.
            metrics.set_model_version(target)
            return {"reloaded": True, "version": target, "previous_version": previous}

    def require(self) -> torch.nn.Module:
        """Return the loaded model or raise 503 (readiness guard for /predict)."""
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail=(f"No model loaded. The registry ({TRACKING_URI}) had no "
                        f"'{MODEL_NAME}@{ALIAS}' at startup. Promote a champion, "
                        f"then POST /reload."),
            )
        return self.model


state = ChampionState()
predictions_log = PredictionLogger()

app = FastAPI(
    title="TerraOps EuroSAT API",
    description="Land-use classifier served from the MLflow registry by @champion alias.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    """Try to load the champion, but NEVER crash if it fails (graceful degrade)."""
    try:
        report = state.load()
        print(f"[startup] loaded {MODEL_NAME} v{report['version']}")
    except Exception as exc:  # registry down, no champion yet, etc.
        print(f"[startup] no model loaded ({type(exc).__name__}: {exc}). "
              f"Serving in degraded mode; POST /reload once a champion exists.")

    # Same policy for the prediction log: start() never raises, and the writer
    # thread reconnects on its own if Postgres is not up yet.
    predictions_log.start()
    print(f"[startup] prediction log ready={predictions_log.ready} "
          f"({predictions_log.last_error or 'connected'})")


@app.on_event("shutdown")
def _shutdown():
    """Flush queued prediction records so a clean stop does not lose the tail."""
    predictions_log.stop()


# --- Response schemas --------------------------------------------------------

class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict           # class -> probability (all 10, sums to 1)
    model_version: str            # WHICH registry version produced this answer


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    model_version: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    tracking_uri: str


class ModelInfo(BaseModel):
    registry_model: str
    alias: str
    model_version: str
    num_classes: int
    classes: list[str]
    image_size: int
    device: str
    loaded_at: float
    tracking_uri: str


class ReloadResult(BaseModel):
    reloaded: bool
    version: str
    previous_version: Optional[str]


class MonitoringStatus(BaseModel):
    """Health of the observability path itself — 'is my monitoring monitoring?'."""
    log_ready: bool
    rows_written: int
    rows_dropped: int          # queue saturation or DB down too long
    flush_failures: int
    queue_depth: int
    last_error: Optional[str]


# --- Inference helper --------------------------------------------------------

@torch.no_grad()
def _predict_tensor(batch: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) -> (N,10) softmax probabilities, on CPU."""
    logits = state.require()(batch.to(DEVICE))
    return torch.softmax(logits, dim=1).cpu()


def _to_prediction(probs_row: torch.Tensor) -> Prediction:
    top = int(probs_row.argmax())
    return Prediction(
        predicted_class=EUROSAT_CLASSES[top],
        confidence=round(float(probs_row[top]), 4),
        probabilities={cls: round(float(p), 4)
                       for cls, p in zip(EUROSAT_CLASSES, probs_row, strict=True)},
        model_version=state.version,
    )


def _decode_all(raws: list[bytes]) -> list[Image.Image]:
    """Bytes -> canonical RGB PIL images, decoded ONCE per request.

    Decoding here (instead of letting preprocess_batch do it internally) is what
    lets monitoring features and the model tensor come from the exact same
    decoded pixels. Handing the PIL objects on to preprocess_batch keeps the
    shared-contract guarantee intact — decode_image is idempotent on a PIL
    input, so the serving path is unchanged, not merely equivalent.
    """
    try:
        return [decode_image(raw) for raw in raws]
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Cannot decode image: {exc}") from exc


@contextmanager
def _counted(endpoint: str):
    """Count a request's OUTCOME, mapping HTTP status to a bounded label set.

    Separating client_error (bad upload) from unavailable (no champion loaded)
    matters operationally: one is the caller's problem, the other is ours, and a
    single `errors_total` would hide a degraded API behind users sending junk.
    """
    try:
        yield
    except HTTPException as exc:
        outcome = "unavailable" if exc.status_code == 503 else "client_error"
        metrics.observe_request(endpoint, outcome)
        raise
    except Exception:
        metrics.observe_request(endpoint, "server_error")
        raise
    else:
        metrics.observe_request(endpoint, "success")


def _record(images: list[Image.Image], raws: list[bytes],
            predictions: list[Prediction], *, endpoint: str,
            source: Optional[str], elapsed_ms: float) -> None:
    """Enqueue one monitoring row per image. Best-effort: never fails a request.

    Latency is charged per image (total / n) so /predict and /predict/batch feed
    the SAME distribution — mixing a 32-image batch's wall time with a single
    image's would make the p95 a function of client batching habits rather than
    of the service.
    """
    n = max(len(images), 1)
    per_image_ms = elapsed_ms / n
    for img, raw, pred in zip(images, raws, predictions, strict=True):
        entropy = prediction_entropy(pred.probabilities)
        # Prometheus first and outside the try: an in-memory counter cannot fail,
        # and the live signal must not be lost because feature extraction did.
        metrics.observe_prediction(
            endpoint=endpoint,
            predicted_class=pred.predicted_class,
            model_version=state.version or "unknown",
            confidence=pred.confidence,
            entropy=entropy,
            latency_seconds=per_image_ms / 1000.0,
        )
        try:
            features = extract_features(img, DATA_CFG)
            row = build_row(
                model_version=state.version or "unknown",
                endpoint=endpoint,
                source=source,
                predicted_class=pred.predicted_class,
                confidence=pred.confidence,
                entropy=entropy,
                latency_ms=per_image_ms,
                batch_size=n,
                n_bytes=len(raw),
                width=img.width,
                height=img.height,
                features=features,
            )
            predictions_log.log(row)
        except Exception as exc:  # feature extraction must never break serving
            print(f"[monitoring] row skipped ({type(exc).__name__}: {exc})")


# --- Endpoints ---------------------------------------------------------------

@app.get("/health", response_model=Health)
def health():
    """Liveness + readiness. Always 200; model_loaded reflects readiness."""
    return Health(
        status="ok",
        model_loaded=state.model is not None,
        model_version=state.version,
        tracking_uri=TRACKING_URI,
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    """Which model is served right now — the traceability endpoint."""
    state.require()  # 503 if degraded
    return ModelInfo(
        registry_model=MODEL_NAME,
        alias=ALIAS,
        model_version=state.version,
        num_classes=len(EUROSAT_CLASSES),
        classes=EUROSAT_CLASSES,
        image_size=DATA_CFG["image_size"],
        device=str(DEVICE),
        loaded_at=state.loaded_at,
        tracking_uri=TRACKING_URI,
    )


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...),
                  x_terraops_source: Optional[str] = Header(default=None)):
    """One image -> class + full probability vector + serving model version.

    The optional X-TerraOps-Source header tags the traffic in the prediction log
    (e.g. `ui`, `sim:cloud:0.4`). It exists so the drift simulator's synthetic
    load can be isolated from genuine uploads at query time — mixing them would
    corrupt the reference the CT loop reacts to.
    """
    with _counted("/predict"):
        state.require()
        raw = await file.read()
        images = _decode_all([raw])

        started = time.perf_counter()
        try:
            batch = preprocess_batch(images, DATA_CFG)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Cannot decode image: {exc}") from exc
        probs = _predict_tensor(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        prediction = _to_prediction(probs[0])
        _record(images, [raw], [prediction], endpoint="/predict",
                source=x_terraops_source, elapsed_ms=elapsed_ms)
        return prediction


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_batch(files: list[UploadFile] = File(...),
                        x_terraops_source: Optional[str] = Header(default=None)):
    """Many images -> many predictions in a single batched forward pass."""
    with _counted("/predict/batch"):
        state.require()
        raws = [await f.read() for f in files]
        images = _decode_all(raws)

        started = time.perf_counter()
        try:
            batch = preprocess_batch(images, DATA_CFG)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Cannot decode an image: {exc}") from exc
        probs = _predict_tensor(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        predictions = [_to_prediction(row) for row in probs]
        _record(images, raws, predictions, endpoint="/predict/batch",
                source=x_terraops_source, elapsed_ms=elapsed_ms)
        return BatchPrediction(predictions=predictions, model_version=state.version)


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint (text exposition format).

    Not a JSON response_model on purpose: Prometheus consumes its own text
    format, and wrapping it would break every scraper.
    """
    metrics.sync_log_gauges(predictions_log)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/monitoring/status", response_model=MonitoringStatus)
def monitoring_status():
    """Is the observability path itself healthy?

    Deliberately separate from /health: the API is healthy even when the log is
    down (serving is not affected), but a drift report computed over a period
    with dropped rows is not trustworthy — that has to be visible somewhere.
    """
    return MonitoringStatus(
        log_ready=predictions_log.ready,
        rows_written=predictions_log.written,
        rows_dropped=predictions_log.dropped,
        flush_failures=predictions_log.failed,
        queue_depth=predictions_log.queue_depth,
        last_error=predictions_log.last_error,
    )


@app.post("/reload", response_model=ReloadResult)
def reload():
    """Re-resolve @champion and hot-swap weights if the version changed.

    This is what makes 'promote a new champion -> API serves it, zero code change'
    true without a restart. No-op (reloaded=false) if the alias still points to the
    version already in memory.
    """
    try:
        report = state.load()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Reload failed — registry unreachable or no champion: {exc}",
        ) from exc
    return ReloadResult(**report)
