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
"""

import os
import threading
import time
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import mlflow
from mlflow import MlflowClient

from dataset import EUROSAT_CLASSES
from preprocessing import preprocess_batch
from utils import load_params, get_device


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
                       for cls, p in zip(EUROSAT_CLASSES, probs_row)},
        model_version=state.version,
    )


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
async def predict(file: UploadFile = File(...)):
    """One image -> class + full probability vector + serving model version.

    Bytes go straight into the shared preprocessing module — the API never decodes
    the image itself, so it cannot introduce channel/interpolation skew.
    """
    state.require()
    raw = await file.read()
    try:
        batch = preprocess_batch([raw], DATA_CFG)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")
    probs = _predict_tensor(batch)
    return _to_prediction(probs[0])


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_batch(files: list[UploadFile] = File(...)):
    """Many images -> many predictions in a single batched forward pass."""
    state.require()
    raws = [await f.read() for f in files]
    try:
        batch = preprocess_batch(raws, DATA_CFG)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode an image: {exc}")
    probs = _predict_tensor(batch)
    return BatchPrediction(
        predictions=[_to_prediction(row) for row in probs],
        model_version=state.version,
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
        )
    return ReloadResult(**report)
