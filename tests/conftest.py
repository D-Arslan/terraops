"""Shared pytest fixtures for the TerraOps test suite.

Two tiers of tests live here:
  - UNIT tests (preprocessing, API contract): pure, fast, no external state. Always run.
  - NON-REGRESSION tests: exercise the SERVED champion (models:/...@champion) on the
    frozen set. They need the MLflow stack up + the DVC data materialized, so they
    AUTO-SKIP (never fail) when those are absent — CI without the stack stays green,
    and the safety net engages exactly where it can.

The frozen set and thresholds come from the same params.yaml the promotion gate uses,
so these tests are the pytest replay of promote.py's spirit, not a parallel truth.
"""

import json
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
# Same flat-import convention the scripts use (python src/train.py puts src/ on path).
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import load_params  # noqa: E402

# --- Config --------------------------------------------------------------------

@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def data_cfg(params):
    return params["data"]


@pytest.fixture(scope="session")
def tracking_uri():
    import os
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# --- Availability gates (turn missing infra into skips, not failures) ----------

def _mlflow_up(uri: str) -> bool:
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_mlflow(tracking_uri):
    if not _mlflow_up(tracking_uri):
        pytest.skip(f"MLflow not reachable at {tracking_uri} — non-regression tests skipped")


@pytest.fixture(scope="session")
def frozen(params):
    """The committed frozen validation set, with the same data-hash guard as the gate."""
    path = REPO_ROOT / params["promote"]["frozen_val_path"]
    if not path.exists():
        pytest.skip(f"frozen set {path} missing")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from utils import get_dvc_data_hash
    try:
        current = get_dvc_data_hash()
    except Exception as exc:
        pytest.skip(f"cannot read dvc.lock: {exc}")
    if data["dvc_data_hash"] != current:
        pytest.skip("dataset changed vs frozen set (dvc_data_hash mismatch) — "
                    "indices into different data are meaningless")
    return data


@pytest.fixture(scope="session")
def eurosat_raw(params):
    """The EuroSAT dataset with NO transform: yields raw PIL images (for invariance)."""
    from torchvision import datasets
    root = params["prepare"]["data_dir"]
    try:
        return datasets.EuroSAT(root=root, download=False, transform=None)
    except Exception as exc:
        pytest.skip(f"EuroSAT data not materialized ({exc}); run `dvc pull`")


# --- The served champion + its behavior on the frozen set ----------------------

@pytest.fixture(scope="session")
def champion(require_mlflow, tracking_uri, params):
    """Load the model the API would serve: models:/<name>@champion, by alias.

    Records the accuracy it was GATED at (its gate_accuracy tag) as the regression
    baseline — the non-regression test checks the champion hasn't drifted below it.
    """
    import mlflow
    from mlflow import MlflowClient

    from utils import get_device

    mlflow.set_tracking_uri(tracking_uri)
    name = params["promote"]["registry_model"]
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except Exception as exc:
        pytest.skip(f"no @champion in registry: {exc}")

    device = get_device()
    model = mlflow.pytorch.load_model(f"models:/{name}/{mv.version}").to(device).eval()
    baseline = mv.tags.get("gate_accuracy")
    return {
        "model": model,
        "version": mv.version,
        "baseline_acc": float(baseline) if baseline else None,
        "device": device,
    }


@pytest.fixture(scope="session")
def champion_scores(champion, frozen, params, data_cfg):
    """Score the champion over the FULL frozen set once; reuse across assertions."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    from dataset import EUROSAT_CLASSES
    from preprocessing import build_eval_transform

    # A transformed view of the data (shared eval transform — the serving path).
    full = datasets.EuroSAT(
        root=params["prepare"]["data_dir"], download=False,
        transform=build_eval_transform(data_cfg),
    )
    indices = frozen["indices"]
    y_true = np.array([full.targets[i] for i in indices])
    loader = DataLoader(Subset(full, indices),
                        batch_size=params["evaluate"]["batch_size"],
                        shuffle=False, num_workers=0)

    model, device = champion["model"], champion["device"]
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            preds.append(model(images.to(device)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return {"y_true": y_true, "y_pred": y_pred, "classes": EUROSAT_CLASSES}


@pytest.fixture(scope="session")
def raw_frozen_pils(eurosat_raw, frozen, params):
    """A spread-out sample of raw PIL images from the frozen indices.

    Evenly spaced (not random) so the sample covers many classes deterministically,
    and so the run is reproducible without seeding.
    """
    indices = frozen["indices"]
    n = max(params["nonreg"]["invariance_sample"], params["nonreg"]["latency_sample"])
    step = max(1, len(indices) // n)
    picked = indices[::step][:n]
    return [eurosat_raw[i][0] for i in picked]  # (PIL, label) -> PIL
