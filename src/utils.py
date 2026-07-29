"""Utility functions for reproducibility, logging, and lineage."""

import os
import random
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_params(path: str = None) -> dict:
    """Load hyperparameters from params.yaml (the single source of truth).

    Resolves params.yaml at the repo root (parent of src/) regardless of the
    current working directory, so the same call works whether a script runs
    from the repo root (as DVC does) or from src/ during manual debugging.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Pipeline OUTPUTS: rewritten during `dvc repro` itself, so they are always
# "modified" while a stage runs. Only INPUT drift (code, params) invalidates
# lineage — flagging outputs would make git_dirty fire on every legitimate run.
_DIRTY_IGNORED = ("dvc.lock", "metrics/")


def get_git_commit() -> tuple:
    """Return (commit_sha, is_dirty) for MLflow lineage tags.

    is_dirty=True means an INPUT (code, params) differs from the commit, so the
    sha does NOT fully identify what ran — the run must be flagged.
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).splitlines()
    # porcelain format: "XY path" (or "XY old -> new" for renames)
    dirty_paths = [line[3:].split(" -> ")[-1] for line in status if line.strip()]
    dirty = any(not p.startswith(_DIRTY_IGNORED) for p in dirty_paths)
    return sha, dirty


def get_dvc_data_hash(stage: str = "prepare", out_path: str = "data/raw") -> str:
    """Read the dataset's content hash from dvc.lock (not recomputed: hashing
    27k files on every run would be slow — dvc.lock IS the frozen record,
    and it is itself committed, so git_dirty covers a stale lock).
    """
    with open(REPO_ROOT / "dvc.lock", "r", encoding="utf-8") as f:
        lock = yaml.safe_load(f)
    for out in lock["stages"][stage]["outs"]:
        if out["path"] == out_path:
            return out["md5"]
    raise KeyError(f"{out_path} not found in dvc.lock stage '{stage}'")


def flatten_params(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict for mlflow.log_params: {'train': {'lr': 1e-3}}
    -> {'train.lr': 1e-3}. Lists (e.g. norm_mean) are logged as strings."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_params(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def set_seed(seed: int = 42):
    """Set seeds for reproducibility across all libraries.

    Remaining nondeterminism: CUDA convolution algorithms may vary
    across GPU architectures even with deterministic mode enabled.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("eurosat")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
