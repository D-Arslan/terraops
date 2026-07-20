"""Utility functions for reproducibility and logging."""

import os
import random
import logging
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import torch


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
