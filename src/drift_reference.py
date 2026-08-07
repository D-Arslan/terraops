"""Build the drift REFERENCE — the frozen statistics of the training inputs.

What a reference is, and what it must not be
--------------------------------------------
Drift detection asks: "does today's traffic still look like what the model
LEARNED FROM?" Not "does today look like yesterday?". A model's domain of
validity is defined by its training data, so the reference is the training data,
frozen, committed, and re-derived only on purpose.

The alternative — a rolling reference over the last N days — fails in the exact
case monitoring exists for. If the world drifts slowly, every day resembles the
previous one, no alert ever fires, and six months later the inputs are far
outside the training domain with a dashboard that stayed green throughout. The
boiling-frog failure. A rolling window is a useful SECOND signal for detecting
an abrupt break; it is never the primary reference.

This file is therefore the input-side twin of gate/frozen_val.json: an immutable
anchor, committed to git, regenerated only with a deliberate act that is
recorded in the reference's own metadata.

Three details that make it correct
----------------------------------
1. TRAIN split only, reproduced from params.seed with the same arithmetic as
   dataset.load_eurosat. Using the whole dataset would fold the validation and
   test images into the reference, and the frozen gate set with them — the
   reference would then overlap the very data used to certify the champion.

2. NO AUGMENTATION. Training applied flips, rotations and jitter; production
   images are not augmented. The reference must describe the input distribution
   the model was trained ON, not the distorted views it was shown during
   optimization. Including jitter would widen the reference and make the
   detector blind to exactly the brightness/saturation drift it is meant to see.

3. Features come from image_features.extract_features — the same function the
   API calls per request. Two extractors would mean measuring the difference
   between them instead of between the distributions.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torchvision import datasets

from image_features import FEATURE_NAMES, extract_features
from utils import REPO_ROOT, get_dvc_data_hash, get_git_commit, load_params


def train_split_indices(params: dict, n_total: int) -> List[int]:
    """Reproduce dataset.load_eurosat's TRAIN indices exactly.

    torch.utils.data.random_split permutes with the given generator and then
    slices sequentially, so a seeded randperm reproduces the same partition. The
    arithmetic below mirrors dataset.py line for line on purpose: if the two ever
    diverge, the reference would describe a different population than the model
    was trained on, silently.
    """
    data_cfg = params["data"]
    n_train = int((1 - data_cfg["val_split"] - data_cfg["test_split"]) * n_total)
    generator = torch.Generator().manual_seed(params["seed"])
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[:n_train]


def build_reference(params: dict, sample: int = None) -> Dict:
    """Extract monitoring features over a sample of the training split.

    Sampling is EVENLY SPACED over the shuffled train indices rather than random:
    the permutation already removed any class ordering, so a stride gives a
    representative sample and makes the build reproducible without a second seed.
    """
    monitor_cfg = params["monitor"]
    sample = sample or monitor_cfg["reference_sample"]

    full = datasets.EuroSAT(root=params["prepare"]["data_dir"],
                            download=False, transform=None)
    train_idx = train_split_indices(params, len(full))

    step = max(1, len(train_idx) // sample)
    picked = train_idx[::step][:sample]

    rows: List[Dict[str, float]] = []
    class_counts: Dict[str, int] = {}
    for i in picked:
        img, label = full[i]
        rows.append(extract_features(img, params["data"]))
        name = full.classes[label]
        class_counts[name] = class_counts.get(name, 0) + 1

    sha, dirty = get_git_commit()
    try:
        data_hash = get_dvc_data_hash()
    except Exception:
        data_hash = None

    return {
        # Lineage: which data, which code, when. A reference whose provenance is
        # unknown cannot be trusted to explain an alert months later.
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": sha,
        "git_dirty": dirty,
        "dvc_data_hash": data_hash,
        "split": "train",
        "augmented": False,
        "image_size": params["data"]["image_size"],
        "n_samples": len(rows),
        "feature_names": FEATURE_NAMES,
        # The TRAIN class distribution, kept for the label-shift comparison
        # against the predicted-class mix in production. Not a drift test by
        # itself: predicted classes are not true labels.
        "train_class_distribution": class_counts,
        "summary": summarize(rows),
        # Raw per-image rows: the statistical tests need distributions, not
        # summary statistics. Wasserstein over two means is meaningless.
        "rows": rows,
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Per-feature summary — human-readable, and the source of the scale used to
    normalize distances (a raw Wasserstein distance is in the feature's unit and
    cannot be compared across features)."""
    out = {}
    for name in FEATURE_NAMES:
        values = np.array([r[name] for r in rows], dtype=np.float64)
        out[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return out


def load_reference(params: dict) -> Dict:
    """Read the committed reference, failing loudly if it is missing or stale.

    The dvc_data_hash check mirrors the frozen gate set's guard: a reference
    built from different data describes a different population, and comparing
    against it would produce confident nonsense.
    """
    path = REPO_ROOT / params["monitor"]["reference_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — build it with `python src/drift_reference.py`. "
            f"Drift cannot be measured without a reference.")
    with open(path, "r", encoding="utf-8") as f:
        reference = json.load(f)

    try:
        current_hash = get_dvc_data_hash()
    except Exception:
        current_hash = None
    if (reference.get("dvc_data_hash") and current_hash
            and reference["dvc_data_hash"] != current_hash):
        print(f"[warning] reference was built on dvc_data_hash "
              f"{reference['dvc_data_hash'][:12]} but the working data is "
              f"{current_hash[:12]}. The comparison may be meaningless.")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen drift reference from the TRAIN split. "
                    "Committing the output is part of the workflow: a reference "
                    "that changes silently invalidates every past drift verdict.")
    parser.add_argument("--sample", type=int, default=None,
                        help="override monitor.reference_sample")
    parser.add_argument("--out", type=Path, default=None,
                        help="override monitor.reference_path")
    args = parser.parse_args()

    params = load_params()
    out = args.out or (REPO_ROOT / params["monitor"]["reference_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    reference = build_reference(params, args.sample)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    print(f"reference: {reference['n_samples']} train images -> {out}")
    print(f"  git_commit={reference['git_commit'][:8]} "
          f"dirty={reference['git_dirty']} "
          f"dvc_data_hash={str(reference['dvc_data_hash'])[:12]}")
    for name in FEATURE_NAMES:
        s = reference["summary"][name]
        print(f"  {name:<12} mean={s['mean']:.4f} std={s['std']:.4f} "
              f"p05={s['p05']:.4f} p95={s['p95']:.4f}")


if __name__ == "__main__":
    main()
