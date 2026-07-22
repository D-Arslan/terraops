"""Generate the FROZEN validation set for the promotion gate.

Run ONCE, commit the output JSON to Git. Every future champion/challenger duel
is judged on exactly these samples — models may change their seed, split code,
or augmentation, but the referee's ground never moves. The file records the DVC
data hash it was built from, so promote.py can refuse to judge if the
underlying dataset has changed (indices into different data are meaningless).
"""

import json

import torch
from torch.utils.data import random_split
from torchvision import datasets

from utils import load_params, get_dvc_data_hash, get_git_commit, REPO_ROOT


def main():
    params = load_params()
    seed = params["seed"]
    data_cfg = params["data"]
    val_split, test_split = data_cfg["val_split"], data_cfg["test_split"]

    # Same split arithmetic + same seeded generator as dataset.load_eurosat:
    # this freezes the val partition the Sprint-1 baseline was selected on.
    full = datasets.EuroSAT(root=params["prepare"]["data_dir"], download=False)
    n_total = len(full)
    n_train = int((1 - val_split - test_split) * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    _, val_set, _ = random_split(full, [n_train, n_val, n_test], generator=generator)

    commit, dirty = get_git_commit()
    payload = {
        "description": "Frozen validation set for the promotion gate (promote.py)",
        "dvc_data_hash": get_dvc_data_hash(),
        "built_from_commit": commit,
        "built_from_dirty_tree": dirty,
        "seed": seed,
        "val_split": val_split,
        "n_samples": len(val_set.indices),
        "indices": sorted(val_set.indices),
    }

    out = REPO_ROOT / params["promote"]["frozen_val_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Frozen validation set: {payload['n_samples']} samples -> {out}")
    print("Commit this file. It must never be regenerated casually: doing so "
          "invalidates every past gate decision.")


if __name__ == "__main__":
    main()
