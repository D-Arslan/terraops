"""Promotion gate: challenger vs @champion on the FROZEN validation set.

Usage:
    python src/promote.py --run-id <MLFLOW_RUN_ID>   # register the run's model
                                                     # as a new version, then gate it
    python src/promote.py --version <N>              # gate an existing registry version

The gate REFUSES promotion (exit code 1) unless ALL rules pass:
  1. absolute floor:  accuracy >= promote.min_accuracy
                      (also the bootstrap rule when no champion exists yet)
  2. margin:          accuracy beats the champion by >= promote.min_delta
                      — a margin above run-to-run noise, not "greater by a hair"
  3. no class regression: no class loses more than promote.max_class_recall_drop
                      of recall vs the champion (a global average can improve
                      while a minority class collapses)

A refused candidate STAYS in the registry as a documented, versioned attempt —
refusal is a recorded decision, not a deletion. Inference cost (ms/image) is
reported for visibility but is not (yet) a blocking rule.
"""

import argparse
import json
import os
import sys
import time

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from dataset import EUROSAT_CLASSES, get_transforms
from utils import REPO_ROOT, get_device, get_dvc_data_hash, load_params

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
CHAMPION_ALIAS = "champion"


def load_frozen_set(params):
    """Build the gate's DataLoader from the committed frozen-indices file."""
    path = REPO_ROOT / params["promote"]["frozen_val_path"]
    if not path.exists():
        sys.exit(f"GATE ERROR: {path} not found. Run `python src/freeze_val.py` "
                 f"once and commit the file.")
    with open(path, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    current_hash = get_dvc_data_hash()
    if frozen["dvc_data_hash"] != current_hash:
        sys.exit(
            "GATE ERROR: the dataset has changed since the frozen set was built\n"
            f"  frozen set built on: {frozen['dvc_data_hash']}\n"
            f"  current dvc.lock:    {current_hash}\n"
            "Frozen indices into different data are meaningless. Rebuild the "
            "frozen set DELIBERATELY (freeze_val.py) and re-baseline the champion."
        )

    full = datasets.EuroSAT(
        root=params["prepare"]["data_dir"], download=False,
        transform=get_transforms(params["data"], train=False),  # eval transforms: NO augmentation
    )
    indices = frozen["indices"]
    y_true = np.array([full.targets[i] for i in indices])
    loader = DataLoader(
        Subset(full, indices),
        batch_size=params["evaluate"]["batch_size"], shuffle=False,
        num_workers=params["data"]["num_workers"],
    )
    return loader, y_true, frozen


@torch.no_grad()
def score(model, loader, device):
    """Predictions + mean inference latency on the frozen set."""
    model.eval()
    preds = []
    n = 0
    t0 = time.perf_counter()
    for images, _ in loader:
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        n += len(images)
    ms_per_image = (time.perf_counter() - t0) / n * 1000
    return np.array(preds), ms_per_image


def per_class_recall(y_true, y_pred):
    return {
        cls: float((y_pred[y_true == c] == c).mean())
        for c, cls in enumerate(EUROSAT_CLASSES)
    }


def main():
    parser = argparse.ArgumentParser(description="Champion/challenger promotion gate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="MLflow run to register as a new version, then gate")
    group.add_argument("--version", help="Existing registry version to gate")
    args = parser.parse_args()

    params = load_params()
    pcfg = params["promote"]
    name = pcfg["registry_model"]

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # --- Resolve candidate ---------------------------------------------------
    if args.run_id:
        mv = mlflow.register_model(f"runs:/{args.run_id}/model", name)
        version = mv.version
        print(f"Registered run {args.run_id} as {name} v{version}")
    else:
        version = args.version

    # --- Resolve current champion (may not exist: bootstrap case) ------------
    try:
        champ_mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        champ_mv = None

    if champ_mv is not None and champ_mv.version == str(version):
        sys.exit(f"{name} v{version} is already the champion. Nothing to do.")

    device = get_device()
    loader, y_true, frozen = load_frozen_set(params)
    print(f"Frozen validation set: {frozen['n_samples']} samples "
          f"(data {frozen['dvc_data_hash'][:12]}...)")

    print(f"\nScoring candidate {name} v{version}...")
    candidate = mlflow.pytorch.load_model(f"models:/{name}/{version}").to(device)
    cand_pred, cand_ms = score(candidate, loader, device)
    cand_acc = float((cand_pred == y_true).mean())
    cand_recall = per_class_recall(y_true, cand_pred)

    champ_acc, champ_recall, champ_ms = None, None, None
    if champ_mv is not None:
        print(f"Scoring champion {name} v{champ_mv.version}...")
        champion = mlflow.pytorch.load_model(f"models:/{name}@{CHAMPION_ALIAS}").to(device)
        champ_pred, champ_ms = score(champion, loader, device)
        champ_acc = float((champ_pred == y_true).mean())
        champ_recall = per_class_recall(y_true, champ_pred)

    # --- Report --------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"{'':24}{'CANDIDATE':>12}{'CHAMPION':>12}{'DELTA':>12}")
    print("=" * 68)
    def fmt(v):
        """Right-aligned metric, or an em dash when there is no champion yet."""
        return f"{v:>12.4f}" if v is not None else f"{'—':>12}"

    delta_acc = None if champ_acc is None else cand_acc - champ_acc
    print(f"{'accuracy':24}{fmt(cand_acc)}{fmt(champ_acc)}{fmt(delta_acc)}")
    print(f"{'ms / image':24}{fmt(cand_ms)}{fmt(champ_ms)}"
          f"{fmt(None if champ_ms is None else cand_ms - champ_ms)}")
    print("-" * 68)
    for cls in EUROSAT_CLASSES:
        c_r = cand_recall[cls]
        ch_r = None if champ_recall is None else champ_recall[cls]
        d = None if ch_r is None else c_r - ch_r
        print(f"{'recall ' + cls:24}{fmt(c_r)}{fmt(ch_r)}{fmt(d)}")
    print("=" * 68)

    # --- Decision ------------------------------------------------------------
    reasons = []
    if cand_acc < pcfg["min_accuracy"]:
        reasons.append(f"accuracy {cand_acc:.4f} < absolute floor "
                       f"{pcfg['min_accuracy']:.4f}")
    if champ_acc is not None:
        if delta_acc < pcfg["min_delta"]:
            reasons.append(
                f"margin {delta_acc:+.4f} < required {pcfg['min_delta']:.4f} "
                f"(a win inside the noise band is not a win)")
        for cls in EUROSAT_CLASSES:
            drop = champ_recall[cls] - cand_recall[cls]
            if drop > pcfg["max_class_recall_drop"]:
                reasons.append(
                    f"recall regression on {cls}: -{drop:.4f} "
                    f"(max allowed {pcfg['max_class_recall_drop']:.4f})")

    # Record the gate's verdict on the version itself — auditable either way.
    client.set_model_version_tag(name, version, "gate_accuracy", f"{cand_acc:.4f}")
    client.set_model_version_tag(name, version, "gate_ms_per_image", f"{cand_ms:.2f}")
    client.set_model_version_tag(name, version, "gate_data_hash", frozen["dvc_data_hash"])

    if reasons:
        client.set_model_version_tag(name, version, "gate_result", "refused")
        print("\nPROMOTION REFUSED:")
        for r in reasons:
            print(f"  - {r}")
        print(f"\n{name} v{version} stays registered (documented attempt), "
              f"champion unchanged.")
        sys.exit(1)

    client.set_model_version_tag(name, version, "gate_result", "promoted")
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    if champ_mv is None:
        print(f"\nPROMOTED (bootstrap): {name} v{version} is the first champion.")
    else:
        print(f"\nPROMOTED: {name} v{version} replaces v{champ_mv.version} "
              f"as @{CHAMPION_ALIAS}.")


if __name__ == "__main__":
    main()
