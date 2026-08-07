"""The sprint's central experiment: does drift detection fire BEFORE accuracy falls?

The question
------------
Production never shows accuracy — there are no labels. All it shows is the input
distribution. Every monitoring stack therefore bets that a shift in the inputs is
a usable proxy for a loss of performance. That bet is almost never verified; the
threshold is copied from a blog post and the dashboard is trusted because it is
green.

Here it can be verified, because the drift is simulated and the labels are known.
For each perturbation and each intensity this script measures both curves on the
SAME images:

    intensity -> accuracy of the served champion    (unknowable in production)
    intensity -> drift verdict from the reference   (all production ever sees)

and reports the gap between the intensity at which the detector fires and the
intensity at which the model breaks. Positive gap = early warning. Negative gap =
the monitoring is decorative, and saying so is worth more than hiding it.

Protocol decisions
------------------
* FROZEN GATE SET, not a fresh sample: the intensity-0 accuracy is then directly
  comparable to the number the champion was promoted with, which makes the
  baseline auditable instead of self-declared.
* The champion is loaded BY ALIAS from the registry — the experiment measures
  what is actually served, not a .pth someone left on disk.
* Perturbation is applied to the RAW PIL image, before the shared preprocessing.
  That is where real drift happens: in the world, not after the resize.
* Features are extracted by image_features and compared with drift_report's
  compute_drift — the same code the production report runs. An experiment that
  validated a different implementation would validate nothing.
* Offline, in-process: the sweep does NOT go through the API. 20k HTTP round
  trips would add hours and change no number; the API path is exercised by the
  live traffic generator instead. The consequence is stated rather than hidden:
  this measures the model and the detector, not the serving stack.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torchvision import datasets

import drift_sim
from drift_reference import load_reference
from drift_report import compute_drift
from image_features import extract_features
from preprocessing import preprocess_batch
from utils import REPO_ROOT, get_device, load_params


def load_champion(params: dict, tracking_uri: str):
    """Load models:/<name>@champion — the model production actually serves."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    name = params["promote"]["registry_model"]
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, "champion")
    model = mlflow.pytorch.load_model(f"models:/{name}/{version.version}")
    return model.to(get_device()).eval(), version


def frozen_sample(params: dict, sample: int) -> List[int]:
    """An evenly spaced subsample of the frozen gate set.

    Evenly spaced rather than random: the frozen indices are already shuffled, so
    a stride is representative, needs no extra seed, and makes the sweep
    reproducible by construction.
    """
    path = REPO_ROOT / params["promote"]["frozen_val_path"]
    with open(path, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    indices = frozen["indices"]
    step = max(1, len(indices) // sample)
    return indices[::step][:sample]


@torch.no_grad()
def evaluate_at(model, images, labels: np.ndarray, params: dict,
                batch_size: int) -> Dict:
    """Accuracy + mean confidence/entropy over a list of perturbed PIL images."""
    device = get_device()
    predictions, confidences, entropies = [], [], []

    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        batch = preprocess_batch(chunk, params["data"]).to(device)
        probs = torch.softmax(model(batch), dim=1).cpu()
        predictions.append(probs.argmax(1).numpy())
        confidences.append(probs.max(1).values.numpy())
        # Normalized entropy, same definition as the API's monitoring signal.
        safe = probs.clamp_min(1e-12)
        entropies.append((-(safe * safe.log()).sum(1) /
                          np.log(probs.shape[1])).numpy())

    y_pred = np.concatenate(predictions)
    counts = np.bincount(y_pred, minlength=10)
    return {
        "accuracy": float((y_pred == labels).mean()),
        "mean_confidence": float(np.concatenate(confidences).mean()),
        "mean_entropy": float(np.concatenate(entropies).mean()),
        # Prediction-side collapse: as inputs leave the training domain the model
        # tends to funnel everything into one class. This is measurable in
        # production WITHOUT labels (it is just the class mix Prometheus already
        # counts), which makes it the third candidate detector alongside input
        # drift and entropy — and the first run showed entropy going back DOWN at
        # extreme drift, so a third signal is not a luxury.
        "majority_class_share": float(counts.max() / len(y_pred)),
        "distinct_classes_predicted": int((counts > 0).sum()),
    }


def sweep(params: dict, tracking_uri: str, sample: Optional[int] = None,
          kinds: Optional[List[str]] = None) -> Dict:
    """Run the full intensity sweep and return the measurement record."""
    exp_cfg = params["experiment"]
    sample = sample or exp_cfg["sample"]
    kinds = kinds or exp_cfg["kinds"]
    intensities = exp_cfg["intensities"]
    batch_size = exp_cfg["batch_size"]

    # Fail fast rather than produce a curve that cannot mean anything: below
    # monitor.min_current_rows every window comes back "insufficient_data", and
    # a sweep of non-verdicts looks exactly like a sweep of "no drift" unless
    # something refuses up front. Learned the hard way — see learning.md.
    floor = int(params["monitor"]["min_current_rows"])
    if sample < floor:
        raise SystemExit(
            f"sample={sample} is below monitor.min_current_rows={floor}: every "
            f"drift window would be inconclusive and the curve would be "
            f"meaningless. Raise --sample or lower the floor deliberately.")

    reference = load_reference(params)
    model, version = load_champion(params, tracking_uri)

    raw = datasets.EuroSAT(root=params["prepare"]["data_dir"],
                           download=False, transform=None)
    indices = frozen_sample(params, sample)
    originals = [raw[i][0] for i in indices]
    labels = np.array([raw.targets[i] for i in indices])

    results: Dict[str, List[Dict]] = {}
    for kind in kinds:
        points = []
        for intensity in intensities:
            # index=i keeps each image's random perturbation tied to the image
            # rather than to its position in this particular run.
            perturbed = [drift_sim.apply_named(img, kind, float(intensity),
                                               params, index=i)
                         for i, img in zip(indices, originals, strict=True)]

            scores = evaluate_at(model, perturbed, labels, params, batch_size)
            feature_rows = [extract_features(img, params["data"])
                            for img in perturbed]
            drift, _ = compute_drift(reference["rows"], feature_rows, params)

            points.append({
                "intensity": float(intensity),
                **scores,
                "dataset_drift": bool(drift.get("dataset_drift")),
                "drifted_share": drift.get("drifted_share"),
                "max_distance": max(
                    (v["distance"] for v in drift.get("features", {}).values()),
                    default=0.0),
                "top_feature": (drift.get("top_drifted") or [{}])[0].get("feature"),
                "drift_status": drift.get("status"),
            })
            print(f"  {kind:<11} i={intensity:.2f} "
                  f"acc={scores['accuracy']:.4f} "
                  f"entropy={scores['mean_entropy']:.3f} "
                  f"drift={'YES' if points[-1]['dataset_drift'] else 'no ':<3} "
                  f"share={points[-1]['drifted_share']} "
                  f"top={points[-1]['top_feature']}")
        results[kind] = points

    baseline = results[kinds[0]][0]["accuracy"]     # intensity 0, any kind
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "champion_version": version.version,
        "champion_gate_accuracy": version.tags.get("gate_accuracy"),
        "n_images": len(indices),
        "baseline_accuracy": baseline,
        "intensities": list(intensities),
        "reference": {
            "created_at": reference.get("created_at"),
            "git_commit": reference.get("git_commit"),
            "n_samples": reference.get("n_samples"),
        },
        "stattest": params["monitor"]["stattest"],
        "stattest_threshold": params["monitor"]["stattest_threshold"],
        "drift_share_threshold": params["monitor"]["drift_share_threshold"],
        "accuracy_drop_tolerance": exp_cfg["accuracy_drop_tolerance"],
        "curves": results,
        "verdicts": [verdict(kind, points, baseline, exp_cfg)
                     for kind, points in results.items()],
    }


def verdict(kind: str, points: List[Dict], baseline: float,
            exp_cfg: dict) -> Dict:
    """Compare WHEN the detector fires against WHEN the model breaks.

    A window that could not be evaluated (fewer rows than
    monitor.min_current_rows) is NOT evidence of absence: if any point in the
    sweep is inconclusive, the whole verdict is inconclusive. Treating a missing
    measurement as "no drift" would manufacture a blind-spot finding out of a
    sample-size mistake — which is precisely what the first run of this script
    did before this guard existed.

    Both bounds are reported as the lowest tested intensity at which the
    condition holds — a lower bound on a grid, not a continuous crossing point.
    The lead is their difference, and its SIGN is the result that matters:

        lead > 0   the alert precedes the failure -> usable early warning
        lead == 0  they coincide -> the alert is a symptom, not a warning
        lead < 0   accuracy falls first -> the detector is blind to this
                   perturbation, and no threshold on these features fixes it
        detection None while failure exists -> the false-negative case, the one
                   worth reporting loudest
    """
    tolerance = exp_cfg["accuracy_drop_tolerance"]

    inconclusive = [p["intensity"] for p in points
                    if p.get("drift_status") != "ok"]
    if inconclusive:
        return {
            "kind": kind,
            "conclusive": False,
            "detection_intensity": None,
            "failure_intensity": None,
            "lead": None,
            "final_accuracy": points[-1]["accuracy"],
            "accuracy_drop": round(baseline - points[-1]["accuracy"], 4),
            "summary": (f"INCONCLUSIVE: drift could not be evaluated at "
                        f"intensities {inconclusive} (window below "
                        f"monitor.min_current_rows) — this is NOT evidence of "
                        f"absence of drift"),
        }

    detection = next((p["intensity"] for p in points if p["dataset_drift"]), None)
    failure = next((p["intensity"] for p in points
                    if p["accuracy"] < baseline - tolerance), None)

    lead = None
    if detection is not None and failure is not None:
        lead = round(failure - detection, 4)

    if detection is None and failure is None:
        summary = "neither fired within the tested range"
    elif detection is None:
        summary = "BLIND SPOT: accuracy collapsed with no drift alert"
    elif failure is None:
        summary = "alert fired while accuracy held (over-sensitive, or robust model)"
    elif lead > 0:
        summary = "early warning: drift detected before accuracy collapsed"
    elif lead == 0:
        summary = "simultaneous: the alert is a symptom, not a warning"
    else:
        summary = "LATE: accuracy collapsed before the alert fired"

    return {
        "kind": kind,
        "conclusive": True,
        "detection_intensity": detection,
        "failure_intensity": failure,
        "lead": lead,
        "final_accuracy": points[-1]["accuracy"],
        "accuracy_drop": round(baseline - points[-1]["accuracy"], 4),
        "summary": summary,
    }


def plot(record: Dict, out_path: Path) -> None:
    """One panel per perturbation: accuracy and drift share on a shared x-axis.

    Both curves on the same axes on purpose — the whole point is the horizontal
    distance between two vertical lines (alert fires / model breaks), and two
    separate figures would hide it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(record["curves"])
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    baseline = record["baseline_accuracy"]
    tolerance = record["accuracy_drop_tolerance"]

    for ax, kind in zip(axes, kinds, strict=True):
        points = record["curves"][kind]
        x = [p["intensity"] for p in points]
        ax.plot(x, [p["accuracy"] for p in points], "o-", color="#1f77b4",
                label="accuracy (needs labels)")
        ax.plot(x, [p["drifted_share"] for p in points], "s--", color="#d62728",
                label="drifted feature share")
        ax.plot(x, [p["mean_entropy"] for p in points], "^:", color="#7f7f7f",
                label="mean entropy", alpha=0.7)
        ax.plot(x, [p.get("majority_class_share") for p in points], "v-.",
                color="#2ca02c", label="majority predicted class", alpha=0.8)

        ax.axhline(record["drift_share_threshold"], color="#d62728", lw=0.8,
                   alpha=0.4)
        ax.axhline(baseline - tolerance, color="#1f77b4", lw=0.8, alpha=0.4)

        found = next((v for v in record["verdicts"] if v["kind"] == kind), {})
        if found.get("detection_intensity") is not None:
            ax.axvline(found["detection_intensity"], color="#d62728", lw=1.2,
                       ls="-", alpha=0.8)
        if found.get("failure_intensity") is not None:
            ax.axvline(found["failure_intensity"], color="#1f77b4", lw=1.2,
                       ls="-", alpha=0.8)

        ax.set_title(f"{kind}\nlead = {found.get('lead')}")
        ax.set_xlabel("perturbation intensity")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("value")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"Drift detection vs accuracy — champion v{record['champion_version']}, "
        f"{record['n_images']} frozen-set images "
        f"(vertical lines: red = alert fires, blue = accuracy collapses)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def log_to_mlflow(record: Dict, params: dict, tracking_uri: str,
                  artifacts: List[Path]) -> None:
    """Track the sweep like any other experiment — it IS one.

    The measurement that calibrates the CT threshold deserves the same lineage
    treatment as a training run: if the threshold is ever questioned, the run
    that produced it can be pulled up with its parameters and artifacts.
    """
    import mlflow

    from utils import get_git_commit

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment"]["mlflow_experiment"])
    sha, dirty = get_git_commit()

    with mlflow.start_run(run_name="drift-curve"):
        mlflow.set_tags({
            "git_commit": sha,
            "git_dirty": str(dirty),
            "champion_version": record["champion_version"],
            "reference_commit": record["reference"].get("git_commit"),
        })
        mlflow.log_params({
            "n_images": record["n_images"],
            "stattest": record["stattest"],
            "stattest_threshold": record["stattest_threshold"],
            "drift_share_threshold": record["drift_share_threshold"],
            "accuracy_drop_tolerance": record["accuracy_drop_tolerance"],
            "intensities": str(record["intensities"]),
        })
        mlflow.log_metric("baseline_accuracy", record["baseline_accuracy"])
        for item in record["verdicts"]:
            kind = item["kind"]
            if item["lead"] is not None:
                mlflow.log_metric(f"lead_{kind}", item["lead"])
            if item["detection_intensity"] is not None:
                mlflow.log_metric(f"detect_{kind}", item["detection_intensity"])
            if item["failure_intensity"] is not None:
                mlflow.log_metric(f"fail_{kind}", item["failure_intensity"])
            mlflow.log_metric(f"final_acc_{kind}", item["final_accuracy"])
        for path in artifacts:
            mlflow.log_artifact(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--kinds", nargs="*", default=None)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments" / "drift_curve")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    params = load_params()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    record = sweep(params, args.tracking_uri, args.sample, args.kinds)

    json_path = args.out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    figure_path = args.out_dir / "drift_curve.png"
    plot(record, figure_path)

    print(f"\nbaseline accuracy (intensity 0): {record['baseline_accuracy']:.4f} "
          f"| champion v{record['champion_version']} "
          f"gated at {record['champion_gate_accuracy']}")
    print(f"{'kind':<12}{'detect':>8}{'fail':>8}{'lead':>8}  verdict")
    for item in record["verdicts"]:
        print(f"{item['kind']:<12}"
              f"{str(item['detection_intensity']):>8}"
              f"{str(item['failure_intensity']):>8}"
              f"{str(item['lead']):>8}  {item['summary']}")
    print(f"\nresults -> {json_path}\nfigure  -> {figure_path}")

    if not args.no_mlflow:
        try:
            log_to_mlflow(record, params, args.tracking_uri,
                          [json_path, figure_path])
            print("logged to MLflow")
        except Exception as exc:
            print(f"[warning] MLflow logging failed ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
