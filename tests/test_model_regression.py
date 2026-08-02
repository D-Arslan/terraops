"""Model non-regression tests — the CI-side replay of the promotion gate.

Unlike a unit test, these can go red while the code is 100% correct: they assert a
STATISTICAL PROPERTY of the served champion, not a deterministic output. A red here
means "the behavior of what we serve changed unacceptably" — a genuinely better
model that regressed one class, a dependency bump that introduced skew, or a drifted
dataset. That is the point.

All tests auto-skip (never fail) when the MLflow stack or DVC data is absent — see
the fixtures in conftest.py.
"""

import io
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from preprocessing import build_eval_transform, decode_image


def _predict_pil(model, device, transform, pil: Image.Image) -> int:
    with torch.no_grad():
        t = transform(pil).unsqueeze(0).to(device)
        return int(model(t).argmax(1).item())


def test_global_accuracy_no_regression(champion_scores, champion, params):
    """Global accuracy holds its absolute floor AND hasn't drifted below the
    baseline the champion was gated at (skew tripwire)."""
    y_true, y_pred = champion_scores["y_true"], champion_scores["y_pred"]
    acc = float((y_pred == y_true).mean())

    floor = params["promote"]["min_accuracy"]
    assert acc >= floor, f"accuracy {acc:.4f} below absolute floor {floor}"

    baseline = champion["baseline_acc"]
    if baseline is not None:
        tol = params["nonreg"]["regression_tolerance"]
        assert acc >= baseline - tol, (
            f"REGRESSION: accuracy {acc:.4f} dropped more than {tol} below the "
            f"gated baseline {baseline:.4f} — suspect train/serving skew")


def test_per_class_recall_floor(champion_scores, params):
    """No single class may collapse — the check a global average would hide."""
    y_true, y_pred = champion_scores["y_true"], champion_scores["y_pred"]
    classes = champion_scores["classes"]
    floor = params["nonreg"]["min_class_recall"]

    below = {}
    for c, name in enumerate(classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recall = float((y_pred[mask] == c).mean())
        if recall < floor:
            below[name] = round(recall, 4)
    assert not below, f"classes below recall floor {floor}: {below}"


def test_hflip_invariance(raw_frozen_pils, champion, data_cfg, params):
    """EuroSAT is invariant to horizontal flip (it's in the train augmentation),
    so the champion's prediction should almost never change under a flip."""
    model, device = champion["model"], champion["device"]
    transform = build_eval_transform(data_cfg)
    sample = raw_frozen_pils[:params["nonreg"]["invariance_sample"]]

    agree = sum(
        _predict_pil(model, device, transform, pil)
        == _predict_pil(model, device, transform, ImageOps.mirror(pil))
        for pil in sample
    )
    rate = agree / len(sample)
    assert rate >= params["nonreg"]["invariance_agreement"], (
        f"hflip invariance {rate:.3f} < {params['nonreg']['invariance_agreement']}")


def test_jpeg_recompression_invariance(raw_frozen_pils, champion, data_cfg, params):
    """A light JPEG re-encode (what an upload pipeline does) must not flip the class."""
    model, device = champion["model"], champion["device"]
    transform = build_eval_transform(data_cfg)
    sample = raw_frozen_pils[:params["nonreg"]["invariance_sample"]]

    agree = 0
    for pil in sample:
        base = _predict_pil(model, device, transform, pil)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        recompressed = decode_image(buf.getvalue())
        if _predict_pil(model, device, transform, recompressed) == base:
            agree += 1
    rate = agree / len(sample)
    assert rate >= params["nonreg"]["invariance_agreement"], (
        f"JPEG invariance {rate:.3f} < {params['nonreg']['invariance_agreement']}")


def test_single_image_latency_budget(raw_frozen_pils, champion, data_cfg, params):
    """p95 latency of a batch-of-1 forward pass stays within the CPU budget."""
    model, device = champion["model"], champion["device"]
    transform = build_eval_transform(data_cfg)
    sample = raw_frozen_pils[:params["nonreg"]["latency_sample"]]

    tensors = [transform(pil).unsqueeze(0).to(device) for pil in sample]
    with torch.no_grad():
        for t in tensors[:3]:              # warm up (lazy init, cache)
            model(t)
        times_ms = []
        for t in tensors:
            t0 = time.perf_counter()
            model(t)
            times_ms.append((time.perf_counter() - t0) * 1000)

    p95 = float(np.percentile(times_ms, 95))
    budget = params["nonreg"]["max_latency_ms"]
    assert p95 <= budget, f"p95 latency {p95:.1f} ms exceeds budget {budget} ms"
