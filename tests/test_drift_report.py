"""Unit tests for the drift computation — no database, no Evidently service.

These pin the behaviours the CT loop depends on. A retraining pipeline fires on
`dataset_drift`, so that boolean must be right for the right reason: it must
stay False on identical populations (or the loop retrains on noise), it must go
True on a real shift, and it must be UNSET when the evidence is too thin (or
"quiet" gets confused with "healthy").

Evidently is imported lazily inside compute_drift, so these tests skip cleanly
in an environment that only installed the serving requirements.
"""

import numpy as np
import pytest

from drift_report import compute_drift, rows_to_frame, summarize_predictions
from image_features import FEATURE_NAMES
from utils import load_params

pytest.importorskip("evidently", reason="monitoring extras not installed")


@pytest.fixture(scope="module")
def params():
    return load_params()


def _rows(n: int, seed: int, shift: dict = None) -> list:
    """Synthetic feature rows: independent gaussians, optionally shifted.

    Gaussian rather than real image features on purpose — these tests are about
    the drift MACHINERY (thresholds, shares, refusal to conclude), and real
    images would make them slow and couple them to the dataset.
    """
    rng = np.random.default_rng(seed)
    shift = shift or {}
    return [
        {name: float(rng.normal(0.5 + shift.get(name, 0.0), 0.1))
         for name in FEATURE_NAMES}
        for _ in range(n)
    ]


# --- frame alignment -----------------------------------------------------------

def test_frame_is_ordered_by_feature_names():
    """Column order must come from FEATURE_NAMES, not from dict insertion order:
    a misaligned frame would compare brightness against sharpness."""
    rows = [{name: float(i) for i, name in enumerate(reversed(FEATURE_NAMES))}]
    frame = rows_to_frame(rows)
    assert list(frame.columns) == list(FEATURE_NAMES)


def test_missing_feature_column_is_a_loud_failure():
    """Silently dropping a monitored dimension would shrink the denominator of
    the drift share and make drift look less widespread than it is."""
    rows = [{name: 0.5 for name in FEATURE_NAMES[:-1]}]
    with pytest.raises(ValueError):
        rows_to_frame(rows)


# --- the decision bit ----------------------------------------------------------

def test_identical_populations_are_not_flagged(params):
    """The false-positive guard: same distribution, different sample -> quiet.

    If this ever fails, the CT loop retrains on sampling noise — the single most
    expensive failure mode of an automated retraining pipeline.
    """
    summary, _ = compute_drift(_rows(1500, seed=1), _rows(1500, seed=2), params)
    assert summary["status"] == "ok"
    assert summary["dataset_drift"] is False
    assert summary["drifted_share"] < params["monitor"]["drift_share_threshold"]


def test_a_broad_shift_is_flagged(params):
    """Every feature displaced by several reference standard deviations."""
    shift = {name: 0.5 for name in FEATURE_NAMES}       # 5 sigma
    summary, _ = compute_drift(_rows(1500, seed=1),
                               _rows(1500, seed=2, shift=shift), params)
    assert summary["dataset_drift"] is True
    assert summary["drifted_share"] == 1.0


def test_a_single_drifted_feature_does_not_trip_the_dataset_verdict(params):
    """One column moving is weather; the share threshold is what makes the
    trigger require a regime change rather than a noisy dimension."""
    summary, _ = compute_drift(_rows(1500, seed=1),
                               _rows(1500, seed=2, shift={"brightness": 0.5}),
                               params)
    assert summary["features"]["brightness"]["drifted"] is True
    assert summary["dataset_drift"] is False


def test_the_most_shifted_feature_is_ranked_first(params):
    """top_drifted is what a human reads first during an incident; it must point
    at the actual culprit, not at an arbitrary column."""
    shift = {"sharpness": 0.8, "brightness": 0.3}
    summary, _ = compute_drift(_rows(1500, seed=1),
                               _rows(1500, seed=2, shift=shift), params)
    assert summary["top_drifted"][0]["feature"] == "sharpness"


def test_distance_and_flag_cannot_disagree(params):
    """The boolean is recomputed from the same threshold Evidently was given."""
    summary, _ = compute_drift(_rows(1500, seed=1),
                               _rows(1500, seed=2, shift={"contrast": 0.6}),
                               params)
    threshold = summary["stattest_threshold"]
    for values in summary["features"].values():
        assert values["drifted"] == (values["distance"] > threshold)


# --- refusing to conclude ------------------------------------------------------

def test_thin_traffic_returns_insufficient_data_not_no_drift(params):
    """'Not enough evidence' must be distinguishable from 'everything is fine'.

    Collapsing the two would let a dead traffic source read as a healthy system
    — the silent-failure shape this whole sprint is built to avoid.
    """
    summary, snapshot = compute_drift(_rows(1500, seed=1), _rows(10, seed=2), params)
    assert summary["status"] == "insufficient_data"
    assert summary["dataset_drift"] is False
    assert snapshot is None


# --- model-side signals are labelled as non-evidence ---------------------------

def test_prediction_summary_states_it_is_not_ground_truth():
    """The wording is load-bearing: predicted-class share is not a label
    distribution, and a reader must not treat it as one."""
    rows = [{"predicted_class": "Forest", "confidence": 0.9, "entropy": 0.1,
             "latency_ms": 30.0, "model_version": "1"},
            {"predicted_class": "River", "confidence": 0.5, "entropy": 0.6,
             "latency_ms": 40.0, "model_version": "1"}]
    summary = summarize_predictions(rows)
    assert "NOT ground-truth" in summary["note"]
    assert summary["predicted_class_share"]["Forest"] == pytest.approx(0.5)
    assert summary["mean_confidence"] == pytest.approx(0.7)
    assert summary["model_versions"] == ["1"]
