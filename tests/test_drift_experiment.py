"""Unit tests for the experiment's VERDICT logic — no model, no registry.

verdict() is the function that turns two curves into the sprint's headline
claim ("the detector warns before the model breaks"). It is therefore the place
where a reasoning bug becomes a false claim in a README, which is worse than a
crash: a crash is noticed.

The first run of drift_experiment.py demonstrated exactly that. The sample was
below monitor.min_current_rows, every drift window came back inconclusive, and
verdict() read the missing measurements as "no drift" — reporting a confident
BLIND SPOT that was really a sample-size mistake. The last test here pins the
fix, and it is the most important one in the file.
"""

import pytest

from drift_experiment import verdict

EXP_CFG = {"accuracy_drop_tolerance": 0.05}
BASELINE = 0.98


def _points(rows):
    """rows: list of (intensity, accuracy, dataset_drift[, status])."""
    return [{"intensity": i, "accuracy": a, "dataset_drift": d,
             "drift_status": rows[k][3] if len(rows[k]) > 3 else "ok"}
            for k, (i, a, d, *_rest) in enumerate(rows)]


def test_early_warning_is_a_positive_lead():
    """The result the project hopes for: the alert precedes the collapse."""
    points = _points([
        (0.0, 0.98, False), (0.2, 0.97, False),
        (0.4, 0.96, True),                       # detector fires here
        (0.6, 0.90, True),                       # accuracy collapses here
    ])
    result = verdict("cloud", points, BASELINE, EXP_CFG)
    assert result["conclusive"] is True
    assert result["detection_intensity"] == 0.4
    assert result["failure_intensity"] == 0.6
    assert result["lead"] == pytest.approx(0.2)
    assert "early warning" in result["summary"]


def test_accuracy_collapsing_first_is_reported_as_late():
    """The uncomfortable result must be stated, not smoothed over."""
    points = _points([
        (0.0, 0.98, False), (0.2, 0.90, False), (0.4, 0.80, True),
    ])
    result = verdict("blur", points, BASELINE, EXP_CFG)
    assert result["lead"] < 0
    assert "LATE" in result["summary"]


def test_no_alert_at_all_while_accuracy_collapses_is_a_blind_spot():
    points = _points([
        (0.0, 0.98, False), (0.5, 0.60, False), (1.0, 0.30, False),
    ])
    result = verdict("blur", points, BASELINE, EXP_CFG)
    assert result["detection_intensity"] is None
    assert result["failure_intensity"] == 0.5
    assert "BLIND SPOT" in result["summary"]


def test_alert_without_any_accuracy_loss_is_flagged_as_over_sensitive():
    """A firing detector on a robust model is a false positive, and the CT loop
    would retrain for nothing. It must not be reported as a success."""
    points = _points([
        (0.0, 0.98, False), (0.5, 0.975, True), (1.0, 0.97, True),
    ])
    result = verdict("band_shift", points, BASELINE, EXP_CFG)
    assert result["failure_intensity"] is None
    assert "over-sensitive" in result["summary"]


def test_simultaneous_is_not_sold_as_an_early_warning():
    points = _points([(0.0, 0.98, False), (0.5, 0.90, True)])
    result = verdict("seasonal", points, BASELINE, EXP_CFG)
    assert result["lead"] == 0
    assert "symptom" in result["summary"]


def test_an_inconclusive_window_never_becomes_a_finding():
    """THE regression test for the bug this file's docstring describes.

    Points whose drift could not be evaluated must invalidate the whole verdict.
    Reading a missing measurement as "no drift" manufactures a blind-spot claim
    out of a sample-size mistake — a false finding, stated confidently.
    """
    points = _points([
        (0.0, 0.98, False, "insufficient_data"),
        (0.5, 0.60, False, "insufficient_data"),
    ])
    result = verdict("blur", points, BASELINE, EXP_CFG)
    assert result["conclusive"] is False
    assert result["detection_intensity"] is None
    assert result["failure_intensity"] is None
    assert "INCONCLUSIVE" in result["summary"]
    assert "NOT evidence of absence" in result["summary"]
