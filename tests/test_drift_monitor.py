"""Unit tests for the CT trigger (src/drift_monitor.py): persistence, cooldown,
inconclusive windows, and the two OR-ed detectors.

Everything here is pure: no database, no Evidently, no GitHub. The trigger's
job is mostly to REFUSE, so most of these tests check that nothing fires.
Thresholds come from params.yaml (`monitor.trigger`).
"""

from datetime import datetime, timedelta, timezone

import pytest

from drift_monitor import decide, evaluate

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cfg(params):
    return params["monitor"]["trigger"]


def _summary(drift=False, share=0.0, status="ok"):
    return {"status": status, "dataset_drift": drift, "drifted_share": share,
            "drift_share_threshold": 0.5,
            "top_drifted": [{"feature": "mean_b", "distance": 1.4}],
            "message": "too few rows"}


def _predictions(top_share=0.12):
    rest = (1 - top_share) / 9
    shares = {f"class_{i}": rest for i in range(9)}
    shares["SeaLake"] = top_share
    return {"predicted_class_share": shares}


def _fresh():
    return {"consecutive_drift": 0, "last_dispatch": None, "history": []}


# --- evaluate: one window -> one verdict -------------------------------------------

def test_quiet_window_is_conclusive_and_clean(params):
    v = evaluate(_summary(), _predictions(), params)
    assert v["conclusive"] and not v["drift_detected"] and v["reasons"] == []


def test_thin_window_is_inconclusive_not_clean(params):
    # "no data" must never read as "no drift"
    v = evaluate(_summary(status="insufficient_data"), _predictions(), params)
    assert not v["conclusive"] and not v["drift_detected"]
    assert v["note"] == "too few rows"


def test_drift_share_detector(params):
    v = evaluate(_summary(drift=True, share=0.92), _predictions(), params)
    assert v["drift_detected"] and len(v["reasons"]) == 1
    assert "drift share 0.92" in v["reasons"][0] and "mean_b" in v["reasons"][0]


def test_class_collapse_detector_fires_alone(params, cfg):
    v = evaluate(_summary(), _predictions(cfg["majority_class_share_max"] + 0.05), params)
    assert v["drift_detected"] and "collapse" in v["reasons"][0]
    assert "SeaLake" in v["reasons"][0]


def test_class_share_at_threshold_does_not_fire(params, cfg):
    v = evaluate(_summary(), _predictions(cfg["majority_class_share_max"]), params)
    assert not v["drift_detected"]


def test_both_detectors_are_or_ed(params):
    v = evaluate(_summary(drift=True, share=0.92), _predictions(0.9), params)
    assert len(v["reasons"]) == 2


# --- decide: persistence ----------------------------------------------------------

def _drift():
    return {"conclusive": True, "drift_detected": True, "reasons": ["x"], "note": None}


def _clean():
    return {"conclusive": True, "drift_detected": False, "reasons": [], "note": None}


def _inconclusive():
    return {"conclusive": False, "drift_detected": False, "reasons": [], "note": "thin"}


def test_one_window_is_noise(params, cfg):
    fire, why, state = decide(_drift(), _fresh(), params, NOW)
    assert not fire and state["consecutive_drift"] == 1
    assert f"1/{cfg['consecutive_windows']}" in why


def test_streak_reaches_threshold_then_fires(params, cfg):
    state = _fresh()
    needed = int(cfg["consecutive_windows"])
    for _ in range(needed - 1):
        fire, _, state = decide(_drift(), state, params, NOW)
        assert not fire
    fire, why, state = decide(_drift(), state, params, NOW)
    assert fire and state["consecutive_drift"] == needed
    assert "cooldown clear" in why


def test_clean_window_resets_the_streak(params):
    state = {"consecutive_drift": 2, "last_dispatch": None, "history": []}
    fire, why, state = decide(_clean(), state, params, NOW)
    assert not fire and state["consecutive_drift"] == 0 and "reset" in why


def test_inconclusive_window_neither_confirms_nor_resets(params):
    state = {"consecutive_drift": 2, "last_dispatch": None, "history": []}
    fire, why, new = decide(_inconclusive(), state, params, NOW)
    assert not fire and new["consecutive_drift"] == 2 and "unchanged" in why


def test_decide_does_not_mutate_the_input_state(params):
    state = _fresh()
    decide(_drift(), state, params, NOW)
    assert state["consecutive_drift"] == 0     # dry runs must be side-effect free


# --- decide: cooldown -------------------------------------------------------------

def test_cooldown_blocks_a_reached_streak(params, cfg):
    needed = int(cfg["consecutive_windows"])
    recent = (NOW - timedelta(minutes=1)).isoformat()
    state = {"consecutive_drift": needed - 1, "last_dispatch": recent, "history": []}
    fire, why, new = decide(_drift(), state, params, NOW)
    assert not fire and "cooldown active" in why
    # the evidence is kept: the streak is NOT reset by a blocked dispatch
    assert new["consecutive_drift"] == needed


def test_cooldown_expired_lets_it_fire(params, cfg):
    needed = int(cfg["consecutive_windows"])
    old = (NOW - timedelta(minutes=int(cfg["cooldown_minutes"]) + 1)).isoformat()
    state = {"consecutive_drift": needed - 1, "last_dispatch": old, "history": []}
    fire, _, _ = decide(_drift(), state, params, NOW)
    assert fire
