"""Unit tests for the promotion gate's decision rules (src/promote.py).

The gate is the one place where an automated retrain is allowed to change what
production serves, so its three rules are tested as a pure function, with no
registry and no model: what is checked here is the DECISION, not the scoring.
Thresholds come from params.yaml (section `promote`), the same file the gate
reads, so a threshold change is reflected here without a code change.
"""

import numpy as np
import pytest

from dataset import EUROSAT_CLASSES
from promote import gate_reasons, per_class_recall


@pytest.fixture(scope="module")
def pcfg(params):
    return params["promote"]


def _recall(value, **overrides):
    """A per-class recall dict, one value for every class unless overridden."""
    r = {cls: value for cls in EUROSAT_CLASSES}
    r.update(overrides)
    return r


# --- rule 1: absolute floor (also the bootstrap rule) ---------------------------

def test_bootstrap_above_floor_is_promoted(pcfg):
    assert gate_reasons(pcfg["min_accuracy"] + 0.01, _recall(0.95), None, None, pcfg) == []


def test_bootstrap_below_floor_is_refused(pcfg):
    reasons = gate_reasons(pcfg["min_accuracy"] - 0.01, _recall(0.95), None, None, pcfg)
    assert len(reasons) == 1 and "absolute floor" in reasons[0]


def test_untrained_candidate_is_refused_even_without_champion(pcfg):
    # The sprint-2 v2 case: a model at chance level (11.5%) must never bootstrap.
    reasons = gate_reasons(0.115, _recall(0.1), None, None, pcfg)
    assert reasons and "absolute floor" in reasons[0]


# --- rule 2: margin above noise ---------------------------------------------------

def test_win_inside_the_noise_band_is_not_a_win(pcfg):
    champ = 0.98
    cand = champ + pcfg["min_delta"] / 2
    reasons = gate_reasons(cand, _recall(0.98), champ, _recall(0.98), pcfg)
    assert len(reasons) == 1 and "margin" in reasons[0]


def test_margin_exactly_at_threshold_passes(pcfg):
    champ = 0.98
    cand = champ + pcfg["min_delta"]
    assert gate_reasons(cand, _recall(0.98), champ, _recall(0.98), pcfg) == []


def test_equal_accuracy_is_refused(pcfg):
    reasons = gate_reasons(0.98, _recall(0.98), 0.98, _recall(0.98), pcfg)
    assert any("margin" in r for r in reasons)


# --- rule 3: no single class may regress ------------------------------------------

def test_global_gain_with_one_class_collapse_is_refused(pcfg):
    champ_recall = _recall(0.98)
    drop = pcfg["max_class_recall_drop"] + 0.05
    cand_recall = _recall(0.99, Highway=0.98 - drop)
    cand_acc = 0.98 + pcfg["min_delta"] + 0.01      # clears rules 1 and 2
    reasons = gate_reasons(cand_acc, cand_recall, 0.98, champ_recall, pcfg)
    assert len(reasons) == 1
    assert "Highway" in reasons[0] and "recall regression" in reasons[0]


def test_class_drop_within_limit_passes(pcfg):
    champ_recall = _recall(0.98)
    cand_recall = _recall(0.99, Highway=0.98 - pcfg["max_class_recall_drop"] + 0.001)
    cand_acc = 0.98 + pcfg["min_delta"] + 0.01
    assert gate_reasons(cand_acc, cand_recall, 0.98, champ_recall, pcfg) == []


def test_every_failing_rule_is_reported(pcfg):
    # A candidate below the floor, below the champion, and collapsing every
    # class: the report must list all of it, not stop at the first rule.
    reasons = gate_reasons(0.5, _recall(0.5), 0.98, _recall(0.98), pcfg)
    assert any("absolute floor" in r for r in reasons)
    assert any("margin" in r for r in reasons)
    assert sum("recall regression" in r for r in reasons) == len(EUROSAT_CLASSES)


# --- per-class recall helper ----------------------------------------------------

def test_per_class_recall_counts_only_the_true_class():
    n = len(EUROSAT_CLASSES)
    # two samples per class, all correct, except: class 0 half right, class 2 all wrong
    y_true = np.repeat(np.arange(n), 2)
    y_pred = y_true.copy()
    y_pred[0] = 1                      # one of the two class-0 samples -> class 1
    y_pred[y_true == 2] = 0            # both class-2 samples -> class 0
    r = per_class_recall(y_true, y_pred)
    assert r[EUROSAT_CLASSES[0]] == 0.5
    assert r[EUROSAT_CLASSES[1]] == 1.0    # extra wrong hits do not inflate recall
    assert r[EUROSAT_CLASSES[2]] == 0.0
    assert set(r) == set(EUROSAT_CLASSES)
