"""Unit tests for the monitoring feature extractor.

These matter more than they look. Every drift number in the sprint is a
comparison of these values between two populations; if a feature is not
deterministic, not bounded, or reacts to something other than what its name
says, the drift report is confidently wrong and the CT loop retrains for a
reason that does not exist.

Each test below pins one property the drift analysis silently assumes:
  - determinism            (same image -> same numbers, always)
  - resolution invariance  (a 64px reference vs a 224px upload must not
                            fabricate drift out of thin air)
  - monotonic response     (each feature moves the way its name promises when
                            the matching perturbation is applied)
  - schema agreement       (FEATURE_NAMES is the single source of truth)

Infra-free and fast: no DB, no model, no MLflow.
"""

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from image_features import FEATURE_NAMES, extract_features, prediction_entropy

CFG = {"image_size": 224}


def _texture(size: int = 64, seed: int = 0) -> Image.Image:
    """A deterministic pseudo-satellite tile: colored noise + structure.

    Flat color patches would give sharpness == 0 and hide every regression in
    the focus measure, so the fixture deliberately carries high-frequency
    content.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(size, size, 3), dtype=np.uint8)
    # Add a low-frequency gradient so contrast is not purely noise-driven.
    grad = np.linspace(0, 60, size, dtype=np.float32)[:, None, None]
    arr = np.clip(base.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# --- contract ------------------------------------------------------------------

def test_returns_exactly_the_declared_features():
    """extract_features and FEATURE_NAMES must not be able to disagree.

    The Postgres columns and the Evidently column mapping are generated from
    FEATURE_NAMES; a key present in one and not the other would silently log
    NULLs or drop a monitored dimension.
    """
    feats = extract_features(_texture(), CFG)
    assert set(feats) == set(FEATURE_NAMES)
    assert len(feats) == len(FEATURE_NAMES)


def test_all_features_are_finite_floats():
    feats = extract_features(_texture(), CFG)
    for name, value in feats.items():
        assert isinstance(value, float), name
        assert np.isfinite(value), name


def test_deterministic():
    """Two calls on the same image must be bit-identical, or drift is noise."""
    img = _texture(seed=7)
    assert extract_features(img, CFG) == extract_features(img, CFG)


def test_black_image_does_not_produce_nan():
    """The degenerate case: max == 0 makes saturation and the ratios 0/0.

    A single NaN row poisons the aggregate statistics of a whole window, so the
    guards in the extractor are load-bearing, not defensive decoration.
    """
    feats = extract_features(Image.new("RGB", (32, 32), (0, 0, 0)), CFG)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["saturation"] == 0.0
    assert feats["red_ratio"] == pytest.approx(1 / 3)


# --- resolution invariance -----------------------------------------------------

def test_features_are_resolution_invariant():
    """The same content at 64px and at 224px must yield near-identical features.

    This is why extraction resizes to image_size FIRST. Without it, sharpness
    alone would differ by an order of magnitude between the 64x64 EuroSAT
    reference and a 224x224 upload, and the drift report would scream about a
    difference the model never sees (it resizes too).
    """
    small = _texture(size=64, seed=3)
    large = small.resize((224, 224), Image.BILINEAR)

    f_small = extract_features(small, CFG)
    f_large = extract_features(large, CFG)

    for name in ["brightness", "contrast", "saturation",
                 "mean_r", "mean_g", "mean_b", "red_ratio", "blue_ratio"]:
        assert f_small[name] == pytest.approx(f_large[name], abs=0.02), name


# --- monotonic response to the perturbations the simulator will apply ----------

def test_brightness_increases_when_image_is_brightened():
    img = _texture(seed=1)
    bright = ImageEnhance.Brightness(img).enhance(1.6)
    assert extract_features(bright, CFG)["brightness"] > \
           extract_features(img, CFG)["brightness"]


def test_sharpness_collapses_under_blur():
    """The focus measure must be the feature that reacts to blur and haze.

    Blur is the perturbation most likely to hurt a CNN while leaving colour
    statistics untouched — if sharpness did not move here, the whole monitoring
    setup would be blind to it.
    """
    img = _texture(seed=2)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    assert extract_features(blurred, CFG)["sharpness"] < \
           0.5 * extract_features(img, CFG)["sharpness"]


def test_saturation_drops_when_desaturated():
    img = _texture(seed=4)
    washed = ImageEnhance.Color(img).enhance(0.2)
    assert extract_features(washed, CFG)["saturation"] < \
           extract_features(img, CFG)["saturation"]


def test_channel_ratios_detect_a_band_shift():
    """Boosting the red channel must move red_ratio, not just mean_r."""
    arr = np.asarray(_texture(seed=5), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.4, 0, 255)
    shifted = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    base = extract_features(_texture(seed=5), CFG)
    after = extract_features(shifted, CFG)
    assert after["red_ratio"] > base["red_ratio"] + 0.01
    assert after["blue_ratio"] < base["blue_ratio"]


def test_channel_ratios_are_robust_to_a_global_exposure_change():
    """red_ratio must isolate a BAND shift from a plain exposure change.

    Scaling all three channels together is an exposure change, not a sensor
    band drift. mean_r moves; red_ratio must not — that separation is the whole
    point of carrying both kinds of feature.
    """
    img = _texture(seed=6)
    arr = np.asarray(img, dtype=np.float32) * 0.7
    darker = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    base = extract_features(img, CFG)
    after = extract_features(darker, CFG)
    assert after["mean_r"] < base["mean_r"] - 0.05          # exposure moved
    assert after["red_ratio"] == pytest.approx(base["red_ratio"], abs=0.01)


# --- the model-side signal -----------------------------------------------------

def test_entropy_is_zero_for_a_confident_prediction():
    probs = {f"c{i}": 0.0 for i in range(10)}
    probs["c0"] = 1.0
    assert prediction_entropy(probs) == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_one_for_a_uniform_prediction():
    """Normalization by log(K) is what makes the threshold class-count agnostic."""
    probs = {f"c{i}": 0.1 for i in range(10)}
    assert prediction_entropy(probs) == pytest.approx(1.0, abs=1e-9)


def test_entropy_is_ordered_between_the_two_extremes():
    sharp = {"a": 0.9, "b": 0.05, "c": 0.05}
    fuzzy = {"a": 0.4, "b": 0.35, "c": 0.25}
    assert prediction_entropy(sharp) < prediction_entropy(fuzzy)
