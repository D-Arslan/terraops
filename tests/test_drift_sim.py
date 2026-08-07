"""Unit tests for the drift simulator.

The simulator is the experiment's independent variable. If it is not exactly the
identity at intensity 0, not monotonic, or not reproducible, the drift curve
measures the simulator's bugs rather than the model's fragility — and every
conclusion drawn from it, including the CT threshold, is invalid.

So these tests pin the CONTRACT stated in drift_sim's docstring, one test per
property, plus the specific feature response each perturbation is supposed to
produce (a perturbation that moves nothing would give a flat, reassuring, and
completely meaningless curve).
"""

import numpy as np
import pytest
from PIL import Image

from drift_sim import apply_named, perturb, perturb_all, perturbation_names
from image_features import extract_features
from utils import load_params

CFG = {"image_size": 224}


@pytest.fixture(scope="module")
def params():
    return load_params()


def _tile(size: int = 64, seed: int = 0) -> Image.Image:
    """Deterministic textured tile — flat colour would make sharpness a constant 0."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 190, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# --- the contract --------------------------------------------------------------

@pytest.mark.parametrize("kind", perturbation_names())
def test_intensity_zero_is_the_exact_identity(kind, params):
    """The baseline point of the curve must be the UNTOUCHED image.

    Not 'almost identical': byte-identical. A baseline that already perturbs
    shifts the whole curve and makes its first point incomparable to the
    accuracy recorded in the model registry.
    """
    img = _tile()
    out = apply_named(img, kind, 0.0, params)
    assert np.array_equal(np.asarray(out), np.asarray(img))


@pytest.mark.parametrize("kind", perturbation_names())
def test_output_geometry_and_mode_are_preserved(kind, params):
    """Label-preserving means: same pixels grid, same channels, no crop, no flip.

    Any geometric change would risk altering the semantics of the tile, turning
    a covariate shift experiment into an unlabelled mess.
    """
    img = _tile()
    out = apply_named(img, kind, 1.0, params)
    assert out.size == img.size
    assert out.mode == "RGB"


@pytest.mark.parametrize("kind", perturbation_names())
def test_deterministic_for_a_given_index(kind, params):
    """(image, intensity, seed, index) must fully determine the output."""
    img = _tile()
    a = apply_named(img, kind, 0.6, params, index=3)
    b = apply_named(img, kind, 0.6, params, index=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_random_perturbation_varies_across_images():
    """The cloud mask must differ per image; a shared mask would make the veil a
    systematic overlay and let the model learn around it."""
    params = load_params()
    img = _tile()
    a = perturb(img, "cloud", 0.8, params, index=0)
    b = perturb(img, "cloud", 0.8, params, index=1)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("kind", perturbation_names())
def test_effect_is_monotonic_in_intensity(kind, params):
    """Stronger intensity = strictly further from the original.

    Measured as mean absolute pixel deviation. Without monotonicity the x-axis
    of the drift curve has no meaning and no single threshold can exist.
    """
    img = _tile()
    base = np.asarray(img, dtype=np.float32)
    deviations = [
        float(np.abs(np.asarray(apply_named(img, kind, i, params),
                                dtype=np.float32) - base).mean())
        for i in (0.25, 0.5, 0.75, 1.0)
    ]
    assert deviations == sorted(deviations), f"{kind}: {deviations}"
    assert deviations[-1] > deviations[0]


def test_intensity_outside_the_unit_interval_is_rejected(params):
    """Fail loudly: a silently clipped 1.5 would produce a curve with two
    identical points at the top and an invisible plateau."""
    img = _tile()
    with pytest.raises(ValueError):
        perturb(img, "blur", 1.5, params)
    with pytest.raises(ValueError):
        perturb(img, "blur", -0.1, params)


def test_unknown_perturbation_raises():
    with pytest.raises(KeyError):
        perturb(_tile(), "snow", 0.5, load_params())


# --- each perturbation must move the feature it claims to move ------------------

def test_cloud_brightens_and_desaturates_and_softens(params):
    """The three-way signature is what makes cloud the EASY case for detection."""
    img = _tile(seed=1)
    before = extract_features(img, CFG)
    after = extract_features(perturb(img, "cloud", 1.0, params), CFG)

    assert after["brightness"] > before["brightness"]
    assert after["saturation"] < before["saturation"]
    assert after["sharpness"] < before["sharpness"]


def test_winter_seasonal_darkens_and_washes_out(params):
    img = _tile(seed=2)
    before = extract_features(img, CFG)
    after = extract_features(perturb(img, "seasonal", 1.0, params), CFG)

    assert params["drift_sim"]["seasonal"]["direction"] == "winter"
    assert after["brightness"] < before["brightness"]
    assert after["saturation"] < before["saturation"]


def test_blur_is_the_hard_case_it_is_meant_to_be(params):
    """Blur must crush sharpness while leaving colour statistics nearly intact.

    This is the point of including it: it is the perturbation that a
    colour-statistics drift detector is most likely to MISS. If this test ever
    starts failing because blur also moves brightness, the experiment loses its
    adversarial case and the conclusion gets easier than it should be.
    """
    img = _tile(seed=3)
    before = extract_features(img, CFG)
    after = extract_features(perturb(img, "blur", 1.0, params), CFG)

    assert after["sharpness"] < 0.3 * before["sharpness"]
    assert after["brightness"] == pytest.approx(before["brightness"], abs=0.02)
    assert after["red_ratio"] == pytest.approx(before["red_ratio"], abs=0.02)


def test_band_shift_moves_the_channel_ratios(params):
    """The feature designed for it must be the feature that reacts."""
    img = _tile(seed=4)
    before = extract_features(img, CFG)
    after = extract_features(perturb(img, "band_shift", 1.0, params), CFG)

    assert after["red_ratio"] > before["red_ratio"] + 0.01
    assert after["blue_ratio"] < before["blue_ratio"] - 0.01


def test_composite_perturbs_every_monitored_dimension_at_once(params):
    """'all' is the MULTI-FACTOR scenario, not the maximum-pixel-distance one.

    A first version of this test asserted that the composite deviates from the
    original at least as much as any single perturbation. It fails, and the
    failure is physical rather than a bug: the cloud veil brightens while the
    winter shift darkens, so in pixel space the two partially CANCEL. The
    composite is measurably closer to the original than the cloud alone.

    That cancellation is worth keeping and worth knowing:
      - it is realistic (a hazy winter scene is not a hazy summer scene);
      - it is a warning about mean-pixel-distance as a drift proxy — two strong,
        opposed perturbations can look mild in aggregate while the model sees
        neither of the two conditions it was trained on. Aggregate distance
        hides compensating shifts; per-feature comparison does not.

    The same effect appears a second time, in the opposite direction: the band
    shift's asymmetric channel gains pull the channels APART, which raises
    saturation and overwhelms the desaturation from cloud and winter. So under
    composition the DIRECTION a feature moves is not predictable from the
    individual perturbations — only that it moves.

    Hence what this test pins: the composite displaces every monitored dimension
    (that is what makes it the hardest case for a detector), asserted as a
    distance from baseline rather than as a sign. Asserting signs here would be
    asserting something the physics does not guarantee.

    The uncomfortable corollary, which belongs in the README's honesty section:
    two real drifts can compensate on a monitored feature and return it to its
    baseline value while both remain bad news for the model. That is a genuine
    blind spot of feature-level drift detection, not a flaw in this simulator.
    """
    img = _tile(seed=5)
    before = extract_features(img, CFG)
    after = extract_features(perturb_all(img, 1.0, params), CFG)

    # Blur + veil crush high frequencies; nothing in the panel restores them,
    # so this one direction IS guaranteed.
    assert after["sharpness"] < 0.5 * before["sharpness"]
    # The rest: displaced, sign unspecified by construction.
    for name in ["saturation", "brightness", "red_ratio", "blue_ratio"]:
        assert abs(after[name] - before[name]) > 0.005, name


def test_opposed_perturbations_cancel_in_aggregate_distance(params):
    """Pin the cancellation itself, so a future refactor cannot erase the lesson.

    This is the experimental caveat in test form: a single scalar 'how different
    is it' number can go DOWN while the situation gets worse for the model.
    """
    img = _tile(seed=5)
    base = np.asarray(img, dtype=np.float32)

    def deviation(out):
        return float(np.abs(np.asarray(out, dtype=np.float32) - base).mean())

    assert deviation(perturb_all(img, 1.0, params)) < deviation(
        perturb(img, "cloud", 1.0, params))
