"""Drift simulator — controlled, realistic degradation of EuroSAT tiles.

Why simulate at all
-------------------
The honest reason, stated up front and repeated in the README: this project has
no production stream. There is no second satellite, no seasonal archive, no
label feedback loop. Without simulation there is nothing to detect, and a drift
dashboard over a stationary replay of the training set is theatre.

What simulation buys that real drift would not
----------------------------------------------
Something real production can never give: the GROUND TRUTH stays known. Every
perturbed tile keeps its label, so both curves can be drawn on the same axis —

    perturbation intensity  ->  real accuracy      (unknowable in production)
    perturbation intensity  ->  drift score        (all production ever sees)

which is the only way to answer the question that actually matters: does the
detector fire BEFORE the model breaks? A monitoring stack that has never been
calibrated against a known degradation is a stack whose threshold was copied
from a blog post.

What it costs, and this is the limitation to state rather than bury: these are
PARAMETRIC perturbations of the same 27k tiles. They are drawn from a family the
author chose, and a detector tuned on them is tuned on that family. Real drift
can be shaped differently — a new sensor's spectral response, a new geography,
a different atmospheric correction pipeline. The curve below is evidence, not
proof, and the threshold it produces is a starting point, not a guarantee.

Design contract (each property is pinned by a test)
---------------------------------------------------
  intensity = 0    -> the IDENTITY, byte for byte. A "baseline" that already
                      perturbs makes the whole curve unreadable.
  monotonic        -> higher intensity means a strictly stronger effect.
  deterministic    -> (image, intensity, seed) fully determines the output, so a
                      sweep is reproducible and a curve can be re-derived.
  label-preserving -> geometry and semantics are untouched: this is DATA drift
                      by construction. A forest under haze is still a forest,
                      so P(Y|X) is unchanged and any accuracy drop is the model
                      failing to generalize, not the label being wrong.

That last point is the one worth defending in an interview: the simulator can
only produce covariate shift. Simulating concept drift would mean changing the
labels, and then the experiment would be measuring a different phenomenon.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageFilter

from utils import load_params

# --- helpers -------------------------------------------------------------------

def _as_array(img: Image.Image) -> np.ndarray:
    """PIL RGB -> (H, W, 3) float32 in [0,1]. All perturbations work in this space."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _as_image(arr: np.ndarray) -> Image.Image:
    """Back to 8-bit PIL, clipped. Rounding, not truncation: truncating biases
    every pixel downward and would inject a spurious brightness drift of its own."""
    return Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def _rng(seed: int, index: int) -> np.random.Generator:
    """A per-image generator derived from (seed, index).

    Derived rather than shared: a single generator threaded through a sweep makes
    an image's perturbation depend on how many images preceded it, so re-running
    a subset would produce different pixels and the curve would not be
    reproducible.
    """
    return np.random.default_rng([seed, index])


def _clamp_intensity(intensity: float) -> float:
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"intensity must be in [0, 1], got {intensity}")
    return float(intensity)


# --- perturbations -------------------------------------------------------------
# Signature: (image, intensity, cfg, rng) -> image. Same shape for all four so the
# sweep code treats them uniformly and adding a fifth is a registry entry.

def cloud_veil(img: Image.Image, intensity: float, cfg: dict,
               rng: np.random.Generator) -> Image.Image:
    """Partial cloud / haze: a low-frequency white veil that also softens edges.

    Modelled as a smooth random mask rather than a uniform whitening, because
    real cloud is patchy: parts of the tile stay pristine while others vanish.
    That patchiness matters for detection — it raises brightness and drops
    saturation and sharpness *simultaneously*, which is a signature no single
    feature captures alone.
    """
    scale = int(cfg["blob_scale"])
    max_opacity = float(cfg["max_opacity"])
    max_blur = float(cfg["max_blur"])

    arr = _as_array(img)
    h, w = arr.shape[:2]

    # Low-resolution noise upsampled bilinearly = smooth blobs, no FFT needed.
    coarse = rng.random((scale, scale)).astype(np.float32)
    mask = np.asarray(
        Image.fromarray((coarse * 255).astype(np.uint8), "L").resize(
            (w, h), Image.BILINEAR),
        dtype=np.float32) / 255.0
    # Smoothstep sharpens the blob edges a little; a purely linear mask reads as
    # a flat wash rather than as cloud.
    mask = mask * mask * (3.0 - 2.0 * mask)

    alpha = (max_opacity * intensity) * mask[..., None]
    veiled = arr * (1.0 - alpha) + alpha            # blend toward white (1.0)

    out = _as_image(veiled)
    radius = max_blur * intensity
    if radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=radius))
    return out


def seasonal(img: Image.Image, intensity: float, cfg: dict,
             rng: np.random.Generator) -> Image.Image:
    """Seasonal radiometry: a joint luminance and saturation shift.

    Winter (the default) darkens and washes out; summer brightens and saturates.
    Applied as a shift around the per-image mean luminance, so the effect is
    relative to the tile's own exposure rather than an absolute offset that would
    clip bright tiles and leave dark ones untouched.
    """
    gain = float(cfg["max_brightness_gain"]) * intensity
    drop = float(cfg["max_saturation_drop"]) * intensity
    if str(cfg["direction"]).lower() == "winter":
        gain = -gain                                # darker
        sat_factor = 1.0 - drop                     # washed out
    else:
        sat_factor = 1.0 + drop                     # vivid

    arr = _as_array(img)
    luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # Desaturate toward the pixel's own luminance: the standard, hue-preserving
    # way to change saturation. Blending toward gray would also shift brightness
    # and confound the two factors.
    arr = luma[..., None] + (arr - luma[..., None]) * sat_factor
    arr = arr * (1.0 + gain)
    return _as_image(arr)


def blur(img: Image.Image, intensity: float, cfg: dict,
         rng: np.random.Generator) -> Image.Image:
    """Defocus / coarser ground sampling distance.

    The perturbation most likely to be a FALSE NEGATIVE for colour-based drift
    detection: it leaves brightness, saturation and channel ratios essentially
    untouched while destroying the high-frequency texture a CNN leans on. It is
    in the panel precisely because it is the adversarial case for the monitoring
    setup, not because it is the easiest to detect.
    """
    radius = float(cfg["max_radius"]) * intensity
    if radius <= 0:
        return img.copy()
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def band_shift(img: Image.Image, intensity: float, cfg: dict,
               rng: np.random.Generator) -> Image.Image:
    """Sensor calibration drift: per-channel gains that pull apart.

    The weights are deliberately asymmetric (R up, B down). A uniform gain across
    all three channels is an EXPOSURE change, which the channel-ratio features
    are built to ignore; the asymmetry is what makes this a genuine band shift
    and what red_ratio/blue_ratio exist to catch.
    """
    weights = np.asarray(cfg["channel_weights"], dtype=np.float32)
    gains = 1.0 + weights * float(cfg["max_gain"]) * intensity
    arr = _as_array(img) * gains[None, None, :]
    return _as_image(arr)


PERTURBATIONS: Dict[str, Callable] = {
    "cloud": cloud_veil,
    "seasonal": seasonal,
    "blur": blur,
    "band_shift": band_shift,
}


# --- public API ----------------------------------------------------------------

def perturb(img: Image.Image, kind: str, intensity: float, params: dict,
            index: int = 0) -> Image.Image:
    """Apply one named perturbation at a given intensity.

    intensity == 0 short-circuits to a copy. That is not an optimization: it is
    the contract that the sweep's first point is the untouched baseline, so the
    accuracy it measures is comparable to the number in the model registry.
    """
    if kind not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation '{kind}'; "
                       f"available: {sorted(PERTURBATIONS)}")
    intensity = _clamp_intensity(intensity)
    if intensity == 0.0:
        return img.convert("RGB").copy()

    cfg = params["drift_sim"]
    return PERTURBATIONS[kind](img, intensity, cfg[kind],
                               _rng(cfg["seed"], index))


def perturb_all(img: Image.Image, intensity: float, params: dict,
                index: int = 0) -> Image.Image:
    """Apply every perturbation in sequence — the 'everything at once' scenario.

    Order is fixed (registry order) so the composition is reproducible; note that
    these operations do NOT commute (blurring a veiled tile differs from veiling
    a blurred one), which is exactly why the order is pinned rather than left to
    dict iteration luck.

    Careful with the intuition that 'all' is the worst case: it is the worst case
    for the MODEL (every input statistic is off at once), but not necessarily the
    largest pixel-space deviation. The cloud veil brightens and the winter shift
    darkens, so in aggregate distance they partially cancel — measured, and
    pinned by tests/test_drift_sim.py. That is a useful reminder in itself: a
    single scalar distance can shrink while the situation gets worse, which is
    why drift is compared feature by feature rather than as one aggregate number.

    The same compensation happens per feature: the band shift's asymmetric gains
    RAISE saturation, cancelling the drop caused by cloud and winter. Under
    composition, only the magnitude of a feature's displacement is predictable,
    not its sign — and two real drifts can in principle return a monitored
    feature to its baseline value while both hurt the model. That blind spot is
    a property of feature-level drift detection itself, and it is stated in the
    README rather than hidden.
    """
    intensity = _clamp_intensity(intensity)
    if intensity == 0.0:
        return img.convert("RGB").copy()
    out = img
    for kind in PERTURBATIONS:
        out = perturb(out, kind, intensity, params, index)
    return out


def perturbation_names() -> List[str]:
    """Registry order + the composite, as used by the sweep and the CLI."""
    return list(PERTURBATIONS) + ["all"]


def apply_named(img: Image.Image, kind: str, intensity: float, params: dict,
                index: int = 0) -> Image.Image:
    """perturb() plus the 'all' pseudo-kind, so callers need one entry point."""
    if kind == "all":
        return perturb_all(img, intensity, params, index)
    return perturb(img, kind, intensity, params, index)


# --- CLI: eyeball the perturbations before trusting any curve ------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a contact sheet of a tile under increasing drift. "
                    "Looking at the images is part of the protocol: a "
                    "perturbation nobody has eyeballed may be physically absurd "
                    "and still produce a beautiful curve.")
    parser.add_argument("image", type=Path, help="input image (any format)")
    parser.add_argument("--kind", default="all", choices=perturbation_names())
    parser.add_argument("--steps", type=int, default=6,
                        help="intensities sampled from 0 to 1 inclusive")
    parser.add_argument("--out", type=Path, default=Path("drift_preview.png"))
    args = parser.parse_args()

    params = load_params()
    src = Image.open(args.image).convert("RGB")
    intensities = np.linspace(0.0, 1.0, args.steps)
    frames = [apply_named(src, args.kind, float(i), params) for i in intensities]

    w, h = frames[0].size
    sheet = Image.new("RGB", (w * len(frames), h))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (i * w, 0))
    sheet.save(args.out)
    print(f"wrote {args.out} — {args.kind} at "
          f"{', '.join(f'{i:.2f}' for i in intensities)}")


if __name__ == "__main__":
    main()
