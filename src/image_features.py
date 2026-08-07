"""Monitoring features — the single extractor used for BOTH sides of drift.

Why this module exists at all
-----------------------------
Drift detection compares a REFERENCE distribution (statistics of the training
set) against a CURRENT one (the production stream). Those two sides are computed
at different times, by different processes, from different code paths — which is
the textbook recipe for measuring the difference between two *extractors*
instead of the difference between two *distributions*.

So this module plays for monitoring the role preprocessing.py plays for the
model: ONE definition, imported by the API (per-request logging), by the
reference builder, and by the drift report. If a feature is ever redefined
elsewhere, the drift numbers become meaningless.

What is measured, and on WHICH image
------------------------------------
Features are computed on the image AS THE MODEL SEES IT: decoded to RGB by
preprocessing.decode_image, then resized to params.data.image_size with the same
bilinear resize as the eval transform — but BEFORE ToTensor/Normalize, so the
values stay in interpretable [0,1] pixel space rather than in ImageNet z-scores.

Two reasons for resizing first:
  - sharpness (Laplacian variance) is resolution-dependent; a 64x64 reference
    compared against a 224x224 upload would show massive fake drift;
  - what drifts should be what the model consumes. Anything the resize destroys
    cannot affect the prediction, so it has no business raising an alert.

Feature choice: cheap, interpretable, and each one maps to a physical
perturbation the drift simulator can produce (cloud veil -> brightness up +
saturation down + sharpness down; band shift -> channel means diverge).
Deliberately NOT deep embeddings: a p95 latency budget of 400 ms leaves no room
for a second forward pass, and an uninterpretable drift score is an alert nobody
can action.
"""

from typing import Dict, List

import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from preprocessing import ImageSource, decode_image

# THE canonical feature order. The Postgres schema, the reference JSON and the
# Evidently column mapping are all generated from this list, so adding a feature
# is a one-line change that propagates everywhere instead of three edits that
# can silently disagree.
FEATURE_NAMES: List[str] = [
    "brightness",    # mean luminance          [0,1]
    "contrast",      # std of luminance        [0,~0.5]
    "saturation",    # mean HSV saturation     [0,1]
    "sharpness",     # variance of Laplacian   [0,~0.1] - collapses under blur/cloud
    "mean_r", "mean_g", "mean_b",   # per-channel means  [0,1]
    "std_r", "std_g", "std_b",      # per-channel spread [0,1]
    "red_ratio",     # R / (R+G+B) - band-shift detector, invariant to exposure
    "blue_ratio",    # B / (R+G+B) - idem (green is 1 - red - blue, so omitted:
                     # a linearly dependent third column would double-count the
                     # same drift in any multivariate test)
]

# ITU-R BT.601 luma weights — the standard RGB -> perceived-brightness mapping.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# 4-neighbour Laplacian: the classic focus measure. Its variance drops toward 0
# as an image loses high-frequency content (blur, haze, cloud veil).
_LAPLACIAN = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]], dtype=np.float32)


def _to_array(img: Image.Image, image_size: int) -> np.ndarray:
    """PIL RGB -> (H, W, 3) float32 in [0,1], resized exactly as the model sees it."""
    resized = TF.resize(img, [image_size, image_size],
                        interpolation=InterpolationMode.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian, computed with pure numpy (no scipy in the API image).

    Valid-region convolution via shifted slices: cheap, allocation-light, and it
    avoids the border artefacts a zero-padded convolution would introduce (which
    would make sharpness depend on the padding rather than on the content).
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    c = gray[1:-1, 1:-1]
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] +
           gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * c)
    return float(lap.var())


def extract_features(source: ImageSource, data_cfg: dict) -> Dict[str, float]:
    """Any image input -> the canonical monitoring feature dict.

    Accepts the same union preprocessing does (bytes, path, PIL), and decodes
    through preprocessing.decode_image so that monitoring and inference agree on
    channel order and orientation — a BGR mix-up here would swap mean_r/mean_b
    and fabricate a permanent band shift.
    """
    img = decode_image(source)
    arr = _to_array(img, data_cfg["image_size"])

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr @ _LUMA

    # HSV saturation, chroma form: (max - min) / max. Defined as 0 for black
    # pixels (max == 0) — guarded to keep pure-black tiles from yielding NaN and
    # poisoning the aggregate, which is how a monitoring pipeline dies quietly.
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-6)

    total = r + g + b
    # Same guard on the ratios: an all-black tile gets the neutral 1/3 rather
    # than a division by zero.
    red_ratio = np.divide(r, total, out=np.full_like(r, 1 / 3), where=total > 1e-6)
    blue_ratio = np.divide(b, total, out=np.full_like(b, 1 / 3), where=total > 1e-6)

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(sat.mean()),
        "sharpness": _laplacian_var(gray),
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
        "red_ratio": float(red_ratio.mean()),
        "blue_ratio": float(blue_ratio.mean()),
    }


def prediction_entropy(probabilities: Dict[str, float]) -> float:
    """Normalized Shannon entropy of a softmax vector, in [0,1].

    The one monitoring signal that reacts to the MODEL's behavior rather than to
    the input: 0 = fully confident, 1 = uniform over the classes. Kept next to
    the input features because it is the natural fallback when input statistics
    fail to move but the model has quietly lost its footing (the false-negative
    case the drift simulator is built to expose).
    """
    p = np.asarray(list(probabilities.values()), dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(probabilities)))
