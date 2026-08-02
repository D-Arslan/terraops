"""Shared preprocessing — the single train/serving contract.

This module owns the ENTIRE inference preprocessing chain, from raw bytes to a
model-ready tensor. train.py (via dataset.py), the promotion gate, and the serving
API all import it, so there is exactly ONE definition of "how an image becomes a
tensor". That makes train/serving skew structurally impossible rather than merely
discouraged by documentation.

Where the guarantee starts — the boundary is the RAW BYTES, not a decoded array.
Decoding lives INSIDE the module on purpose: channel order (RGB vs BGR), alpha
channels, grayscale, and EXIF rotation are the classic silent-skew sources, and
they all happen BEFORE the torchvision transforms. If a caller decoded the image
with its own library and handed us an array, the guarantee would no longer cover
it — so callers hand us bytes/paths, not arrays.

The eval transform here is byte-for-byte the one that produced the champion
(dataset.py's train=False branch), lifted into one place. Two hardenings were
added that are NO-OPs on the EuroSAT tiles (so they introduce zero skew vs the
existing champion) but make real-world uploads robust:
  - convert("RGB")     -> neutralizes BGR / alpha / grayscale by construction
  - exif_transpose     -> applies phone-photo orientation (EuroSAT tiles carry
                          no EXIF, so this changes nothing for them)
  - interpolation pinned to BILINEAR explicitly, because torchvision's Resize
    default has drifted across versions — a silent interpolation change is exactly
    the kind of resize skew that lowers accuracy without raising any error.
"""

import io
from pathlib import Path
from typing import Iterable, Union

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Anything a caller may legitimately hand us as "an image to classify".
ImageSource = Union[bytes, bytearray, str, Path, Image.Image]


def decode_image(source: ImageSource) -> Image.Image:
    """Raw input -> a canonical RGB PIL image (the module's true entry point).

    Accepts raw bytes (API upload), a filesystem path, or an already-open PIL
    image (tests, dataset loading). Whatever comes in leaves as a 3-channel,
    R,G,B, upright PIL image — the exact form torchvision's datasets.EuroSAT fed
    the transforms during training.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:  # str or Path
        img = Image.open(source)

    # Orientation must be resolved BEFORE any resize, or a 90° phone photo would
    # be squashed into a square while still sideways.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def build_eval_transform(data_cfg: dict) -> transforms.Compose:
    """THE deterministic inference transform, built from params.yaml.

    This is the single definition imported by dataset.py (val/test splits + the
    frozen gate) and by the serving API (/predict). One function, several callers
    — that identity is what forbids skew.

    Takes a PIL RGB image (as produced by decode_image) and returns a normalized
    CHW float tensor. No augmentation: augmentation is a train-only concern and
    deliberately lives elsewhere.
    """
    size = data_cfg["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # PIL RGB uint8 HWC -> float32 [0,1] CHW
        transforms.Normalize(mean=data_cfg["norm_mean"], std=data_cfg["norm_std"]),
    ])


def preprocess(source: ImageSource, data_cfg: dict) -> torch.Tensor:
    """Raw input -> a single (1, 3, H, W) batch tensor, ready for model(...).

    The full serving chain in one call: decode -> eval transform -> add batch dim.
    """
    tensor = build_eval_transform(data_cfg)(decode_image(source))
    return tensor.unsqueeze(0)


def preprocess_batch(sources: Iterable[ImageSource], data_cfg: dict) -> torch.Tensor:
    """List of raw inputs -> a stacked (N, 3, H, W) batch tensor.

    Builds the transform once and reuses it across the batch (cheaper than
    rebuilding the Compose per image on a /predict/batch call).
    """
    transform = build_eval_transform(data_cfg)
    return torch.stack([transform(decode_image(s)) for s in sources])
