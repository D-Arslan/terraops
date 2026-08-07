"""Unit tests for the shared preprocessing contract.

Deterministic and infra-free: these guard the ONE function both train.py and the
API depend on. A single unit test here protects both the train and serving worlds,
because there is only one world.
"""

import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from preprocessing import build_eval_transform, decode_image, preprocess, preprocess_batch


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- decode_image: the module's entry point, where skew is killed --------------

def test_decode_forces_rgb_from_rgba():
    """A 4-channel upload must come out as 3-channel RGB (alpha dropped)."""
    rgba = Image.new("RGBA", (32, 32), (10, 20, 30, 128))
    out = decode_image(rgba)
    assert out.mode == "RGB"
    assert len(out.getbands()) == 3


def test_decode_forces_rgb_from_grayscale():
    """A single-channel (L) upload must be promoted to 3 channels, not crash."""
    gray = Image.new("L", (32, 32), 128)
    out = decode_image(gray)
    assert out.mode == "RGB"


def test_decode_preserves_channel_order():
    """PNG round-trip must keep R,G,B order (no BGR scrambling on the way in)."""
    img = Image.new("RGB", (8, 8), (200, 100, 50))
    decoded = np.array(decode_image(_png_bytes(img)))
    # Top-left pixel stays (R=200, G=100, B=50).
    assert tuple(decoded[0, 0]) == (200, 100, 50)


def test_decode_accepts_bytes_path_and_pil(tmp_path):
    img = Image.new("RGB", (16, 16), (1, 2, 3))
    p = tmp_path / "tile.png"
    img.save(p)
    from_bytes = np.array(decode_image(_png_bytes(img)))
    from_path = np.array(decode_image(str(p)))
    from_pil = np.array(decode_image(img))
    assert np.array_equal(from_bytes, from_path)
    assert np.array_equal(from_bytes, from_pil)


# --- The eval transform: the anti-skew invariant -------------------------------

def test_eval_transform_matches_reference(data_cfg):
    """build_eval_transform must equal the champion's original eval pipeline
    (Resize BILINEAR -> ToTensor -> Normalize). This is the skew tripwire: if the
    transform ever drifts, this fails before a model is ever loaded."""
    size = data_cfg["image_size"]
    reference = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg["norm_mean"], std=data_cfg["norm_std"]),
    ])
    img = Image.new("RGB", (64, 64), (123, 200, 50))
    assert torch.equal(build_eval_transform(data_cfg)(img), reference(img))


def test_output_shape_and_dtype(data_cfg):
    img = Image.new("RGB", (64, 64), (120, 180, 60))
    t = preprocess(_png_bytes(img), data_cfg)
    assert t.shape == (1, 3, data_cfg["image_size"], data_cfg["image_size"])
    assert t.dtype == torch.float32


def test_normalization_is_applied(data_cfg):
    """After ImageNet normalization a pure-black image maps to (0-mean)/std,
    which is strictly negative — proof normalization ran (not just /255)."""
    black = Image.new("RGB", (64, 64), (0, 0, 0))
    t = preprocess(_png_bytes(black), data_cfg)
    mean = torch.tensor(data_cfg["norm_mean"]).view(3, 1, 1)
    std = torch.tensor(data_cfg["norm_std"]).view(3, 1, 1)
    expected = ((0.0 - mean) / std)
    assert t.min() < 0
    # every pixel of a constant image equals the per-channel normalized constant
    assert torch.allclose(t[0, :, 0, 0], expected.view(3), atol=1e-6)


def test_batch_shape(data_cfg):
    imgs = [_png_bytes(Image.new("RGB", (64, 64), c))
            for c in [(10, 20, 30), (40, 50, 60), (70, 80, 90)]]
    batch = preprocess_batch(imgs, data_cfg)
    assert batch.shape == (3, 3, data_cfg["image_size"], data_cfg["image_size"])


def test_resize_upscales_small_tile(data_cfg):
    """EuroSAT tiles are 64x64 native; serving must upscale to the model's size."""
    small = Image.new("RGB", (64, 64), (100, 100, 100))
    t = preprocess(_png_bytes(small), data_cfg)
    assert t.shape[-1] == data_cfg["image_size"] == 224
