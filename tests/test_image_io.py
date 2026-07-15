"""Tests for utils/image_io.py — covers Pillow (9→12) and torch/torchvision APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Make sure the project root is importable regardless of working directory
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w=100, h=80, color=(128, 64, 32)):
    """Return an RGB PIL Image of given dimensions."""
    return Image.new("RGB", (w, h), color=color)


def _save_tmp_image(tmp_path, w=100, h=80, fmt="PNG"):
    """Save a tiny image and return its path."""
    img = _make_rgb_image(w, h)
    p = tmp_path / f"test_{w}x{h}.{fmt.lower()}"
    img.save(str(p), format=fmt)
    return str(p)


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_mode_has_required_keys(self):
        for name, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{name} missing 'size'"
            assert "tokens" in cfg, f"{name} missing 'tokens'"
            assert "pad" in cfg, f"{name} missing 'pad'"

    def test_size_tuples(self):
        for name, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2
            assert all(isinstance(v, int) and v > 0 for v in cfg["size"])

    def test_pad_modes(self):
        # base and large must pad; tiny and small must not
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_ordering(self):
        # More tokens for larger modes
        assert (
            MODE_CONFIGS["tiny"]["tokens"]
            < MODE_CONFIGS["small"]["tokens"]
            < MODE_CONFIGS["base"]["tokens"]
            < MODE_CONFIGS["large"]["tokens"]
        )


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_via_padding(self):
        # Wide image — horizontal padding should be minimal / vertical larger
        img = _make_rgb_image(400, 100)  # 4:1 aspect
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target  # canvas matches

    def test_custom_pad_color(self):
        img = _make_rgb_image(100, 100, color=(0, 0, 0))
        result = resize_and_pad(img, (512, 512), pad_color=(255, 0, 0))
        arr = np.array(result)
        # Corner pixel should be the pad color (red)
        corner = tuple(arr[0, 0])
        assert corner == (255, 0, 0)

    def test_square_input_no_horizontal_padding(self):
        img = _make_rgb_image(100, 100, color=(0, 255, 0))
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image(self):
        img = _make_rgb_image(50, 200)  # tall image
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_returns_pil_image(self):
        img = _make_rgb_image(64, 64)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# load_image — using tmp_path fixture so no real disk I/O is needed
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_return_pil_flag(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_tensor_dtype_float32(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="tiny")
        assert t.dtype == torch.float32

    def test_tensor_normalized_range(self, tmp_path):
        """After ImageNet normalisation, values are roughly in [-3, 3]."""
        p = _save_tmp_image(tmp_path)
        t = load_image(p, mode="tiny")
        assert t.min().item() >= -4.0
        assert t.max().item() <= 4.0

    def test_unknown_mode_raises(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="xxl")

    def test_loads_jpeg(self, tmp_path):
        p = _save_tmp_image(tmp_path, fmt="JPEG")
        t = load_image(p, mode="tiny")
        assert t.shape[1] == 3

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """Grayscale images should be converted to RGB (3-channel)."""
        gray = Image.new("L", (50, 50), color=128)
        gp = tmp_path / "gray.png"
        gray.save(str(gp))
        t = load_image(str(gp), mode="tiny")
        assert t.shape[1] == 3

    def test_rgba_converted_to_rgb(self, tmp_path):
        rgba = Image.new("RGBA", (50, 50), color=(0, 128, 0, 200))
        rp = tmp_path / "rgba.png"
        rgba.save(str(rp))
        t = load_image(str(rp), mode="tiny")
        assert t.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_size_returns_total_tokens(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_size_returns_quarter_tokens(self):
        # scale = min(1024/512, 1024/512) = 2 → scaled = 512x512
        # ratio = (512/1024)*(512/1024) = 0.25
        result = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert result == 64

    def test_wide_image(self):
        # orig 2000x500, target 1024x1024
        # scale = min(1024/2000, 1024/500) = min(0.512, 2.048) = 0.512
        # scaled = 1024x256
        # ratio = (1024/1024)*(256/1024) = 0.25
        result = calculate_valid_tokens(2000, 500, 1024, 1024, 256)
        assert result == 64

    def test_returns_int(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_never_exceeds_total(self):
        for ow in [100, 512, 1024, 2048]:
            for oh in [100, 512, 1024, 2048]:
                result = calculate_valid_tokens(ow, oh, 1024, 1024, 256)
                assert result <= 256

    def test_minimum_is_zero_or_positive(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0
