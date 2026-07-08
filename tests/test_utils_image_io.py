"""Tests for utils/image_io.py"""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# Make sure the project root is importable
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

def make_pil_image(width=200, height=150, color=(128, 64, 32)):
    """Create a small in-memory PIL image."""
    return Image.new("RGB", (width, height), color)


def save_pil_to_tmp(tmp_path, img, filename="test.jpg"):
    p = tmp_path / filename
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_values(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_increase(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens), "Token counts should increase with mode size"


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(200, 150)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_landscape_image_padded(self):
        img = make_pil_image(400, 100)  # wide image
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_portrait_image_padded(self):
        img = make_pil_image(100, 400)  # tall image
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_square_image_padded(self):
        img = make_pil_image(300, 300)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_custom_pad_color(self):
        img = make_pil_image(100, 50)
        target = (200, 200)
        result = resize_and_pad(img, target, pad_color=(0, 0, 0))
        # top-left corner should be black (padding)
        px = result.getpixel((0, 0))
        assert px == (0, 0, 0)

    def test_output_is_rgb(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.mode == "RGB"

    def test_non_square_target(self):
        img = make_pil_image(200, 100)
        target = (800, 400)
        result = resize_and_pad(img, target)
        assert result.size == target


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def _img_path(self, tmp_path):
        img = make_pil_image(256, 256)
        self.path = save_pil_to_tmp(tmp_path, img, "sample.jpg")

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        tensor = load_image(self.path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        tensor = load_image(self.path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        tensor = load_image(self.path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_tiny(self):
        img = load_image(self.path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self):
        img = load_image(self.path, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.path, mode="invalid_mode")

    def test_tensor_dtype_float(self):
        tensor = load_image(self.path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized_range(self):
        # After ImageNet normalization values can be negative
        tensor = load_image(self.path, mode="tiny")
        # Just check it's not 0-255 range
        assert tensor.max().item() < 10.0

    def test_different_image_aspect_ratios(self, tmp_path):
        for w, h in [(100, 400), (400, 100), (256, 256)]:
            img = make_pil_image(w, h)
            path = save_pil_to_tmp(tmp_path, img, f"img_{w}_{h}.png")
            tensor = load_image(path, mode="base")
            th, tw = MODE_CONFIGS["base"]["size"]
            assert tensor.shape == (1, 3, th, tw)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_all_tokens(self):
        # Image exactly fits target → no padding → all tokens valid
        result = calculate_valid_tokens(512, 512, 512, 512, 100)
        assert result == 100

    def test_half_area_roughly_half_tokens(self):
        # 256x512 fits into 512x512: scale = min(2,1) = 1
        # scaled = 256x512, ratio = (256/512)*(512/512) = 0.5
        result = calculate_valid_tokens(256, 512, 512, 512, 100)
        assert result == 50

    def test_returns_integer(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 256)
        assert isinstance(result, int)

    def test_non_square_target(self):
        result = calculate_valid_tokens(400, 200, 800, 400, 200)
        # scale = min(2,2) = 2; scaled=800x400; ratio=1.0
        assert result == 200

    def test_small_image_large_canvas(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 400)
        # scale = min(1024/100, 1024/100) = 10.24 → scaled = 1024x1024
        # but capped at target → ratio = 1.0
        assert result == 400

    def test_zero_tokens_stays_zero(self):
        result = calculate_valid_tokens(100, 100, 512, 512, 0)
        assert result == 0
