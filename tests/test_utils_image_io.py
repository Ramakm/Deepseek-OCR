"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) and torch APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helper: create a small in-memory RGB PIL image
# ---------------------------------------------------------------------------

def make_pil_image(width=64, height=48, color=(128, 64, 32)):
    img = Image.new("RGB", (width, height), color)
    return img


def save_pil_to_tmp(tmp_path, name="test.jpg", width=64, height=48):
    img = make_pil_image(width, height)
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_base_uses_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True

    def test_large_uses_padding(self):
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False

    def test_small_no_padding(self):
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(100, 50)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_output_is_rgb(self):
        img = make_pil_image(100, 50)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"

    def test_square_input_no_padding(self):
        img = make_pil_image(64, 64)
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_wide_image_padded_vertically(self):
        img = make_pil_image(200, 50)
        result = resize_and_pad(img, (200, 200))
        # After scaling 200x50 to fit 200x200: scale = min(1.0, 4.0) = 1.0
        # new_w=200, new_h=50 → top/bottom padding
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = make_pil_image(10, 10, color=(0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        # Corner pixel should be the pad color (red)
        corner = result.getpixel((0, 0))
        assert corner == (255, 0, 0)

    def test_portrait_image(self):
        img = make_pil_image(50, 200)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)


class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny")
        # tiny size = (512, 512)
        assert result.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="base")
        assert result.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="large")
        assert result.shape == (1, 3, 1280, 1280)

    def test_return_pil(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="nonexistent")

    def test_tensor_dtype_float(self, tmp_path):
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_values_normalized(self, tmp_path):
        # After ImageNet normalization values may be outside [0,1]
        # but should be finite
        path = save_pil_to_tmp(tmp_path)
        result = load_image(path, mode="tiny")
        assert torch.isfinite(result).all()

    def test_png_file(self, tmp_path):
        path = save_pil_to_tmp(tmp_path, name="test.png")
        result = load_image(path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (32, 32), (10, 20, 30, 128))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        result = load_image(str(p), mode="tiny")
        assert result.shape == (1, 3, 512, 512)


class TestCalculateValidTokens:
    def test_same_size_returns_total(self):
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_width(self):
        # orig=50x100, target=100x100: scale=min(2,1)=1 → scaled=50x100
        # valid_ratio=(50/100)*(100/100)=0.5 → 128
        result = calculate_valid_tokens(50, 100, 100, 100, 256)
        assert result == 128

    def test_returns_int(self):
        result = calculate_valid_tokens(80, 60, 100, 100, 256)
        assert isinstance(result, int)

    def test_small_image_fewer_tokens(self):
        result_small = calculate_valid_tokens(30, 30, 100, 100, 256)
        result_large = calculate_valid_tokens(90, 90, 100, 100, 256)
        assert result_small < result_large

    def test_zero_tokens_edge(self):
        result = calculate_valid_tokens(100, 100, 100, 100, 0)
        assert result == 0
