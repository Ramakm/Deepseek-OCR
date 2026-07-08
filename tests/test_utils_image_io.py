"""Tests for utils/image_io.py"""
import io
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(width=200, height=150, color=(128, 64, 32)):
    """Create a simple in-memory PIL RGB image."""
    img = Image.new("RGB", (width, height), color)
    return img


def save_tmp_image(tmp_path, width=200, height=150, color=(128, 64, 32), name="test.jpg"):
    """Save a PIL image to a temp file and return the path string."""
    img = make_rgb_image(width, height, color)
    path = tmp_path / name
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_mode_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_sizes_are_tuples_of_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            w, h = cfg["size"]
            assert isinstance(w, int) and isinstance(h, int)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_mode_rgb(self):
        img = make_rgb_image(300, 200)
        result = resize_and_pad(img, (640, 640))
        assert result.mode == "RGB"

    def test_padding_color_default_white(self):
        # Very thin image → lots of padding
        img = make_rgb_image(10, 100)
        result = resize_and_pad(img, (100, 100))
        arr = np.array(result)
        # Top-left corner should be white padding
        assert arr[0, 0, 0] == 255
        assert arr[0, 0, 1] == 255
        assert arr[0, 0, 2] == 255

    def test_custom_pad_color(self):
        img = make_rgb_image(10, 100)
        result = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top-left corner should be black padding
        assert arr[0, 0, 0] == 0

    def test_square_image_no_padding(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_landscape_image(self):
        img = make_rgb_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image(self):
        img = make_rgb_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size(self):
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_smaller_image_fewer_tokens(self):
        result = calculate_valid_tokens(50, 50, 100, 100, 256)
        assert result < 256
        assert result > 0

    def test_result_is_int(self):
        result = calculate_valid_tokens(200, 100, 400, 400, 400)
        assert isinstance(result, int)

    def test_landscape_image(self):
        # 200x100 into 200x200 target: scale=1.0 on width, 2.0 on height → min scale=1.0
        # scaled: 200x100, ratio = (200/200)*(100/200) = 0.5
        result = calculate_valid_tokens(200, 100, 200, 200, 400)
        assert result == int(400 * 0.5)

    def test_zero_tokens_input(self):
        result = calculate_valid_tokens(50, 50, 100, 100, 0)
        assert result == 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor(self, tmp_path):
        path = save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        path = save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_path):
        path = save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_path):
        path = save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_path):
        path = save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_dtype_float(self, tmp_path):
        path = save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_return_pil(self, tmp_path):
        path = save_tmp_image(tmp_path)
        img = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_invalid_mode_raises(self, tmp_path):
        path = save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="xxlarge")

    def test_normalization_applied(self, tmp_path):
        # A pure-white image normalized with ImageNet stats should not be all 1s
        path = save_tmp_image(tmp_path, color=(255, 255, 255))
        tensor = load_image(path, mode="tiny")
        # After normalization, values should differ from raw [0,1] range mean
        assert not torch.allclose(tensor, torch.ones_like(tensor))

    def test_tensor_values_range(self, tmp_path):
        path = save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        # After ImageNet normalization, pixel values can go below 0 or above 1
        # but should be finite
        assert torch.isfinite(tensor).all()

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        path = tmp_path / "gray.png"
        img.save(str(path))
        tensor = load_image(str(path), mode="tiny")
        assert tensor.shape[1] == 3

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (128, 64, 32, 200))
        path = tmp_path / "rgba.png"
        img.save(str(path))
        tensor = load_image(str(path), mode="tiny")
        assert tensor.shape[1] == 3
