"""Tests for utils/image_io.py — exercises Pillow (9→10 MAJOR bump) and torch APIs."""
import io
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=150, mode="RGB", color=(128, 64, 32)):
    """Create a simple in-memory PIL image."""
    img = Image.new(mode, (width, height), color)
    return img


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_config_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_pad_flag_values(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_pil_image(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_output_mode_rgb(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded_correctly(self):
        """Wide image should have horizontal bars of padding on top/bottom."""
        img = make_pil_image(400, 100)  # 4:1 ratio
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_portrait_image_padded_correctly(self):
        img = make_pil_image(100, 400)  # 1:4 ratio
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_custom_pad_color(self):
        img = make_pil_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        # Corner pixels should be the pad color (red)
        r, g, b = result.getpixel((0, 0))
        assert r == 255
        assert g == 0
        assert b == 0

    def test_square_image_no_padding(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# load_image  (exercises Pillow Image.open / convert / resize)
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = make_pil_image(300, 200)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    def test_returns_tensor_by_default(self):
        result = load_image(self.img_path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self):
        result = load_image(self.img_path, mode="tiny")
        # tiny size is (512,512), tensor is [1, 3, H, W]
        assert result.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        result = load_image(self.img_path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        result = load_image(self.img_path, mode="base")
        assert result.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        result = load_image(self.img_path, mode="large")
        assert result.shape == (1, 3, 1280, 1280)

    def test_return_pil_flag(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_tensor_dtype_float(self):
        result = load_image(self.img_path, mode="tiny")
        assert result.dtype == torch.float32

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="nonexistent")

    def test_normalized_values_range(self):
        """After ImageNet normalization values can be outside [0,1]."""
        result = load_image(self.img_path, mode="tiny")
        # Just check it is a finite tensor
        assert torch.isfinite(result).all()

    def test_png_image(self, tmp_path):
        img = make_pil_image(200, 200)
        path = str(tmp_path / "test.png")
        img.save(path)
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        """RGBA images should be converted to RGB without error."""
        img = Image.new("RGBA", (100, 100), (128, 128, 128, 200))
        path = str(tmp_path / "rgba.png")
        img.save(path)
        result = load_image(path, mode="tiny")
        assert result.shape[1] == 3  # channels = 3

    def test_greyscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        path = str(tmp_path / "grey.jpg")
        img.save(path)
        result = load_image(path, mode="tiny")
        assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_same_as_target_returns_all_tokens(self):
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_smaller_image_returns_fewer_tokens(self):
        result = calculate_valid_tokens(256, 256, 512, 512, 256)
        assert result < 256

    def test_result_is_integer(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 256)
        assert isinstance(result, int)

    def test_landscape_image(self):
        result = calculate_valid_tokens(1024, 512, 1024, 1024, 400)
        assert 0 < result <= 400

    def test_portrait_image(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 400)
        assert 0 < result <= 400

    def test_result_non_negative(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0
