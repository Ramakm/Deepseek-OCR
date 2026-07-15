"""Tests for utils/image_io.py — exercises Pillow (9→12) and torch/torchvision APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup so we can import project modules without installing the package
# ---------------------------------------------------------------------------
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

def _make_rgb_image(width: int = 200, height: int = 150, color=(128, 64, 32)) -> Image.Image:
    """Return a small solid-colour RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temporary file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_mode_has_required_keys(self, mode):
        cfg = MODE_CONFIGS[mode]
        assert "size" in cfg
        assert "tokens" in cfg
        assert "pad" in cfg

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_size_is_tuple_of_two_ints(self, mode):
        size = MODE_CONFIGS[mode]["size"]
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert all(isinstance(v, int) for v in size)

    def test_pad_modes(self):
        # base and large use padding; tiny and small do not
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_increase(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens), "Token counts should be non-decreasing"


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded_vertically(self):
        """Wide image should have white padding above/below."""
        img = _make_rgb_image(400, 100, color=(0, 0, 0))
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        # Top-left pixel should be white (padding), not black (image)
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_portrait_image_padded_horizontally(self):
        """Tall image should have white padding left/right."""
        img = _make_rgb_image(100, 400, color=(0, 0, 0))
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_square_image_no_effective_padding(self):
        """A square image fitted into a square target should fill it completely."""
        img = _make_rgb_image(200, 200, color=(100, 100, 100))
        result = resize_and_pad(img, (200, 200))
        # All pixels should be (approximately) grey — no white border
        arr = np.array(result)
        # Check corners
        for corner in [(0, 0), (0, 199), (199, 0), (199, 199)]:
            assert result.getpixel(corner) != (255, 255, 255)

    def test_custom_pad_color(self):
        img = _make_rgb_image(300, 100, color=(50, 50, 50))
        result = resize_and_pad(img, (300, 300), pad_color=(0, 128, 255))
        # Check that the pad colour appears in the top row
        top_pixel = result.getpixel((0, 0))
        assert top_pixel == (0, 128, 255)

    def test_returns_pil_image(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_no_padding(self):
        """When orig size == target size, all tokens are valid."""
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, total_tokens=256)
        assert result == 256

    def test_half_area_image(self):
        """A 512×1024 image in a 1024×1024 target should give ~128 tokens."""
        result = calculate_valid_tokens(512, 1024, 1024, 1024, total_tokens=256)
        # scale = min(1024/512, 1024/1024) = 1.0 (height constrains)
        # scaled_w = 512*1.0=512, scaled_h=1024*1.0=1024
        # valid_ratio = (512/1024)*(1024/1024) = 0.5
        assert result == 128

    def test_tiny_image_scaled_up(self):
        """A very small image will have low valid ratio."""
        result = calculate_valid_tokens(100, 100, 1024, 1024, total_tokens=256)
        assert 0 < result < 256

    def test_result_is_int(self):
        result = calculate_valid_tokens(200, 300, 1024, 1024, total_tokens=256)
        assert isinstance(result, int)

    def test_zero_tokens_edge_case(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, total_tokens=0)
        assert result == 0


# ---------------------------------------------------------------------------
# load_image — tensor output
# ---------------------------------------------------------------------------

class TestLoadImage:
    """Exercises Pillow image loading + numpy + torchvision Normalize."""

    @pytest.fixture()
    def sample_jpg(self):
        img = _make_rgb_image(256, 256)
        path = _save_temp_image(img, ".jpg")
        yield path
        os.unlink(path)

    @pytest.fixture()
    def sample_png(self):
        img = _make_rgb_image(300, 200)
        path = _save_temp_image(img, ".png")
        yield path
        os.unlink(path)

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_returns_tensor_shape(self, sample_jpg, mode):
        tensor = load_image(sample_jpg, mode=mode)
        h, w = MODE_CONFIGS[mode]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_float_tensor(self, sample_jpg):
        tensor = load_image(sample_jpg, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalized(self, sample_jpg):
        """After ImageNet normalisation the values should be in a reasonable range."""
        tensor = load_image(sample_jpg, mode="tiny")
        # ImageNet-normalised pixels usually fall in (-3, 3)
        assert tensor.min().item() >= -5.0
        assert tensor.max().item() <= 5.0

    def test_unknown_mode_raises(self, sample_jpg):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(sample_jpg, mode="nonexistent")

    def test_return_pil_mode(self, sample_jpg):
        result = load_image(sample_jpg, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, sample_jpg):
        result = load_image(sample_jpg, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_png_loads_correctly(self, sample_png):
        tensor = load_image(sample_png, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_rgba_image_converted_to_rgb(self):
        """RGBA images should be converted to RGB without error."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        path = _save_temp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3  # 3 channels
        finally:
            os.unlink(path)

    def test_grayscale_image_converted_to_rgb(self):
        img = Image.new("L", (100, 100), 128)
        path = _save_temp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(path)

    def test_base_mode_uses_padding(self, sample_jpg):
        """Base mode should produce exactly 1024×1024 output."""
        tensor = load_image(sample_jpg, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_large_mode_output_size(self, sample_jpg):
        tensor = load_image(sample_jpg, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)
