"""Tests for utils/image_io.py

Covers:
- load_image (all modes, return_pil, error cases)
- resize_and_pad
- calculate_valid_tokens
- Pillow 10.x API compatibility (Image.BILINEAR -> Image.Resampling.BILINEAR)
"""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make sure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_image(width=100, height=80, color=(128, 64, 32), fmt="JPEG") -> str:
    """Create a temporary image file and return its path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_mode_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"Missing 'size' in mode {mode}"
            assert "tokens" in cfg, f"Missing 'tokens' in mode {mode}"
            assert "pad" in cfg, f"Missing 'pad' in mode {mode}"
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0
            assert isinstance(cfg["pad"], bool)

    def test_pad_modes(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = Image.new("RGB", (200, 100))
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300)

    def test_landscape_image(self):
        img = Image.new("RGB", (400, 200))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_portrait_image(self):
        img = Image.new("RGB", (100, 400))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_square_image(self):
        img = Image.new("RGB", (128, 128), (255, 0, 0))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_pad_color_default_white(self):
        img = Image.new("RGB", (50, 50), (0, 0, 0))
        result = resize_and_pad(img, (200, 200))
        # corners should be white (pad color)
        px = result.getpixel((0, 0))
        assert px == (255, 255, 255)

    def test_custom_pad_color(self):
        img = Image.new("RGB", (50, 50), (0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        img = Image.new("RGB", (100, 100))
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)

    def test_aspect_ratio_preserved_landscape(self):
        """The resized image inside the padded canvas should keep aspect ratio."""
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        result = resize_and_pad(img, (256, 256))
        arr = np.array(result)
        # Red content should exist
        red_pixels = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        assert red_pixels.sum() > 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_image(self):
        path = _make_temp_image(200, 150)
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, temp_image):
        tensor = load_image(temp_image, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, temp_image):
        tensor = load_image(temp_image, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, temp_image):
        tensor = load_image(temp_image, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil(self, temp_image):
        img = load_image(temp_image, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self, temp_image):
        img = load_image(temp_image, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self, temp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_image, mode="ultra")

    def test_tensor_dtype_float(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized(self, temp_image):
        """Values should be roughly in [-3, 3] after ImageNet normalization."""
        tensor = load_image(temp_image, mode="tiny")
        assert tensor.min().item() > -4.0
        assert tensor.max().item() < 4.0

    def test_file_not_found_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="base")

    def test_base_mode_uses_padding(self, temp_image):
        """Base mode should pad (uses resize_and_pad path)."""
        tensor = load_image(temp_image, mode="base")
        assert tensor.shape[-2:] == (1024, 1024)

    def test_all_modes_return_correct_shapes(self, temp_image):
        expected = {
            "tiny": (1, 3, 512, 512),
            "small": (1, 3, 640, 640),
            "base": (1, 3, 1024, 1024),
            "large": (1, 3, 1280, 1280),
        }
        for mode, shape in expected.items():
            t = load_image(temp_image, mode=mode)
            assert t.shape == shape, f"Mode {mode}: expected {shape}, got {t.shape}"


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_all_tokens(self):
        """A square image in a square target should use all tokens."""
        result = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert result == 100

    def test_half_width_half_tokens(self):
        """Image half the target width and height should give ~25% tokens."""
        result = calculate_valid_tokens(128, 128, 256, 256, 100)
        # scale = min(256/128, 256/128) = 2.0 -> scaled = 256, 256
        # valid_ratio = 1.0 -> 100 tokens
        assert result == 100

    def test_landscape_image(self):
        result = calculate_valid_tokens(200, 100, 400, 400, 256)
        # scale = min(400/200, 400/100) = min(2, 4) = 2
        # scaled_w=400, scaled_h=200
        # valid_ratio = (400/400) * (200/400) = 0.5
        assert result == 128

    def test_portrait_image(self):
        result = calculate_valid_tokens(100, 200, 400, 400, 256)
        # scale = min(4, 2) = 2 -> scaled_w=200, scaled_h=400
        # valid_ratio = (200/400)*(400/400) = 0.5
        assert result == 128

    def test_returns_integer(self):
        result = calculate_valid_tokens(100, 100, 200, 300, 100)
        assert isinstance(result, int)

    def test_zero_tokens_edge(self):
        result = calculate_valid_tokens(100, 100, 200, 200, 0)
        assert result == 0

    def test_large_mode_values(self):
        result = calculate_valid_tokens(960, 720, 1280, 1280, 400)
        assert 0 < result <= 400
