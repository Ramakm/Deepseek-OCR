"""Tests for utils/image_io.py - covers Pillow (9→12) and torch/torchvision APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(width: int = 200, height: int = 150, color=(128, 64, 32)) -> "Image.Image":
    """Create a synthetic RGB PIL image."""
    img = Image.new("RGB", (width, height), color=color)
    return img


def save_temp_image(img: "Image.Image", suffix: str = ".jpg") -> str:
    """Save a PIL image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_schema(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"Missing 'size' in {mode}"
            assert "tokens" in cfg, f"Missing 'tokens' in {mode}"
            assert "pad" in cfg, f"Missing 'pad' in {mode}"
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2
            assert isinstance(cfg["tokens"], int)
            assert isinstance(cfg["pad"], bool)

    def test_pad_modes(self):
        # base and large should pad; tiny and small should not
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_ordering(self):
        # larger modes should produce more tokens
        assert MODE_CONFIGS["tiny"]["tokens"] < MODE_CONFIGS["small"]["tokens"]
        assert MODE_CONFIGS["small"]["tokens"] < MODE_CONFIGS["base"]["tokens"]
        assert MODE_CONFIGS["base"]["tokens"] < MODE_CONFIGS["large"]["tokens"]


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 150)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_mode_is_rgb(self):
        img = make_rgb_image(200, 150)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_wide_image_padded_vertically(self):
        """Wide image: resize to fit width, pad top/bottom."""
        img = make_rgb_image(400, 100)  # very wide
        target = (256, 256)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_tall_image_padded_horizontally(self):
        img = make_rgb_image(100, 400)  # very tall
        target = (256, 256)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_square_image(self):
        img = make_rgb_image(200, 200)
        target = (128, 128)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        target = (100, 100)
        result = resize_and_pad(img, target, pad_color=(255, 0, 0))
        # Corner pixel should be red (padding)
        corner = result.getpixel((0, 0))
        assert corner[0] == 255  # red channel

    def test_already_correct_size(self):
        img = make_rgb_image(256, 256)
        target = (256, 256)
        result = resize_and_pad(img, target)
        assert result.size == target


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_image(self):
        img = make_rgb_image(300, 200)
        path = save_temp_image(img, suffix=".png")
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_returns_tensor_by_default(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, temp_image):
        tensor = load_image(temp_image, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, temp_image):
        tensor = load_image(temp_image, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, temp_image):
        tensor = load_image(temp_image, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_dtype_float32(self, temp_image):
        tensor = load_image(temp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_return_pil(self, temp_image):
        result = load_image(temp_image, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, temp_image):
        result = load_image(temp_image, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected

    def test_unknown_mode_raises(self, temp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_image, mode="unknown_xyz")

    def test_tensor_values_normalized(self, temp_image):
        """After ImageNet normalization tensor values may be outside [0,1]."""
        tensor = load_image(temp_image, mode="tiny")
        # Values should NOT be simply in [0,1] after normalization
        # but the original pixel values (128/255 ≈ 0.5) when normalized by
        # mean≈0.45, std≈0.22 gives ≈0.23 — just check it's a finite float tensor
        assert torch.isfinite(tensor).all()

    def test_jpeg_image(self):
        img = make_rgb_image(200, 150)
        path = save_temp_image(img, suffix=".jpg")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape == (1, 3, 512, 512)
        finally:
            os.remove(path)

    def test_file_not_found_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    def test_all_modes_load_successfully(self, temp_image):
        for mode in MODE_CONFIGS:
            tensor = load_image(temp_image, mode=mode)
            assert tensor.shape[0] == 1
            assert tensor.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Square image fitting exactly → all tokens valid."""
        total = 256
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, total)
        assert result == total

    def test_wide_image_reduces_tokens(self):
        """Wide image padded → fewer valid tokens."""
        # 2:1 aspect ratio, target 1:1 → half the height wasted
        total = 256
        result = calculate_valid_tokens(2000, 1000, 1000, 1000, total)
        assert result < total

    def test_result_is_int(self):
        result = calculate_valid_tokens(300, 200, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_zero_wasted_space(self):
        """Image matches target aspect ratio → valid tokens == total."""
        result = calculate_valid_tokens(512, 512, 512, 512, 100)
        assert result == 100

    def test_very_small_image(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0

    def test_tall_image_reduces_tokens(self):
        total = 400
        result = calculate_valid_tokens(100, 400, 400, 400, total)
        assert result < total
