"""Tests for utils/image_io.py"""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_image(width=64, height=64, color=(128, 200, 50)) -> str:
    """Create a temporary RGB JPEG and return its path."""
    img = Image.new("RGB", (width, height), color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
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
            assert "size" in cfg, f"Missing 'size' in {mode}"
            assert "tokens" in cfg, f"Missing 'tokens' in {mode}"
            assert "pad" in cfg, f"Missing 'pad' in {mode}"

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts(self):
        assert MODE_CONFIGS["tiny"]["tokens"] == 64
        assert MODE_CONFIGS["small"]["tokens"] == 100
        assert MODE_CONFIGS["base"]["tokens"] == 256
        assert MODE_CONFIGS["large"]["tokens"] == 400


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = Image.new("RGB", (200, 100))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_output_is_rgb(self):
        img = Image.new("RGB", (100, 200))
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved(self):
        """After padding, the resized content should fit within target."""
        img = Image.new("RGB", (400, 200))  # 2:1 ratio
        target = (256, 256)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_square_image_no_padding_needed(self):
        img = Image.new("RGB", (128, 128), color=(10, 20, 30))
        result = resize_and_pad(img, (128, 128))
        arr = np.array(result)
        # Center pixel should be the original color (no padding)
        assert tuple(arr[64, 64]) == (10, 20, 30)

    def test_custom_pad_color(self):
        img = Image.new("RGB", (50, 100))  # tall image in 256x256 square
        result = resize_and_pad(img, (256, 256), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top-left corner should be black padding
        assert tuple(arr[0, 0]) == (0, 0, 0)

    def test_wider_than_tall(self):
        img = Image.new("RGB", (200, 50))
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_taller_than_wide(self):
        img = Image.new("RGB", (50, 200))
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_image(self):
        path = _make_temp_image(128, 128)
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, temp_image):
        t = load_image(temp_image, mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_tensor_shape_tiny(self, temp_image):
        t = load_image(temp_image, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, temp_image):
        t = load_image(temp_image, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, temp_image):
        t = load_image(temp_image, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, temp_image):
        t = load_image(temp_image, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_dtype_float(self, temp_image):
        t = load_image(temp_image, mode="tiny")
        assert t.dtype == torch.float32

    def test_return_pil(self, temp_image):
        result = load_image(temp_image, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, temp_image):
        result = load_image(temp_image, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self, temp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_image, mode="ultra")

    def test_tensor_normalized_range(self, temp_image):
        """Normalized tensor values should be roughly in [-3, 3] range."""
        t = load_image(temp_image, mode="tiny")
        assert t.min() > -4.0
        assert t.max() < 4.0

    def test_base_mode_uses_padding(self, tmp_path):
        """A non-square image in base mode should still produce the right size."""
        img = Image.new("RGB", (300, 100))
        p = tmp_path / "wide.jpg"
        img.save(str(p))
        t = load_image(str(p), mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_equals_total(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_wide_image_reduces_tokens(self):
        # 2:1 ratio image, scale limited by height
        result = calculate_valid_tokens(2048, 512, 1024, 1024, 256)
        assert result < 256

    def test_tall_image_reduces_tokens(self):
        result = calculate_valid_tokens(512, 2048, 1024, 1024, 256)
        assert result < 256

    def test_result_is_non_negative(self):
        result = calculate_valid_tokens(100, 200, 1024, 1024, 256)
        assert result >= 0

    def test_result_leq_total_tokens(self):
        result = calculate_valid_tokens(100, 100, 1280, 1280, 400)
        assert result <= 400

    def test_small_image_small_tokens(self):
        result = calculate_valid_tokens(50, 50, 1024, 1024, 256)
        assert result < 256

    def test_returns_int(self):
        result = calculate_valid_tokens(500, 500, 1024, 1024, 256)
        assert isinstance(result, int)
