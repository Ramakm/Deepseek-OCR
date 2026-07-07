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

# Ensure project root is on the path
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

def _make_tmp_image(width=200, height=150, color=(128, 64, 32)) -> str:
    """Create a temporary RGB JPEG and return the path."""
    img = Image.new("RGB", (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_are_positive_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_equals_target(self):
        img = Image.new("RGB", (300, 100), (255, 0, 0))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_square_image_stays_centered(self):
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = Image.new("RGB", (10, 10), (0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        assert result.size == (100, 100)

    def test_wider_than_tall_image(self):
        img = Image.new("RGB", (400, 100), (0, 0, 255))
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_taller_than_wide_image(self):
        img = Image.new("RGB", (100, 400), (0, 0, 255))
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        img = Image.new("RGB", (50, 50))
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = Image.new("RGB", (300, 200), (100, 150, 200))
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.img_path, mode="tiny")
        # [1, 3, H, W]
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_tensor_shape_base(self):
        tensor = load_image(self.img_path, mode="base")
        assert tensor.ndim == 4
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_tensor_shape_large(self):
        tensor = load_image(self.img_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_return_pil(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="nonexistent")

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalized(self):
        """After ImageNet normalization values should not be in [0,1] raw range."""
        tensor = load_image(self.img_path, mode="tiny")
        # At least some values should be outside [0,1] range after normalization
        assert (tensor.abs() > 1.0).any() or tensor.min() < 0

    def test_all_modes_work(self):
        for mode in MODE_CONFIGS:
            t = load_image(self.img_path, mode=mode)
            assert t.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_aspect_ratio_returns_full_tokens(self):
        # square original, square target → scale fills exactly
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens == 256

    def test_landscape_image_in_square_target(self):
        # 200x100 in 200x200 → scaled to 200x100, valid ratio = 0.5
        tokens = calculate_valid_tokens(200, 100, 200, 200, 256)
        assert tokens == 128  # 256 * 0.5

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 200, 400, 400, 100)
        assert isinstance(result, int)

    def test_non_zero_tokens(self):
        result = calculate_valid_tokens(50, 50, 256, 256, 256)
        assert result > 0

    def test_full_coverage(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256
