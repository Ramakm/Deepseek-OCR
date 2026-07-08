"""Tests for utils/image_io.py"""
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
import tempfile
import os

# We need to be able to import from the project root
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

def make_temp_image(width=200, height=150, color=(128, 64, 32), fmt="JPEG"):
    """Create a temporary image file and return its path."""
    img = Image.new("RGB", (width, height), color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_pad_flag_correct(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_sizes_are_tuples_of_two_ints(self):
        for cfg in MODE_CONFIGS.values():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_tokens_are_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = Image.new("RGB", (300, 200))
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_output_is_pil_image(self):
        img = Image.new("RGB", (100, 100))
        result = resize_and_pad(img, (512, 512))
        assert isinstance(result, Image.Image)

    def test_padding_color_default_white(self):
        # Small image padded into larger canvas – corners should be near white
        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200))
        pixels = np.array(result)
        # Top-left corner should be white (padding)
        assert pixels[0, 0].tolist() == [255, 255, 255]

    def test_custom_padding_color(self):
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        pixels = np.array(result)
        assert pixels[0, 0].tolist() == [0, 0, 0]

    def test_aspect_ratio_preserved(self):
        # Wide image: 400x100 → padded into 400x400
        img = Image.new("RGB", (400, 100))
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_square_image_no_padding(self):
        img = Image.new("RGB", (100, 100), color=(10, 20, 30))
        result = resize_and_pad(img, (100, 100))
        # Should be identical in size
        assert result.size == (100, 100)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_no_padding(self):
        # When orig == target, valid_tokens == total_tokens
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256

    def test_partial_fill(self):
        # Tall image (100x200) fitted into (200x200): scale=1, fill = (100/200)*(200/200) = 0.5
        tokens = calculate_valid_tokens(100, 200, 200, 200, 100)
        assert 0 < tokens <= 100

    def test_returns_int(self):
        result = calculate_valid_tokens(640, 480, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_valid_tokens_leq_total(self):
        result = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert result <= 256

    def test_valid_tokens_positive(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result > 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        path = make_temp_image(200, 150)
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape(self, tmp_image):
        for mode in MODE_CONFIGS:
            tensor = load_image(tmp_image, mode=mode)
            h, w = MODE_CONFIGS[mode]["size"]
            assert tensor.shape == (1, 3, h, w), f"Failed for mode={mode}"

    def test_return_pil(self, tmp_image):
        img = load_image(tmp_image, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self, tmp_image):
        img = load_image(tmp_image, mode="small", return_pil=True)
        assert img.size == MODE_CONFIGS["small"]["size"]

    def test_tensor_dtype_float(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized_range(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        # After ImageNet normalization values can be negative
        assert tensor.min().item() < 1.0
        assert tensor.max().item() <= 3.0  # upper bound after norm

    def test_unknown_mode_raises(self, tmp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image, mode="ultra")

    def test_all_modes_work(self, tmp_image):
        for mode in MODE_CONFIGS:
            tensor = load_image(tmp_image, mode=mode)
            assert tensor.ndim == 4
