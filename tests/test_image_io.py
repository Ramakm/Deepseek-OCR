"""Tests for utils/image_io.py - covers Pillow (9→12) and torch APIs."""
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(w=200, h=150, mode="RGB"):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def _save_temp_image(img: Image.Image, suffix=".jpg"):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    img.save(f.name)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_config_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_aspect_ratio_preserved_wide_image(self):
        img = _make_pil_image(400, 100)  # very wide
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)
        # Padding should be above/below
        arr = np.array(result)
        # Top rows should be white (padding color default)
        assert arr[0, 100, 0] == 255

    def test_square_image_no_padding_needed(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_custom_pad_color(self):
        img = _make_pil_image(50, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Corners should be black
        assert arr[0, 0, 0] == 0
        assert arr[0, 0, 1] == 0
        assert arr[0, 0, 2] == 0

    def test_returns_pil_image(self):
        img = _make_pil_image(100, 80)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_mode_is_rgb(self):
        img = _make_pil_image(80, 80)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_all_tokens_valid(self):
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_area_image(self):
        # scale = min(512/256, 512/256) = 2 → capped at 1 via min rule
        # orig fits inside target without scaling beyond 1
        result = calculate_valid_tokens(256, 256, 512, 512, 256)
        # scaled_w=256, scaled_h=256 → valid_ratio=(256/512)*(256/512)=0.25
        assert result == 64

    def test_wide_image(self):
        result = calculate_valid_tokens(800, 400, 1024, 1024, 400)
        # scale = min(1024/800, 1024/400) = 1.28
        # scaled_w = int(800*1.28)=1024, scaled_h=int(400*1.28)=512
        # valid_ratio = (1024/1024)*(512/1024) = 0.5
        assert result == 200

    def test_returns_integer(self):
        result = calculate_valid_tokens(300, 200, 1024, 1024, 400)
        assert isinstance(result, int)

    def test_non_zero_for_nonzero_input(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result > 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture()
    def temp_jpg(self):
        img = _make_pil_image(300, 200)
        path = _save_temp_image(img, ".jpg")
        yield path
        os.unlink(path)

    @pytest.fixture()
    def temp_png(self):
        img = _make_pil_image(128, 128)
        path = _save_temp_image(img, ".png")
        yield path
        os.unlink(path)

    def test_returns_tensor_shape(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="tiny")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3

    def test_spatial_size_matches_mode_tiny(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_spatial_size_matches_mode_small(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_spatial_size_matches_mode_base(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_spatial_size_matches_mode_large(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w

    def test_tensor_dtype_float32(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_return_pil_true(self, temp_jpg):
        result = load_image(temp_jpg, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_correct_size(self, temp_jpg):
        result = load_image(temp_jpg, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected

    def test_unknown_mode_raises(self, temp_jpg):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_jpg, mode="ultra")

    def test_png_input(self, temp_png):
        tensor = load_image(temp_png, mode="tiny")
        assert tensor.shape[1] == 3

    def test_normalized_values_range(self, temp_jpg):
        tensor = load_image(temp_jpg, mode="tiny")
        # After ImageNet normalization values can be outside [0,1] but
        # should be in a reasonable range
        assert tensor.min().item() > -5.0
        assert tensor.max().item() < 5.0

    def test_grayscale_converted_to_rgb(self):
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        path = _save_temp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(path)
