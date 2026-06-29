"""Tests for utils/image_io.py - covers Pillow (9.0.0 → 10.0.1 MAJOR bump) usage."""
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

def _make_pil_image(width=200, height=150, color=(128, 64, 32)):
    """Return a simple RGB PIL image."""
    img = Image.new("RGB", (width, height), color=color)
    return img


def _save_temp_image(img, suffix=".jpg"):
    """Save PIL image to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


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
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_base_uses_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True

    def test_large_uses_pad(self):
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False

    def test_small_no_pad(self):
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_reasonable(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(300, 200)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_aspect_ratio_preserved_wide_image(self):
        """Wide image: padded on top/bottom."""
        img = _make_pil_image(400, 100)  # 4:1 aspect
        target = (400, 400)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_aspect_ratio_preserved_tall_image(self):
        img = _make_pil_image(100, 400)  # 1:4 aspect
        target = (400, 400)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_pad_color_default_white(self):
        img = _make_pil_image(10, 10, color=(0, 0, 0))
        target = (100, 100)
        result = resize_and_pad(img, target)
        # Corner pixels should be white (padding)
        corner = result.getpixel((0, 0))
        assert corner == (255, 255, 255)

    def test_custom_pad_color(self):
        img = _make_pil_image(10, 10, color=(0, 0, 0))
        target = (100, 100)
        result = resize_and_pad(img, target, pad_color=(0, 0, 255))
        corner = result.getpixel((0, 0))
        assert corner == (0, 0, 255)

    def test_returns_pil_image(self):
        img = _make_pil_image(50, 50)
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)

    def test_square_image_no_padding_needed(self):
        img = _make_pil_image(64, 64)
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    # Pillow 10 removed the ANTIALIAS constant; Image.BILINEAR should work
    def test_uses_bilinear_resampling(self):
        """Ensure resize_and_pad does not raise AttributeError with Pillow 10+."""
        img = _make_pil_image(200, 200)
        # Should not raise even with Pillow >= 10 (ANTIALIAS removed)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_img(self, tmp_path):
        img = _make_pil_image(300, 200)
        p = tmp_path / "test.jpg"
        img.save(str(p))
        self.img_path = str(p)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.img_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        tensor = load_image(self.img_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        tensor = load_image(self.img_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        tensor = load_image(self.img_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_flag(self):
        pil = load_image(self.img_path, mode="base", return_pil=True)
        assert isinstance(pil, Image.Image)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="ultra")

    def test_tensor_dtype_float(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized_range(self):
        """After ImageNet normalization values can be negative."""
        tensor = load_image(self.img_path, mode="tiny")
        # Normalized tensors can be < 0
        assert tensor.min().item() < 1.0  # not clipped to [0,1]

    def test_png_image(self, tmp_path):
        img = _make_pil_image(100, 100)
        p = tmp_path / "test.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[0] == 1

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """PIL convert('RGB') should handle grayscale input."""
        img = Image.new("L", (100, 100), 128)
        p = tmp_path / "gray.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3  # 3 channels

    def test_rgba_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_returns_total_tokens(self):
        # Image exactly fits target → ratio = 1.0
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_size_image(self):
        # 256x256 in 512x512 target: scale=0.5, ratio=(256/512)*(256/512)=0.25
        result = calculate_valid_tokens(256, 256, 512, 512, 256)
        assert result == 64

    def test_wide_image(self):
        # 1024x512 in 1024x1024: scale=min(1,2)=1 → new=(1024,512), ratio=(1024/1024)*(512/1024)=0.5
        result = calculate_valid_tokens(1024, 512, 1024, 1024, 256)
        assert result == 128

    def test_result_non_negative(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 400)
        assert result >= 0

    def test_result_does_not_exceed_total(self):
        result = calculate_valid_tokens(1000, 1000, 1024, 1024, 256)
        assert result <= 256

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 100)
        assert isinstance(result, int)
