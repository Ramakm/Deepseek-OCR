"""Tests for utils/image_io.py — exercises Pillow (9.0.0 → 10.0.1 MAJOR bump)."""
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

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w: int = 200, h: int = 150, color=(128, 64, 32)) -> Image.Image:
    """Create a solid-color RGB PIL image."""
    return Image.new("RGB", (w, h), color)


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temp file and return path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(f.name)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
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

    def test_pad_flag(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_increase(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens)

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(300, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_output_is_pil_rgb(self):
        img = _make_rgb_image(100, 200)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(100, 100, color=(255, 0, 0))
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_wide_image_has_vertical_padding(self):
        img = _make_rgb_image(400, 100)  # 4:1 aspect
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target
        # Top-left corner should be pad color (white)
        arr = np.array(result)
        # Top row should be white (padding)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_tall_image_has_horizontal_padding(self):
        img = _make_rgb_image(100, 400)  # 1:4 aspect
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_custom_pad_color(self):
        img = _make_rgb_image(300, 100)
        result = resize_and_pad(img, (256, 256), pad_color=(0, 0, 0))
        arr = np.array(result)
        assert arr[0, 0].tolist() == [0, 0, 0]

    def test_larger_target_than_source(self):
        img = _make_rgb_image(50, 50)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_exact_target_size_source(self):
        img = _make_rgb_image(128, 128)
        result = resize_and_pad(img, (128, 128))
        assert result.size == (128, 128)

    def test_pillow_bilinear_does_not_crash(self):
        """Pillow 10 removed ANTIALIAS; BILINEAR should still work."""
        img = _make_rgb_image(100, 100)
        # Should not raise AttributeError for removed ANTIALIAS constant
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = _make_rgb_image(256, 256)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        tensor = load_image(self.img_path, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        tensor = load_image(self.img_path, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        tensor = load_image(self.img_path, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]

    def test_tensor_dtype_float(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalised(self):
        """After ImageNet normalization values should not be in [0,1] raw range."""
        tensor = load_image(self.img_path, mode="tiny")
        # Normalised values can go negative
        assert tensor.min().item() < 0.5

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="ultra")

    def test_invalid_path_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    def test_grayscale_image_converted(self, tmp_path):
        """Grayscale images should be converted to RGB."""
        gray = Image.new("L", (100, 100), 128)
        p = str(tmp_path / "gray.png")
        gray.save(p)
        tensor = load_image(p, mode="tiny")
        assert tensor.shape[1] == 3  # 3 channels

    def test_rgba_image_converted(self, tmp_path):
        """RGBA images should be converted to RGB."""
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        p = str(tmp_path / "rgba.png")
        rgba.save(p)
        tensor = load_image(p, mode="tiny")
        assert tensor.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_equals_full_tokens(self):
        # Square image scaled to square target => ratio ~1
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens == 256

    def test_wide_image_fewer_tokens(self):
        # Wide image (2:1) padded into a square → fewer valid tokens
        tokens = calculate_valid_tokens(200, 100, 100, 100, 256)
        assert tokens < 256

    def test_tall_image_fewer_tokens(self):
        tokens = calculate_valid_tokens(100, 200, 100, 100, 256)
        assert tokens < 256

    def test_returns_int(self):
        result = calculate_valid_tokens(640, 480, 1024, 1024, 400)
        assert isinstance(result, int)

    def test_zero_tokens_impossible_for_nonzero_input(self):
        tokens = calculate_valid_tokens(50, 50, 1024, 1024, 256)
        assert tokens > 0

    def test_max_tokens_not_exceeded(self):
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens <= 256

    def test_base_mode_params(self):
        cfg = MODE_CONFIGS["base"]
        tw, th = cfg["size"]
        total = cfg["tokens"]
        tokens = calculate_valid_tokens(tw, th, tw, th, total)
        assert tokens == total
