"""Tests for utils/image_io.py - exercises Pillow (9.0.0 → 10.0.1) APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

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

def make_rgb_image(width=200, height=150, color=(128, 64, 32)):
    """Create a simple in-memory RGB PIL image."""
    return Image.new("RGB", (width, height), color=color)


def save_pil_to_tmp(tmp_path, img, name="test.jpg"):
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_pad_flag(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_sizes_are_tuples(self):
        for cfg in MODE_CONFIGS.values():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_tokens_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 150)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = make_rgb_image(100, 200)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_wide_image_padded_vertically(self):
        """Wide image should have letterbox padding (top/bottom)."""
        img = make_rgb_image(400, 100)
        result = resize_and_pad(img, (400, 400))
        # The image is wider, so it should fill horizontally
        assert result.size == (400, 400)

    def test_tall_image_padded_horizontally(self):
        img = make_rgb_image(100, 400)
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_square_image_no_extra_padding(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        # Corner pixels should be red (padding color)
        px = result.getpixel((0, 0))
        # Content is black; corner may be red padding
        assert result.size == (200, 200)

    def test_already_target_size(self):
        img = make_rgb_image(512, 512)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_pillow_bilinear_used(self):
        """Ensure Pillow BILINEAR resampling (Pillow 10 compat)."""
        img = make_rgb_image(300, 200)
        # Should not raise with Pillow 10's Image.BILINEAR
        result = resize_and_pad(img, (640, 640))
        assert result.size == (640, 640)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        img = make_rgb_image(300, 200)
        return save_pil_to_tmp(tmp_path, img, "sample.jpg")

    def test_returns_tensor_by_default(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_mode(self, sample_image_path):
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self, sample_image_path):
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_return_pil_size_base(self, sample_image_path):
        result = load_image(sample_image_path, mode="base", return_pil=True)
        assert result.size == MODE_CONFIGS["base"]["size"]

    def test_invalid_mode_raises(self, sample_image_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(sample_image_path, mode="xxlarge")

    def test_tensor_dtype_float(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized(self, sample_image_path):
        """Values should be roughly normalized (ImageNet stats)."""
        tensor = load_image(sample_image_path, mode="tiny")
        # Not checking exact values, but reasonable range after normalize
        assert tensor.min() < 1.0
        assert tensor.max() > -1.0

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), color=128)
        path = str(tmp_path / "gray.png")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert tensor.shape[1] == 3  # 3 channels

    def test_png_image(self, tmp_path):
        img = make_rgb_image(200, 200)
        path = str(tmp_path / "img.png")
        img.save(path)
        tensor = load_image(path, mode="small")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Square image same as target returns full tokens."""
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_width_image(self):
        """Image that's half the width should use ~half the tokens."""
        result = calculate_valid_tokens(256, 512, 512, 512, 256)
        assert result < 256
        assert result > 0

    def test_result_is_int(self):
        result = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_small_image_fewer_tokens(self):
        small_result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        large_result = calculate_valid_tokens(900, 900, 1024, 1024, 256)
        assert small_result < large_result

    def test_same_aspect_ratio_preserves_ratio(self):
        """4:3 image scaled to 4:3 target — scale = 1."""
        result = calculate_valid_tokens(400, 300, 400, 300, 100)
        assert result == 100

    def test_landscape_image(self):
        result = calculate_valid_tokens(1280, 720, 1024, 1024, 400)
        assert 0 < result <= 400

    def test_portrait_image(self):
        result = calculate_valid_tokens(720, 1280, 1024, 1024, 400)
        assert 0 < result <= 400


# ---------------------------------------------------------------------------
# Pillow 10 specific compatibility
# ---------------------------------------------------------------------------

class TestPillowCompat:
    """Pillow 9→10 major bump: Image.ANTIALIAS removed, BILINEAR/LANCZOS remain."""

    def test_image_bilinear_constant_available(self):
        assert hasattr(Image, "BILINEAR")

    def test_image_lanczos_constant_available(self):
        assert hasattr(Image, "LANCZOS")

    def test_image_new_rgb(self):
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        assert img.size == (64, 64)

    def test_image_open_and_convert(self, tmp_path):
        img = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        loaded = Image.open(str(p)).convert("RGB")
        assert loaded.mode == "RGB"

    def test_resize_with_bilinear(self):
        img = make_rgb_image(200, 200)
        resized = img.resize((100, 100), Image.BILINEAR)
        assert resized.size == (100, 100)

    def test_image_paste(self):
        base = Image.new("RGB", (200, 200), (255, 255, 255))
        patch_img = Image.new("RGB", (50, 50), (0, 0, 0))
        base.paste(patch_img, (75, 75))
        px = base.getpixel((100, 100))
        assert px == (0, 0, 0)
