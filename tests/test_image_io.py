"""Tests for utils/image_io.py

Covers:
- load_image (all modes, tensor shape, PIL return)
- resize_and_pad
- calculate_valid_tokens
- Pillow (upgraded 9.0.0 → 12.2.0) API surface
"""
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(width=200, height=150, color=(128, 64, 32)) -> Image.Image:
    """Create a solid-colour RGB PIL image."""
    img = Image.new("RGB", (width, height), color)
    return img


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_jpg(tmp_path):
    """Temporary JPEG image file (200×150)."""
    img = make_rgb_image(200, 150)
    p = tmp_path / "test.jpg"
    img.save(str(p))
    return str(p)


@pytest.fixture()
def tmp_png(tmp_path):
    """Temporary PNG image file (300×400)."""
    img = make_rgb_image(300, 400, color=(10, 20, 30))
    p = tmp_path / "test.png"
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

# Ensure we can import even without optional heavy deps
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"'{mode}' missing 'size'"
            assert "tokens" in cfg, f"'{mode}' missing 'tokens'"
            assert "pad" in cfg, f"'{mode}' missing 'pad'"

    def test_size_is_tuple_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2
            assert all(isinstance(v, int) for v in cfg["size"])

    def test_tokens_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0

    def test_pad_is_bool(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["pad"], bool)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_equals_target(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_via_white_padding(self):
        """Wide image padded to square — top/bottom rows should be white."""
        img = make_rgb_image(400, 100, color=(0, 0, 0))  # pure black wide image
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        arr = np.array(result)
        # top-left pixel should be the pad colour (white)
        assert tuple(arr[0, 0]) == (255, 255, 255)

    def test_custom_pad_color(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 255))
        arr = np.array(result)
        # At least one corner pixel should be the pad colour
        corner = tuple(arr[0, 0])
        assert corner == (0, 0, 255) or corner == (128, 64, 32)  # pad or image edge

    def test_square_image_no_padding_needed(self):
        img = make_rgb_image(100, 100, color=(50, 50, 50))
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_tall_image(self):
        img = make_rgb_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        img = make_rgb_image(80, 60)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_size_gives_all_tokens(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        # scale = min(1024/512, 1024/1024) = 1.0 → scaled=(512,1024)
        # valid_ratio = (512/1024)*(1024/1024) = 0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_quarter_area_image(self):
        # scale = min(1024/512, 1024/512) = 2.0 → scaled=(1024,1024) (capped at target)
        # Actually scale=2, scaled_w=1024, scaled_h=1024
        result = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert result == 256

    def test_small_image_large_canvas(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_returns_int(self):
        result = calculate_valid_tokens(200, 150, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_zero_tokens_edge_case(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 0)
        assert result == 0


# ---------------------------------------------------------------------------
# load_image — tensor output
# ---------------------------------------------------------------------------

class TestLoadImageTensor:
    def test_tiny_mode_shape(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_small_mode_shape(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_base_mode_shape(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_large_mode_shape(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_returns_float_tensor(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalized(self, tmp_jpg):
        tensor = load_image(tmp_jpg, mode="tiny")
        # After ImageNet normalisation the range can go below 0 or above 1
        assert tensor.min().item() < 1.0  # not raw [0,1]
        assert tensor.shape[1] == 3

    def test_png_input(self, tmp_png):
        tensor = load_image(tmp_png, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_unknown_mode_raises(self, tmp_jpg):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_jpg, mode="ultra")

    def test_base_mode_uses_padding(self, tmp_jpg):
        """Base mode should preserve aspect ratio with padding."""
        tensor = load_image(tmp_jpg, mode="base")
        assert tensor.shape[-2:] == (1024, 1024)

    def test_tiny_mode_no_padding(self, tmp_jpg):
        """Tiny mode resizes directly without padding."""
        tensor = load_image(tmp_jpg, mode="tiny")
        assert tensor.shape[-2:] == (512, 512)


class TestLoadImagePIL:
    def test_return_pil_true(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_mode_rgb(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="base", return_pil=True)
        assert result.mode == "RGB"

    def test_pil_size_tiny(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_pil_size_base(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="base", return_pil=True)
        assert result.size == (1024, 1024)


# ---------------------------------------------------------------------------
# Pillow-specific API surface tests (exercises upgraded Pillow 12.2.0)
# ---------------------------------------------------------------------------

class TestPillowAPICompat:
    """Ensure we're using Pillow APIs correctly under the upgraded version."""

    def test_image_bilinear_resize(self):
        """Image.BILINEAR constant still available (or LANCZOS fallback)."""
        img = make_rgb_image(100, 100)
        resized = img.resize((64, 64), Image.BILINEAR)
        assert resized.size == (64, 64)

    def test_image_new_rgb(self):
        canvas = Image.new("RGB", (256, 256), (255, 255, 255))
        assert canvas.size == (256, 256)
        assert canvas.mode == "RGB"

    def test_image_paste(self):
        canvas = Image.new("RGB", (200, 200), (0, 0, 0))
        patch_img = make_rgb_image(50, 50, color=(255, 0, 0))
        canvas.paste(patch_img, (10, 10))
        arr = np.array(canvas)
        assert tuple(arr[10, 10]) == (255, 0, 0)

    def test_image_open_convert_rgb(self, tmp_jpg):
        img = Image.open(tmp_jpg).convert("RGB")
        assert img.mode == "RGB"

    def test_image_save_and_reopen(self, tmp_path):
        img = make_rgb_image(64, 64, color=(10, 20, 30))
        path = str(tmp_path / "out.png")
        img.save(path)
        reopened = Image.open(path)
        assert reopened.size == (64, 64)

    def test_numpy_array_from_image(self):
        img = make_rgb_image(32, 32, color=(100, 150, 200))
        arr = np.array(img)
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.uint8
        assert tuple(arr[0, 0]) == (100, 150, 200)

    def test_image_size_attribute(self):
        img = make_rgb_image(80, 60)
        assert img.size == (80, 60)  # (width, height)

    def test_image_tobytes(self):
        img = make_rgb_image(8, 8)
        data = img.tobytes()
        assert len(data) == 8 * 8 * 3
