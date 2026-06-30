"""Tests for utils/image_io.py — exercises PIL (pillow 10.x) APIs."""
import io
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of how pytest is invoked
# ---------------------------------------------------------------------------
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

def _make_rgb_image(w: int = 100, h: int = 80, color=(128, 64, 32)) -> Image.Image:
    """Create a small solid-color RGB PIL image."""
    img = Image.new("RGB", (w, h), color)
    return img


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"Mode {mode} missing 'size'"
            assert "tokens" in cfg, f"Mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"Mode {mode} missing 'pad'"

    def test_pad_modes(self):
        # Only base and large should use padding per the paper
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_sizes_are_square_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            w, h = cfg["size"]
            assert w > 0 and h > 0

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        out = resize_and_pad(img, (512, 512))
        assert out.size == (512, 512)

    def test_mode_is_rgb(self):
        img = _make_rgb_image(50, 50)
        out = resize_and_pad(img, (256, 256))
        assert out.mode == "RGB"

    def test_landscape_image_padded_vertically(self):
        """Wide image → letterbox padding top/bottom."""
        img = _make_rgb_image(400, 100)  # 4:1 aspect
        target = (400, 400)
        out = resize_and_pad(img, target)
        assert out.size == target
        arr = np.array(out)
        # Top row should be white (default pad colour)
        assert tuple(arr[0, 0]) == (255, 255, 255)

    def test_portrait_image_padded_horizontally(self):
        """Tall image → pillarbox padding left/right."""
        img = _make_rgb_image(100, 400)
        target = (400, 400)
        out = resize_and_pad(img, target)
        assert out.size == target

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 100)
        out = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(out)
        # Corner should be black
        assert tuple(arr[0, 0]) == (0, 0, 0)

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(100, 100)
        out = resize_and_pad(img, (100, 100))
        assert out.size == (100, 100)

    def test_pillow_bilinear_constant_used(self):
        """Ensure Image.BILINEAR (or LANCZOS equivalent) is accepted in Pillow 10."""
        img = _make_rgb_image(64, 64)
        # Pillow 10 removed some old constants but BILINEAR still exists
        result = img.resize((128, 128), Image.BILINEAR)
        assert result.size == (128, 128)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        """Write a small JPEG to disk and clean up afterwards."""
        img = _make_rgb_image(200, 150)
        self.path = _save_temp_image(img, ".jpg")
        yield
        os.unlink(self.path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.path, mode="tiny")
        # tiny mode: (512, 512) → [1, 3, 512, 512]
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        tensor = load_image(self.path, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        tensor = load_image(self.path, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        tensor = load_image(self.path, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil_flag(self):
        result = load_image(self.path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_tensor_dtype_float(self):
        tensor = load_image(self.path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_normalized_range(self):
        """After ImageNet normalization the pixel values may be outside [0,1]."""
        tensor = load_image(self.path, mode="tiny")
        # Not checking exact range but confirming it's been processed
        assert tensor.min() < 1.0  # definitely below 255

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.path, mode="ultramax")

    def test_png_image_loads(self):
        img = _make_rgb_image(80, 60)
        path = _save_temp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert isinstance(tensor, torch.Tensor)
        finally:
            os.unlink(path)

    def test_rgba_image_converted_to_rgb(self):
        """PIL should convert RGBA to RGB without crashing."""
        img = Image.new("RGBA", (100, 100), (10, 20, 30, 128))
        path = _save_temp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(path)

    def test_all_modes_run_without_error(self):
        for mode in MODE_CONFIGS:
            tensor = load_image(self.path, mode=mode)
            assert isinstance(tensor, torch.Tensor)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_same_as_target(self):
        # Perfect fit → all tokens valid
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_landscape_image_fewer_tokens(self):
        result = calculate_valid_tokens(200, 100, 512, 512, 256)
        assert 0 < result < 256

    def test_portrait_image_fewer_tokens(self):
        result = calculate_valid_tokens(100, 300, 512, 512, 256)
        assert 0 < result < 256

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 200, 1024, 1024, 400)
        assert isinstance(result, int)

    def test_zero_tokens_for_degenerate_case(self):
        # If orig_w and orig_h are extremely small, valid_tokens approaches 0
        result = calculate_valid_tokens(1, 1, 1024, 1024, 256)
        assert result >= 0

    def test_large_mode_tokens(self):
        result = calculate_valid_tokens(1280, 1280, 1280, 1280, 400)
        assert result == 400

    def test_asymmetric_target(self):
        result = calculate_valid_tokens(800, 400, 1024, 512, 200)
        assert 0 <= result <= 200
