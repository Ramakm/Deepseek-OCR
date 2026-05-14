"""Tests for utils/image_io.py — covers Pillow (9→10 major bump) usage."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

# Make project root importable
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

def make_rgb_image(w=200, h=150, color=(128, 64, 32)) -> Image.Image:
    """Create a small in-memory RGB image."""
    return Image.new("RGB", (w, h), color)


def save_tmp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_required_keys_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_each_mode_has_expected_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_size_is_tuple_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2
            assert all(isinstance(s, int) for s in cfg["size"])

    def test_base_and_large_use_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(300, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_output_mode_is_rgb(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"

    def test_landscape_image_gets_padded_vertically(self):
        # Wide image → should have top/bottom padding
        img = make_rgb_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image_gets_padded_horizontally(self):
        img = make_rgb_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color_applied(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        pad_color = (255, 0, 0)
        result = resize_and_pad(img, (200, 200), pad_color=pad_color)
        # Corner pixels should be pad_color (red)
        pixel = result.getpixel((0, 0))
        assert pixel == pad_color

    def test_square_image_no_pad_needed(self):
        img = make_rgb_image(64, 64)
        result = resize_and_pad(img, (128, 128))
        assert result.size == (128, 128)

    def test_uses_bilinear_resampling(self):
        """Ensure we can call resize with BILINEAR — Pillow 10 still supports it."""
        img = make_rgb_image(100, 100)
        # Should not raise even on Pillow 10 where ANTIALIAS was removed
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = make_rgb_image(300, 200)
        self.tmp_path = save_tmp_image(self.img)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    # --- happy path ---

    def test_returns_tensor_by_default(self):
        import torch
        tensor = load_image(self.tmp_path, mode="tiny")
        assert hasattr(tensor, "shape"), "Expected a torch.Tensor"
        assert len(tensor.shape) == 4  # [1, 3, H, W]

    def test_tensor_shape_tiny_mode(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        expected_h, expected_w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_small_mode(self):
        tensor = load_image(self.tmp_path, mode="small")
        expected_h, expected_w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_base_mode(self):
        tensor = load_image(self.tmp_path, mode="base")
        expected_h, expected_w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_large_mode(self):
        tensor = load_image(self.tmp_path, mode="large")
        expected_h, expected_w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w)

    def test_return_pil_flag(self):
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_matches_mode(self):
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        expected_size = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected_size

    def test_tensor_dtype_is_float(self):
        import torch
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalized(self):
        """After ImageNet normalization values can be negative / > 1."""
        tensor = load_image(self.tmp_path, mode="tiny")
        # Just verify it contains float values — not raw [0, 255]
        assert tensor.max().item() <= 10.0  # sensible range
        assert tensor.min().item() >= -10.0

    def test_png_file_supported(self):
        tmp_png = save_tmp_image(self.img, suffix=".png")
        try:
            tensor = load_image(tmp_png, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(tmp_png)

    # --- error cases ---

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="ultra")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    # --- Pillow 10 ANTIALIAS removal ---

    def test_pillow_bilinear_constant_accessible(self):
        """Pillow 10 removed Image.ANTIALIAS; Image.BILINEAR must still work."""
        assert hasattr(Image, "BILINEAR") or hasattr(Image, "Resampling")

    def test_pillow_resize_bilinear_does_not_raise(self):
        img = make_rgb_image(200, 200)
        # This is how image_io.py calls resize; must not raise on Pillow 10+
        resized = img.resize((64, 64), Image.BILINEAR)
        assert resized.size == (64, 64)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_fills_canvas(self):
        """A square image in a square canvas → all tokens valid."""
        valid = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert valid == 100

    def test_half_width_image(self):
        valid = calculate_valid_tokens(128, 256, 256, 256, 100)
        # scale = min(2, 1) = 1 → scaled = (128,256), ratio = 0.5*1 = 0.5
        assert valid == 50

    def test_landscape_image(self):
        valid = calculate_valid_tokens(400, 200, 400, 400, 200)
        # scale = min(1, 2) = 1 → scaled=(400,200), ratio=(400/400)*(200/400)=0.5
        assert valid == 100

    def test_valid_tokens_not_exceed_total(self):
        valid = calculate_valid_tokens(100, 100, 1000, 1000, 500)
        assert valid <= 500

    def test_valid_tokens_non_negative(self):
        valid = calculate_valid_tokens(50, 50, 200, 200, 64)
        assert valid >= 0

    def test_returns_int(self):
        valid = calculate_valid_tokens(300, 200, 400, 400, 100)
        assert isinstance(valid, int)
