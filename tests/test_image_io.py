"""Tests for utils/image_io.py — exercises PIL/Pillow, torch, and torchvision APIs."""
import io
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure project root is importable
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

def _make_pil_image(w: int = 200, h: int = 100, color: tuple = (128, 64, 32)) -> Image.Image:
    """Return a solid-colour RGB PIL image."""
    return Image.new("RGB", (w, h), color)


def _save_tmp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save PIL image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_exist(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_size_is_two_tuple(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2

    def test_pad_is_bool(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["pad"], bool)

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
        img = _make_pil_image(300, 150)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_output_is_rgb(self):
        img = _make_pil_image(100, 200)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"

    def test_landscape_image_padded_vertically(self):
        """Wide image should have white bars top/bottom."""
        img = _make_pil_image(200, 50, color=(0, 0, 0))  # pure black landscape
        result = resize_and_pad(img, (200, 200), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Top-left corner should be white (padding)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_portrait_image_padded_horizontally(self):
        """Tall image should have white bars left/right."""
        img = _make_pil_image(50, 200, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Top-left corner should be white (padding)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_square_image_no_padding(self):
        """Square → square: no padding needed, fills completely."""
        img = _make_pil_image(100, 100, color=(10, 20, 30))
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_custom_pad_color(self):
        img = _make_pil_image(10, 100, color=(0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        arr = np.array(result)
        # Left side should be red padding
        assert arr[0, 0, 0] == 255  # R channel

    def test_pillow_version_compatibility(self):
        """Ensure Image.BILINEAR still works (Pillow 12.x may rename to LANCZOS etc)."""
        img = _make_pil_image(64, 64)
        # Should not raise
        result = resize_and_pad(img, (32, 32))
        assert result.size == (32, 32)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def _tmp_image(self, w=200, h=200, color=(100, 150, 200)):
        img = _make_pil_image(w, h, color)
        return _save_tmp_image(img)

    def test_returns_tensor_by_default(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)
        os.unlink(path)

    def test_tensor_shape_4d(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="tiny")
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1   # batch
        assert tensor.shape[1] == 3   # RGB
        os.unlink(path)

    def test_tensor_spatial_matches_mode_tiny(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w
        os.unlink(path)

    def test_tensor_spatial_matches_mode_small(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w
        os.unlink(path)

    def test_tensor_spatial_matches_mode_base(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w
        os.unlink(path)

    def test_tensor_spatial_matches_mode_large(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape[2] == h
        assert tensor.shape[3] == w
        os.unlink(path)

    def test_return_pil_mode(self):
        path = self._tmp_image()
        result = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        os.unlink(path)

    def test_pil_mode_size_matches(self):
        path = self._tmp_image()
        result = load_image(path, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected
        os.unlink(path)

    def test_tensor_dtype_float(self):
        path = self._tmp_image()
        tensor = load_image(path, mode="tiny")
        assert tensor.dtype == torch.float32
        os.unlink(path)

    def test_tensor_is_normalized(self):
        """Values should be roughly in [-2.5, 2.5] after ImageNet normalisation."""
        path = self._tmp_image()
        tensor = load_image(path, mode="tiny")
        assert tensor.min().item() >= -3.0
        assert tensor.max().item() <= 3.0
        os.unlink(path)

    def test_invalid_mode_raises(self):
        path = self._tmp_image()
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="ultra")
        os.unlink(path)

    def test_grayscale_image_converted_to_rgb(self):
        """Grayscale input should still return 3-channel tensor."""
        img = Image.new("L", (100, 100), 128)
        path = _save_tmp_image(img, suffix=".png")
        tensor = load_image(path, mode="tiny")
        assert tensor.shape[1] == 3
        os.unlink(path)

    def test_png_file_works(self):
        img = _make_pil_image(150, 150)
        path = _save_tmp_image(img, suffix=".png")
        tensor = load_image(path, mode="small")
        assert tensor.ndim == 4
        os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_fills_all_tokens(self):
        """A square image in a square target uses all tokens (or very close)."""
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_width_image(self):
        """Image half the width → fewer valid tokens."""
        result = calculate_valid_tokens(50, 100, 100, 100, 256)
        assert 0 < result < 256

    def test_returns_int(self):
        result = calculate_valid_tokens(200, 100, 400, 400, 400)
        assert isinstance(result, int)

    def test_zero_tokens_for_tiny_image(self):
        """Very small image in large target → very few tokens."""
        result = calculate_valid_tokens(1, 1, 1000, 1000, 400)
        assert result >= 0

    def test_full_match(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_landscape_fewer_tokens(self):
        # wide but short — fits width, has vertical padding
        wide = calculate_valid_tokens(200, 50, 200, 200, 256)
        full = calculate_valid_tokens(200, 200, 200, 200, 256)
        assert wide < full

    def test_portrait_fewer_tokens(self):
        tall = calculate_valid_tokens(50, 200, 200, 200, 256)
        full = calculate_valid_tokens(200, 200, 200, 200, 256)
        assert tall < full
