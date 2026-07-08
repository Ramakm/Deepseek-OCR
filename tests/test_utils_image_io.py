"""Tests for utils/image_io.py — covers Pillow (9→10) and torch APIs."""
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the project root is importable without installing the package
# ---------------------------------------------------------------------------
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

def _make_pil_image(width=100, height=80, color=(128, 64, 32)):
    """Create a small solid-color RGB PIL image."""
    img = Image.new("RGB", (width, height), color)
    return img


def _save_pil_to_tmp(tmp_path: Path, img: Image.Image, name="test.jpg") -> str:
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ===========================================================================
# MODE_CONFIGS structure
# ===========================================================================

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"'size' missing for mode {mode}"
            assert "tokens" in cfg, f"'tokens' missing for mode {mode}"
            assert "pad" in cfg, f"'pad' missing for mode {mode}"

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ===========================================================================
# resize_and_pad
# ===========================================================================

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_wide_image_padded_vertically(self):
        """Wider-than-target image should have horizontal bars of padding."""
        img = _make_pil_image(200, 50)  # very wide
        result = resize_and_pad(img, (100, 100), pad_color=(255, 255, 255))
        assert result.size == (100, 100)
        # Top row should be white (padding)
        top_pixel = result.getpixel((50, 0))
        assert top_pixel == (255, 255, 255)

    def test_tall_image_padded_horizontally(self):
        img = _make_pil_image(50, 200)  # very tall
        result = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        assert result.size == (100, 100)
        left_pixel = result.getpixel((0, 50))
        assert left_pixel == (0, 0, 0)

    def test_square_image_no_extra_padding(self):
        img = _make_pil_image(100, 100, color=(200, 200, 200))
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_custom_pad_color(self):
        img = _make_pil_image(10, 100)
        result = resize_and_pad(img, (64, 64), pad_color=(255, 0, 0))
        # Left column should be red padding
        pixel = result.getpixel((0, 32))
        assert pixel == (255, 0, 0)

    def test_output_is_pil_image(self):
        img = _make_pil_image()
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)

    def test_aspect_ratio_preserved_within_canvas(self):
        """The non-padding area should respect the original aspect ratio."""
        img = _make_pil_image(200, 100)  # 2:1
        result = resize_and_pad(img, (100, 100))
        # The resized image should be 100×50, centred
        # Check that pixels at paste_y - 1 are padding and at paste_y are image
        arr = np.array(result)
        # Row 24 should be white (padding), row 25 should be image content
        assert arr[24, 50].tolist() == [255, 255, 255]


# ===========================================================================
# load_image — using a real temporary file
# ===========================================================================

class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        for mode in MODE_CONFIGS:
            result = load_image(path, mode=mode)
            assert result.ndim == 4, f"Expected 4-D tensor for mode {mode}"
            assert result.shape[0] == 1  # batch dim
            assert result.shape[1] == 3  # RGB channels
            h_cfg, w_cfg = MODE_CONFIGS[mode]["size"]
            assert result.shape[2] == h_cfg
            assert result.shape[3] == w_cfg

    def test_returns_pil_when_requested(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_return_size(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_invalid_mode_raises_value_error(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="nonexistent_mode")

    def test_tensor_dtype_float32(self, tmp_path):
        img = _make_pil_image()
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_normalised(self, tmp_path):
        """After ImageNet normalisation values may fall outside [0,1]."""
        # Create a white image; after normalization the tensor should be roughly
        # (1-mean)/std which is ≈ 2.64 for channel 0
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        # White → normalised > 1 (since raw = 1.0, mean ~0.485, std ~0.229)
        assert result[0, 0].mean().item() > 1.0

    def test_grayscale_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 80), 128)
        path = _save_pil_to_tmp(tmp_path, img, name="gray.png")
        result = load_image(path, mode="tiny")
        assert result.shape[1] == 3

    def test_base_mode_pads(self, tmp_path):
        """Base mode should pad to preserve aspect ratio."""
        img = _make_pil_image(200, 50)
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="base")
        assert result.shape[2:] == torch.Size(list(MODE_CONFIGS["base"]["size"]))

    def test_tiny_mode_no_pad(self, tmp_path):
        img = _make_pil_image(200, 50)
        path = _save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        assert result.shape[2:] == torch.Size(list(MODE_CONFIGS["tiny"]["size"]))


# ===========================================================================
# calculate_valid_tokens
# ===========================================================================

class TestCalculateValidTokens:
    def test_square_no_padding_all_valid(self):
        # If orig == target, all tokens should be valid
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens == 256

    def test_half_width_half_valid(self):
        # orig 50x100, target 100x100 → scale=1, scaled=50x100
        # valid_ratio = (50/100) * (100/100) = 0.5
        tokens = calculate_valid_tokens(50, 100, 100, 100, 256)
        assert tokens == 128

    def test_returns_int(self):
        result = calculate_valid_tokens(80, 60, 100, 100, 256)
        assert isinstance(result, int)

    def test_large_original_clipped_to_all_tokens(self):
        # orig larger than target → scale < 1 → valid_ratio < 1
        tokens = calculate_valid_tokens(200, 200, 100, 100, 256)
        assert tokens == 256  # scale=0.5 → scaled=100x100 → ratio=1.0

    def test_wide_image(self):
        # orig 200x100, target 100x100 → scale=min(0.5, 1.0)=0.5
        # scaled = 100x50 → valid_ratio = (100/100)*(50/100) = 0.5
        tokens = calculate_valid_tokens(200, 100, 100, 100, 400)
        assert tokens == 200

    def test_total_tokens_zero(self):
        result = calculate_valid_tokens(100, 100, 100, 100, 0)
        assert result == 0
