"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=100, height=80, color=(128, 64, 32)):
    """Return a small RGB PIL image."""
    img = Image.new("RGB", (width, height), color)
    return img


def save_pil_to_tmp(tmp_path, img, name="test.jpg"):
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"mode {mode} missing 'size'"
            assert "tokens" in cfg, f"mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"mode {mode} missing 'pad'"

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2

    def test_pad_flag_semantics(self):
        # base and large use padding; tiny and small do not
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_tokens_are_positive_integers(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(200, 100)
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300)

    def test_output_mode_rgb(self):
        img = make_pil_image(50, 50)
        result = resize_and_pad(img, (100, 100))
        assert result.mode == "RGB"

    def test_wide_image_padded_vertically(self):
        """Wide image → letterbox padding top/bottom."""
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Top rows should be white padding (255, 255, 255)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_tall_image_padded_horizontally(self):
        """Tall image → pillarbox padding left/right."""
        img = make_pil_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_square_image_no_padding(self):
        """Square image resized to square target → no empty padding."""
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        result = resize_and_pad(img, (100, 100))
        arr = np.array(result)
        # No row should be entirely white (pad color) when image fills target
        assert not np.all(arr[0] == 255)

    def test_custom_pad_color(self):
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        assert arr[0, 0].tolist() == [0, 0, 0]

    def test_aspect_ratio_preserved(self):
        """After resize+pad the non-padding content keeps aspect ratio."""
        img = make_pil_image(200, 100)  # 2:1 ratio
        result = resize_and_pad(img, (400, 400))
        # Scaled width = 400, scaled height = 200 → 200px of padding (100 each side)
        arr = np.array(result)
        # Row 0 is padding
        assert arr[0, 200].tolist() == [255, 255, 255]
        # Middle row (200) should NOT be fully white
        assert not np.all(arr[200] == [255, 255, 255])


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_fills_target(self):
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens == 256

    def test_half_area_image(self):
        # 50x50 scaled to fit in 100x100 → scale=0.5 → 50x50/100x100 = 25% area
        tokens = calculate_valid_tokens(50, 50, 100, 100, 256)
        assert tokens == 64  # 256 * 0.25

    def test_wide_image(self):
        # 200x100 in 100x100: scale=min(0.5,1)=0.5 → 100x50 → 100*50/(100*100)=0.5
        tokens = calculate_valid_tokens(200, 100, 100, 100, 100)
        assert tokens == 50

    def test_result_is_integer(self):
        tokens = calculate_valid_tokens(123, 456, 1000, 1000, 400)
        assert isinstance(tokens, int)

    def test_large_image_full_tokens(self):
        # Large image fills the target → valid_tokens == total_tokens
        tokens = calculate_valid_tokens(1000, 1000, 1000, 1000, 400)
        assert tokens == 400


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_path):
        img = make_pil_image(80, 80)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_path):
        img = make_pil_image(120, 90)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_path):
        img = make_pil_image(64, 64)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_return_pil_flag(self, tmp_path):
        img = make_pil_image(50, 50)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self, tmp_path):
        img = make_pil_image(50, 50)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_normalized_values_in_range(self, tmp_path):
        img = make_pil_image(64, 64)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny")
        # After ImageNet normalization values typically in [-3, 3]
        assert result.min().item() > -5.0
        assert result.max().item() < 5.0

    def test_dtype_float32(self, tmp_path):
        img = make_pil_image(64, 64)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny")
        assert result.dtype == torch.float32

    def test_unknown_mode_raises(self, tmp_path):
        img = make_pil_image(64, 64)
        p = save_pil_to_tmp(tmp_path, img)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="nonexistent")

    def test_png_image_loads(self, tmp_path):
        img = make_pil_image(64, 64)
        p = save_pil_to_tmp(tmp_path, img, name="test.png")
        result = load_image(p, mode="tiny")
        assert result.shape[0] == 1

    def test_rgba_image_converts_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (64, 64), (100, 100, 100, 128))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        result = load_image(str(p), mode="tiny")
        assert result.shape[1] == 3  # 3 channels

    def test_grayscale_image_converts_to_rgb(self, tmp_path):
        img = Image.new("L", (64, 64), 128)
        p = tmp_path / "gray.png"
        img.save(str(p))
        result = load_image(str(p), mode="tiny")
        assert result.shape[1] == 3

    def test_all_modes_produce_correct_shapes(self, tmp_path):
        img = make_pil_image(200, 150)
        p = save_pil_to_tmp(tmp_path, img)
        for mode, cfg in MODE_CONFIGS.items():
            result = load_image(p, mode=mode)
            h, w = cfg["size"]
            assert result.shape == (1, 3, h, w), f"Shape mismatch for mode={mode}"
