"""Tests for utils/image_io.py"""
import io
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# Ensure project root is on path
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

def make_rgb_image(width=200, height=150, color=(128, 64, 32)) -> Image.Image:
    """Create a small RGB PIL image for testing."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_tmp_image(tmp_path, width=200, height=150, name="test.jpg") -> Path:
    img = make_rgb_image(width, height)
    p = tmp_path / name
    img.save(str(p))
    return p


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

    def test_pad_flag_base_large(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_pad_flag_tiny_small(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_shape_tiny(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_shape_small(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_shape_base(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_shape_large(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_float_tensor(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        assert tensor.dtype == torch.float32

    def test_returns_pil_when_requested(self, tmp_path):
        p = save_tmp_image(tmp_path)
        result = load_image(str(p), mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_size_tiny(self, tmp_path):
        p = save_tmp_image(tmp_path)
        result = load_image(str(p), mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_normalized_values(self, tmp_path):
        """After ImageNet normalization, values should differ from [0,1] range."""
        p = save_tmp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        # After normalization some values will be negative
        assert tensor.min() < 0.5

    def test_unknown_mode_raises(self, tmp_path):
        p = save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(str(p), mode="nonexistent")

    def test_all_modes_run_without_error(self, tmp_path):
        p = save_tmp_image(tmp_path)
        for mode in MODE_CONFIGS:
            t = load_image(str(p), mode=mode)
            assert t.ndim == 4

    def test_square_image(self, tmp_path):
        p = save_tmp_image(tmp_path, width=256, height=256)
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tall_image_pad_mode(self, tmp_path):
        """Tall image should be padded in base mode."""
        p = save_tmp_image(tmp_path, width=100, height=400)
        tensor = load_image(str(p), mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_wide_image_pad_mode(self, tmp_path):
        """Wide image should be padded in base mode."""
        p = save_tmp_image(tmp_path, width=400, height=100)
        tensor = load_image(str(p), mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300)

    def test_default_pad_color_white(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Top-left corner should be padded (white)
        assert tuple(arr[0, 0]) == (255, 255, 255)

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 255))
        arr = np.array(result)
        # Padding area should be blue
        assert arr[0, 0, 2] == 255

    def test_returns_pil_image(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_preserves_aspect_ratio(self):
        """A 2:1 wide image padded to square should have equal side padding."""
        img = make_rgb_image(200, 100)  # 2:1 aspect ratio
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Top row should be padding (white)
        assert np.all(arr[0] == 255)

    def test_square_input_no_padding(self):
        img = make_rgb_image(100, 100, color=(100, 150, 200))
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_upscale(self):
        img = make_rgb_image(32, 32)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_no_padding(self):
        """A square image filling the full target gives all tokens valid."""
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_width_image(self):
        """An image that fills half the target width (50% valid area)."""
        result = calculate_valid_tokens(50, 100, 100, 100, 100)
        # scale = min(100/50, 100/100) = 1.0 → scaled 50x100, ratio = (50/100)*(100/100) = 0.5
        assert result == 50

    def test_tiny_image(self):
        """Very small image results in small valid token count."""
        result = calculate_valid_tokens(10, 10, 100, 100, 400)
        assert result < 400
        assert result > 0

    def test_larger_than_target(self):
        """Image larger than target is scaled down, fills most area."""
        result = calculate_valid_tokens(500, 500, 100, 100, 256)
        assert result == 256

    def test_returns_int(self):
        result = calculate_valid_tokens(100, 100, 200, 200, 100)
        assert isinstance(result, int)
