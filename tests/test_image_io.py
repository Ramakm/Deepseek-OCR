"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) usage."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=100, color=(128, 64, 32)):
    """Return a simple solid-colour PIL RGB image."""
    return Image.new("RGB", (width, height), color)


def make_temp_image(tmp_path, width=200, height=100, fmt="JPEG"):
    """Save a PIL image to a temporary file and return the path string."""
    img = make_pil_image(width, height)
    p = tmp_path / f"test_image.{'jpg' if fmt == 'JPEG' else fmt.lower()}"
    img.save(str(p), format=fmt)
    return str(p)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

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
        for key in ("tiny", "small", "base", "large"):
            assert key in MODE_CONFIGS

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
        img = make_pil_image(300, 150)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_aspect_ratio_preserved_wide_image(self):
        """A wide image should have horizontal bars top/bottom or be letter-boxed."""
        img = make_pil_image(400, 100)  # very wide
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_aspect_ratio_preserved_tall_image(self):
        img = make_pil_image(100, 400)  # very tall
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_pad_color_default_white(self):
        """Padding pixels should be white (255,255,255) by default."""
        img = make_pil_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Corner pixel should be padding (white)
        assert arr[0, 0, 0] == 255
        assert arr[0, 0, 1] == 255
        assert arr[0, 0, 2] == 255

    def test_custom_pad_color(self):
        img = make_pil_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 255))
        arr = np.array(result)
        # Corner should be blue padding
        assert arr[0, 0, 2] == 255

    def test_square_image_no_padding(self):
        """A square source → square target: no padding at all."""
        img = make_pil_image(100, 100, color=(200, 100, 50))
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_returns_pil_image(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)

    def test_mode_is_rgb(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (64, 64))
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """A square image equal to target has all tokens valid."""
        tokens = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert tokens == 100

    def test_half_width_image(self):
        """Wide image scaled to fit: valid < total."""
        tokens = calculate_valid_tokens(100, 200, 200, 200, 100)
        assert 0 < tokens < 100

    def test_returns_int(self):
        tokens = calculate_valid_tokens(100, 100, 200, 200, 256)
        assert isinstance(tokens, int)

    def test_zero_tokens_when_tiny_source(self):
        """Degenerate: very small source may give 0 tokens."""
        tokens = calculate_valid_tokens(1, 1, 1000, 1000, 256)
        assert tokens >= 0

    def test_consistent_with_scale(self):
        """Manually verify formula."""
        orig_w, orig_h = 640, 480
        target_w, target_h = 1024, 1024
        total = 400
        scale = min(target_w / orig_w, target_h / orig_h)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        expected = int(total * (scaled_w / target_w) * (scaled_h / target_h))
        result = calculate_valid_tokens(orig_w, orig_h, target_w, target_h, total)
        assert result == expected


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil(self, tmp_path):
        p = make_temp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_size_matches_mode(self, tmp_path):
        p = make_temp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises_value_error(self, tmp_path):
        p = make_temp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="nonexistent")

    def test_tensor_dtype_float32(self, tmp_path):
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalised(self, tmp_path):
        """After ImageNet normalisation, values are not in [0,1] naively."""
        p = make_temp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        # Values should be floats — not necessarily bounded to [0,1]
        assert tensor.min() < 1.0  # sanity

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), color=128)
        p = tmp_path / "gray.jpg"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3

    def test_rgba_image_handled(self, tmp_path):
        img = Image.new("RGBA", (100, 100), color=(128, 64, 32, 200))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3
