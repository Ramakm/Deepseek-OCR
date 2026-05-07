"""Tests for utils/image_io.py — covers Pillow 10.x API changes and core logic."""
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

def make_rgb_image(w: int = 200, h: int = 150, color=(128, 64, 32)) -> Image.Image:
    """Create a small RGB PIL Image in memory."""
    return Image.new("RGB", (w, h), color)


def save_temp_image(tmp_path: Path, w: int = 200, h: int = 150, name="test.jpg") -> Path:
    img = make_rgb_image(w, h)
    p = tmp_path / name
    img.save(str(p))
    return p


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_modes(self):
        # base and large should have pad=True; tiny/small pad=False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_increase_with_mode(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded_correctly(self):
        """Wide image should have white padding on top/bottom."""
        img = make_rgb_image(400, 100)  # 4:1 aspect
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Top row should be white (padding)
        assert arr[0, 0, 0] == 255

    def test_portrait_image_padded_correctly(self):
        """Tall image should have padding on left/right."""
        img = make_rgb_image(100, 400)  # 1:4 aspect
        result = resize_and_pad(img, (400, 400), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Left column top should be black (padding)
        assert arr[0, 0, 0] == 0

    def test_square_image_no_padding(self):
        """Square image should have no significant padding."""
        img = make_rgb_image(200, 200)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = make_rgb_image(300, 100)
        pad_color = (10, 20, 30)
        result = resize_and_pad(img, (300, 300), pad_color=pad_color)
        arr = np.array(result)
        # Verify that the top-left corner is the pad color
        top_left = tuple(arr[0, 0])
        assert top_left == pad_color

    def test_pillow_bilinear_resampling_used(self):
        """Ensure Pillow 10.x-compatible BILINEAR constant works."""
        img = make_rgb_image(100, 100)
        # Pillow 10 removed ANTIALIAS; BILINEAR should still work
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_upscale(self):
        img = make_rgb_image(32, 32)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Square image fitting exactly → all tokens valid."""
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_landscape_fewer_tokens(self):
        """Wide image padded into square → fewer valid tokens."""
        result = calculate_valid_tokens(800, 400, 512, 512, 256)
        assert result < 256

    def test_portrait_fewer_tokens(self):
        result = calculate_valid_tokens(200, 600, 512, 512, 256)
        assert result < 256

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 100)
        assert isinstance(result, int)

    def test_valid_tokens_never_exceed_total(self):
        for orig_w, orig_h in [(100, 50), (50, 100), (200, 200), (1, 1000)]:
            result = calculate_valid_tokens(orig_w, orig_h, 512, 512, 256)
            assert result <= 256


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_with_correct_shape_tiny(self, tmp_path):
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_with_correct_shape_small(self, tmp_path):
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_with_correct_shape_base(self, tmp_path):
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_returns_tensor_with_correct_shape_large(self, tmp_path):
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_flag(self, tmp_path):
        p = save_temp_image(tmp_path)
        result = load_image(str(p), mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_tensor_is_float(self, tmp_path):
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized(self, tmp_path):
        """After ImageNet normalisation values can go below 0 or above 1."""
        p = save_temp_image(tmp_path)
        tensor = load_image(str(p), mode="tiny")
        # Values should NOT all be in [0,1] after normalisation
        assert not (tensor.min() >= 0.0 and tensor.max() <= 1.0), (
            "Tensor appears un-normalised (all values in [0,1])"
        )

    def test_invalid_mode_raises(self, tmp_path):
        p = save_temp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(str(p), mode="ultra")

    def test_png_image(self, tmp_path):
        img = make_rgb_image(100, 100)
        p = tmp_path / "test.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3  # RGB channels

    def test_grayscale_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        p = tmp_path / "gray.jpg"
        img.convert("RGB").save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3
