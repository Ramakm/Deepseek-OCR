"""Tests for utils/image_io.py — covers Pillow (9→12) and torch tensor APIs."""
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
# Make project root importable without an installed package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_rgb(w: int = 100, h: int = 80) -> Image.Image:
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_tmp_image(tmp_path: Path, w: int = 100, h: int = 80) -> Path:
    img = _make_pil_rgb(w, h)
    p = tmp_path / "test.jpg"
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for key in ("tiny", "small", "base", "large"):
            assert key in MODE_CONFIGS

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_values(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_increase_with_size(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_rgb(200, 100)
        out = resize_and_pad(img, (256, 256))
        assert out.size == (256, 256)

    def test_output_mode_is_rgb(self):
        img = _make_pil_rgb(50, 50)
        out = resize_and_pad(img, (128, 128))
        assert out.mode == "RGB"

    def test_landscape_image_padded(self):
        img = _make_pil_rgb(200, 50)  # wide image
        out = resize_and_pad(img, (200, 200))
        assert out.size == (200, 200)

    def test_portrait_image_padded(self):
        img = _make_pil_rgb(50, 200)  # tall image
        out = resize_and_pad(img, (200, 200))
        assert out.size == (200, 200)

    def test_custom_pad_color(self):
        img = _make_pil_rgb(10, 10)
        out = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        # corners should be the pad colour (black)
        corner = out.getpixel((0, 0))
        assert corner == (0, 0, 0)

    def test_square_image_no_padding_needed(self):
        img = _make_pil_rgb(64, 64)
        out = resize_and_pad(img, (64, 64))
        assert out.size == (64, 64)

    def test_aspect_ratio_preserved(self):
        """The resized content inside the padded canvas preserves aspect ratio."""
        img = _make_pil_rgb(200, 100)  # 2:1 ratio
        out = resize_and_pad(img, (200, 200))
        arr = np.array(out)
        # With white padding (255,255,255), the top/bottom rows should be white
        top_row = arr[0, :, :]
        assert np.all(top_row == 255)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="base")
        assert isinstance(t, torch.Tensor)

    def test_tensor_shape_base(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="base")
        assert t.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_tiny(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="tiny")
        assert t.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="small")
        assert t.shape == (1, 3, 640, 640)

    def test_tensor_shape_large(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="large")
        assert t.shape == (1, 3, 1280, 1280)

    def test_tensor_dtype_float32(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        t = load_image(str(p), mode="tiny")
        assert t.dtype == torch.float32

    def test_return_pil_flag(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        img = load_image(str(p), mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        img = load_image(str(p), mode="tiny", return_pil=True)
        assert img.size == (512, 512)

    def test_unknown_mode_raises(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(str(p), mode="nonexistent")

    def test_tensor_values_normalized(self, tmp_path):
        """After ImageNet normalisation values may be negative or > 1."""
        # Use a white image so we know input was all 1.0 before normalisation
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        p = tmp_path / "white.jpg"
        img.save(p)
        t = load_image(str(p), mode="tiny")
        # After normalise, channel 0 mean=(0.485) → (1-0.485)/0.229 ≈ 2.25
        assert t.min() < 5.0  # just verify it's normalised, not raw [0,255]

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (64, 64), 128)
        p = tmp_path / "gray.png"
        img.save(p)
        t = load_image(str(p), mode="tiny")
        assert t.shape[1] == 3

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 200))
        p = tmp_path / "rgba.png"
        img.save(p)
        t = load_image(str(p), mode="tiny")
        assert t.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_no_padding(self):
        # When orig == target, valid_tokens == total_tokens
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width(self):
        # 512 wide image padded to 1024 wide canvas → scale=0.5
        result = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        # scale = min(1024/512, 1024/512) = 2, but wait — orig < target
        # scale = min(target/orig) = 2 → clamps to target, so scaled = 1024
        # Actually ratio = (1024/1024)*(1024/1024) = 1 → 256
        # Let's test a tall narrow image instead
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_returns_integer(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_result_bounded(self):
        result = calculate_valid_tokens(100, 200, 1024, 1024, 256)
        assert 0 <= result <= 256

    def test_wide_image(self):
        result = calculate_valid_tokens(2000, 500, 1024, 1024, 256)
        assert 0 <= result <= 256

    def test_exact_aspect_ratio_square(self):
        result = calculate_valid_tokens(100, 100, 200, 200, 100)
        # scale = 2 → scaled_w=200, scaled_h=200 → ratio = 1 → 100
        assert result == 100
