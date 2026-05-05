"""Tests for utils/image_io.py — exercises Pillow (major upgrade 9→12) and torch APIs."""
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make the project root importable regardless of working directory
# ---------------------------------------------------------------------------
import sys
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

def make_rgb_image(width: int = 200, height: int = 150, color=(128, 64, 32)) -> Image.Image:
    """Return a solid-colour RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save *img* to a temp file and return its path (caller must delete)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_keys_present(self):
        for key in ("tiny", "small", "base", "large"):
            assert key in MODE_CONFIGS

    def test_each_mode_has_required_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_values(self):
        # tiny / small → no padding; base / large → padding
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_are_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2
            w, h = cfg["size"]
            assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# resize_and_pad — Pillow API
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_pil_image(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        img = make_rgb_image(100, 80)
        result = resize_and_pad(img, (200, 200))
        assert result.mode == "RGB"

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        arr = np.array(result)
        # The top-left corner should be the pad colour (red) because the
        # 50×50 image is small relative to the 200×200 canvas.
        assert arr[0, 0, 0] == 255  # R
        assert arr[0, 0, 1] == 0    # G
        assert arr[0, 0, 2] == 0    # B

    def test_aspect_ratio_preserved_wide_image(self):
        """A wide image padded to a square should have top/bottom bars."""
        img = make_rgb_image(400, 100, color=(0, 255, 0))
        result = resize_and_pad(img, (400, 400), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top row should be black (padding)
        assert arr[0, 0, 0] == 0 and arr[0, 0, 1] == 0

    def test_aspect_ratio_preserved_tall_image(self):
        """A tall image padded to a square should have left/right bars."""
        img = make_rgb_image(100, 400, color=(255, 0, 0))
        result = resize_and_pad(img, (400, 400), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Left column should be black (padding)
        assert arr[0, 0, 0] == 0

    def test_square_image_no_padding_needed(self):
        """A square image should fill the target entirely (no padding)."""
        img = make_rgb_image(100, 100, color=(10, 20, 30))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Corner pixel should NOT be white (all pixels are image content)
        assert not (arr[0, 0, 0] == 255 and arr[0, 0, 1] == 255 and arr[0, 0, 2] == 255)


# ---------------------------------------------------------------------------
# load_image — exercises both Pillow and torch
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_image(self, tmp_path):
        img = make_rgb_image(300, 200)
        p = tmp_path / "test.jpg"
        img.save(str(p))
        self.img_path = str(p)

    def test_returns_tensor_by_default(self):
        t = load_image(self.img_path, mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_tensor_shape_tiny(self):
        t = load_image(self.img_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        t = load_image(self.img_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        t = load_image(self.img_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        t = load_image(self.img_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_dtype_float32(self):
        t = load_image(self.img_path, mode="tiny")
        assert t.dtype == torch.float32

    def test_return_pil_flag(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="nonexistent_mode")

    def test_tensor_normalized_range(self):
        """After ImageNet normalization values should be outside [0,1]."""
        t = load_image(self.img_path, mode="tiny")
        # At least some values should be outside [0,1] after normalization
        outside = (t < 0) | (t > 1)
        assert outside.any()

    def test_png_image(self, tmp_path):
        img = make_rgb_image(100, 100)
        p = tmp_path / "test.png"
        img.save(str(p))
        t = load_image(str(p), mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        """RGBA images should still load correctly (converted via .convert('RGB'))."""
        img = Image.new("RGBA", (100, 100), (128, 64, 32, 200))
        p = tmp_path / "test_rgba.png"
        img.save(str(p))
        t = load_image(str(p), mode="tiny")
        assert t.shape[1] == 3  # 3 channels

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        p = tmp_path / "test_gray.png"
        img.save(str(p))
        t = load_image(str(p), mode="tiny")
        assert t.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """A square image exactly matching target should use all tokens."""
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, total_tokens=256)
        assert result == 256

    def test_half_width_image(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, total_tokens=256)
        # scale = min(1024/512, 1024/1024) = 1.0 (height is the constraint)
        # scaled_w = 1024, scaled_h = 1024
        # valid_ratio = (1024/1024) * (1024/1024) = 1.0
        assert result == 256

    def test_small_image_fewer_tokens(self):
        result = calculate_valid_tokens(512, 512, 1024, 1024, total_tokens=256)
        # scale = min(2, 2) = 2; scaled_w = 1024, scaled_h = 1024
        # valid_ratio = 1.0
        assert result == 256

    def test_narrow_wide_image(self):
        # 2048x512, target 1024x1024
        result = calculate_valid_tokens(2048, 512, 1024, 1024, total_tokens=256)
        # scale = min(1024/2048, 1024/512) = min(0.5, 2) = 0.5
        # scaled_w=1024, scaled_h=256
        # valid_ratio = (1024/1024) * (256/1024) = 0.25
        assert result == int(256 * 0.25)

    def test_returns_integer(self):
        result = calculate_valid_tokens(300, 200, 1024, 1024, total_tokens=256)
        assert isinstance(result, int)

    def test_zero_tokens_edge(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, total_tokens=0)
        assert result == 0
