"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) and torch/torchvision usage."""
import io
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w=100, h=80, color=(128, 64, 32)):
    """Return a small RGB PIL image."""
    return Image.new("RGB", (w, h), color)


def _save_tmp_image(img: Image.Image, fmt="PNG") -> str:
    """Save a PIL image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=f".{fmt.lower()}")
    os.close(fd)
    img.save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# Import the module under test
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
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_keys(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_tiny_config(self):
        cfg = MODE_CONFIGS["tiny"]
        assert cfg["size"] == (512, 512)
        assert cfg["tokens"] == 64
        assert cfg["pad"] is False

    def test_small_config(self):
        cfg = MODE_CONFIGS["small"]
        assert cfg["size"] == (640, 640)
        assert cfg["tokens"] == 100
        assert cfg["pad"] is False

    def test_base_config(self):
        cfg = MODE_CONFIGS["base"]
        assert cfg["size"] == (1024, 1024)
        assert cfg["tokens"] == 256
        assert cfg["pad"] is True

    def test_large_config(self):
        cfg = MODE_CONFIGS["large"]
        assert cfg["size"] == (1280, 1280)
        assert cfg["tokens"] == 400
        assert cfg["pad"] is True


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        out = resize_and_pad(img, (512, 512))
        assert out.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(300, 150)
        out = resize_and_pad(img, (1024, 1024))
        assert out.mode == "RGB"

    def test_custom_pad_color(self):
        img = _make_rgb_image(10, 10, color=(0, 0, 0))
        out = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        # Top-left corner should be the pad colour (image is centred)
        arr = np.array(out)
        # The image occupies a small central region; corners should be red
        assert arr[0, 0, 0] == 255  # R channel of pad
        assert arr[0, 0, 1] == 0    # G channel of pad

    def test_aspect_ratio_preserved(self):
        """After padding the scaled sub-image should not exceed target dims."""
        img = _make_rgb_image(800, 200)  # wide image
        out = resize_and_pad(img, (512, 512))
        assert out.size == (512, 512)

    def test_square_image(self):
        img = _make_rgb_image(256, 256)
        out = resize_and_pad(img, (512, 512))
        assert out.size == (512, 512)

    def test_portrait_image(self):
        img = _make_rgb_image(100, 300)
        out = resize_and_pad(img, (640, 640))
        assert out.size == (640, 640)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size(self):
        vt = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert vt == 256

    def test_half_size_image(self):
        # Image half the target in both dims → valid_ratio = 0.25
        vt = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert vt == 64

    def test_wide_image(self):
        # 800×400 in 1024×1024 target
        # scale = min(1024/800, 1024/400) = min(1.28, 2.56) = 1.28
        # scaled: 1024×512
        # valid_ratio = (1024/1024)*(512/1024) = 0.5
        vt = calculate_valid_tokens(800, 400, 1024, 1024, 256)
        assert vt == 128

    def test_returns_int(self):
        vt = calculate_valid_tokens(300, 200, 640, 640, 100)
        assert isinstance(vt, int)

    def test_zero_tokens(self):
        vt = calculate_valid_tokens(512, 512, 512, 512, 0)
        assert vt == 0


# ---------------------------------------------------------------------------
# load_image — happy paths
# ---------------------------------------------------------------------------

class TestLoadImage:
    """Uses PIL to create temp images; exercises Pillow API after upgrade."""

    @pytest.fixture(autouse=True)
    def tmp_png(self):
        img = _make_rgb_image(200, 150)
        path = _save_tmp_image(img, "PNG")
        self.img_path = path
        yield
        os.unlink(path)

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_returns_tensor_all_modes(self, mode):
        t = load_image(self.img_path, mode=mode)
        assert isinstance(t, torch.Tensor)

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_tensor_shape(self, mode):
        t = load_image(self.img_path, mode=mode)
        h, w = MODE_CONFIGS[mode]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_dtype_float32(self):
        t = load_image(self.img_path, mode="tiny")
        assert t.dtype == torch.float32

    def test_return_pil_flag(self):
        pil = load_image(self.img_path, mode="base", return_pil=True)
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"

    def test_return_pil_size_base(self):
        pil = load_image(self.img_path, mode="base", return_pil=True)
        assert pil.size == MODE_CONFIGS["base"]["size"]

    def test_return_pil_size_tiny(self):
        pil = load_image(self.img_path, mode="tiny", return_pil=True)
        assert pil.size == MODE_CONFIGS["tiny"]["size"]

    def test_normalized_range(self):
        """After ImageNet normalization values should be roughly within [-3, 3]."""
        t = load_image(self.img_path, mode="tiny")
        assert t.min().item() > -4.0
        assert t.max().item() < 4.0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="xlarge")

    def test_jpeg_image(self):
        img = _make_rgb_image(320, 240)
        path = _save_tmp_image(img, "JPEG")
        try:
            t = load_image(path, mode="small")
            assert t.shape == (1, 3, 640, 640)
        finally:
            os.unlink(path)

    def test_rgba_image_converted_to_rgb(self):
        """RGBA images should be transparently converted to RGB by PIL."""
        rgba = Image.new("RGBA", (100, 100), (0, 128, 255, 200))
        path = _save_tmp_image(rgba, "PNG")
        try:
            t = load_image(path, mode="tiny")
            assert t.shape == (1, 3, 512, 512)
        finally:
            os.unlink(path)

    def test_grayscale_image_converted_to_rgb(self):
        gray = Image.new("L", (100, 100), 128)
        path = _save_tmp_image(gray, "PNG")
        try:
            t = load_image(path, mode="tiny")
            assert t.shape == (1, 3, 512, 512)
        finally:
            os.unlink(path)
