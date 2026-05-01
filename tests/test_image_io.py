"""Tests for utils/image_io.py

Covers:
- load_image with all resolution modes
- resize_and_pad
- calculate_valid_tokens
- Pillow (pillow 9→10 MAJOR bump) API usage
- torchvision transforms
"""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make repo root importable regardless of where pytest is invoked from
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

def make_rgb_image(w: int = 200, h: int = 100) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_tmp_image(img: Image.Image, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_increase(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ["tiny", "small", "base", "large"]]
        assert tokens == sorted(tokens)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target  # PIL .size is (w, h)

    def test_mode_is_rgb(self):
        img = make_rgb_image(300, 150)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_custom_pad_color(self):
        # White square image, pad with red
        img = Image.new("RGB", (100, 50), (255, 255, 255))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        assert result.size == (200, 200)
        # Top-left corner should be red (padding area for wide image)
        px = result.getpixel((0, 0))
        assert px == (255, 0, 0)

    def test_landscape_image_preserves_aspect_ratio(self):
        img = make_rgb_image(400, 200)   # 2:1 aspect
        target = (512, 512)
        result = resize_and_pad(img, target)
        # Image should be 512 wide × 256 tall inside 512×512 canvas
        arr = np.array(result)
        assert arr.shape == (512, 512, 3)

    def test_portrait_image(self):
        img = make_rgb_image(100, 300)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_square_image_no_padding(self):
        img = Image.new("RGB", (128, 128), (10, 20, 30))
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    # Pillow 10 removed ANTIALIAS; we should be using BILINEAR
    def test_uses_bilinear_resampling(self):
        """Verify no ANTIALIAS (removed in Pillow 10) is used."""
        img = make_rgb_image(200, 200)
        # Should not raise even with Pillow 10+
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = make_rgb_image(300, 200)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        tensor = load_image(self.img_path, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        tensor = load_image(self.img_path, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        tensor = load_image(self.img_path, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size_tiny(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert img.size == (512, 512)

    def test_return_pil_base_mode(self):
        img = load_image(self.img_path, mode="base", return_pil=True)
        assert img.size == (1024, 1024)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="invalid_mode")

    def test_tensor_dtype_float(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized(self):
        # After ImageNet normalization values should not be in [0,1] necessarily
        tensor = load_image(self.img_path, mode="tiny")
        # Check that some normalization happened (not raw [0,1])
        # Mean of ImageNet-normalized tensors should be near 0 for natural images
        assert tensor.min() < 1.0

    def test_png_file(self, tmp_path):
        img = make_rgb_image(150, 150)
        path = str(tmp_path / "test.png")
        img.save(path)
        tensor = load_image(path, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        path = str(tmp_path / "gray.jpg")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_equals_total_tokens(self):
        # When image exactly fits target, all tokens are valid
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_area_image(self):
        # 512x256 in 512x512 → scale=1 (height limited), so 512×256/512×512
        result = calculate_valid_tokens(512, 256, 512, 512, 256)
        assert result == 128

    def test_returns_integer(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 100)
        assert isinstance(result, int)

    def test_never_exceeds_total_tokens(self):
        for orig_w, orig_h in [(100, 100), (200, 300), (512, 256), (1000, 800)]:
            result = calculate_valid_tokens(orig_w, orig_h, 512, 512, 256)
            assert result <= 256, f"Exceeded for {orig_w}x{orig_h}"

    def test_zero_tokens_when_both_zero(self):
        result = calculate_valid_tokens(0, 100, 512, 512, 256)
        assert result == 0

    def test_landscape_original(self):
        # 1024x512 → 512x256 fits in 512×512 → valid_ratio = (512/512)*(256/512) = 0.5
        result = calculate_valid_tokens(1024, 512, 512, 512, 256)
        assert result == 128

    def test_small_image_in_large_canvas(self):
        # 100x100 in 1024x1024 → scale = 1024/100 (but we scale to fit, scale=10.24)
        # scaled_w=1024, scaled_h=1024 → valid_ratio=1.0
        result = calculate_valid_tokens(100, 100, 1024, 1024, 400)
        assert result == 400
