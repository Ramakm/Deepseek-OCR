"""Tests for utils/image_io.py — covers Pillow (major bump 9→10) usage."""
import io
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width=200, height=150, color=(128, 64, 32)):
    """Return a small solid-color RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def _save_tmp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------
class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_are_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0, f"{mode} token count must be positive"


# ---------------------------------------------------------------------------
# resize_and_pad  (heavy Pillow usage)
# ---------------------------------------------------------------------------
class TestResizeAndPad:
    def test_output_matches_target_size(self):
        img = _make_pil_image(300, 200)
        target = (256, 256)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (64, 64))
        assert result.mode == "RGB"

    def test_custom_pad_color(self):
        img = _make_pil_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        arr = np.array(result)
        # corners should be (approximately) the pad color
        assert arr[0, 0, 0] == 255  # R channel at top-left corner

    def test_wide_image_padded_vertically(self):
        """Wide image → pillarbox padding on top/bottom."""
        img = _make_pil_image(200, 50)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_tall_image_padded_horizontally(self):
        """Tall image → letterbox padding on left/right."""
        img = _make_pil_image(50, 200)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_square_image_no_real_padding(self):
        """Square → scale only, no significant padding."""
        img = _make_pil_image(64, 64, color=(100, 150, 200))
        result = resize_and_pad(img, (128, 128))
        assert result.size == (128, 128)

    def test_pillow_bilinear_resize_used(self):
        """Confirm Image.BILINEAR is still a valid resampling filter (Pillow 10+)."""
        img = _make_pil_image(100, 100)
        # Should not raise
        img.resize((50, 50), Image.BILINEAR)


# ---------------------------------------------------------------------------
# load_image  (Pillow + torchvision + NumPy)
# ---------------------------------------------------------------------------
class TestLoadImage:
    @pytest.fixture()
    def tmp_jpg(self):
        img = _make_pil_image(300, 200)
        path = _save_tmp_image(img, ".jpg")
        yield path
        os.unlink(path)

    @pytest.fixture()
    def tmp_png(self):
        img = _make_pil_image(128, 128)
        path = _save_tmp_image(img, ".png")
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny")
        expected_h, expected_w = MODE_CONFIGS["tiny"]["size"]
        assert result.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_small(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="small")
        expected_h, expected_w = MODE_CONFIGS["small"]["size"]
        assert result.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_base(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="base")
        expected_h, expected_w = MODE_CONFIGS["base"]["size"]
        assert result.shape == (1, 3, expected_h, expected_w)

    def test_tensor_shape_large(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="large")
        expected_h, expected_w = MODE_CONFIGS["large"]["size"]
        assert result.shape == (1, 3, expected_h, expected_w)

    def test_return_pil_mode(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_invalid_mode_raises(self, tmp_jpg):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_jpg, mode="ultra")

    def test_tensor_dtype_float(self, tmp_jpg):
        result = load_image(tmp_jpg, mode="tiny")
        assert result.dtype == torch.float32

    def test_png_loads_correctly(self, tmp_png):
        result = load_image(tmp_png, mode="tiny")
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1

    def test_tensor_values_normalized(self, tmp_jpg):
        """After ImageNet normalization, values need not be in [0,1]."""
        result = load_image(tmp_jpg, mode="tiny")
        # Values can be negative due to normalization; just check they're finite
        assert torch.isfinite(result).all()

    def test_grayscale_image_converted_to_rgb(self, tmp_png):
        """Grayscale images should be converted to RGB."""
        gray = Image.new("L", (100, 100), 128)
        path = _save_tmp_image(gray, ".png")
        try:
            result = load_image(path, mode="tiny")
            assert result.shape[1] == 3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------
class TestCalculateValidTokens:
    def test_square_image_fills_target(self):
        """Square image that exactly fills target → all tokens valid."""
        valid = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert valid == 100

    def test_half_size_image(self):
        """Image half the target dimensions in both axes → 25 % tokens valid."""
        valid = calculate_valid_tokens(128, 128, 256, 256, 100)
        assert valid == 25

    def test_wide_image(self):
        """Wide image constrained by width."""
        valid = calculate_valid_tokens(400, 100, 200, 200, 100)
        # scale = min(200/400, 200/100) = 0.5
        # scaled_w=200, scaled_h=50 → ratio = (200/200)*(50/200)=0.25
        assert valid == 25

    def test_returns_integer(self):
        valid = calculate_valid_tokens(300, 200, 1024, 1024, 256)
        assert isinstance(valid, int)

    def test_result_leq_total_tokens(self):
        for orig_w, orig_h in [(100, 100), (500, 300), (1200, 800)]:
            valid = calculate_valid_tokens(orig_w, orig_h, 1024, 1024, 256)
            assert valid <= 256

    def test_result_positive(self):
        valid = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert valid >= 0
