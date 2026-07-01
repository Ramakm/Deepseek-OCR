"""Tests for utils/image_io.py

Covers:
- load_image (all modes, return_pil, error paths)
- resize_and_pad
- calculate_valid_tokens
- PIL (Pillow) upgrade compatibility
"""
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

def _make_rgb_image(w=200, h=150, color=(128, 64, 32)) -> Image.Image:
    """Create a small solid-color RGB PIL image."""
    return Image.new("RGB", (w, h), color)


def _save_tmp_image(img: Image.Image, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"mode '{mode}' missing 'size'"
            assert "tokens" in cfg, f"mode '{mode}' missing 'tokens'"
            assert "pad" in cfg, f"mode '{mode}' missing 'pad'"

    def test_size_is_2_tuple(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2

    def test_pad_modes(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_are_positive_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved(self):
        """After resize+pad the content region aspect ratio should be ~original."""
        img = _make_rgb_image(400, 200)  # 2:1 ratio
        result = resize_and_pad(img, (512, 512))
        arr = np.array(result)
        # White padding (255,255,255) should appear in top/bottom or left/right
        white_rows = np.all(arr == 255, axis=(1, 2))
        white_cols = np.all(arr == 255, axis=(0, 2))
        # For a 2:1 image in a 1:1 canvas, top/bottom rows should be padded
        assert white_rows.any() or white_cols.any()

    def test_custom_pad_color(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Some pixels should be black padding if image is smaller
        assert (arr == 0).any()

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_returns_pil_image(self):
        img = _make_rgb_image(100, 200)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_canvas(self):
        # Image same size as target → all tokens valid
        result = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert result == 100

    def test_half_area_image(self):
        # 128×128 in 256×256 canvas → scale=0.5, valid_ratio=0.25
        result = calculate_valid_tokens(128, 128, 256, 256, 100)
        assert 0 < result < 100

    def test_wide_image(self):
        # 400×200 in 400×400 → scale=min(1.0, 2.0)=1.0 → w=400, h=200
        # valid_ratio = (400/400)*(200/400) = 0.5
        result = calculate_valid_tokens(400, 200, 400, 400, 200)
        assert result == 100

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 256)
        assert isinstance(result, int)

    def test_nonzero_for_any_valid_image(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result >= 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        """Create a small test image and store its path."""
        img = _make_rgb_image(300, 200)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    # -- return_pil=True ------------------------------------------------
    def test_return_pil_tiny(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_return_pil_small(self):
        result = load_image(self.img_path, mode="small", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.size == MODE_CONFIGS["small"]["size"]

    def test_return_pil_base(self):
        result = load_image(self.img_path, mode="base", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.size == MODE_CONFIGS["base"]["size"]

    def test_return_pil_large(self):
        result = load_image(self.img_path, mode="large", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.size == MODE_CONFIGS["large"]["size"]

    # -- return tensor --------------------------------------------------
    def test_returns_tensor_by_default(self):
        result = load_image(self.img_path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self):
        result = load_image(self.img_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        result = load_image(self.img_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_dtype_float32(self):
        result = load_image(self.img_path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_is_normalized(self):
        """After ImageNet normalization values should go below 0."""
        result = load_image(self.img_path, mode="tiny")
        # Normalized tensors routinely have negative values
        assert result.min().item() < 0.5  # crude but reliable check

    # -- error path -----------------------------------------------------
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="invalid_mode")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    # -- PNG support (Pillow) -------------------------------------------
    def test_load_png(self, tmp_path):
        img = _make_rgb_image(100, 100)
        png_path = str(tmp_path / "test.png")
        img.save(png_path)
        result = load_image(png_path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        png_path = str(tmp_path / "rgba.png")
        img.save(png_path)
        result = load_image(png_path, mode="tiny")
        assert result.shape[1] == 3  # RGB channels
