"""Tests for utils/image_io.py — exercises PIL (Pillow 10.x) APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w: int = 200, h: int = 150) -> Image.Image:
    """Return a small solid-colour RGB PIL image."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_temp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save *img* to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


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

    def test_pad_flag(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_are_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0

    def test_sizes_are_tuples(self):
        for cfg in MODE_CONFIGS.values():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 150)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_output_mode_is_rgb(self):
        img = _make_rgb_image(200, 150)
        result = resize_and_pad(img, (512, 512))
        assert result.mode == "RGB"

    def test_custom_pad_color(self):
        img = _make_rgb_image(10, 10)
        result = resize_and_pad(img, (100, 100), pad_color=(128, 0, 0))
        arr = np.array(result)
        # Corner pixels should be the pad colour
        assert tuple(arr[0, 0]) == (128, 0, 0)

    def test_square_image_no_padding(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_wide_image(self):
        img = _make_rgb_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_tall_image(self):
        img = _make_rgb_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    # Pillow 10 removed the ANTIALIAS constant — verify BILINEAR still works
    def test_pillow10_bilinear_resampling(self):
        """Ensure Image.BILINEAR is still accessible (Pillow 10 compat)."""
        assert hasattr(Image, "BILINEAR")
        img = _make_rgb_image(300, 200)
        result = img.resize((150, 100), Image.BILINEAR)
        assert result.size == (150, 100)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = _make_rgb_image(300, 200)
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
        pil_img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(pil_img, Image.Image)

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_normalization_applied(self):
        # Values should NOT be in [0, 1] after ImageNet normalisation
        tensor = load_image(self.img_path, mode="tiny")
        # After normalisation some values should be negative
        assert tensor.min().item() < 0.0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="ultra")

    def test_png_file(self, tmp_path):
        img = _make_rgb_image(100, 100)
        path = str(tmp_path / "test.png")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_rgba_image_converted(self, tmp_path):
        arr = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        path = str(tmp_path / "rgba.png")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        # Square image to square target — no padding → all tokens valid
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        # Image is half-width: scaled_w=512, scaled_h=1024, valid_ratio=0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_returns_int(self):
        result = calculate_valid_tokens(200, 150, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_valid_tokens_lte_total(self):
        for orig_w, orig_h in [(100, 200), (300, 100), (1024, 1024), (50, 50)]:
            result = calculate_valid_tokens(orig_w, orig_h, 1024, 1024, 400)
            assert result <= 400

    def test_valid_tokens_positive(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result > 0
