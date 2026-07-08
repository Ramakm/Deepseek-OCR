"""Tests for utils/image_io.py — covers PIL (pillow 12.x), torch, torchvision."""
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

def _make_pil(w=200, h=150, color=(128, 64, 32)):
    """Create a small solid-colour RGB PIL image."""
    img = Image.new("RGB", (w, h), color)
    return img


def _save_pil(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return the path."""
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
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

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

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            w, h = cfg["size"]
            assert w > 0 and h > 0

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_pil_image(self):
        img = _make_pil(300, 200)
        result = resize_and_pad(img, (640, 640))
        assert isinstance(result, Image.Image)

    def test_aspect_ratio_preserved_via_padding(self):
        """Wide image → horizontal bars should be white."""
        img = _make_pil(400, 100, color=(0, 0, 0))  # very wide, black
        result = resize_and_pad(img, (512, 512), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Top-left corner should be white (padding)
        assert arr[0, 0, 0] == 255

    def test_custom_pad_color(self):
        img = _make_pil(100, 100)
        result = resize_and_pad(img, (300, 300), pad_color=(0, 128, 255))
        arr = np.array(result)
        # Corner pixel should be the pad colour (image is square, but smaller)
        assert arr[0, 0, 0] == 0
        assert arr[0, 0, 1] == 128
        assert arr[0, 0, 2] == 255

    def test_mode_is_rgb(self):
        img = _make_pil(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_square_image_no_padding(self):
        """A 100×100 image padded into a 200×200 canvas — centre."""
        img = _make_pil(100, 100, color=(255, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 255))
        arr = np.array(result)
        # centre pixel should be red (content)
        assert arr[100, 100, 0] == 255
        assert arr[100, 100, 2] == 0


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size_returns_total(self):
        total = 256
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, total)
        assert result == total

    def test_half_area_returns_half_tokens(self):
        # orig 512×512 into 1024×1024 → scale=0.5, ratio=0.25, tokens=64
        result = calculate_valid_tokens(512, 512, 1024, 1024, total_tokens=256)
        assert result == 64

    def test_wide_image(self):
        # orig 2000×500 into 1024×1024 → scale = min(1024/2000, 1024/500) = 0.512
        # scaled = 1024×256, ratio = (1024/1024)*(256/1024) = 0.25 → 64
        result = calculate_valid_tokens(2000, 500, 1024, 1024, total_tokens=256)
        assert result == 64

    def test_returns_int(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_small_image_zero_or_positive(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def temp_image(self, tmp_path):
        img = _make_pil(256, 256)
        p = tmp_path / "test.jpg"
        img.save(str(p))
        self.img_path = str(p)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.img_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        tensor = load_image(self.img_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        tensor = load_image(self.img_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        tensor = load_image(self.img_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_flag(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_return_pil_size(self):
        img = load_image(self.img_path, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="nonexistent")

    def test_tensor_dtype_float(self):
        tensor = load_image(self.img_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_normalized_range(self):
        """After ImageNet normalisation, values needn't be in [0,1]."""
        tensor = load_image(self.img_path, mode="tiny")
        # Just check it is a finite float tensor
        assert torch.isfinite(tensor).all()

    def test_png_input(self, tmp_path):
        img = _make_pil(128, 128)
        p = tmp_path / "test.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[0] == 1

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (128, 128), (10, 20, 30, 128))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3
