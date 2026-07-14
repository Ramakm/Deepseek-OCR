"""Tests for utils/image_io.py — covers Pillow (9→12) and torch/torchvision usage."""
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

# Ensure project root is on the path
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

def make_rgb_image(width=100, height=80, color=(128, 64, 32)) -> Image.Image:
    """Create a small solid-color RGB PIL image."""
    return Image.new("RGB", (width, height), color)


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


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

    def test_pad_modes(self):
        """base and large should use padding; tiny and small should not."""
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        target = (300, 300)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = make_rgb_image(50, 50)
        result = resize_and_pad(img, (100, 100))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_landscape(self):
        """A wide image should have horizontal content fill the width."""
        img = make_rgb_image(200, 50, color=(255, 0, 0))
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Top rows should be white padding
        assert arr[0, 100, 0] == 255  # red channel ≈ 255 (white pad)

    def test_square_image_no_padding_needed(self):
        img = make_rgb_image(100, 100, color=(10, 20, 30))
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Corners should be black
        assert arr[0, 0].tolist() == [0, 0, 0]

    def test_small_image_upscale(self):
        img = make_rgb_image(10, 10)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        img = make_rgb_image(200, 150)
        path = save_temp_image(img, suffix=".png")
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_image):
        tensor = load_image(tmp_image, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_image):
        tensor = load_image(tmp_image, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_image):
        tensor = load_image(tmp_image, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_flag(self, tmp_image):
        result = load_image(tmp_image, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_mode_rgb(self, tmp_image):
        result = load_image(tmp_image, mode="tiny", return_pil=True)
        assert result.mode == "RGB"

    def test_tensor_dtype_float(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_unknown_mode_raises(self, tmp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image, mode="ultra")

    def test_normalized_range(self, tmp_image):
        """After ImageNet normalization values may be outside [0,1]."""
        tensor = load_image(tmp_image, mode="tiny")
        # Just ensure it's a finite float tensor
        assert torch.isfinite(tensor).all()

    def test_jpeg_file(self):
        img = make_rgb_image(300, 200)
        path = save_temp_image(img, suffix=".jpg")
        try:
            tensor = load_image(path, mode="small")
            assert tensor.shape[0] == 1
        finally:
            os.unlink(path)

    def test_rgba_image_converted_to_rgb(self):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        path = save_temp_image(img, suffix=".png")
        try:
            # The function calls .convert("RGB") internally
            tensor = load_image(path, mode="tiny")
            assert tensor.shape == (1, 3, 512, 512)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_size_all_tokens_valid(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        # orig is 512×1024, target 1024×1024 → scale=1, scaled=(512,1024), ratio=0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_small_image(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result >= 0
        assert result <= 256

    def test_returns_int(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_zero_tokens_input(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 0)
        assert result == 0

    def test_landscape_image(self):
        # 2:1 landscape → scale constrained by height → scaled=(1024,512)
        result = calculate_valid_tokens(2048, 1024, 1024, 1024, 256)
        assert 0 < result <= 256
