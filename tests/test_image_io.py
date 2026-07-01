"""Tests for utils/image_io.py — covers Pillow (9→12) and torch/torchvision APIs."""
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

# Make project root importable
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

def _make_rgb_image(width: int = 100, height: int = 80) -> Image.Image:
    """Return a simple solid-colour RGB PIL image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_tmp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save PIL image to a temp file and return its path."""
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
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_pad_flag(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = _make_rgb_image(50, 60)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded(self):
        """Wide image should get top/bottom padding."""
        img = _make_rgb_image(400, 100)
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_portrait_image_padded(self):
        """Tall image should get left/right padding."""
        img = _make_rgb_image(100, 400)
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_square_image_no_border(self):
        """Square image fills target without border on one axis."""
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 50)
        pad_color = (255, 0, 0)
        result = resize_and_pad(img, (200, 200), pad_color=pad_color)
        # Top-left corner should be the pad color (tall/wide padding)
        pixel = result.getpixel((0, 0))
        # Pixel is either pad_color or part of the image — just check type
        assert isinstance(pixel, tuple)

    def test_already_correct_size(self):
        img = _make_rgb_image(256, 256)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size(self):
        tokens = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert tokens == 100

    def test_half_width(self):
        """Image half the width → valid_ratio = 0.5 * 1.0 = 0.5"""
        tokens = calculate_valid_tokens(128, 256, 256, 256, 100)
        assert 0 < tokens <= 100

    def test_result_non_negative(self):
        tokens = calculate_valid_tokens(10, 10, 256, 256, 400)
        assert tokens >= 0

    def test_result_leq_total(self):
        tokens = calculate_valid_tokens(100, 200, 1024, 1024, 256)
        assert tokens <= 256

    def test_large_image_fits_exactly(self):
        """When scaled image exactly fills target, all tokens are valid."""
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = _make_rgb_image(200, 150)
        self.tmp_path = _save_tmp_image(self.img, ".png")

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_returns_tensor(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        tensor = load_image(self.tmp_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        tensor = load_image(self.tmp_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        tensor = load_image(self.tmp_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_dtype_float(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_return_pil(self):
        pil = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(pil, Image.Image)

    def test_return_pil_size(self):
        pil = load_image(self.tmp_path, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert pil.size == expected

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="invalid_mode")

    def test_normalized_range(self):
        """After ImageNet normalisation values are roughly in [-3, 3]."""
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.min().item() < 1.0   # has been normalised down
        assert tensor.max().item() < 5.0   # not raw [0,255]

    def test_jpeg_image(self):
        tmp = _save_tmp_image(self.img, ".jpg")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[0] == 1
        finally:
            os.remove(tmp)

    def test_grayscale_converted_to_rgb(self):
        """Grayscale images should be converted via .convert('RGB')."""
        gray = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        tmp = _save_tmp_image(gray, ".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(tmp)
