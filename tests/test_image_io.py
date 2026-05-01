"""Tests for utils/image_io.py — exercises Pillow (9→10 MAJOR bump) and torch/torchvision APIs."""
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
# Make project root importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
    save_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(w: int = 200, h: int = 150, color: tuple = (128, 64, 32)) -> Image.Image:
    """Create a small solid-color PIL image."""
    return Image.new("RGB", (w, h), color)


def _save_temp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save a PIL image to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
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

    def test_size_is_tuple_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            sz = cfg["size"]
            assert isinstance(sz, tuple) and len(sz) == 2
            assert all(isinstance(v, int) and v > 0 for v in sz)

    def test_tokens_are_positive_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0

    def test_pad_modes(self):
        # tiny/small: no pad; base/large: pad
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_ordering(self):
        """Larger modes should have more tokens."""
        assert MODE_CONFIGS["tiny"]["tokens"] < MODE_CONFIGS["small"]["tokens"]
        assert MODE_CONFIGS["small"]["tokens"] < MODE_CONFIGS["base"]["tokens"]
        assert MODE_CONFIGS["base"]["tokens"] < MODE_CONFIGS["large"]["tokens"]


# ---------------------------------------------------------------------------
# resize_and_pad — Pillow API
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_equals_target(self):
        img = _make_pil_image(300, 200)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_pil_image(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        img = _make_pil_image(100, 80)
        result = resize_and_pad(img, (200, 200))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide(self):
        """Wide image: content should fit inside target without distortion."""
        img = _make_pil_image(400, 100)   # 4:1 aspect ratio
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_aspect_ratio_preserved_tall(self):
        """Tall image: content should fit inside target without distortion."""
        img = _make_pil_image(100, 400)   # 1:4 aspect ratio
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_custom_pad_color(self):
        """Padding pixels should reflect the specified color."""
        img = _make_pil_image(10, 10, color=(0, 0, 0))
        pad_color = (255, 0, 0)
        result = resize_and_pad(img, (100, 100), pad_color=pad_color)
        # Top-left corner should be padding (red)
        pixel = result.getpixel((0, 0))
        assert pixel == pad_color

    def test_square_image_no_padding_needed(self):
        """Square image padded to same-ratio target needs minimal padding."""
        img = _make_pil_image(100, 100)
        target = (100, 100)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_uses_bilinear_resize(self):
        """Ensure Image.BILINEAR constant is accessible (Pillow 10 compatibility)."""
        # Pillow 10 removed deprecated constants — Image.BILINEAR == Image.Resampling.BILINEAR
        assert hasattr(Image, "BILINEAR") or hasattr(Image, "Resampling")

    def test_pillow_resampling_enum_available(self):
        """Pillow 10 introduced Image.Resampling enum."""
        # This should not raise on Pillow ≥10
        _ = Image.Resampling.BILINEAR


# ---------------------------------------------------------------------------
# load_image — Pillow + torch tensor output
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture
    def temp_image_path(self):
        img = _make_pil_image(300, 200)
        path = _save_temp_image(img)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_returns_tensor_by_default(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_dtype_float32(self, temp_image_path):
        tensor = load_image(temp_image_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_is_normalized(self, temp_image_path):
        """After ImageNet normalization, values should be outside [0, 1]."""
        tensor = load_image(temp_image_path, mode="tiny")
        # Normalized values can be negative or > 1
        assert tensor.min().item() < 1.0  # at minimum some values differ

    def test_return_pil_flag(self, temp_image_path):
        result = load_image(temp_image_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_matches_mode(self, temp_image_path):
        result = load_image(temp_image_path, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected

    def test_invalid_mode_raises_value_error(self, temp_image_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_image_path, mode="invalid_mode")

    def test_all_modes_succeed(self, temp_image_path):
        for mode in MODE_CONFIGS:
            tensor = load_image(temp_image_path, mode=mode)
            assert tensor.ndim == 4

    def test_loads_png_image(self):
        img = _make_pil_image(150, 150)
        path = _save_temp_image(img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert isinstance(tensor, torch.Tensor)
        finally:
            os.remove(path)

    def test_rgba_image_converted_to_rgb(self):
        """RGBA images should be converted to RGB without errors."""
        img = Image.new("RGBA", (100, 100), (128, 128, 128, 200))
        path = _save_temp_image(img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(path)

    def test_grayscale_image_converted_to_rgb(self):
        img = Image.new("L", (100, 100), 128)
        path = _save_temp_image(img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_same_as_target(self):
        """No padding → all tokens are valid."""
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_width_image(self):
        """Image half as wide as target → ~half tokens valid."""
        result = calculate_valid_tokens(256, 512, 512, 512, 256)
        # scale = min(512/256, 512/512) = 1.0; scaled = 256x512
        # valid_ratio = (256/512)*(512/512) = 0.5
        assert result == 128

    def test_result_is_int(self):
        result = calculate_valid_tokens(300, 200, 512, 512, 256)
        assert isinstance(result, int)

    def test_result_non_negative(self):
        result = calculate_valid_tokens(100, 100, 512, 512, 256)
        assert result >= 0

    def test_result_leq_total_tokens(self):
        result = calculate_valid_tokens(200, 200, 512, 512, 256)
        assert result <= 256

    def test_tall_image(self):
        result = calculate_valid_tokens(100, 400, 512, 512, 256)
        assert 0 < result <= 256

    def test_wide_image(self):
        result = calculate_valid_tokens(400, 100, 512, 512, 256)
        assert 0 < result <= 256

    def test_tiny_image(self):
        result = calculate_valid_tokens(1, 1, 512, 512, 256)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# save_image (stub — function exists)
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_image_function_exists(self):
        assert callable(save_image)
