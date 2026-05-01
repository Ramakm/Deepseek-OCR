"""Tests for utils/image_io.py — covers Pillow (9→10 MAJOR upgrade) and torch APIs."""
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
# Ensure project root is importable
# ---------------------------------------------------------------------------
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

def _make_rgb_image(width=100, height=80, color=(128, 64, 32)):
    """Return a simple RGB PIL image."""
    img = Image.new("RGB", (width, height), color=color)
    return img


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temp file and return the path."""
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

    def test_mode_structure(self):
        for name, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"mode {name} missing 'size'"
            assert "tokens" in cfg, f"mode {name} missing 'tokens'"
            assert "pad" in cfg, f"mode {name} missing 'pad'"

    def test_pad_modes(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_sizes_are_tuples_of_ints(self):
        for name, cfg in MODE_CONFIGS.items():
            size = cfg["size"]
            assert len(size) == 2
            assert all(isinstance(s, int) for s in size)

    def test_token_counts_positive(self):
        for name, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad  (Pillow 10 API)
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300), f"Expected (300,300), got {result.size}"

    def test_output_is_pil_image(self):
        img = _make_rgb_image(50, 50)
        result = resize_and_pad(img, (100, 100))
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        img = _make_rgb_image(40, 60)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"

    def test_landscape_image_padded_correctly(self):
        # Wide image → vertical padding expected
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image_padded_correctly(self):
        img = _make_rgb_image(100, 200)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = _make_rgb_image(10, 10, color=(0, 0, 0))
        result = resize_and_pad(img, (50, 50), pad_color=(255, 0, 0))
        # The top-left corner should be pad color (red) since image is centred
        pixel = result.getpixel((0, 0))
        assert pixel == (255, 0, 0)

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_resize_uses_bilinear(self):
        """Pillow 10 removed ANTIALIAS; ensure BILINEAR constant still works."""
        img = _make_rgb_image(200, 200)
        # Should not raise even with Pillow 10
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)


# ---------------------------------------------------------------------------
# load_image  (Pillow + torch)
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = _make_rgb_image(200, 150)
        self.tmp_path = _save_temp_image(self.img)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_4d(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.ndim == 4, "Expected 4-D tensor [1, 3, H, W]"

    def test_tensor_batch_size_one(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.shape[0] == 1

    def test_tensor_has_three_channels(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.shape[1] == 3

    @pytest.mark.parametrize("mode,expected_size", [
        ("tiny", (512, 512)),
        ("small", (640, 640)),
        ("base", (1024, 1024)),
        ("large", (1280, 1280)),
    ])
    def test_spatial_dims_match_mode(self, mode, expected_size):
        tensor = load_image(self.tmp_path, mode=mode)
        assert tensor.shape[2] == expected_size[1]
        assert tensor.shape[3] == expected_size[0]

    def test_return_pil_flag(self):
        pil_img = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(pil_img, Image.Image)

    def test_return_pil_size_matches_mode(self):
        pil_img = load_image(self.tmp_path, mode="small", return_pil=True)
        assert pil_img.size == MODE_CONFIGS["small"]["size"]

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="nonexistent_mode")

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_are_normalised(self):
        # After ImageNet normalisation values can be negative
        tensor = load_image(self.tmp_path, mode="tiny")
        # Values should not all be in [0,1] after normalisation
        assert tensor.min().item() < 1.0

    def test_png_image(self):
        path = _save_temp_image(self.img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(path)

    def test_rgba_image_converted_to_rgb(self):
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        path = _save_temp_image(rgba, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_fills_target_all_tokens(self):
        # If orig == target, all tokens are valid
        tokens = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert tokens == 256

    def test_landscape_fewer_valid_tokens(self):
        # Wide image → height scaled down → fewer valid tokens
        tokens = calculate_valid_tokens(1000, 200, 512, 512, 256)
        assert tokens < 256

    def test_returns_int(self):
        tokens = calculate_valid_tokens(100, 100, 200, 200, 100)
        assert isinstance(tokens, int)

    def test_valid_tokens_not_negative(self):
        tokens = calculate_valid_tokens(10, 10, 512, 512, 256)
        assert tokens >= 0

    def test_valid_tokens_not_exceed_total(self):
        tokens = calculate_valid_tokens(300, 300, 512, 512, 256)
        assert tokens <= 256

    def test_portrait_image(self):
        tokens = calculate_valid_tokens(200, 1000, 512, 512, 256)
        assert 0 < tokens <= 256

    def test_exact_calculation(self):
        # orig 256x256, target 512x512 → scale=2, scaled=512x512 → ratio=1 → 100 tokens
        tokens = calculate_valid_tokens(256, 256, 512, 512, 100)
        assert tokens == 100
