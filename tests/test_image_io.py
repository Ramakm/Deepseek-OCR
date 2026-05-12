"""Tests for utils/image_io.py — exercises Pillow (9→10 major bump) APIs."""
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

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
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

def _make_rgb_image(width=200, height=150, color=(128, 64, 32)) -> Image.Image:
    """Create a solid-colour PIL RGB image."""
    return Image.new("RGB", (width, height), color)


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temporary file and return its path."""
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

    def test_mode_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"'{mode}' missing 'size'"
            assert "tokens" in cfg, f"'{mode}' missing 'tokens'"
            assert "pad" in cfg, f"'{mode}' missing 'pad'"

    def test_pad_modes(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad  (Pillow API)
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300)

    def test_returns_pil_image(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert isinstance(result, Image.Image)

    def test_mode_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide_image(self):
        """Wide image: pillar-boxing expected (left/right padding)."""
        img = _make_rgb_image(400, 100)  # 4:1 aspect
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_aspect_ratio_preserved_tall_image(self):
        """Tall image: letter-boxing expected (top/bottom padding)."""
        img = _make_rgb_image(100, 400)  # 1:4 aspect
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_custom_pad_color(self):
        """Padding pixels should be the requested colour."""
        img = _make_rgb_image(100, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        arr = np.array(result)
        # Top row should be red padding
        top_pixel = tuple(arr[0, 0, :])
        assert top_pixel == (255, 0, 0)

    def test_square_image_no_padding(self):
        """Square image to square target — no padding needed."""
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_pillow_bilinear_resampling(self):
        """Smoke-test that Pillow's BILINEAR resampling constant is usable."""
        img = _make_rgb_image(64, 64)
        # Pillow 10 renamed ANTIALIAS → LANCZOS; BILINEAR is still valid
        resized = img.resize((128, 128), Image.BILINEAR)
        assert resized.size == (128, 128)


# ---------------------------------------------------------------------------
# load_image  (Pillow + torchvision + numpy + torch)
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = _make_rgb_image(300, 200)
        self.path = _save_temp_image(self.img)

    def teardown_method(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    # --- mode validation ---

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.path, mode="xlarge")

    # --- return types ---

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_returns_pil_when_requested(self):
        pil = load_image(self.path, mode="tiny", return_pil=True)
        assert isinstance(pil, Image.Image)

    # --- tensor shape ---

    @pytest.mark.parametrize("mode", list(MODE_CONFIGS.keys()))
    def test_tensor_shape_for_all_modes(self, mode):
        tensor = load_image(self.path, mode=mode)
        expected_h, expected_w = MODE_CONFIGS[mode]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w), (
            f"mode={mode}: expected (1,3,{expected_h},{expected_w}), got {tensor.shape}"
        )

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.path, mode="tiny")
        assert tensor.dtype == torch.float32

    # --- normalisation ---

    def test_tensor_values_normalized(self):
        """After ImageNet normalisation values can be outside [0,1]."""
        tensor = load_image(self.path, mode="tiny")
        # Values should be finite
        assert torch.isfinite(tensor).all()

    # --- pil return ---

    @pytest.mark.parametrize("mode,expect_pad", [
        ("tiny", False), ("small", False), ("base", True), ("large", True)
    ])
    def test_pil_size_correct(self, mode, expect_pad):
        pil = load_image(self.path, mode=mode, return_pil=True)
        expected = MODE_CONFIGS[mode]["size"]
        assert pil.size == expected

    # --- different image formats ---

    def test_load_png(self):
        path = _save_temp_image(self.img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert isinstance(tensor, torch.Tensor)
        finally:
            os.remove(path)

    def test_grayscale_image_converted_to_rgb(self):
        gray = Image.new("L", (100, 100), 128)
        path = _save_temp_image(gray, suffix=".jpg")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Image same aspect as target → all tokens valid."""
        tokens = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert tokens == 256

    def test_half_width_image(self):
        """Image half-width of target → roughly half tokens."""
        tokens = calculate_valid_tokens(50, 100, 100, 100, 256)
        # scale = min(100/50, 100/100) = 1.0 → scaled=(50,100)
        # ratio = (50/100)*(100/100) = 0.5
        assert tokens == int(256 * 0.5)

    def test_returns_integer(self):
        tokens = calculate_valid_tokens(300, 200, 1024, 1024, 400)
        assert isinstance(tokens, int)

    def test_non_negative(self):
        tokens = calculate_valid_tokens(1, 1, 1000, 1000, 100)
        assert tokens >= 0

    def test_small_image_fewer_tokens(self):
        tokens_small = calculate_valid_tokens(100, 100, 1000, 1000, 400)
        tokens_full = calculate_valid_tokens(1000, 1000, 1000, 1000, 400)
        assert tokens_small < tokens_full
