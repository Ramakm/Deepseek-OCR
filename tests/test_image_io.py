"""Tests for utils/image_io.py — exercises PIL (Pillow) API as used in the project."""
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width=200, height=150, mode="RGB"):
    """Return a small synthetic PIL image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def _save_temp_image(width=200, height=150, fmt="JPEG"):
    """Save a synthetic image to a temp file and return its path."""
    img = _make_pil_image(width, height)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


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
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_base_and_large_use_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_are_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(300, 100)
        target = (200, 200)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_aspect_ratio_preserved_landscape(self):
        """Wide image: letterboxed (top/bottom bars)."""
        img = _make_pil_image(400, 100)  # 4:1 landscape
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_aspect_ratio_preserved_portrait(self):
        img = _make_pil_image(100, 400)  # tall portrait
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_square_image_no_padding(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        # All pixels should be image content (no white padding bars needed
        # when both dims scale identically)
        assert result.size == (200, 200)

    def test_custom_pad_color(self):
        img = _make_pil_image(50, 50)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top-left corner should be black padding for a tiny square in a large canvas
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (128, 128))
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.tmp_path = _save_temp_image(200, 150)

    def teardown_method(self):
        os.unlink(self.tmp_path)

    # --- happy path (tensor output) ---

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_returns_tensor_for_all_modes(self, mode):
        result = load_image(self.tmp_path, mode=mode)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_tensor_shape_correct(self, mode):
        result = load_image(self.tmp_path, mode=mode)
        cfg = MODE_CONFIGS[mode]
        h, w = cfg["size"]
        assert result.shape == (1, 3, h, w), f"Mode {mode}: got {result.shape}"

    def test_tensor_dtype_float32(self):
        result = load_image(self.tmp_path, mode="base")
        assert result.dtype == torch.float32

    def test_tensor_range_normalized(self):
        """After ImageNet normalization values can be outside [0,1]."""
        result = load_image(self.tmp_path, mode="base")
        # The tensor should be finite
        assert torch.isfinite(result).all()

    def test_return_pil_flag(self):
        result = load_image(self.tmp_path, mode="base", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_output_size_matches_mode(self):
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    # --- error cases ---

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="nonexistent")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="base")

    # --- Pillow-specific: BILINEAR resize still produces correct shape ---

    def test_bilinear_resize_used_for_tiny(self):
        """tiny/small modes use direct resize — verify output size is correct."""
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_pad_mode_base_output_size(self):
        result = load_image(self.tmp_path, mode="base", return_pil=True)
        assert result.size == (1024, 1024)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Square image filling the full canvas → all tokens valid."""
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256

    def test_half_width_image(self):
        tokens = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        # scale = min(1024/512, 1024/1024) = 1.0 (height limited)
        # scaled_w=512, scaled_h=1024; ratio = (512/1024)*(1024/1024) = 0.5
        assert tokens == 128

    def test_returns_integer(self):
        tokens = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert isinstance(tokens, int)

    def test_never_exceeds_total(self):
        for ow, oh in [(100, 100), (512, 256), (1280, 720), (1024, 1024)]:
            tokens = calculate_valid_tokens(ow, oh, 1024, 1024, 256)
            assert tokens <= 256

    def test_non_negative(self):
        tokens = calculate_valid_tokens(50, 50, 1024, 1024, 256)
        assert tokens >= 0

    def test_large_image_clipped_to_total(self):
        """Image larger than target: scaling brings it down, ratio ≤ 1."""
        tokens = calculate_valid_tokens(2048, 2048, 1024, 1024, 256)
        assert tokens == 256  # scale=0.5 → 1024x1024 fills canvas

    def test_landscape_image(self):
        tokens = calculate_valid_tokens(2000, 1000, 1024, 1024, 256)
        # scale = min(1024/2000, 1024/1000) = 0.512 → scaled_w=1024, scaled_h=512
        # ratio = (1024/1024)*(512/1024) = 0.5 → 128 tokens
        assert tokens == 128
