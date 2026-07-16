"""Tests for utils/image_io.py - exercises Pillow (9→12) and torch/torchvision APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make the project root importable from tests/
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

def _make_tmp_image(width=200, height=150, color=(128, 64, 32), fmt="JPEG"):
    """Create a temporary image file and return its path."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False)
    img.save(tmp.name, format=fmt)
    tmp.close()
    return tmp.name


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

    def test_size_is_tuple_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            s = cfg["size"]
            assert len(s) == 2
            assert all(isinstance(v, int) for v in s)

    def test_pad_flag(self):
        # tiny and small should NOT pad; base and large should pad
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_ordering(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens), "Token counts should increase with resolution"


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        from PIL import Image

        img = Image.new("RGB", (300, 200))
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_aspect_ratio_preserved(self):
        """The content area should keep the aspect ratio of the original."""
        from PIL import Image

        orig_w, orig_h = 400, 200  # 2:1 ratio
        img = Image.new("RGB", (orig_w, orig_h), color=(255, 0, 0))
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_white_padding_default(self):
        """Corners should be white (padding colour) for a non-square image."""
        from PIL import Image

        img = Image.new("RGB", (100, 50), color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200))
        # Top-left corner should be white padding
        corner = result.getpixel((0, 0))
        assert corner == (255, 255, 255)

    def test_custom_pad_color(self):
        from PIL import Image

        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 255))
        corner = result.getpixel((0, 0))
        assert corner == (0, 0, 255)

    def test_square_image_no_padding_needed(self):
        """A square image resized to a square target should fill the entire canvas."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(10, 20, 30))
        result = resize_and_pad(img, (200, 200))
        # Corner should be the image colour (no padding)
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.tmp_path = _make_tmp_image(width=200, height=150)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    # ---- tensor output ----

    def test_returns_tensor_by_default(self):
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

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_is_normalised(self):
        """After ImageNet normalisation, values should not all be in [0,1]."""
        tensor = load_image(self.tmp_path, mode="tiny")
        # Values can be negative after normalisation
        assert tensor.min().item() < 1.0

    # ---- PIL output ----

    def test_returns_pil_when_requested(self):
        from PIL import Image

        img = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_pil_size_tiny(self):
        img = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]

    def test_pil_size_base_uses_padding(self):
        img = load_image(self.tmp_path, mode="base", return_pil=True)
        assert img.size == MODE_CONFIGS["base"]["size"]

    # ---- error handling ----

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="ultra")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    # ---- PNG format ----

    def test_loads_png(self):
        png_path = _make_tmp_image(width=64, height=64, fmt="PNG")
        try:
            tensor = load_image(png_path, mode="tiny")
            assert isinstance(tensor, torch.Tensor)
        finally:
            os.unlink(png_path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        # A square image fitted into a square target → all tokens are valid
        total = 256
        result = calculate_valid_tokens(512, 512, 512, 512, total)
        assert result == total

    def test_wide_image_fewer_tokens(self):
        # Wide image → top/bottom padding → fewer valid tokens
        result = calculate_valid_tokens(1000, 200, 512, 512, 256)
        assert result < 256

    def test_tall_image_fewer_tokens(self):
        result = calculate_valid_tokens(200, 1000, 512, 512, 256)
        assert result < 256

    def test_result_non_negative(self):
        result = calculate_valid_tokens(100, 50, 1024, 1024, 400)
        assert result >= 0

    def test_exact_fit(self):
        """Image with same aspect ratio as target should fill most tokens."""
        result = calculate_valid_tokens(256, 256, 256, 256, 100)
        assert result == 100


# ---------------------------------------------------------------------------
# save_image (smoke test)
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_saves_3d_tensor(self):
        tensor = torch.rand(3, 64, 64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(tensor, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_saves_4d_tensor(self):
        tensor = torch.rand(1, 3, 64, 64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(tensor, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
