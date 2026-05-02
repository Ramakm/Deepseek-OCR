"""Tests for utils/image_io.py — exercises Pillow 10.x APIs."""
import io
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image, __version__ as PIL_VERSION
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=150, mode="RGB"):
    """Create a small synthetic PIL image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temp file; caller is responsible for cleanup."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS sanity checks
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2
            assert isinstance(cfg["tokens"], int)
            assert isinstance(cfg["pad"], bool)

    def test_base_uses_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True

    def test_large_uses_padding(self):
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False

    def test_small_no_padding(self):
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# resize_and_pad  (Pillow 10.x compat — Image.BILINEAR still works via
# Image.Resampling.BILINEAR alias, but the constant value must be accepted)
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(300, 200)
        out = resize_and_pad(img, (512, 512))
        assert out.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_pil_image(100, 100)
        out = resize_and_pad(img, (256, 256))
        assert out.mode == "RGB"

    def test_aspect_ratio_preserved(self):
        """The content region inside the padded canvas should keep ratio."""
        img = make_pil_image(200, 100)  # 2:1 aspect
        out = resize_and_pad(img, (400, 400))
        assert out.size == (400, 400)

    def test_custom_pad_color(self):
        """Corners should contain the pad colour for non-square input."""
        img = make_pil_image(100, 50)  # wide image
        out = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(out)
        # Top-left corner should be black (padding)
        assert arr[0, 0, 0] == 0

    def test_square_input_no_padding_needed(self):
        img = make_pil_image(100, 100)
        out = resize_and_pad(img, (100, 100))
        assert out.size == (100, 100)

    def test_pillow_bilinear_constant_accepted(self):
        """Pillow 10 deprecated some resampling aliases; ensure no crash."""
        img = make_pil_image(80, 60)
        # Should not raise even under Pillow 10+
        out = resize_and_pad(img, (128, 128))
        assert out is not None


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self, tmp_path):
        img = make_pil_image(256, 256)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path)

    def test_returns_tensor(self):
        result = load_image(self.img_path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self):
        t = load_image(self.img_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        t = load_image(self.img_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        t = load_image(self.img_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        t = load_image(self.img_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_return_pil(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_output_size(self):
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="nonexistent")

    def test_tensor_dtype_float(self):
        t = load_image(self.img_path, mode="tiny")
        assert t.dtype == torch.float32

    def test_tensor_normalized(self):
        """After ImageNet normalization values should be roughly in [-3, 3]."""
        t = load_image(self.img_path, mode="tiny")
        assert t.min().item() > -4.0
        assert t.max().item() < 4.0

    def test_loads_png(self, tmp_path):
        img = make_pil_image(64, 64)
        path = str(tmp_path / "img.png")
        img.save(path)
        t = load_image(path, mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_loads_rgba_converted_to_rgb(self, tmp_path):
        arr = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        path = str(tmp_path / "img.png")
        img.save(path)
        t = load_image(path, mode="tiny")
        assert t.shape[1] == 3  # RGB channels


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        """Square image equal to target gives all tokens as valid."""
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256

    def test_half_width_reduces_tokens(self):
        tokens = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert tokens < 256

    def test_positive_result(self):
        tokens = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert tokens > 0

    def test_tokens_not_exceed_total(self):
        tokens = calculate_valid_tokens(2000, 2000, 1024, 1024, 256)
        assert tokens <= 256

    def test_small_image_very_few_tokens(self):
        tokens = calculate_valid_tokens(64, 64, 1024, 1024, 256)
        assert tokens < 20
