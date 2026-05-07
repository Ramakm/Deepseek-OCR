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

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    save_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w=200, h=150, color=(128, 64, 32)) -> Image.Image:
    """Create a simple solid-color PIL RGB image."""
    img = Image.new("RGB", (w, h), color)
    return img


def _save_tmp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_correctness(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_count_ordering(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ("tiny", "small", "base", "large")]
        assert tokens == sorted(tokens), "Token counts should increase with mode size"


# ---------------------------------------------------------------------------
# resize_and_pad  (Pillow API)
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(300, 200)
        out = resize_and_pad(img, (256, 256))
        assert out.size == (256, 256)

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        out = resize_and_pad(img, (64, 64))
        assert out.mode == "RGB"

    def test_aspect_ratio_preserved_wide_image(self):
        """Wide image: pillar-boxed (left/right padding)."""
        img = _make_rgb_image(400, 100)  # 4:1
        out = resize_and_pad(img, (100, 100))
        arr = np.array(out)
        # Corners should be white (pad_color default)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_aspect_ratio_preserved_tall_image(self):
        """Tall image: letter-boxed (top/bottom padding)."""
        img = _make_rgb_image(100, 400)  # 1:4
        out = resize_and_pad(img, (100, 100))
        assert out.size == (100, 100)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 50)
        out = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        arr = np.array(out)
        # Top-left corner should be black (pad)
        assert arr[0, 0].tolist() == [0, 0, 0]

    def test_square_image_no_padding(self):
        """Square → square target: no padding needed."""
        img = _make_rgb_image(100, 100, color=(10, 20, 30))
        out = resize_and_pad(img, (64, 64))
        arr = np.array(out)
        assert out.size == (64, 64)
        # No pure-white pixels since no padding
        assert not np.all(arr == 255)

    def test_uses_bilinear_resampling(self):
        """Smoke-test: resize_and_pad completes without error using BILINEAR."""
        img = _make_rgb_image(123, 77)
        out = resize_and_pad(img, (128, 128))
        assert out is not None

    def test_pillow_image_bilinear_constant(self):
        """Verify Image.BILINEAR is accessible (Pillow 10 compat)."""
        assert hasattr(Image, "BILINEAR") or hasattr(Image, "Resampling")


# ---------------------------------------------------------------------------
# load_image  (Pillow + torchvision)
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        img = _make_rgb_image(300, 200)
        self.tmp_path = _save_tmp_image(img, ".jpg")

    def teardown_method(self):
        os.unlink(self.tmp_path)

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

    def test_return_pil(self):
        pil = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(pil, Image.Image)

    def test_return_pil_size_matches_mode(self):
        pil = load_image(self.tmp_path, mode="small", return_pil=True)
        assert pil.size == MODE_CONFIGS["small"]["size"]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="xlarge")

    def test_normalized_values_in_range(self):
        """After ImageNet normalization values can be outside [0,1] but finite."""
        tensor = load_image(self.tmp_path, mode="tiny")
        assert torch.isfinite(tensor).all()

    def test_png_image(self):
        img = _make_rgb_image(100, 100)
        tmp = _save_tmp_image(img, ".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(tmp)

    def test_rgba_image_converted_to_rgb(self):
        """RGBA images should be loadable (convert to RGB internally)."""
        img = Image.new("RGBA", (100, 100), (10, 20, 30, 128))
        tmp = _save_tmp_image(img, ".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(tmp)

    def test_grayscale_image_converted_to_rgb(self):
        """Grayscale images should be converted to RGB."""
        img = Image.new("L", (100, 100), 128)
        tmp = _save_tmp_image(img, ".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_no_padding(self):
        """Square image to square target → all tokens valid."""
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_size_image(self):
        """Image half target size → ~25% of tokens valid."""
        result = calculate_valid_tokens(50, 50, 100, 100, 256)
        assert result == 64  # 0.5 * 0.5 * 256 = 64

    def test_wide_image(self):
        result = calculate_valid_tokens(200, 100, 200, 200, 400)
        # scale = min(1.0, 2.0) = 1.0; scaled_w=200, scaled_h=100
        # ratio = (200/200) * (100/200) = 0.5
        assert result == 200

    def test_result_non_negative(self):
        result = calculate_valid_tokens(10, 10, 1000, 1000, 256)
        assert result >= 0

    def test_result_lte_total_tokens(self):
        result = calculate_valid_tokens(50, 80, 100, 100, 256)
        assert result <= 256

    def test_integer_result(self):
        result = calculate_valid_tokens(64, 48, 100, 100, 100)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# save_image  (Pillow write path)
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_3d_tensor(self):
        tensor = torch.rand(3, 64, 64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_image(tensor, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_4d_tensor(self):
        tensor = torch.rand(1, 3, 64, 64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            save_image(tensor, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
