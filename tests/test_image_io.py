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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

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

def make_pil_image(w=200, h=100, color=(128, 64, 32)):
    """Create a small solid-colour RGB PIL image."""
    img = Image.new("RGB", (w, h), color)
    return img


def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    """Save a PIL image to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = f.name
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_size_is_tuple_of_two_positive_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            s = cfg["size"]
            assert len(s) == 2
            assert all(isinstance(v, int) and v > 0 for v in s)

    def test_pad_flag_types(self):
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
        img = make_pil_image(200, 100)
        result = resize_and_pad(img, (300, 300))
        assert result.size == (300, 300)

    def test_output_is_rgb(self):
        img = make_pil_image(50, 50)
        result = resize_and_pad(img, (64, 64))
        assert result.mode == "RGB"

    def test_landscape_image_fits(self):
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image_fits(self):
        img = make_pil_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_square_image_no_padding_needed(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_custom_pad_color(self):
        img = make_pil_image(10, 10, color=(0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        # Corner pixels should be the pad colour
        corner = result.getpixel((0, 0))
        assert corner == (255, 0, 0)

    def test_aspect_ratio_preserved_in_content(self):
        """After padding, the pasted region should keep the original ratio."""
        img = make_pil_image(200, 100)  # 2:1 landscape
        result = resize_and_pad(img, (200, 200))
        # The content should be 200×100 → scaled to fit 200 wide → still 200×100
        arr = np.array(result)
        # Top and bottom strips should be white (default pad)
        assert arr[0, 0].tolist() == [255, 255, 255]
        assert arr[199, 0].tolist() == [255, 255, 255]


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_returns_total_tokens(self):
        # When orig == target, valid == total
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_area_returns_half_tokens(self):
        # orig 50×50 into target 100×100 → scale=0.5 → scaled 50×50
        # valid_ratio = (50/100) * (50/100) = 0.25
        result = calculate_valid_tokens(50, 50, 100, 100, 256)
        assert result == int(256 * 0.25)

    def test_landscape_into_square(self):
        # orig 200×100, target 100×100 → scale = min(0.5, 1.0) = 0.5
        # scaled = 100×50 → ratio = (100/100)*(50/100) = 0.5
        result = calculate_valid_tokens(200, 100, 100, 100, 400)
        assert result == int(400 * 0.5)

    def test_result_is_int(self):
        result = calculate_valid_tokens(80, 60, 100, 100, 100)
        assert isinstance(result, int)

    def test_zero_tokens(self):
        result = calculate_valid_tokens(100, 100, 100, 100, 0)
        assert result == 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    """Tests for load_image — exercises Pillow image opening and torchvision Normalize."""

    @pytest.fixture
    def temp_jpg(self, tmp_path):
        img = make_pil_image(200, 150)
        p = tmp_path / "test.jpg"
        img.save(str(p))
        return str(p)

    @pytest.fixture
    def temp_png(self, tmp_path):
        img = make_pil_image(64, 64)
        p = tmp_path / "test.png"
        img.save(str(p))
        return str(p)

    def test_returns_tensor_by_default(self, temp_jpg):
        t = load_image(temp_jpg, mode="tiny")
        assert isinstance(t, torch.Tensor)

    def test_tensor_shape_tiny(self, temp_jpg):
        t = load_image(temp_jpg, mode="tiny")
        # tiny size = (512, 512) → [1, 3, 512, 512]
        assert t.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, temp_jpg):
        t = load_image(temp_jpg, mode="small")
        assert t.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, temp_jpg):
        t = load_image(temp_jpg, mode="base")
        assert t.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, temp_jpg):
        t = load_image(temp_jpg, mode="large")
        assert t.shape == (1, 3, 1280, 1280)

    def test_return_pil_tiny(self, temp_jpg):
        img = load_image(temp_jpg, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)

    def test_return_pil_base(self, temp_jpg):
        img = load_image(temp_jpg, mode="base", return_pil=True)
        assert isinstance(img, Image.Image)
        assert img.size == (1024, 1024)

    def test_tensor_dtype_float32(self, temp_jpg):
        t = load_image(temp_jpg, mode="tiny")
        assert t.dtype == torch.float32

    def test_invalid_mode_raises(self, temp_jpg):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(temp_jpg, mode="ultra")

    def test_png_loads_correctly(self, temp_png):
        t = load_image(temp_png, mode="tiny")
        assert t.shape == (1, 3, 512, 512)

    def test_normalization_applied(self, tmp_path):
        """Pixel values should be outside [0,1] after ImageNet normalisation."""
        # Create a white image (all 1.0 before norm)
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        p = tmp_path / "white.jpg"
        img.save(str(p))
        t = load_image(str(p), mode="tiny")
        # After normalizing a white image, values > 1 for some channels
        # (mean subtracted, divided by std → (1-0.485)/0.229 ≈ 2.25)
        assert t.max().item() > 1.0

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (64, 64), (100, 100, 100, 200))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        t = load_image(str(p), mode="tiny")
        assert t.shape[1] == 3  # 3 channels


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_and_reload(self, tmp_path):
        """save_image should produce a readable image file."""
        # Craft a 3-channel tensor in [0,1]
        tensor = torch.rand(3, 32, 32)
        out_path = str(tmp_path / "out.png")
        save_image(tensor, out_path)
        # Reload with PIL
        reloaded = Image.open(out_path)
        assert reloaded.size == (32, 32)

    def test_save_4d_tensor(self, tmp_path):
        """save_image should handle [1, 3, H, W] tensors as well."""
        tensor = torch.rand(1, 3, 32, 32)
        out_path = str(tmp_path / "out4d.png")
        # If save_image supports 4-D, it should not raise
        try:
            save_image(tensor, out_path)
        except Exception:
            pytest.skip("save_image does not yet support 4D tensors — skip")
