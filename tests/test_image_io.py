"""Tests for utils/image_io.py — exercises PIL (pillow 10.x) APIs."""
import io
import sys
import os
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(w: int = 200, h: int = 150) -> Image.Image:
    """Create a small solid-colour RGB PIL image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_tmp_image(tmp_path, name="test.jpg", w=200, h=150) -> str:
    img = make_rgb_image(w, h)
    p = str(tmp_path / name)
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_mode_has_required_keys(self, mode):
        cfg = MODE_CONFIGS[mode]
        assert "size" in cfg
        assert "tokens" in cfg
        assert "pad" in cfg

    def test_pad_flag_base_and_large(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_pad_flag_tiny_and_small(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    @pytest.mark.parametrize("mode,expected_tokens", [
        ("tiny", 64),
        ("small", 100),
        ("base", 256),
        ("large", 400),
    ])
    def test_token_counts(self, mode, expected_tokens):
        assert MODE_CONFIGS[mode]["tokens"] == expected_tokens


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide_image(self):
        """Wide image: horizontal padding should be minimal vs vertical."""
        img = make_rgb_image(400, 100)  # 4:1 landscape
        result = resize_and_pad(img, (400, 400))
        # Should not raise; output must be 400×400
        assert result.size == (400, 400)

    def test_aspect_ratio_preserved_tall_image(self):
        img = make_rgb_image(100, 400)  # portrait
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_custom_pad_color(self):
        img = make_rgb_image(50, 50)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        # Corners should be black (padding)
        px = result.getpixel((0, 0))
        assert px == (0, 0, 0)

    def test_square_image_no_distortion(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_pillow_bilinear_constant(self):
        """Ensure Image.BILINEAR still resolves (Pillow 10 renamed some attrs)."""
        assert hasattr(Image, "BILINEAR") or hasattr(Image, "Resampling")


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.mark.parametrize("mode", ["tiny", "small", "base", "large"])
    def test_returns_tensor_shape(self, tmp_path, mode):
        p = save_tmp_image(tmp_path, w=300, h=200)
        tensor = load_image(p, mode=mode)
        expected_h, expected_w = MODE_CONFIGS[mode]["size"]
        assert tensor.shape == (1, 3, expected_h, expected_w)

    def test_returns_float_tensor(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_return_pil_flag(self, tmp_path):
        p = save_tmp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_unknown_mode_raises_value_error(self, tmp_path):
        p = save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="xxxxx")

    def test_normalized_values_in_reasonable_range(self, tmp_path):
        """After ImageNet normalization values should be roughly in [-3, 3]."""
        p = save_tmp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        assert tensor.min().item() > -5.0
        assert tensor.max().item() < 5.0

    def test_base_mode_uses_padding(self, tmp_path):
        """For a non-square image in base mode the tensor must still be square."""
        p = save_tmp_image(tmp_path, w=600, h=200)
        tensor = load_image(p, mode="base")
        assert tensor.shape[-1] == tensor.shape[-2]  # square

    def test_tiny_mode_direct_resize(self, tmp_path):
        p = save_tmp_image(tmp_path, w=800, h=600)
        tensor = load_image(p, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_batch_dim_is_1(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(p, mode="small")
        assert tensor.shape[0] == 1

    def test_channels_are_3(self, tmp_path):
        p = save_tmp_image(tmp_path)
        tensor = load_image(p, mode="small")
        assert tensor.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size_full_tokens(self):
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_width_image(self):
        """Image is half the target width; tokens should be ≤ 256."""
        result = calculate_valid_tokens(256, 512, 512, 512, 256)
        assert 0 < result <= 256

    def test_returns_int(self):
        result = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_very_small_image(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0

    def test_landscape_image(self):
        result = calculate_valid_tokens(1000, 500, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_portrait_image(self):
        result = calculate_valid_tokens(500, 1000, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_exact_target_size(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 400)
        assert result == 400
