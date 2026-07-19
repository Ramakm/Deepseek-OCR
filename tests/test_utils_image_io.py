"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) and torch/torchvision APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=150, color=(128, 64, 32)):
    """Create a small solid-colour RGB PIL image."""
    img = Image.new("RGB", (width, height), color)
    return img


def save_pil_to_tmp(tmp_path, img, name="test.jpg"):
    """Save a PIL image to a temp file and return the path."""
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts(self):
        assert MODE_CONFIGS["tiny"]["tokens"] == 64
        assert MODE_CONFIGS["small"]["tokens"] == 100
        assert MODE_CONFIGS["base"]["tokens"] == 256
        assert MODE_CONFIGS["large"]["tokens"] == 400


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide_image(self):
        """A wide image should have white bars top/bottom."""
        img = make_pil_image(400, 100)  # 4:1 aspect ratio
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        # The centre row should contain image data; top-left pixel should be pad
        arr = np.array(result)
        # top-left corner is padding (white)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_aspect_ratio_preserved_tall_image(self):
        img = make_pil_image(100, 400)
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        arr = np.array(result)
        assert arr[0, 0].tolist() == [255, 255, 255]

    def test_square_image_no_padding(self):
        """A square source image into a square target shouldn't need padding."""
        img = make_pil_image(100, 100, color=(10, 20, 30))
        result = resize_and_pad(img, (256, 256), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Centre pixel should NOT be the pad colour
        cx, cy = 128, 128
        assert arr[cy, cx].tolist() != [255, 255, 255]

    def test_custom_pad_color(self):
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (400, 400), pad_color=(0, 0, 0))
        arr = np.array(result)
        assert arr[0, 0].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_same_as_target(self):
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256

    def test_wide_image(self):
        # 2048x1024 into 1024x1024 → scale=0.5 → scaled_w=1024,scaled_h=512
        # valid_ratio = (1024/1024)*(512/1024) = 0.5
        tokens = calculate_valid_tokens(2048, 1024, 1024, 1024, 256)
        assert tokens == 128

    def test_tall_image(self):
        tokens = calculate_valid_tokens(512, 2048, 1024, 1024, 256)
        assert tokens == 64

    def test_small_image_same_ratio(self):
        # 512x512 into 1024x1024 → scale=2 but capped to fit? Actually scale=min(2,2)=2
        # scaled_w=1024, scaled_h=1024, valid_ratio=1.0
        tokens = calculate_valid_tokens(512, 512, 1024, 1024, 100)
        assert tokens == 100

    def test_returns_int(self):
        result = calculate_valid_tokens(640, 480, 1024, 1024, 256)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        img = make_pil_image(300, 200)
        return save_pil_to_tmp(tmp_path, img)

    def test_returns_tensor_by_default(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil(self, sample_image_path):
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, sample_image_path):
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_tensor_dtype_is_float(self, sample_image_path):
        tensor = load_image(sample_image_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_invalid_mode_raises(self, sample_image_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(sample_image_path, mode="xlarge")

    def test_normalized_values_range(self, sample_image_path):
        """After ImageNet normalization values should roughly be in [-3, 3]."""
        tensor = load_image(sample_image_path, mode="tiny")
        assert tensor.min().item() > -4.0
        assert tensor.max().item() < 4.0

    def test_png_image(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img, "test.png")
        tensor = load_image(path, mode="tiny")
        assert tensor.shape[0] == 1

    def test_greyscale_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        p = tmp_path / "grey.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3  # 3 channels

    def test_rgba_image_converted(self, tmp_path):
        img = Image.new("RGBA", (100, 100), (128, 64, 32, 200))
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape[1] == 3
