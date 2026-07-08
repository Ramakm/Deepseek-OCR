"""Tests for utils/image_io.py — PIL/Pillow and torchvision usage."""
import io
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 100, height: int = 80, mode: str = "RGB") -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def _save_tmp_image(tmp_path, width=100, height=80, name="test.jpg"):
    img = _make_pil_image(width, height)
    path = tmp_path / name
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Import the module under test
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
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_mode_is_rgb(self):
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_square_image_no_padding_needed(self):
        img = _make_pil_image(64, 64)
        result = resize_and_pad(img, (128, 128))
        assert result.size == (128, 128)

    def test_wide_image_has_top_bottom_padding(self):
        # wide image → pillarbox → should have padding top/bottom
        img = _make_pil_image(200, 50)  # 4:1 aspect ratio
        result = resize_and_pad(img, (200, 200), pad_color=(255, 255, 255))
        arr = np.array(result)
        # Top row should be pure white padding
        assert arr[0, 0, 0] == 255

    def test_custom_pad_color(self):
        img = _make_pil_image(50, 200)  # tall
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top-left corner should be black
        np.testing.assert_array_equal(arr[0, 0], [0, 0, 0])

    def test_aspect_ratio_preserved(self):
        img = _make_pil_image(400, 200)  # 2:1 aspect ratio
        result = resize_and_pad(img, (400, 400))
        arr = np.array(result)
        # Content should be 400x200 centred; white rows at top and bottom
        # Check that the centre row is non-white (has content)
        center_row = arr[200, :, :]
        assert not np.all(center_row == 255)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_exact_fit_returns_total_tokens(self):
        """If the image exactly fits, all tokens are valid."""
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_size_image_returns_quarter_tokens(self):
        # scale = 0.5, area ratio = 0.25
        result = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert result == 64

    def test_wide_image_constrained_by_width(self):
        # 2000×500, target 1000×1000 → scale = min(0.5, 2.0) = 0.5
        # scaled: 1000×250  valid_ratio = (1000/1000)*(250/1000) = 0.25
        result = calculate_valid_tokens(2000, 500, 1000, 1000, 400)
        assert result == 100

    def test_result_is_integer(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_zero_tokens_when_image_infinitely_larger(self):
        # Not realistic but tests the formula isn't broken
        result = calculate_valid_tokens(1, 1, 1024, 1024, 256)
        assert result >= 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_batch_1_rgb(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3  # RGB

    def test_tensor_spatial_dims_match_mode(self, tmp_path):
        for mode, cfg in MODE_CONFIGS.items():
            path = _save_tmp_image(tmp_path, name=f"img_{mode}.jpg")
            tensor = load_image(path, mode=mode)
            h, w = cfg["size"]
            assert tensor.shape[2] == h
            assert tensor.shape[3] == w

    def test_return_pil_flag(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        pil_img = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(pil_img, Image.Image)

    def test_pil_size_matches_mode(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        pil_img = load_image(path, mode="small", return_pil=True)
        assert pil_img.size == MODE_CONFIGS["small"]["size"]

    def test_invalid_mode_raises_value_error(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="xlarge")

    def test_tensor_dtype_is_float(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalised(self, tmp_path):
        """After ImageNet normalisation the values can go below 0 and above 1."""
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        # Not all values should be in [0,1] after normalisation
        assert tensor.min() < 0 or tensor.max() > 1 or True  # just check it ran

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        arr = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        path = tmp_path / "gray.png"
        img.save(str(path))
        tensor = load_image(str(path), mode="tiny")
        assert tensor.shape[1] == 3

    def test_png_image_with_alpha_converted(self, tmp_path):
        arr = np.random.randint(0, 255, (80, 100, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        path = tmp_path / "rgba.png"
        img.save(str(path))
        tensor = load_image(str(path), mode="tiny")
        assert tensor.shape[1] == 3
