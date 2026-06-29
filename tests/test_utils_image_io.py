"""Tests for utils/image_io.py — exercises Pillow (9→10 major bump) APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=150, mode="RGB"):
    """Create a simple in-memory PIL image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def save_pil_to_tmp(tmp_path, img, name="test.jpg"):
    p = tmp_path / name
    img.save(str(p))
    return str(p)


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
# MODE_CONFIGS structure
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

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_pad_flag_types(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["pad"], bool)

    def test_base_and_large_use_pad(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_pad(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------
class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_pil_image(300, 200)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_pil_image(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_output_mode_rgb(self):
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded_to_square(self):
        img = make_pil_image(400, 100)  # wide
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_portrait_image_padded_to_square(self):
        img = make_pil_image(100, 400)  # tall
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_custom_pad_color(self):
        img = make_pil_image(50, 50)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        assert result.size == (200, 200)

    def test_square_image_fits_exactly(self):
        img = make_pil_image(256, 256)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_aspect_ratio_preserved_landscape(self):
        """Resized content area should honour the original aspect ratio."""
        img = make_pil_image(400, 200)  # 2:1 ratio
        result = resize_and_pad(img, (400, 400))
        arr = np.array(result)
        # Top and bottom rows should be white (default pad colour)
        top_row = arr[0, :, :]
        assert np.all(top_row == 255) or np.all(top_row >= 250)


# ---------------------------------------------------------------------------
# load_image — uses Pillow Image.open / resize / BILINEAR
# ---------------------------------------------------------------------------
class TestLoadImage:
    def test_returns_tensor(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_path):
        img = make_pil_image(200, 150)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_path):
        img = make_pil_image(300, 400)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_path):
        img = make_pil_image(640, 480)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_return_pil(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_unknown_mode_raises(self, tmp_path):
        img = make_pil_image(100, 100)
        path = save_pil_to_tmp(tmp_path, img)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="xlarge")

    def test_tensor_dtype_float(self, tmp_path):
        img = make_pil_image(50, 50)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_normalized(self, tmp_path):
        """After ImageNet normalisation values can be negative."""
        img = make_pil_image(50, 50)
        path = save_pil_to_tmp(tmp_path, img)
        result = load_image(path, mode="tiny")
        # Not all values should be in [0,1] because of normalisation
        assert result.min() < 1.0  # sanity check

    def test_png_image(self, tmp_path):
        img = make_pil_image(80, 80)
        path = save_pil_to_tmp(tmp_path, img, name="test.png")
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        arr = np.random.randint(0, 255, (60, 60, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        path = tmp_path / "rgba.png"
        img.save(str(path))
        result = load_image(str(path), mode="tiny")
        assert result.shape[1] == 3  # RGB channels

    def test_grayscale_image_converted(self, tmp_path):
        arr = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        path = tmp_path / "gray.png"
        img.save(str(path))
        result = load_image(str(path), mode="tiny")
        assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------
class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        # 1024x1024 image into 1024x1024 target → all tokens valid
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result > 0
        assert result <= 256

    def test_landscape_image(self):
        result = calculate_valid_tokens(2000, 1000, 1280, 1280, 400)
        assert 0 < result <= 400

    def test_portrait_image(self):
        result = calculate_valid_tokens(800, 1600, 1280, 1280, 400)
        assert 0 < result <= 400

    def test_returns_int(self):
        result = calculate_valid_tokens(640, 480, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_tiny_image_small_tokens(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result < 256
