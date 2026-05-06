"""Tests for utils/image_io.py — covers PIL (pillow 10.x), torch, torchvision usage."""
import io
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width=100, height=80, mode="RGB", color=(128, 64, 32)):
    img = Image.new(mode, (width, height), color)
    return img


def _save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
    save_image,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"mode {mode} missing 'size'"
            assert "tokens" in cfg, f"mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"mode {mode} missing 'pad'"

    def test_sizes_are_tuples_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_pad_modes(self):
        # base and large should pad; tiny and small should not
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_increase_with_resolution(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ["tiny", "small", "base", "large"]]
        assert tokens == sorted(tokens), "Token counts should increase with mode"


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_has_correct_size(self):
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_pil_image(self):
        img = _make_pil_image(50, 50)
        result = resize_and_pad(img, (256, 256))
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        img = _make_pil_image(50, 50)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide_image(self):
        # Wide image: 200×100 padded into 256×256 — height padding expected
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (256, 256))
        arr = np.array(result)
        # Top and bottom rows should be mostly white (padding)
        assert arr[0, 0, 0] == 255  # top-left pixel is pad colour

    def test_custom_pad_color(self):
        img = _make_pil_image(50, 100)
        result = resize_and_pad(img, (256, 256), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Corner pixels should be black
        assert arr[0, 0, 0] == 0

    def test_square_image_no_effective_padding(self):
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture
    def tmp_image_path(self, tmp_path):
        img = _make_pil_image(200, 150)
        path = str(tmp_path / "test.jpg")
        img.save(path)
        return path

    def test_returns_tensor_by_default(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_has_batch_dimension(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="tiny")
        assert tensor.ndim == 4  # [1, 3, H, W]

    def test_tensor_has_three_channels(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="tiny")
        assert tensor.shape[1] == 3

    def test_tiny_mode_output_size(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="tiny")
        expected = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape[2] == expected[1]
        assert tensor.shape[3] == expected[0]

    def test_small_mode_output_size(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="small")
        expected = MODE_CONFIGS["small"]["size"]
        assert tensor.shape[2] == expected[1]
        assert tensor.shape[3] == expected[0]

    def test_base_mode_output_size(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="base")
        expected = MODE_CONFIGS["base"]["size"]
        assert tensor.shape[2] == expected[1]
        assert tensor.shape[3] == expected[0]

    def test_large_mode_output_size(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="large")
        expected = MODE_CONFIGS["large"]["size"]
        assert tensor.shape[2] == expected[1]
        assert tensor.shape[3] == expected[0]

    def test_return_pil_flag(self, tmp_image_path):
        result = load_image(tmp_image_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_pil_return_has_correct_size(self, tmp_image_path):
        result = load_image(tmp_image_path, mode="tiny", return_pil=True)
        expected = MODE_CONFIGS["tiny"]["size"]
        assert result.size == expected

    def test_unknown_mode_raises_value_error(self, tmp_image_path):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image_path, mode="ultra")

    def test_tensor_dtype_is_float(self, tmp_image_path):
        tensor = load_image(tmp_image_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_are_normalised(self, tmp_image_path):
        # After ImageNet normalisation values can be negative or >1
        tensor = load_image(tmp_image_path, mode="tiny")
        # At minimum, should not be in raw [0, 255] range
        assert tensor.min() < 1.0

    def test_png_file_loads_correctly(self, tmp_path):
        img = _make_pil_image(100, 100)
        path = str(tmp_path / "test.png")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        path = str(tmp_path / "gray.jpg")
        img.save(path)
        tensor = load_image(path, mode="tiny")
        assert tensor.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        # When orig == target the entire canvas is valid
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_half_area_image(self):
        # orig is half size → ratio ≈ 0.5 * 0.5 = 0.25
        # scale = min(512/256, 512/256) = 2 → scaled = 512×512 → ratio = 1.0
        # Actually orig 256×256, target 512×512: scale=2, scaled=512×512, ratio=1.0
        result = calculate_valid_tokens(256, 256, 512, 512, 256)
        assert result == 256

    def test_wide_image_reduces_valid_tokens(self):
        # 1000×100 in 512×512 → scale=min(0.512, 5.12)=0.512
        # scaled = 512×51, ratio ≈ (512/512)*(51/512) ≈ 0.0996
        result = calculate_valid_tokens(1000, 100, 512, 512, 256)
        assert result < 256

    def test_returns_integer(self):
        result = calculate_valid_tokens(200, 100, 512, 512, 256)
        assert isinstance(result, int)

    def test_result_non_negative(self):
        result = calculate_valid_tokens(50, 50, 512, 512, 256)
        assert result >= 0

    def test_result_at_most_total(self):
        result = calculate_valid_tokens(100, 100, 512, 512, 256)
        assert result <= 256


# ---------------------------------------------------------------------------
# save_image  (uses PIL Image.fromarray — pillow 10.x API)
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_saves_file(self, tmp_path):
        tensor = torch.rand(3, 64, 64)
        path = str(tmp_path / "out.png")
        save_image(tensor, path)
        assert os.path.exists(path)

    def test_saved_file_can_be_reopened(self, tmp_path):
        tensor = torch.rand(3, 64, 64)
        path = str(tmp_path / "out.png")
        save_image(tensor, path)
        img = Image.open(path)
        assert img is not None

    def test_saved_image_dimensions(self, tmp_path):
        tensor = torch.rand(3, 48, 64)
        path = str(tmp_path / "out.png")
        save_image(tensor, path)
        img = Image.open(path)
        assert img.size == (64, 48)
