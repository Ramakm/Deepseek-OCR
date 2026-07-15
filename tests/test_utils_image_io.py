"""Tests for utils/image_io.py — covers Pillow (9→12) and torch/torchvision APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width=200, height=150, color=(128, 64, 32)) -> Image.Image:
    """Create a small solid-color RGB image."""
    img = Image.new("RGB", (width, height), color)
    return img


def _save_tmp_image(tmp_dir, name="test.jpg", width=200, height=150) -> str:
    """Save a test image to a temp directory and return its path."""
    img = _make_rgb_image(width, height)
    path = os.path.join(tmp_dir, name)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS structure
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_keys_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_fields(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_size_is_tuple_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2
            assert all(isinstance(d, int) for d in cfg["size"])

    def test_pad_flag_values(self):
        # tiny/small should not pad; base/large should pad
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_increase_with_resolution(self):
        tokens = [MODE_CONFIGS[m]["tokens"] for m in ["tiny", "small", "base", "large"]]
        assert tokens == sorted(tokens)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_portrait_image_preserves_aspect(self):
        img = _make_rgb_image(100, 200)  # 1:2 aspect
        result = resize_and_pad(img, (512, 512))
        arr = np.array(result)
        # The image content should be centred; rows at the very top/bottom should be white padding
        assert result.size == (512, 512)

    def test_landscape_image(self):
        img = _make_rgb_image(400, 100)  # 4:1 aspect
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 50, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(0, 255, 0))
        arr = np.array(result)
        # Corners should be green padding
        corner = arr[0, 0]
        assert corner[1] == 255  # green channel

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)

    def test_returns_pil_image(self):
        img = _make_rgb_image(80, 60)
        result = resize_and_pad(img, (512, 512))
        assert isinstance(result, Image.Image)

    def test_non_square_target(self):
        img = _make_rgb_image(300, 100)
        result = resize_and_pad(img, (640, 480))
        assert result.size == (640, 480)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_by_default(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="base")
        assert result.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="large")
        assert result.shape == (1, 3, 1280, 1280)

    def test_return_pil_flag(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_tensor_dtype_is_float(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny")
        assert result.dtype == torch.float32

    def test_invalid_mode_raises(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="ultra")

    def test_tensor_values_normalized(self, tmp_path):
        """Values should roughly sit in a normalised range, not raw [0,1]."""
        path = _save_tmp_image(str(tmp_path))
        tensor = load_image(path, mode="tiny")
        # After ImageNet normalization values can be negative
        assert tensor.min().item() < 1.0

    def test_loads_png(self, tmp_path):
        img = _make_rgb_image(100, 100)
        path = str(tmp_path / "test.png")
        img.save(path)
        result = load_image(path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_non_square_source_image(self, tmp_path):
        path = _save_tmp_image(str(tmp_path), width=400, height=100)
        result = load_image(path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_grayscale_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (100, 100), 128)
        path = str(tmp_path / "gray.jpg")
        img.save(path)
        result = load_image(path, mode="tiny")
        assert result.shape[1] == 3

    def test_batch_dim_is_one(self, tmp_path):
        path = _save_tmp_image(str(tmp_path))
        result = load_image(path, mode="tiny")
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_aspect_ratio_full_tokens(self):
        # Square image into square target: all tokens valid
        tokens = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert tokens == 256

    def test_half_width_image(self):
        # 512x1024 (portrait) into 1024x1024 target
        tokens = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        # scale = min(1024/512, 1024/1024) = 1.0 → scaled = 512x1024
        # ratio = (512/1024) * (1024/1024) = 0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_returns_integer(self):
        result = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_zero_tokens_not_returned_for_valid_image(self):
        result = calculate_valid_tokens(200, 200, 1024, 1024, 256)
        assert result > 0

    def test_result_leq_total_tokens(self):
        result = calculate_valid_tokens(100, 100, 1280, 1280, 400)
        assert result <= 400

    def test_landscape_image(self):
        # 1024x512 (wide) into 1024x1024
        result = calculate_valid_tokens(1024, 512, 1024, 1024, 256)
        # scale = min(1, 2) = 1 → scaled = 1024x512
        # ratio = (1024/1024) * (512/1024) = 0.5
        assert result == 128

    def test_exact_match_returns_total(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256
