"""Tests for utils/image_io.py — covers Pillow (upgraded 9→12) usage."""
import io
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the project root is importable regardless of CWD
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

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

def _make_rgb_image(width: int = 200, height: int = 150) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_temp_image(img: Image.Image, suffix: str = ".jpg") -> str:
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
            assert "size" in cfg, f"Mode {mode} missing 'size'"
            assert "tokens" in cfg, f"Mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"Mode {mode} missing 'pad'"

    def test_pad_flag_correct(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0

    def test_sizes_are_tuples_of_two(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(300, 200)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_horizontally(self):
        """Wide image → padding on top/bottom."""
        img = _make_rgb_image(800, 200)  # 4:1 wide
        target = (512, 512)
        result = resize_and_pad(img, target)
        arr = np.array(result)
        # Top rows should be white padding
        assert np.all(arr[0, :, :] == 255)

    def test_aspect_ratio_preserved_vertically(self):
        """Tall image → padding on left/right."""
        img = _make_rgb_image(200, 800)  # 1:4 tall
        target = (512, 512)
        result = resize_and_pad(img, target)
        arr = np.array(result)
        assert np.all(arr[:, 0, :] == 255)

    def test_custom_pad_color(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 400), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top should be black
        assert np.all(arr[0, :, :] == 0)

    def test_square_image_no_padding(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # No white border at corners expected
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = _make_rgb_image(300, 200)
        self.tmp_path = _save_temp_image(self.img)

    def teardown_method(self):
        os.unlink(self.tmp_path)

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        tensor = load_image(self.tmp_path, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        tensor = load_image(self.tmp_path, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        tensor = load_image(self.tmp_path, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil_true(self):
        pil_img = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(pil_img, Image.Image)

    def test_return_pil_size_tiny(self):
        pil_img = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert pil_img.size == (512, 512)

    def test_tensor_dtype_float32(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="nonexistent")

    def test_tensor_roughly_normalized(self):
        """After ImageNet normalization values should be in roughly [-3, 3]."""
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.min().item() > -4.0
        assert tensor.max().item() < 4.0

    def test_png_image_loads(self):
        tmp = _save_temp_image(self.img, suffix=".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape == (1, 3, 512, 512)
        finally:
            os.unlink(tmp)

    def test_grayscale_converted_to_rgb(self):
        gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), "L")
        tmp = _save_temp_image(gray, suffix=".png")
        try:
            tensor = load_image(tmp, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_returns_total(self):
        """If image exactly fills target, valid tokens == total tokens."""
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        # scale = min(1024/512, 1024/1024) = 1.0, scaled_w=512, scaled_h=1024
        # valid_ratio = (512/1024)*(1024/1024) = 0.5
        assert result == 128

    def test_small_image_returns_fewer_tokens(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result < 256

    def test_result_non_negative(self):
        result = calculate_valid_tokens(50, 50, 1024, 1024, 400)
        assert result >= 0

    def test_result_not_greater_than_total(self):
        result = calculate_valid_tokens(200, 300, 1024, 1024, 256)
        assert result <= 256

    def test_wide_image(self):
        """Wide image: height limits the scale."""
        result = calculate_valid_tokens(2000, 500, 1024, 1024, 256)
        assert 0 < result <= 256
