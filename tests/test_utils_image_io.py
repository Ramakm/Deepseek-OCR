"""Tests for utils/image_io.py"""
import io
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(w=100, h=80, color=(128, 64, 32)):
    img = Image.new("RGB", (w, h), color=color)
    return img


def save_pil_to_tmp(tmp_path, img, name="test.jpg"):
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        from utils.image_io import MODE_CONFIGS
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_keys(self):
        from utils.image_io import MODE_CONFIGS
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg

    def test_tiny_no_pad(self):
        from utils.image_io import MODE_CONFIGS
        assert MODE_CONFIGS["tiny"]["pad"] is False

    def test_small_no_pad(self):
        from utils.image_io import MODE_CONFIGS
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_base_pad(self):
        from utils.image_io import MODE_CONFIGS
        assert MODE_CONFIGS["base"]["pad"] is True

    def test_large_pad(self):
        from utils.image_io import MODE_CONFIGS
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_positive(self):
        from utils.image_io import MODE_CONFIGS
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_mode_rgb(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_wide(self):
        from utils.image_io import resize_and_pad
        # Wide image: 400x100 → fit inside 256x256
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (256, 256))
        arr = np.array(result)
        # Corners should be white (padding)
        assert tuple(arr[0, 0]) == (255, 255, 255)

    def test_custom_pad_color(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(50, 200)
        result = resize_and_pad(img, (256, 256), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Top-left corner should be black
        assert tuple(arr[0, 0]) == (0, 0, 0)

    def test_square_image_no_padding_needed(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(512, 512)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_shape_tiny(self, tmp_path):
        from utils.image_io import load_image, MODE_CONFIGS
        img = make_pil_image(200, 200)
        p = save_pil_to_tmp(tmp_path, img)
        t = load_image(p, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_returns_tensor_shape_base(self, tmp_path):
        from utils.image_io import load_image, MODE_CONFIGS
        img = make_pil_image(200, 200)
        p = save_pil_to_tmp(tmp_path, img)
        t = load_image(p, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_returns_tensor_shape_large(self, tmp_path):
        from utils.image_io import load_image, MODE_CONFIGS
        img = make_pil_image(300, 150)
        p = save_pil_to_tmp(tmp_path, img)
        t = load_image(p, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert t.shape == (1, 3, h, w)

    def test_returns_pil_when_requested(self, tmp_path):
        from utils.image_io import load_image
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_tensor_dtype_float(self, tmp_path):
        from utils.image_io import load_image
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        t = load_image(p, mode="tiny")
        assert t.dtype == torch.float32

    def test_unknown_mode_raises(self, tmp_path):
        from utils.image_io import load_image
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="nonexistent")

    def test_all_valid_modes(self, tmp_path):
        from utils.image_io import load_image, MODE_CONFIGS
        img = make_pil_image(200, 200)
        p = save_pil_to_tmp(tmp_path, img)
        for mode in MODE_CONFIGS:
            t = load_image(p, mode=mode)
            assert isinstance(t, torch.Tensor)

    def test_tensor_normalized_range(self, tmp_path):
        """After ImageNet normalization the tensor can go below 0."""
        from utils.image_io import load_image
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img)
        t = load_image(p, mode="tiny")
        # Values should not be in [0,1] range after normalization
        # but should be finite
        assert torch.isfinite(t).all()

    def test_png_image(self, tmp_path):
        from utils.image_io import load_image
        img = make_pil_image(100, 100)
        p = save_pil_to_tmp(tmp_path, img, name="test.png")
        t = load_image(p, mode="small")
        assert t.shape[0] == 1 and t.shape[1] == 3


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        from utils.image_io import calculate_valid_tokens
        # Square image that exactly fits target → full tokens
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        from utils.image_io import calculate_valid_tokens
        # Image half the width: scale=1.0 in height, 0.5 in width-limited
        # scale = min(1024/512, 1024/1024) = 1.0  → scaled: 512x1024
        # valid_ratio = (512/1024) * (1024/1024) = 0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_returns_int(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_non_negative(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result >= 0

    def test_small_image(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(64, 64, 256, 256, 100)
        assert 0 <= result <= 100
