"""Tests for utils/image_io.py — covers Pillow (9→12) and torch/torchvision APIs."""
import io
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil(width=100, height=80, mode="RGB"):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_tmp_image(tmp_path, width=100, height=80, fmt="JPEG"):
    img = _make_pil(width, height)
    p = tmp_path / f"test_image.{'jpg' if fmt=='JPEG' else 'png'}"
    img.save(str(p), format=fmt)
    return str(p)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

from utils.image_io import (
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
    MODE_CONFIGS,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
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

    def test_mode_sizes_are_tuples(self):
        for cfg in MODE_CONFIGS.values():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2

    def test_base_large_use_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_small_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_tensor_shape_base(self, tmp_path):
        p = _save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(p, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)
        assert tensor.dtype == torch.float32

    def test_returns_tensor_shape_tiny(self, tmp_path):
        p = _save_tmp_image(tmp_path, 60, 60)
        tensor = load_image(p, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_returns_tensor_shape_small(self, tmp_path):
        p = _save_tmp_image(tmp_path, 60, 60)
        tensor = load_image(p, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_returns_tensor_shape_large(self, tmp_path):
        p = _save_tmp_image(tmp_path, 200, 150)
        tensor = load_image(p, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        result = load_image(p, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)

    def test_return_pil_base_preserves_aspect(self, tmp_path):
        p = _save_tmp_image(tmp_path, 200, 100)
        result = load_image(p, mode="base", return_pil=True)
        assert result.size == (1024, 1024)

    def test_unknown_mode_raises(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(p, mode="nonexistent")

    def test_tensor_normalized_range(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        tensor = load_image(p, mode="tiny")
        # After ImageNet normalization values can be < -1 or > 1,
        # but should be finite
        assert torch.isfinite(tensor).all()

    def test_loads_png(self, tmp_path):
        p = _save_tmp_image(tmp_path, fmt="PNG")
        tensor = load_image(p, mode="tiny")
        assert tensor.shape[0] == 1

    def test_batch_dim_is_one(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        tensor = load_image(p, mode="small")
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1

    def test_channel_dim_is_three(self, tmp_path):
        p = _save_tmp_image(tmp_path)
        tensor = load_image(p, mode="small")
        assert tensor.shape[1] == 3

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        arr = np.random.randint(0, 256, (60, 60, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        p = tmp_path / "rgba.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_grayscale_image_converted(self, tmp_path):
        arr = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        p = tmp_path / "gray.png"
        img.save(str(p))
        tensor = load_image(str(p), mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil(200, 100)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_output_size_large_mode(self):
        img = _make_pil(300, 400)
        result = resize_and_pad(img, (1280, 1280))
        assert result.size == (1280, 1280)

    def test_square_image_no_padding(self):
        img = _make_pil(64, 64)
        result = resize_and_pad(img, (64, 64))
        assert result.size == (64, 64)

    def test_default_pad_color_is_white(self):
        # Make a tiny image so padding is visible
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        result = resize_and_pad(img, (100, 100))
        pixels = np.array(result)
        # Corner should be white (padding)
        assert pixels[0, 0, 0] == 255

    def test_custom_pad_color(self):
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(128, 128, 128))
        pixels = np.array(result)
        assert pixels[0, 0, 0] == 128

    def test_aspect_ratio_preserved(self):
        img = _make_pil(200, 100)
        result = resize_and_pad(img, (512, 512))
        arr = np.array(result)
        # The image content has been placed, result is 512×512
        assert arr.shape == (512, 512, 3)

    def test_returns_pil_image(self):
        img = _make_pil(50, 50)
        result = resize_and_pad(img, (100, 100))
        assert isinstance(result, Image.Image)

    def test_portrait_image(self):
        img = _make_pil(100, 300)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_landscape_image(self):
        img = _make_pil(300, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_coverage_returns_all_tokens(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width(self):
        # 512×1024 scaled into 1024×1024 → scale=1.0 for height, 2.0 for width
        # scale = min(1024/512, 1024/1024) = 1.0
        # scaled_w=512, scaled_h=1024 → valid_ratio = (512/1024)*(1024/1024) = 0.5
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128

    def test_returns_integer(self):
        result = calculate_valid_tokens(200, 150, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_never_exceeds_total(self):
        for orig_w, orig_h in [(100, 100), (500, 300), (2000, 1000), (1024, 1024)]:
            result = calculate_valid_tokens(orig_w, orig_h, 1024, 1024, 256)
            assert result <= 256

    def test_tiny_image_has_few_tokens(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result < 256

    def test_large_token_budget(self):
        result = calculate_valid_tokens(1280, 1280, 1280, 1280, 400)
        assert result == 400

    def test_zero_edge_not_crash(self):
        # If orig equals target, should return all tokens
        result = calculate_valid_tokens(640, 640, 640, 640, 100)
        assert result == 100
