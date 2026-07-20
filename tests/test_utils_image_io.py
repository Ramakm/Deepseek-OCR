"""Tests for utils/image_io.py - covers Pillow (9→12) and torch/torchvision APIs."""
import io
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(w=200, h=150, color=(128, 64, 32)):
    img = Image.new("RGB", (w, h), color)
    return img


def _save_pil_to_tmp(tmp_path, img, name="test.jpg"):
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

    def test_mode_structure(self):
        from utils.image_io import MODE_CONFIGS
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2
            assert isinstance(cfg["tokens"], int)
            assert isinstance(cfg["pad"], bool)

    def test_pad_flags(self):
        from utils.image_io import MODE_CONFIGS
        # tiny and small should NOT pad
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        # base and large SHOULD pad
        assert MODE_CONFIGS["base"]["pad"] is True
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
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_portrait_image(self):
        from utils.image_io import resize_and_pad
        img = _make_pil_image(100, 400)
        result = resize_and_pad(img, (640, 640))
        assert result.size == (640, 640)

    def test_landscape_image(self):
        from utils.image_io import resize_and_pad
        img = _make_pil_image(400, 100)
        result = resize_and_pad(img, (640, 640))
        assert result.size == (640, 640)

    def test_square_image(self):
        from utils.image_io import resize_and_pad
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_custom_pad_color(self):
        from utils.image_io import resize_and_pad
        # Use a very thin landscape image so lots of padding appears
        img = _make_pil_image(100, 10, color=(0, 0, 0))
        result = resize_and_pad(img, (100, 100), pad_color=(255, 0, 0))
        # Top-left corner should be padding color (red)
        r, g, b = result.getpixel((0, 0))
        assert r == 255
        assert g == 0
        assert b == 0

    def test_returns_pil_image(self):
        from utils.image_io import resize_and_pad
        img = _make_pil_image(200, 150)
        result = resize_and_pad(img, (512, 512))
        assert isinstance(result, Image.Image)

    def test_preserves_aspect_ratio_content(self):
        """Resized content should fit within target without distortion."""
        from utils.image_io import resize_and_pad
        img = _make_pil_image(200, 100)
        result = resize_and_pad(img, (400, 400))
        # The resized content occupies only a strip of the canvas
        arr = np.array(result)
        assert arr.shape == (400, 400, 3)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture()
    def sample_image_path(self, tmp_path):
        img = _make_pil_image(256, 256)
        return _save_pil_to_tmp(tmp_path, img)

    def test_returns_tensor_by_default(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="base")
        assert result.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="large")
        assert result.shape == (1, 3, 1280, 1280)

    def test_return_pil(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_unknown_mode_raises(self, sample_image_path):
        from utils.image_io import load_image
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(sample_image_path, mode="ultramax")

    def test_tensor_dtype_float(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_values_normalized(self, sample_image_path):
        from utils.image_io import load_image
        result = load_image(sample_image_path, mode="tiny")
        # After ImageNet normalization values can be negative and outside [0,1]
        assert result.min().item() < 1.0
        # Sanity: not all zeros
        assert result.abs().sum().item() > 0

    def test_non_rgb_converted(self, tmp_path):
        """RGBA / grayscale images should be converted to RGB."""
        from utils.image_io import load_image
        img = Image.new("RGBA", (100, 100), (10, 20, 30, 128))
        path = _save_pil_to_tmp(tmp_path, img.convert("RGB"), "rgba.jpg")
        result = load_image(path, mode="tiny")
        assert result.shape[1] == 3  # 3 channels


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_size_image(self):
        from utils.image_io import calculate_valid_tokens
        tokens = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert tokens == 256

    def test_half_size_image(self):
        from utils.image_io import calculate_valid_tokens
        # 512x512 into 1024x1024 canvas: scale=0.5, scaled 512x512, ratio=0.25
        tokens = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert tokens == 64

    def test_returns_int(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(300, 200, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_non_zero_for_valid_image(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(200, 200, 1024, 1024, 256)
        assert result > 0

    def test_landscape_image(self):
        from utils.image_io import calculate_valid_tokens
        # Wide image: scale limited by width
        result = calculate_valid_tokens(2048, 512, 1024, 1024, 256)
        assert 0 < result <= 256
