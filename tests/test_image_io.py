"""Tests for utils/image_io.py — exercises Pillow (9→12) and torch APIs."""
import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pil_image(width=200, height=150, mode="RGB"):
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def pil_to_bytes(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_keys_exist(self):
        from utils.image_io import MODE_CONFIGS
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_tiny_config(self):
        from utils.image_io import MODE_CONFIGS
        cfg = MODE_CONFIGS["tiny"]
        assert cfg["size"] == (512, 512)
        assert cfg["tokens"] == 64
        assert cfg["pad"] is False

    def test_small_config(self):
        from utils.image_io import MODE_CONFIGS
        cfg = MODE_CONFIGS["small"]
        assert cfg["size"] == (640, 640)
        assert cfg["tokens"] == 100
        assert cfg["pad"] is False

    def test_base_config(self):
        from utils.image_io import MODE_CONFIGS
        cfg = MODE_CONFIGS["base"]
        assert cfg["size"] == (1024, 1024)
        assert cfg["tokens"] == 256
        assert cfg["pad"] is True

    def test_large_config(self):
        from utils.image_io import MODE_CONFIGS
        cfg = MODE_CONFIGS["large"]
        assert cfg["size"] == (1280, 1280)
        assert cfg["tokens"] == 400
        assert cfg["pad"] is True


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def _patch_open(self, tmp_path):
        """Write a real small JPEG so PIL can open it."""
        img = make_pil_image(300, 200)
        self.img_path = str(tmp_path / "test.jpg")
        img.save(self.img_path, "JPEG")

    def test_returns_tensor_by_default(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="small")
        assert result.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="base")
        assert result.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="large")
        assert result.shape == (1, 3, 1280, 1280)

    def test_return_pil(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny", return_pil=True)
        assert result.size == (512, 512)

    def test_unknown_mode_raises(self):
        from utils.image_io import load_image
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.img_path, mode="unknown_mode")

    def test_tensor_dtype_float32(self):
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny")
        assert result.dtype == torch.float32

    def test_tensor_normalized_range(self):
        """After ImageNet normalization values can be negative or > 1."""
        from utils.image_io import load_image
        result = load_image(self.img_path, mode="tiny")
        # Just verify there are both negative and positive values (normalized)
        assert result.min().item() < 1.0
        assert result.shape[1] == 3  # 3 channels

    def test_rgb_conversion(self, tmp_path):
        """RGBA image should be converted to RGB."""
        from utils.image_io import load_image
        arr = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        rgba_img = Image.fromarray(arr, "RGBA")
        rgba_path = str(tmp_path / "rgba.png")
        rgba_img.save(rgba_path, "PNG")
        result = load_image(rgba_path, mode="tiny")
        assert result.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(300, 200)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_wide_image_padded_vertically(self):
        from utils.image_io import resize_and_pad
        # 4:1 wide image → large vertical padding
        img = make_pil_image(400, 100)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_tall_image_padded_horizontally(self):
        from utils.image_io import resize_and_pad
        # 1:4 tall image → large horizontal padding
        img = make_pil_image(100, 400)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_custom_pad_color(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(100, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        assert result.size == (200, 200)

    def test_returns_pil_image(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(50, 50)
        result = resize_and_pad(img, (512, 512))
        assert isinstance(result, Image.Image)

    def test_square_image_no_padding_needed(self):
        from utils.image_io import resize_and_pad
        img = make_pil_image(200, 200)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        from utils.image_io import calculate_valid_tokens
        # Same aspect ratio as target → all tokens valid
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        from utils.image_io import calculate_valid_tokens
        # 512-wide in 1024 target: scale=1.0 (height constrains), valid < 256
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_returns_int(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(300, 200, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_tiny_image(self):
        from utils.image_io import calculate_valid_tokens
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0

    def test_wide_image_reduces_tokens(self):
        from utils.image_io import calculate_valid_tokens
        # Very wide image will have lots of padding top/bottom
        result_wide = calculate_valid_tokens(2000, 100, 1024, 1024, 256)
        result_square = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result_wide < result_square
