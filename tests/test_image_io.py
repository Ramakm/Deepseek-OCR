"""Tests for utils/image_io.py — covers Pillow (9→10 MAJOR bump) and torch APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 200, height: int = 100) -> Image.Image:
    """Return a simple solid-colour RGB PIL image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_tmp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save a PIL image to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


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

    def test_pad_flags(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_sizes_are_tuples_of_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["size"], tuple)
            assert len(cfg["size"]) == 2
            assert all(isinstance(v, int) for v in cfg["size"])

    def test_token_counts_are_positive(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 100)
        target = (300, 300)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded(self):
        # Wide image — should have vertical padding
        img = _make_rgb_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_portrait_image_padded(self):
        # Tall image — should have horizontal padding
        img = _make_rgb_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_square_image_no_padding(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (100, 100))
        # No padding expected — original pixels should dominate
        assert result.size == (100, 100)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 50)
        result = resize_and_pad(img, (100, 100), pad_color=(0, 0, 0))
        arr = np.array(result)
        # Corner should be black (pad colour)
        assert arr[0, 0].tolist() == [0, 0, 0]

    def test_pillow_bilinear_not_deprecated(self):
        """Pillow 10 removed ANTIALIAS; BILINEAR must still work."""
        img = _make_rgb_image(200, 200)
        # Should not raise with Pillow ≥ 10
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        img = _make_rgb_image(300, 200)
        path = _save_tmp_image(img, ".jpg")
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert tensor.shape == (1, 3, 512, 512)

    def test_tensor_shape_small(self, tmp_image):
        tensor = load_image(tmp_image, mode="small")
        assert tensor.shape == (1, 3, 640, 640)

    def test_tensor_shape_base(self, tmp_image):
        tensor = load_image(tmp_image, mode="base")
        assert tensor.shape == (1, 3, 1024, 1024)

    def test_tensor_shape_large(self, tmp_image):
        tensor = load_image(tmp_image, mode="large")
        assert tensor.shape == (1, 3, 1280, 1280)

    def test_return_pil(self, tmp_image):
        pil_img = load_image(tmp_image, mode="tiny", return_pil=True)
        assert isinstance(pil_img, Image.Image)

    def test_pil_size_tiny(self, tmp_image):
        pil_img = load_image(tmp_image, mode="tiny", return_pil=True)
        assert pil_img.size == (512, 512)

    def test_tensor_dtype_float(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_unknown_mode_raises(self, tmp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image, mode="ultraHD")

    def test_normalized_range(self, tmp_image):
        """After ImageNet normalization values may fall outside [0,1]."""
        tensor = load_image(tmp_image, mode="tiny")
        # Not all zeros / ones — some normalization occurred
        assert tensor.min().item() < 1.0

    def test_png_image(self):
        img = _make_rgb_image(100, 100)
        path = _save_tmp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape == (1, 3, 512, 512)
        finally:
            os.unlink(path)

    def test_pillow_open_convert_rgb(self, tmp_image):
        """Ensure RGBA images are properly converted (Pillow 10 compat)."""
        img = Image.new("RGBA", (100, 100), (128, 64, 32, 200))
        path = _save_tmp_image(img, ".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_full_tokens(self):
        # Square image matching target → all tokens valid
        result = calculate_valid_tokens(100, 100, 100, 100, 256)
        assert result == 256

    def test_half_area_image(self):
        # Image half width → valid ratio = 0.5 * 1.0 = 0.5
        result = calculate_valid_tokens(50, 100, 100, 100, 256)
        assert result == 128

    def test_result_is_int(self):
        result = calculate_valid_tokens(200, 150, 300, 300, 400)
        assert isinstance(result, int)

    def test_result_leq_total_tokens(self):
        for orig_w in [50, 100, 200, 300]:
            result = calculate_valid_tokens(orig_w, 150, 300, 300, 400)
            assert result <= 400

    def test_result_positive(self):
        result = calculate_valid_tokens(10, 10, 100, 100, 256)
        assert result > 0

    def test_landscape_image(self):
        result = calculate_valid_tokens(200, 100, 200, 200, 400)
        # scale=1 (width fits), scaled_h=100 → ratio=(200/200)*(100/200)=0.5
        assert result == 200

    def test_portrait_image(self):
        result = calculate_valid_tokens(100, 200, 200, 200, 400)
        # scale=1 (height fits), scaled_w=100 → ratio=(100/200)*(200/200)=0.5
        assert result == 200
