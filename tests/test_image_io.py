"""Tests for utils/image_io.py - PIL/Pillow usage (upgraded 9.0.0 -> 12.3.0)."""
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is on path
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

def _make_pil_image(width=100, height=80, mode="RGB"):
    """Return a synthetic PIL image without touching the filesystem."""
    from PIL import Image
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode)


def _save_tmp_image(tmp_path, width=200, height=150, name="test.jpg"):
    """Save a synthetic JPEG to tmp_path and return its path string."""
    img = _make_pil_image(width, height)
    p = tmp_path / name
    img.save(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"mode {mode} missing 'size'"
            assert "tokens" in cfg, f"mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"mode {mode} missing 'pad'"

    def test_size_is_two_tuple(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2

    def test_pad_flag_for_base_and_large(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_pad_flag_for_tiny_and_small(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_ordering(self):
        # Larger modes should have more tokens
        assert MODE_CONFIGS["tiny"]["tokens"] < MODE_CONFIGS["small"]["tokens"]
        assert MODE_CONFIGS["small"]["tokens"] < MODE_CONFIGS["base"]["tokens"]
        assert MODE_CONFIGS["base"]["tokens"] < MODE_CONFIGS["large"]["tokens"]


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        from PIL import Image
        img = _make_pil_image(300, 200)
        target = (512, 512)
        result = resize_and_pad(img, target)
        assert result.size == target

    def test_mode_preserved(self):
        from PIL import Image
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_square_image_fills_mostly(self):
        from PIL import Image
        img = _make_pil_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Center pixel should not be the pad colour (255,255,255)
        # because a square image scaled to fill a square canvas
        center_pixel = arr[100, 100]
        assert not np.all(center_pixel == 255)

    def test_custom_pad_color(self):
        from PIL import Image
        img = _make_pil_image(10, 10)
        # Very wide target so lots of vertical padding
        result = resize_and_pad(img, (100, 10), pad_color=(0, 0, 0))
        arr = np.array(result)
        # First column should be black padding
        assert np.all(arr[0, 0] == [0, 0, 0])

    def test_wide_image(self):
        from PIL import Image
        img = _make_pil_image(400, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_tall_image(self):
        from PIL import Image
        img = _make_pil_image(100, 400)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)


# ---------------------------------------------------------------------------
# load_image – tensor output
# ---------------------------------------------------------------------------

class TestLoadImageTensor:
    def test_returns_tensor(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_shape_tiny(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        h, w = MODE_CONFIGS["tiny"]["size"]
        tensor = load_image(path, mode="tiny")
        assert tensor.shape == (1, 3, h, w)

    def test_shape_small(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        h, w = MODE_CONFIGS["small"]["size"]
        tensor = load_image(path, mode="small")
        assert tensor.shape == (1, 3, h, w)

    def test_shape_base(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        h, w = MODE_CONFIGS["base"]["size"]
        tensor = load_image(path, mode="base")
        assert tensor.shape == (1, 3, h, w)

    def test_shape_large(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        h, w = MODE_CONFIGS["large"]["size"]
        tensor = load_image(path, mode="large")
        assert tensor.shape == (1, 3, h, w)

    def test_dtype_float(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_normalized_range(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        tensor = load_image(path, mode="tiny")
        # After ImageNet normalisation values can be slightly outside [0,1]
        # but should be in a reasonable range
        assert tensor.min() > -3.0
        assert tensor.max() < 3.0

    def test_unknown_mode_raises(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(path, mode="nonexistent")


# ---------------------------------------------------------------------------
# load_image – PIL output
# ---------------------------------------------------------------------------

class TestLoadImagePIL:
    def test_return_pil_flag(self, tmp_path):
        from PIL import Image
        path = _save_tmp_image(tmp_path)
        img = load_image(path, mode="tiny", return_pil=True)
        assert isinstance(img, Image.Image)

    def test_pil_size_matches_mode(self, tmp_path):
        path = _save_tmp_image(tmp_path)
        img = load_image(path, mode="tiny", return_pil=True)
        assert img.size == MODE_CONFIGS["tiny"]["size"]


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size(self):
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_wide_image(self):
        # Wide image in square canvas → tall padding → fewer valid tokens
        result = calculate_valid_tokens(1024, 512, 1024, 1024, 256)
        assert result < 256

    def test_positive_result(self):
        result = calculate_valid_tokens(200, 150, 512, 512, 400)
        assert result > 0

    def test_result_leq_total(self):
        result = calculate_valid_tokens(100, 100, 512, 512, 100)
        assert result <= 100

    def test_larger_original_gives_more_tokens(self):
        small = calculate_valid_tokens(100, 100, 512, 512, 400)
        large = calculate_valid_tokens(400, 400, 512, 512, 400)
        assert large >= small
