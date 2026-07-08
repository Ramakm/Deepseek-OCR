"""Tests for utils/image_io.py

Covers load_image, resize_and_pad, calculate_valid_tokens, MODE_CONFIGS,
and the Pillow (upgraded 9→12) APIs used therein.
"""
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

# Make sure project root is importable
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

def _make_rgb_image(w: int = 200, h: int = 150) -> Image.Image:
    """Return a solid-colour PIL RGB image."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_tmp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save *img* to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_required_keys_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_each_mode_has_size_tokens_pad(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_values(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_token_counts_are_positive_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert isinstance(cfg["tokens"], int) and cfg["tokens"] > 0

    def test_sizes_are_tuples_of_two_ints(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert len(cfg["size"]) == 2
            assert all(isinstance(v, int) and v > 0 for v in cfg["size"])


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_matches_target_size(self):
        img = _make_rgb_image(300, 200)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_padding_with_non_square_source(self):
        """Wide image padded into square should have rows of pad colour."""
        img = Image.new("RGB", (400, 100), color=(0, 128, 0))
        result = resize_and_pad(img, (400, 400), pad_color=(255, 255, 255))
        assert result.size == (400, 400)
        # Top-left corner should be white (padding)
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 50)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        # corners should be black
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_square_image_no_padding_needed(self):
        img = _make_rgb_image(128, 128)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_aspect_ratio_preserved_in_content(self):
        """The resized content inside the padded canvas should keep ratio."""
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        result = resize_and_pad(img, (400, 400))
        # centre pixel should be red (content), not white (padding)
        cx, cy = 200, 200
        r, g, b = result.getpixel((cx, cy))
        assert r > 200  # strongly red


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.img = _make_rgb_image(200, 150)
        self.tmp_path = _save_tmp_image(self.img)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)

    # --- return type ----------------------------------------------------------
    def test_returns_tensor_by_default(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_batch_1(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3

    def test_tensor_spatial_matches_mode_size(self):
        for mode, cfg in MODE_CONFIGS.items():
            tensor = load_image(self.tmp_path, mode=mode)
            _, _, h, w = tensor.shape
            assert (w, h) == cfg["size"], f"Mode {mode} size mismatch"

    def test_return_pil_true(self):
        pil = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(pil, Image.Image)

    def test_return_pil_size(self):
        pil = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert pil.size == MODE_CONFIGS["tiny"]["size"]

    # --- normalisation --------------------------------------------------------
    def test_tensor_values_float32(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_tensor_values_normalised(self):
        """After ImageNet normalisation values should not be [0,1]."""
        tensor = load_image(self.tmp_path, mode="base")
        # mean is subtracted, so some values will be negative
        assert tensor.min().item() < 0.5

    # --- error path -----------------------------------------------------------
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="xlarge")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path.jpg", mode="tiny")

    # --- mode coverage --------------------------------------------------------
    @pytest.mark.parametrize("mode", list(MODE_CONFIGS.keys()))
    def test_all_modes_produce_correct_shape(self, mode):
        tensor = load_image(self.tmp_path, mode=mode)
        _, _, h, w = tensor.shape
        assert (w, h) == MODE_CONFIGS[mode]["size"]

    # --- PNG support (Pillow) -------------------------------------------------
    def test_png_image_loads(self):
        png_path = _save_tmp_image(self.img, suffix=".png")
        try:
            tensor = load_image(png_path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(png_path)

    # --- grayscale auto-conversion --------------------------------------------
    def test_grayscale_converted_to_rgb(self):
        grey = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8), "L")
        grey_path = _save_tmp_image(grey, suffix=".png")
        try:
            tensor = load_image(grey_path, mode="tiny")
            assert tensor.shape[1] == 3
        finally:
            os.remove(grey_path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_full_image_returns_all_tokens(self):
        """When orig == target, all tokens should be valid."""
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result == 128  # 0.5 * 1.0 * 256

    def test_portrait_image(self):
        """Tall portrait image: scale factor determined by height."""
        result = calculate_valid_tokens(500, 1000, 1000, 1000, 400)
        # scale = min(1000/500, 1000/1000) = 1.0
        # scaled_w=500, scaled_h=1000
        # ratio = (500/1000) * (1000/1000) = 0.5
        assert result == 200

    def test_result_is_integer(self):
        result = calculate_valid_tokens(300, 400, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_result_non_negative(self):
        result = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert result >= 0

    def test_result_lte_total_tokens(self):
        result = calculate_valid_tokens(200, 200, 1024, 1024, 256)
        assert result <= 256
