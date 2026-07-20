"""Image I/O utilities for DeepSeek-OCR.

Handles image loading, resizing, padding, and preprocessing for different
resolution modes.
"""
import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Mode configurations
# ---------------------------------------------------------------------------

MODE_CONFIGS = {
    "tiny": {
        "size": (224, 224),
        "tokens": 64,
        "pad": False,
    },
    "small": {
        "size": (448, 448),
        "tokens": 256,
        "pad": False,
    },
    "base": {
        "size": (896, 896),
        "tokens": 1024,
        "pad": True,
    },
    "large": {
        "size": (1344, 1344),
        "tokens": 2304,
        "pad": True,
    },
}

# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

def resize_and_pad(
    img: Image.Image,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize image maintaining aspect ratio and pad to target size.

    Args:
        img: PIL image to resize.
        target_size: (width, height) target dimensions.
        pad_color: RGB color to use for padding.

    Returns:
        PIL image of exactly target_size.
    """
    img = img.convert("RGB")
    target_w, target_h = target_size

    # Compute scale to fit within target preserving aspect ratio
    scale = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Create padded canvas
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))

    return canvas


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

def calculate_valid_tokens(
    img: Image.Image,
    target_size: Tuple[int, int],
    max_tokens: int,
) -> int:
    """Calculate the number of valid (non-padding) tokens for an image.

    Computes the fraction of the target canvas covered by the resized image
    (i.e., the area ratio) and multiplies by max_tokens.

    Args:
        img: PIL image.
        target_size: (width, height) target canvas dimensions.
        max_tokens: total number of tokens for the full canvas.

    Returns:
        Integer number of valid tokens.
    """
    target_w, target_h = target_size
    scale = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    area_ratio = (new_w * new_h) / (target_w * target_h)
    return int(math.floor(area_ratio * max_tokens))


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

def load_image(
    path: Union[str, Path],
    mode: str = "base",
    return_pil: bool = False,
) -> Union[torch.Tensor, Image.Image]:
    """Load and preprocess an image for DeepSeek-OCR.

    Args:
        path: path to image file.
        mode: one of 'tiny', 'small', 'base', 'large'.
        return_pil: if True, return PIL image instead of tensor.

    Returns:
        Float32 tensor of shape [1, 3, H, W] normalized to [0, 1],
        or PIL image if return_pil=True.
    """
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(MODE_CONFIGS.keys())}")

    cfg = MODE_CONFIGS[mode]
    target_size = cfg["size"]  # (W, H)

    img = Image.open(path).convert("RGB")

    if cfg["pad"]:
        img = resize_and_pad(img, target_size)
    else:
        img = img.resize(target_size, Image.LANCZOS)

    if return_pil:
        return img

    # Convert to tensor [1, 3, H, W] float32 in [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor
"""Image I/O utilities for DeepSeek-OCR.

Handles image loading, preprocessing, and mode-specific resizing/padding
according to the paper's multi-resolution support.
"""
from PIL import Image
import numpy as np
import torch
from typing import Union, Tuple, Optional
import torchvision.transforms as T


# Resolution modes from paper (Section 3.2.2)
MODE_CONFIGS = {
    "tiny": {"size": (512, 512), "tokens": 64, "pad": False},
    "small": {"size": (640, 640), "tokens": 100, "pad": False},
    "base": {"size": (1024, 1024), "tokens": 256, "pad": True},
    "large": {"size": (1280, 1280), "tokens": 400, "pad": True},
}


def load_image(
    path: str,
    mode: str = "base",
    return_pil: bool = False
) -> Union[torch.Tensor, Image.Image]:
    """Load and preprocess image for DeepSeek-OCR.
    
    Args:
        path: path to image file
        mode: resolution mode (tiny/small/base/large)
        return_pil: return PIL image instead of tensor
        
    Returns:
        preprocessed image as tensor [1, 3, H, W] or PIL image
    """
    img = Image.open(path).convert("RGB")
    
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(MODE_CONFIGS.keys())}")
    
    config = MODE_CONFIGS[mode]
    target_size = config["size"]
    use_pad = config["pad"]
    
    if use_pad:
        # Pad to preserve aspect ratio (base/large modes)
        img = resize_and_pad(img, target_size)
    else:
        # Direct resize (tiny/small modes)
        img = img.resize(target_size, Image.BILINEAR)
    
    if return_pil:
        return img
    
    # Convert to tensor
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    
    # Normalize (ImageNet stats)
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    tensor = normalize(tensor)
    
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def resize_and_pad(
    img: Image.Image,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Resize image and pad to target size, preserving aspect ratio.
    
    Paper reference: Section 3.2.2 - Base and Large modes use padding
    to preserve original image aspect ratio.
    
    Args:
        img: input PIL image
        target_size: (width, height) target size
        pad_color: RGB color for padding
        
    Returns:
        padded image
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    
    # Calculate scaling factor to fit within target
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Create padded canvas
    padded = Image.new("RGB", target_size, pad_color)
    
    # Paste resized image in center
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(img_resized, (paste_x, paste_y))
    
    return padded


def calculate_valid_tokens(
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    total_tokens: int
) -> int:
    """Calculate number of valid (non-padding) tokens.
    
    Paper formula: valid_tokens = total_tokens * (orig_w/target_w) * (orig_h/target_h)
    
    Args:
        orig_w, orig_h: original image dimensions
        target_w, target_h: target dimensions
        total_tokens: total token count for this mode
        
    Returns:
        number of valid tokens
    """
    scale = min(target_w / orig_w, target_h / orig_h)
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    
    valid_ratio = (scaled_w / target_w) * (scaled_h / target_h)
    valid_tokens = int(total_tokens * valid_ratio)
    
    return valid_tokens


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image.
    
    Args:
        tensor: [3, H, W] or [1, 3, H, W] tensor
        path: output path
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize if needed
    if tensor.min() < 0:
        denorm = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        tensor = denorm(tensor)
    
    # Clamp and convert
    tensor = torch.clamp(tensor, 0, 1)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
