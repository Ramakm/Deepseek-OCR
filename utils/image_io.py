"""Image I/O utilities for DeepSeek-OCR.

Handles loading, resizing, padding, and converting images for model input.
Supports multiple resolution modes for the encoder.
"""
import os
from typing import Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image


# Resolution mode configurations
# Each mode defines: target size, token count, and whether to pad
MODE_CONFIGS = {
    "tiny":  {"size": (224, 224),  "tokens": 256,  "pad": False},
    "small": {"size": (448, 448),  "tokens": 1024, "pad": False},
    "base":  {"size": (672, 672),  "tokens": 2304, "pad": True},
    "large": {"size": (896, 896),  "tokens": 4096, "pad": True},
}


def _get_resampling_filter():
    """Return the best available resampling filter."""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def resize_and_pad(
    image: Image.Image,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize image to fit within target_size, padding if necessary.

    Args:
        image: PIL Image to resize
        target_size: (width, height) target dimensions
        pad_color: RGB color for padding

    Returns:
        PIL Image of exactly target_size
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    # Compute scale to fit within target while preserving aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resample = _get_resampling_filter()
    resized = image.convert("RGB").resize((new_w, new_h), resample)

    # Create padded canvas and paste resized image
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def load_image(
    path: str,
    mode: str = "base",
    return_pil: bool = False,
) -> Union[torch.Tensor, Image.Image]:
    """Load and preprocess an image for model input.

    Args:
        path: path to image file
        mode: resolution mode (tiny/small/base/large)
        return_pil: if True, return PIL Image instead of tensor

    Returns:
        torch.Tensor of shape [1, 3, H, W] with values in [0, 1],
        or PIL Image if return_pil=True
    """
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(MODE_CONFIGS.keys())}")

    cfg = MODE_CONFIGS[mode]
    target_size = cfg["size"]  # (width, height)
    should_pad = cfg["pad"]

    # Load image and convert to RGB
    img = Image.open(path).convert("RGB")

    if should_pad:
        img = resize_and_pad(img, target_size)
    else:
        resample = _get_resampling_filter()
        img = img.resize(target_size, resample)

    if return_pil:
        return img

    # Convert to tensor [1, 3, H, W] in [0, 1]
    # Use numpy to avoid torchvision.transforms.ToTensor() removal in newer versions
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

    return tensor.unsqueeze(0)  # [1, 3, H, W]


def calculate_valid_tokens(
    image_size: Tuple[int, int],
    mode: str = "base",
) -> int:
    """Calculate the number of valid (non-padding) tokens for an image.

    Args:
        image_size: (width, height) of the original image
        mode: resolution mode

    Returns:
        Number of valid tokens (int)
    """
    cfg = MODE_CONFIGS[mode]
    target_w, target_h = cfg["size"]
    total_tokens = cfg["tokens"]

    orig_w, orig_h = image_size

    # Compute the fraction of the canvas actually covered by the image
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    valid_fraction = (new_w * new_h) / (target_w * target_h)
    valid_tokens = int(total_tokens * valid_fraction)

    return max(1, valid_tokens)

    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    # Token grid side (assumes square token grid from square canvas)
    grid_side = int(math.sqrt(total_tokens))
    token_w = round(new_w / target_w * grid_side)
    token_h = round(new_h / target_h * grid_side)

    valid = token_w * token_h
    # Clamp to [1, total_tokens]
    return max(1, min(valid, total_tokens))
"""Image I/O utilities for DeepSeek-OCR.

Handles image loading, preprocessing, and mode-specific resizing/padding
Supports Pillow backend (OpenCV used only for reading, with BGR→RGB conversion).
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
    canvas = Image.new("RGB", (target_w, target_h), color=tuple(pad_color) if not isinstance(pad_color, tuple) else pad_color)
    
    # Normalize (ImageNet stats)
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    tensor = normalize(tensor)
    
    return tensor.unsqueeze(0)  # [1, 3, H, W]
    resample = getattr(Image, 'LANCZOS', getattr(Image.Resampling, 'LANCZOS', Image.BICUBIC))
    resized = img.resize((new_w, new_h), resample=resample)

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
