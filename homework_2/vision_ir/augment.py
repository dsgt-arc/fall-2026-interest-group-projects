"""Edits used to build copy-detection queries: "find the original of this altered photo".

Each edit is a deterministic PIL function, so every student and every model sees
exactly the same query images. Some edits match augmentations used while
pretraining certain models (crops, flips, color changes); others do not (90°
rotation). Comparing them reveals which invariances each objective learned.
"""

import io
from collections.abc import Callable

from PIL import Image, ImageFilter, ImageOps


def _crop(image: Image.Image) -> Image.Image:
    """Keep a centered window with 40% of the original area."""
    width, height = image.size
    scale = 0.4**0.5
    new_w, new_h = round(width * scale), round(height * scale)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def _hue_shift(image: Image.Image) -> Image.Image:
    """Rotate every hue by half the color wheel while keeping brightness."""
    hue, saturation, value = image.convert("HSV").split()
    hue = hue.point(lambda h: (h + 128) % 256)
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


def _jpeg(image: Image.Image, quality: int = 8) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _pixelate(image: Image.Image, size: int = 48) -> Image.Image:
    small = image.resize((size, size), Image.Resampling.BILINEAR)
    return small.resize(image.size, Image.Resampling.NEAREST)


AUGMENTATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "flip": ImageOps.mirror,
    "crop 40%": _crop,
    "grayscale": lambda image: ImageOps.grayscale(image).convert("RGB"),
    "hue shift": _hue_shift,
    "blur": lambda image: image.filter(ImageFilter.GaussianBlur(radius=4)),
    "JPEG q=8": _jpeg,
    "pixelate": _pixelate,
    "rotate 90°": lambda image: image.rotate(90, expand=True),
}


def augment(image: Image.Image, name: str) -> Image.Image:
    """Apply one named edit to an RGB image and return a new RGB image."""
    if name not in AUGMENTATIONS:
        raise KeyError(f"Unknown augmentation {name!r}; choose from {sorted(AUGMENTATIONS)}.")
    return AUGMENTATIONS[name](image.convert("RGB")).convert("RGB")
