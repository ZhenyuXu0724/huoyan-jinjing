"""Shared RGB raster input, independent of torch and the model backend."""
from pathlib import Path

from PIL import Image


def read_image(source):
    # No EXIF rotation: xyxy refers to the stored raster coordinate system.
    if isinstance(source, Image.Image):
        image = source.convert("RGB").copy()
    elif isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            image = opened.convert("RGB").copy()
    else:
        raise ValueError("image must be a path or PIL.Image")
    if min(image.size) < 1:
        raise ValueError("empty image")
    return image
