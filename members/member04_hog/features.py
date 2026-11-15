"""Shared image preparation for training and inference."""
import numpy as np
from PIL import Image, ImageOps
from skimage.exposure import equalize_adapthist
from skimage.feature import hog


def read_image(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255


def extract(image, config, turns=0, mirror=False, gamma=1.0):
    image = np.rot90(image, turns)
    if mirror:
        image = np.fliplr(image)
    if gamma != 1.0:
        image = image ** gamma
    if config["clahe"]:
        image = equalize_adapthist(image, clip_limit=0.01)
    cell = config["cell"]
    return hog(image, orientations=9, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)
