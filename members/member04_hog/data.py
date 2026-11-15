"""Explicit manifests and checks for leakage across splits."""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    scene: str
    label: str
    digest: str


def load_manifest(path):
    path = Path(path)
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "scene_id", "label"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected image_path, scene_id, label")
        for number, row in enumerate(reader, 2):
            raw, scene, label = [(row.get(key) or "").strip()
                                 for key in ("image_path", "scene_id", "label")]
            relative = Path(raw)
            if not raw or relative.drive or relative.root or not scene or label not in ("PASS", "FAIL"):
                raise ValueError(f"{path}:{number}: invalid relative image path, scene or label")
            image_path = (path.parent / relative).resolve(strict=True)
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("L")
                digest = hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).hexdigest()
            samples.append(Sample(relative.name, image_path, scene, label, digest))
    if not samples or len({s.image_id for s in samples}) != len(samples):
        raise ValueError(f"{path}: need nonempty manifest with unique image filenames")
    if len({s.path for s in samples}) != len(samples):
        raise ValueError(f"{path}: repeated image path")
    return samples


def identity(samples):
    return {"scenes": sorted({s.scene for s in samples}),
            "images": sorted({s.digest for s in samples})}


def reject_overlap(left, right):
    for key in ("scenes", "images"):
        if set(left[key]).intersection(right[key]):
            raise ValueError(f"Dataset leakage: overlapping {key}")
