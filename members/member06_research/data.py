"""No implicit splitting: the upstream scene partition is authoritative."""
import csv
import hashlib
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from .image_io import read_image


class Preprocess:
    def __init__(self, config):
        self.config = config

    def __call__(self, source):
        image = read_image(source)
        size = self.config.image_size
        if self.config.resize_mode == "letterbox":
            scale = min(size / image.width, size / image.height)
            width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            image = Image.new("RGB", (size, size), tuple(self.config.pad_rgb))
            image.paste(resized, ((size - width) // 2, (size - height) // 2))
        else:
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(image), self.config.mean, self.config.std)
        if tensor.shape != (3, size, size) or not torch.isfinite(tensor).all():
            raise ValueError("invalid preprocessed tensor")
        return tensor


def image_digest(path):
    image = read_image(path)
    digest = hashlib.sha256()
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


class ManifestDataset(Dataset):
    def __init__(self, manifest, config, split):
        if split not in {"train", "val"}:
            raise ValueError("training/selection only accepts train or val, never test")
        self.manifest = Path(manifest).resolve()
        self.transform = Preprocess(config)
        self.rows = []
        with self.manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != {"image_path", "label", "scene_id", "split", "data_kind"}:
                raise ValueError("manifest columns must be image_path,label,scene_id,split,data_kind")
            for row in reader:
                if row["split"] != split or row["data_kind"] != config.data_kind:
                    raise ValueError("manifest role/data_kind disagrees with experiment")
                if row["label"] not in config.class_to_idx or not row["scene_id"].strip():
                    raise ValueError("invalid label or empty scene_id")
                path = (self.manifest.parent / row["image_path"]).resolve()
                self.rows.append({**row, "path": path, "target": config.class_to_idx[row["label"]],
                                  "sha256_rgb": image_digest(path)})
        if not self.rows or {row["label"] for row in self.rows} != {"PASS", "FAIL"}:
            raise ValueError("each partition must contain both labels and at least two images")
        if len({row["path"] for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate path within partition")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(row["path"]), row["target"]


def check_disjoint(train, val):
    for key in ("scene_id", "path", "sha256_rgb"):
        if {row[key] for row in train.rows} & {row[key] for row in val.rows}:
            raise ValueError(f"train/val overlap in {key}")
