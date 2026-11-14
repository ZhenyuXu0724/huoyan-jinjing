"""Explicit, serializable experiment settings. Paths are relative to the shell cwd."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


def normalize_counts(value):
    if isinstance(value, dict):
        if set(value) != {str(i) for i in range(1, 6)}:
            raise ValueError("counts require exactly string keys 1..5")
        value = [value[str(i)] for i in range(1, 6)]
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError("counts require five entries")
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError("counts must be nonnegative integers, not bool")
    return list(value)


@dataclass
class Config:
    class_to_idx: dict = field(default_factory=lambda: {"PASS": 0, "FAIL": 1})
    fail_label: str = "FAIL"
    image_size: int = 224
    resize_mode: str = "letterbox"
    mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    pad_rgb: list = field(default_factory=lambda: [0, 0, 0])
    weights: str = "IMAGENET1K_V1"
    data_kind: str = "REAL"
    supported_counts: list = field(default_factory=lambda: [2, 2, 2, 2, 2])
    counts_rule_status: str = "PROVISIONAL_PENDING_PILOT"
    train_manifest: str = "data/member06/train.csv"
    val_manifest: str = "data/member06/val.csv"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "auto"
    cpu_threads: int = 2

    def __post_init__(self):
        if set(self.class_to_idx) != {"PASS", "FAIL"} or any(
            type(x) is not int for x in self.class_to_idx.values()
        ) or set(self.class_to_idx.values()) != {0, 1}:
            raise ValueError("class_to_idx must map PASS/FAIL explicitly to 0 and 1")
        if self.fail_label != "FAIL":
            raise ValueError("FAIL is the positive class for reported P/R/F1")
        for name, minimum in (("image_size", 32), ("epochs", 1), ("batch_size", 1),
                              ("num_workers", 0), ("cpu_threads", 1), ("seed", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        if self.resize_mode not in {"letterbox", "stretch"}:
            raise ValueError("resize_mode must be letterbox or stretch; no center crop")
        for name in ("mean", "std"):
            values = getattr(self, name)
            if len(values) != 3 or any(not math.isfinite(x) for x in values):
                raise ValueError(f"invalid {name}")
        if any(x <= 0 for x in self.std):
            raise ValueError("std must be positive")
        if len(self.pad_rgb) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in self.pad_rgb):
            raise ValueError("pad_rgb must contain three bytes")
        if self.weights not in {"IMAGENET1K_V1", "RANDOM_SMOKE_ONLY"}:
            raise ValueError("invalid weights")
        if self.data_kind not in {"REAL", "SMOKE_ONLY"}:
            raise ValueError("invalid data_kind")
        if self.weights == "RANDOM_SMOKE_ONLY" and self.data_kind != "SMOKE_ONLY":
            raise ValueError("random weights require SMOKE_ONLY data_kind")
        self.supported_counts = normalize_counts(self.supported_counts)
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto/cpu/cuda")
        for name in ("learning_rate", "weight_decay"):
            x = getattr(self, name)
            if not math.isfinite(x) or x < 0 or (name == "learning_rate" and x == 0):
                raise ValueError(f"invalid {name}")

    @property
    def evidence_status(self):
        return "SMOKE_ONLY" if self.data_kind == "SMOKE_ONLY" else "TRAIN_VAL_ONLY"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
