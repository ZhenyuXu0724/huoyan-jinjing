"""Reproducibility and checkpoint helpers; loading never downloads pretrained weights."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

from .config import Config
from .model import FrozenResNet18


def seed_everything(seed, cpu_threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def choose_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(request)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment(device):
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    return {"python": sys.version, "platform": platform.platform(), "git_head": git_head,
            "git_working_tree": "may contain uncommitted changes; source hashes below identify module files",
            "source_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "packages": {x: importlib.metadata.version(x) for x in
                         ("torch", "torchvision", "numpy", "Pillow", "opencv-python-headless", "matplotlib")},
            "device": str(device), "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_checkpoint(path, device="auto", *, allow_smoke=False):
    # Only load trusted project checkpoints. weights_only restricts pickle globals.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("architecture") != "frozen_resnet18_linear":
        raise ValueError("unsupported checkpoint format")
    config = Config(**payload["config"])
    if payload.get("evidence_status") != config.evidence_status:
        raise ValueError("checkpoint evidence status disagrees with config")
    if config.evidence_status == "SMOKE_ONLY" and not allow_smoke:
        raise ValueError("SMOKE_ONLY checkpoint requires explicit allow_smoke=True / --allow-smoke")
    seed_everything(config.seed, config.cpu_threads)
    selected = choose_device(device)
    model = FrozenResNet18(config.weights, initialize=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(selected).eval()
    return model, config, selected, payload
