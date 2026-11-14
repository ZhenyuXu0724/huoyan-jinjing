"""Shared v0.1 entry point. No model download/load happens during import."""
from functools import lru_cache
import os


@lru_cache(maxsize=2)
def _adapter(checkpoint, device):
    from .inference import create_adapter
    return create_adapter(checkpoint, device)


def predict(image_path: str, expected_counts=None) -> dict:
    return predict_batch([image_path], expected_counts)[0]


def predict_batch(image_paths, expected_counts=None) -> list[dict]:
    from .adapter import result
    images = list(image_paths)
    checkpoint = os.environ.get("MEMBER06_CHECKPOINT")
    if not checkpoint:
        return [result("MODEL_NOT_CONFIGURED", error="set MEMBER06_CHECKPOINT to a trusted real-data checkpoint") for _ in images]
    return _adapter(checkpoint, os.environ.get("MEMBER06_DEVICE", "auto")).predict_batch(images, expected_counts)
