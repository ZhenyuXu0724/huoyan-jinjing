"""Inference prototype; no downloads or training at import/inference time."""
import argparse
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

NAMES = {0: "non_catalogue", 1: "long_screw", 2: "thick_screw",
         3: "thin_screw", 4: "hex_nut", 5: "flat_washer"}
DEFAULT_CONFIG = Path(__file__).with_name("inference.json")


def validate_names(names):
    actual = dict(enumerate(names)) if isinstance(names, list) else dict(names)
    actual = {int(k): v for k, v in actual.items()}
    if actual != NAMES:
        raise ValueError("Checkpoint must have project classes 0..5 in the documented order; COCO weights are not a project detector")


def convert_boxes(xyxy, scores, classes, width, height):
    """YOLO xyxy already uses original-image coordinates; do not rescale twice."""
    if not (len(xyxy) == len(scores) == len(classes)):
        raise ValueError("Inconsistent detection arrays")
    objects = []
    for box, score, cls in zip(xyxy, scores, classes):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in [*box, score, cls]):
            raise ValueError("Non-finite or malformed detection")
        if int(cls) != cls or int(cls) not in NAMES or not 0 <= score <= 1:
            raise ValueError("Invalid class or score")
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Degenerate box after clipping")
        objects.append({"class_id": int(cls), "bbox": [x1, y1, x2, y2], "score": float(score)})
    return objects


def count_objects(objects):
    counts = {str(i): 0 for i in NAMES}
    for obj in objects:
        counts[str(obj["class_id"])] += 1
    return counts


@lru_cache(maxsize=1)
def _load_model(path, mtime_ns, size):
    from ultralytics import YOLO
    model = YOLO(path, task="detect")
    validate_names(model.names)
    return model


def predict(image_path: str) -> dict:
    """Real detections, REVIEW_REQUIRED until physical rejection is implemented.

    Config: MEMBER05_CONFIG; weights: MEMBER05_WEIGHTS or config.weights.
    Relative weight paths resolve against config directory.
    expected_counts and ground truth are intentionally not accepted.
    """
    image = Path(image_path)
    if not image.is_file():
        return {"status": "INPUT_ERROR", "objects": [], "message": "Input image does not exist"}
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "INPUT_ERROR", "objects": [], "message": "Cannot decode input image"}
    cfg_path = Path(os.environ.get("MEMBER05_CONFIG", DEFAULT_CONFIG)).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    value = os.environ.get("MEMBER05_WEIGHTS") or cfg.get("weights")
    if not value:
        raise RuntimeError("NOT_IMPLEMENTED: no project-trained checkpoint; set MEMBER05_WEIGHTS after training")
    weights = Path(value).expanduser()
    if not weights.is_absolute():
        weights = cfg_path.parent / weights
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Local checkpoint missing: {weights}; no automatic download")
    if cfg.get("weights_sha256") and hashlib.sha256(weights.read_bytes()).hexdigest() != cfg["weights_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from frozen configuration")
    conf, iou, imgsz = cfg["conf"], cfg["iou"], cfg["imgsz"]
    if not 0 <= conf <= 1 or not 0 <= iou <= 1 or type(imgsz) is not int or imgsz <= 0:
        raise ValueError("Invalid inference settings")
    stat = weights.stat()
    model = _load_model(str(weights), stat.st_mtime_ns, stat.st_size)
    result = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz,
                           device=cfg["device"], max_det=cfg["max_det"], verbose=False)[0]
    boxes = result.boxes
    h, w = frame.shape[:2]
    objects = convert_boxes(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                            boxes.cls.cpu().tolist(), w, h)
    return {"status": "REVIEW_REQUIRED", "objects": objects,
            "score_type": "uncalibrated_model_score", "counts": count_objects(objects),
            "message": "Prototype: physical-size rejection/candidate coverage not validated; not eligible for PASS or official metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    try:
        result = predict(args.image)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
