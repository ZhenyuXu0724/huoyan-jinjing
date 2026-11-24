"""Single-image baseline1 entry point for the v4 frozen default model."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
ACTIVE_CONFIG = MODULE_DIR / "configs/active_model.json"
MODEL_PATH = ROOT / "weights/member04_hog/final_v4/representatives/default.joblib"


@lru_cache(maxsize=4)
def _load_cached(path, modified, size):
    import joblib

    return joblib.load(path)


def load_model(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


def load_active_model():
    """Read configs/active_model.json, verify the frozen file and bundle metadata."""
    from .final_v4.common import PREFIX, sha256, validate_bundle

    record = json.loads(ACTIVE_CONFIG.read_text(encoding="utf-8"))
    if record.get("name") != PREFIX + "_Rebuild":
        raise ValueError("active model record has an unexpected name")
    path = (ROOT / record["model_path"]).resolve()
    weights_root = (ROOT / "weights/member04_hog/final_v4/representatives").resolve()
    if weights_root not in path.parents:
        raise ValueError("active model path is outside the frozen representatives directory")
    if sha256(path) != record["model_sha256"]:
        raise ValueError("active model checksum mismatch")
    bundle = load_model(path)
    ok, reason = validate_bundle(bundle, expected_version=record["version"])
    if not ok:
        raise ValueError(reason)
    if float(bundle["config"]["threshold"]) != float(record["threshold"]):
        raise ValueError("active model threshold mismatch")
    if bundle["score_type"] != record["score_type"]:
        raise ValueError("active model score_type mismatch")
    return bundle


def predict_with_model(image_path, bundle):
    from PIL import Image

    from .features import extract, read_image
    from .final_v4.common import reconstruction_scores

    config = bundle["config"]
    try:
        image = read_image(image_path, config["size"])
    except (OSError, ValueError, Image.DecompressionBombError):
        return {"status": "INPUT_ERROR", "predicted_label": None,
                "prediction_score": None, "score_type": None}
    vector = extract(image, config)
    features = bundle["scaler"].transform(vector.reshape(1, -1))
    if bundle["kind"] == "hog_autoencoder":
        score = float(reconstruction_scores(bundle["model"], features)[0])
        score_type = "hog_reconstruction_mse"
    else:
        score = float(bundle["model"].predict_proba(features)[0, 1])
        score_type = "uncalibrated_fail_probability"
    return {"status": "OK",
            "predicted_label": "FAIL" if score >= float(config["threshold"]) else "PASS",
            "prediction_score": score, "score_type": score_type}


def predict(image_path: str) -> dict:
    try:
        bundle = load_active_model()
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return {"status": "MODEL_UNAVAILABLE", "predicted_label": None,
                "prediction_score": None, "score_type": None}
    return predict_with_model(image_path, bundle)
