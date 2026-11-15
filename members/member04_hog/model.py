"""Single-image baseline1 entry point."""
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "weights" / "member04_hog" / "baseline1.joblib"


@lru_cache(maxsize=4)
def _load(path, modified, size):
    import joblib

    return joblib.load(path)


def load_model(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _load(path, stat.st_mtime_ns, stat.st_size)


def predict_with_model(image_path, bundle):
    from PIL import Image
    from .features import extract, read_image

    config = bundle["config"]
    try:
        image = read_image(image_path, config["size"])
    except (OSError, ValueError, Image.DecompressionBombError):
        return {"status": "INPUT_ERROR", "predicted_label": None,
                "prediction_score": None, "score_type": None}
    vector = extract(image, config)
    model = bundle["model"]
    features = bundle["scaler"].transform(vector.reshape(1, -1))
    if bundle.get("kind") == "hog_autoencoder":
        from .one_class.scoring import reconstruction_scores

        score = float(reconstruction_scores(model, features)[0])
        score_type = "hog_reconstruction_mse"
    else:
        score = float(model.predict_proba(features)[0, 1])
        score_type = "uncalibrated_fail_probability"
    return {"status": "OK",
            "predicted_label": "FAIL" if score >= config["threshold"] else "PASS",
            "prediction_score": score, "score_type": score_type}

def predict(image_path: str) -> dict:
    try:
        bundle = load_model(MODEL_PATH)
    except FileNotFoundError:
        return {"status": "MODEL_UNAVAILABLE", "predicted_label": None,
                "prediction_score": None, "score_type": None}
    return predict_with_model(image_path, bundle)
