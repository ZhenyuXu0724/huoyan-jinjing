import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from members.member04_hog import model as model_module
from members.member04_hog import inference
from members.member04_hog.final_v4.common import EXPECTED_COUNTS, PREFIX, validate_bundle
from tmpdirs import workspace_temp


class StubScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class StubModel:
    def __init__(self, probability=0.7):
        self.probability = probability

    def predict_proba(self, values):
        return np.tile([1 - self.probability, self.probability], (len(values), 1))

    def predict(self, values):
        return np.asarray(values, dtype=float) * 0.5


def fake_bundle(probability=0.7, kind="supervised"):
    return {
        "format_version": 2,
        "version": PREFIX + ("_Supervised_MLP" if kind == "supervised"
                             else "_PASSOnly_Autoencoder"),
        "kind": "hog_supervised_classifier" if kind == "supervised" else "hog_autoencoder",
        "method": kind,
        "role": "local",
        "group": "A0",
        "seed": 42,
        "epoch": 200,
        "model": StubModel(probability),
        "scaler": StubScaler(),
        "config": {"size": 256, "cell": 16, "threshold": 0.5,
                   "expected_counts": EXPECTED_COUNTS, "clahe": False,
                   "gamma": False, "geometry": "A0", "epochs": 200},
        "score_type": ("uncalibrated_fail_probability" if kind == "supervised"
                       else "hog_reconstruction_mse"),
        "classes": {"PASS": 0, "FAIL": 1},
        "capabilities": {"image_level": True, "objects": False, "arbitrary_counts": False},
        "dependencies": {},
        "code_sha256": {},
        "data": {},
        "protocol_sha256": "x",
    }


class InferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = workspace_temp()
        self.root = temporary.__enter__()
        self.addCleanup(lambda: temporary.__exit__(None, None, None))
        self.image = self.root / "case.png"
        Image.new("L", (48, 32), 220).save(self.image)

    def test_single_image_scores_and_input_errors(self):
        result = model_module.predict_with_model(self.image, fake_bundle())
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["predicted_label"], "FAIL")
        self.assertAlmostEqual(result["prediction_score"], 0.7)
        self.assertEqual(result["score_type"], "uncalibrated_fail_probability")
        broken = self.root / "broken.png"
        broken.write_text("not an image", encoding="utf-8")
        self.assertEqual(model_module.predict_with_model(broken, fake_bundle())["status"],
                         "INPUT_ERROR")
        self.assertEqual(model_module.predict_with_model(self.root / "missing.png",
                                                         fake_bundle())["status"],
                         "INPUT_ERROR")

    def test_active_model_missing_returns_model_unavailable(self):
        original = model_module.ACTIVE_CONFIG
        try:
            config = self.root / "active.json"
            config.write_text(json.dumps({
                "name": PREFIX + "_Rebuild",
                "model_path": "weights/member04_hog/final_v4/representatives/missing.joblib",
                "model_sha256": "0" * 64,
                "version": PREFIX + "_Supervised_MLP",
                "threshold": 0.5,
                "score_type": "uncalibrated_fail_probability",
            }), encoding="utf-8")
            model_module.ACTIVE_CONFIG = config
            self.assertEqual(model_module.predict(self.image)["status"], "MODEL_UNAVAILABLE")
        finally:
            model_module.ACTIVE_CONFIG = original

    def test_batch_statuses_and_missing_model(self):
        import joblib

        counts = self.root / "requests.json"
        counts.write_text(json.dumps([
            {"image_id": "case.png", "expected_counts": {str(i): 2 for i in range(1, 6)}},
            {"image_id": "missing.png", "expected_counts": {str(i): 2 for i in range(1, 6)}},
            {"image_id": "other.png", "expected_counts": {str(i): 1 for i in range(1, 6)}},
        ]), encoding="utf-8")
        artifact = self.root / "bundle.joblib"
        joblib.dump(fake_bundle(), artifact)
        rows = inference.batch(self.root, counts, self.root / "out", artifact)
        self.assertEqual([row["status"] for row in rows],
                         ["OK", "INPUT_ERROR", "UNSUPPORTED_COUNTS"])
        self.assertTrue(all(row["predicted_label"] is None for row in rows[1:]))
        with self.assertRaises(FileNotFoundError):
            inference.batch(self.root, counts, self.root / "no_model",
                            self.root / "missing.joblib")
        self.assertFalse((self.root / "no_model").exists())

    def test_legacy_and_inconsistent_bundles_are_rejected(self):
        legacy = fake_bundle()
        legacy["format_version"] = 1
        self.assertFalse(validate_bundle(legacy)[0])
        wrong = fake_bundle()
        wrong["score_type"] = "hog_reconstruction_mse"
        self.assertFalse(validate_bundle(wrong)[0])
        missing = fake_bundle()
        del missing["config"]["threshold"]
        self.assertFalse(validate_bundle(missing)[0])


if __name__ == "__main__":
    unittest.main()
