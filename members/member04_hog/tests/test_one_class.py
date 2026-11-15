import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image, ImageDraw

from members.member04_hog.data import load_manifest
from members.member04_hog.evaluate import freeze, validate_freeze
from members.member04_hog.model import predict_with_model
from members.member04_hog.one_class.train import (
    fit_group, load_development, reconstruction_scores, threshold_from_validation)


class OneClassTests(unittest.TestCase):
    def test_threshold_and_score_direction(self):
        threshold = threshold_from_validation(["PASS", "PASS", "FAIL", "FAIL"], [1, 2, 5, 7])
        self.assertTrue(2 < threshold <= 5)
        with self.assertRaisesRegex(ValueError, "both validation"):
            threshold_from_validation(["PASS"], [1])
        class Identity:
            def predict(self, vectors):
                return vectors
        np.testing.assert_array_equal(reconstruction_scores(Identity(), np.ones((2, 3))), [0, 0])

    def test_training_uses_pass_only_and_predicts_with_error_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for split, offset in (("train", 0), ("valid", 10)):
                manifest = root / f"{split}.csv"
                paths.append(manifest)
                with manifest.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image_path", "scene_id", "label"])
                    for index in range(4):
                        image = Image.new("L", (32, 32), 255)
                        ImageDraw.Draw(image).rectangle((index + offset, 3, index + offset + 4, 9), fill=0)
                        name = f"{split}{index}.png"
                        image.save(root / name)
                        label = "FAIL" if split == "valid" and index % 2 else "PASS"
                        writer.writerow([name, split, label])
            train, valid = load_development(*paths)
            weights = root / "model.joblib"
            config = {"size": 32, "cell": 8, "epochs": 2,
                      "expected_counts": {str(i): 2 for i in range(1, 6)}}
            from sklearn.neural_network import MLPRegressor
            original = MLPRegressor.partial_fit
            calls = []
            def capture(model, x, y, *args, **kwargs):
                np.testing.assert_array_equal(x, y)
                self.assertEqual(x.shape[0], len(train))
                calls.append(len(x))
                return original(model, x, y, *args, **kwargs)
            with patch.object(MLPRegressor, "partial_fit", capture):
                fit_group(train, valid, config, "A0", 42, root / "run", weights)
            self.assertEqual(calls, [4, 4])
            bundle = joblib.load(weights)
            self.assertEqual(bundle["scaler"].n_samples_seen_, 4)
            result = predict_with_model(valid[0].path, bundle)
            self.assertEqual(result["score_type"], "hog_reconstruction_mse")
            self.assertGreaterEqual(result["prediction_score"], 0)
            with (root / "run/valid_predictions.csv").open(newline="", encoding="utf-8") as stream:
                saved = next(csv.DictReader(stream))
            self.assertEqual(result["predicted_label"], saved["predicted_label"])
            self.assertAlmostEqual(result["prediction_score"], float(saved["prediction_score"]), places=6)
            self.assertEqual(result, predict_with_model(valid[0].path, joblib.load(weights)))
            freeze(root / "freeze.json", [weights])
            record, frozen_models = validate_freeze(root / "freeze.json")
            self.assertEqual(len(frozen_models), 1)
            self.assertIn("unverified", record["assessment_scope"])
            with weights.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen model changed"):
                validate_freeze(root / "freeze.json")
            text = paths[0].read_text().replace(",PASS", ",FAIL", 1)
            paths[0].write_text(text)
            with self.assertRaisesRegex(ValueError, "PASS samples only"):
                load_development(*paths)
