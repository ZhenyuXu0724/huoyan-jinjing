import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from members.member04_hog.inference import batch


class BatchTests(unittest.TestCase):
    def test_counts_are_checked_without_changing_model_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").touch()
            artifact = root / "model.joblib"
            artifact.write_bytes(b"test")
            counts = {str(i): 2 for i in range(1, 6)}
            bundle = {"config": {"expected_counts": counts}, "dependencies": {},
                      "capabilities": {"image_level": True, "objects": False}}
            requests = root / "counts.json"
            requests.write_text(json.dumps([{"image_id": "image.png", "expected_counts": counts}]))
            result = {"status": "OK", "predicted_label": "PASS", "prediction_score": 0.1,
                      "score_type": "hog_reconstruction_mse"}
            with patch("members.member04_hog.inference.load_model", return_value=bundle), patch(
                    "members.member04_hog.inference.predict_with_model", return_value=result) as predict:
                rows = batch(root, requests, root / "first", artifact)
                self.assertEqual(rows[0]["predicted_label"], "PASS")
                changed = dict(counts, **{"1": 3})
                requests.write_text(json.dumps([{"image_id": "image.png", "expected_counts": changed}]))
                rows = batch(root, requests, root / "changed", artifact)
                self.assertEqual(rows[0]["status"], "UNSUPPORTED_COUNTS")
                self.assertIsNone(rows[0]["prediction_score"])
                self.assertEqual(predict.call_count, 1)

    def test_rejects_truth_in_inference_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "counts.json"
            requests.write_text(json.dumps([{"image_id": "x.png", "expected_counts": {},
                                             "label": "PASS"}]))
            with self.assertRaisesRegex(ValueError, "only image_id"):
                batch(root, requests, root / "out")
