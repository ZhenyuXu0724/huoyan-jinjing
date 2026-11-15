import csv
import json
import tempfile
import unittest
from pathlib import Path

from members.member04_hog.compare import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.truth = self.root / "truth.csv"
        self.counts = self.root / "counts.json"
        self.prediction = self.root / "images.csv"
        self.write(self.truth, [
            {"image_id": "good.jpg", "annotation_status": "VERIFIED",
             **{f"count_{i}": 2 for i in range(1, 6)}, "non_catalogue_count": 0},
            {"image_id": "extra.jpg", "annotation_status": "VERIFIED",
             **{f"count_{i}": 2 for i in range(1, 6)}, "non_catalogue_count": 1}])
        self.cases = [{"image_id": name, "expected_counts": {str(i): 2 for i in range(1, 6)}}
                      for name in ("good.jpg", "extra.jpg")]
        self.counts.write_text(json.dumps(self.cases))
        self.rows = [{"image_id": case["image_id"], "model_id": "baseline3",
                      "status": "OK", "predicted_label": "PASS"} for case in self.cases]
        self.write(self.prediction, self.rows)

    def write(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_extra_unknown_produces_false_pass(self):
        compare(self.truth, self.counts, [self.prediction], self.root / "out")
        result = json.loads((self.root / "out/baseline3/metrics.json").read_text())
        self.assertEqual(result["false_pass"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["coverage"], 1)

    def test_rejects_different_count_rules_and_missing_rows(self):
        self.cases[1]["expected_counts"]["1"] = 3
        self.counts.write_text(json.dumps(self.cases))
        with self.assertRaisesRegex(ValueError, "fixed-count"):
            compare(self.truth, self.counts, [self.prediction], self.root / "bad")
        self.cases[1]["expected_counts"]["1"] = 2
        self.counts.write_text(json.dumps(self.cases))
        self.write(self.prediction, self.rows[:1])
        with self.assertRaisesRegex(ValueError, "common input"):
            compare(self.truth, self.counts, [self.prediction], self.root / "missing")

    def test_failed_prediction_is_unjudged(self):
        self.rows[1]["status"] = "INPUT_ERROR"
        self.rows[1]["predicted_label"] = ""
        self.write(self.prediction, self.rows)
        compare(self.truth, self.counts, [self.prediction], self.root / "out")
        result = json.loads((self.root / "out/baseline3/metrics.json").read_text())
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["unjudged"], ["extra.jpg"])
