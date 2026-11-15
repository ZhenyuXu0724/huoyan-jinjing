import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from members.member04_hog.data import identity, load_manifest, reject_overlap
from members.member04_hog.evaluate import metrics
from members.member04_hog.features import extract, read_image
from members.member04_hog.one_class.settings import GEOMETRY


class DataFeatureTests(unittest.TestCase):
    def test_cross_split_repeated_pixels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                Image.new("L", (30, 15), 220).save(root / f"{name}.png")
                with (root / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image_path", "scene_id", "label"])
                    writer.writerow([f"{name}.png", name, "PASS"])
            a, b = [load_manifest(root / f"{name}.csv") for name in ("a", "b")]
            with self.assertRaisesRegex(ValueError, "overlapping images"):
                reject_overlap(identity(a), identity(b))

    def test_transforms_preserve_pixels_and_vertical_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            image = Image.new("L", (25, 15), 255)
            ImageDraw.Draw(image).rectangle((2, 1, 5, 7), fill=0)
            image.save(path)
            original = read_image(path, 32)
            settings = {"cell": 8, "clahe": False}
            representations = []
            with patch("members.member04_hog.features.hog", side_effect=lambda pixels, **kw: pixels):
                for turns, mirror in GEOMETRY["A5"]:
                    pixels = extract(original, settings, turns, mirror)
                    np.testing.assert_array_equal(np.sort(pixels.ravel()), np.sort(original.ravel()))
                    representations.append(pixels.tobytes())
                np.testing.assert_array_equal(extract(original, settings, 2, True), np.flipud(original))
            self.assertEqual(len(set(representations)), 8)

    def test_fail_positive_metrics_and_coverage(self):
        pairs = [("FAIL", "FAIL"), ("PASS", "FAIL"), ("FAIL", "PASS"), ("PASS", "PASS"),
                 ("FAIL", None)]
        rows = [{"image_id": str(i), "label": truth, "predicted_label": label,
                 "status": "OK" if label else "INPUT_ERROR"}
                for i, (truth, label) in enumerate(pairs)]
        result = metrics(rows)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["coverage"], 0.8)
