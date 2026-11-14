"""Offline invariants plus actual optimization, checkpoint reload and CLI integration."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from members.member06_research.adapter import ClassificationAdapter
from members.member06_research.config import Config
from members.member06_research.data import ManifestDataset, Preprocess, check_disjoint
from members.member06_research.inference import create_adapter
from members.member06_research.model import FrozenResNet18
from members.member06_research.research_tools import LocalFeatureExtractor, crop_roi, geometry_features, segment_contours
from members.member06_research.runtime import choose_device, load_checkpoint, seed_everything
from members.member06_research.smoke_data import generate
from members.member06_research.train import fit, run_epoch, summarize


ROOT = Path(__file__).resolve().parents[3]


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(6)
        self.config = Config(weights="RANDOM_SMOKE_ONLY", data_kind="SMOKE_ONLY", image_size=64)
        self.model = FrozenResNet18(self.config.weights)

    def test_output_and_local_feature_dimensions(self):
        x = torch.randn(2, 3, 64, 64)
        self.assertEqual(self.model(x).shape, (2, 2))
        self.assertEqual(self.model.extract_features(x).shape, (2, 512))
        self.assertEqual(self.model.head.in_features, 512)

    def test_only_head_updates_and_bn_buffers_stay_fixed(self):
        before = {k: v.clone() for k, v in self.model.backbone.state_dict().items()}
        head = self.model.head.weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.head.parameters(), lr=0.1)
        self.model.train()
        self.assertTrue(self.model.head.training)
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(all(not m.training for m in self.model.backbone.modules() if isinstance(m, nn.BatchNorm2d)))
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.head.parameters()))
        for _ in range(2):
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(self.model(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 1, 0])).backward()
            optimizer.step()
        self.assertFalse(torch.equal(head, self.model.head.weight))
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))
        for key, value in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.model.eval().train()
        self.assertFalse(self.model.backbone.training)

    def test_model_rejects_malformed_inputs(self):
        for tensor in (torch.zeros(3, 64, 64), torch.zeros(1, 1, 64, 64),
                       torch.zeros(1, 3, 16, 16), torch.full((1, 3, 64, 64), float("nan")),
                       torch.zeros(1, 3, 64, 64, dtype=torch.int64)):
            with self.subTest(shape=tensor.shape), self.assertRaises(ValueError):
                self.model(tensor)

    def test_pretrained_failure_never_falls_back(self):
        with patch("members.member06_research.model.resnet18", side_effect=OSError("simulated offline")):
            with self.assertRaisesRegex(RuntimeError, "PRETRAINED_WEIGHTS_UNAVAILABLE.*simulated offline"):
                FrozenResNet18("IMAGENET1K_V1")

    def test_preprocessing_keeps_both_horizontal_edges(self):
        image = Image.new("RGB", (200, 50), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 15, 49), fill="red")
        draw.rectangle((185, 0, 199, 49), fill="green")
        config = Config(image_size=64, mean=[0, 0, 0], std=[1, 1, 1])
        tensor = Preprocess(config)(image)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertGreater(tensor[0, :, 0].max().item(), 0.9)
        self.assertGreater(tensor[1, :, -1].max().item(), 0.4)
        self.assertTrue(torch.equal(tensor, Preprocess(config)(image)))
        self.assertEqual(Preprocess(Config(image_size=64, resize_mode="stretch"))(image).shape, tensor.shape)

    def test_local_cnn_uses_features_and_explicit_smoke_status(self):
        image = Image.new("RGB", (100, 70), "white")
        extractor = LocalFeatureExtractor(self.model, self.config, torch.device("cpu"))
        result = extractor.extract(image, [10, 10, 80, 60])
        self.assertEqual(result["dimension"], 512)
        self.assertEqual(result["status"], "SMOKE_ONLY")
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertEqual(extractor.extract(image, [-1, 0, 10, 10])["status"], "INVALID_ROI")

    def test_device_selection(self):
        self.assertEqual(str(choose_device("cpu")), "cpu")
        self.assertEqual(str(choose_device("auto")), "cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                choose_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware/build unavailable; CPU is tested")
    def test_cuda_forward(self):
        self.assertEqual(self.model.cuda()(torch.randn(2, 3, 64, 64, device="cuda")).shape, (2, 2))


class DataAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = generate(Path(self.tmp.name) / "fixtures")
        self.config = Config.load(self.directory / "smoke_config.json")

    def rewrite(self, split, fn):
        path = self.directory / f"{split}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fn(rows)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_manifest_labels_and_shapes(self):
        train = ManifestDataset(self.config.train_manifest, self.config, "train")
        val = ManifestDataset(self.config.val_manifest, self.config, "val")
        check_disjoint(train, val)
        self.assertEqual(len(train), 8)
        self.assertEqual(train[0][0].shape, (3, 64, 64))
        self.assertEqual(train[0][1], self.config.class_to_idx["PASS"])

    def test_reject_test_role_and_smoke_real_mixing(self):
        with self.assertRaises(ValueError):
            ManifestDataset(self.config.val_manifest, self.config, "test")
        self.config.data_kind = "REAL"
        with self.assertRaises(ValueError):
            ManifestDataset(self.config.train_manifest, self.config, "train")

    def test_scene_overlap_is_rejected(self):
        self.rewrite("val", lambda rows: rows[0].update(scene_id="synthetic_train_0"))
        with self.assertRaisesRegex(ValueError, "scene_id"):
            check_disjoint(ManifestDataset(self.config.train_manifest, self.config, "train"),
                           ManifestDataset(self.config.val_manifest, self.config, "val"))

    def test_identical_pixels_under_different_filename_rejected(self):
        image = Image.open(self.directory / "train/synthetic_0.png")
        image.save(self.directory / "val/synthetic_0.png")
        with self.assertRaisesRegex(ValueError, "sha256_rgb"):
            check_disjoint(ManifestDataset(self.config.train_manifest, self.config, "train"),
                           ManifestDataset(self.config.val_manifest, self.config, "val"))

    def test_invalid_label_and_missing_file(self):
        self.rewrite("train", lambda rows: rows[0].update(label="UNKNOWN"))
        with self.assertRaisesRegex(ValueError, "label"):
            ManifestDataset(self.config.train_manifest, self.config, "train")
        self.rewrite("train", lambda rows: rows[0].update(label="PASS", image_path="missing.png"))
        with self.assertRaises(OSError):
            ManifestDataset(self.config.train_manifest, self.config, "train")

    def test_invalid_configuration(self):
        for kwargs in ({"class_to_idx": {"PASS": 0, "FAIL": 0}}, {"class_to_idx": {"PASS": False, "FAIL": 1}},
                       {"resize_mode": "center_crop"}, {"weights": "RANDOM_SMOKE_ONLY"}, {"batch_size": 0},
                       {"std": [0, 1, 1]}, {"learning_rate": float("nan")}, {"supported_counts": [True, 2, 2, 2, 2]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        self.assertEqual(Config(class_to_idx={"PASS": 1, "FAIL": 0}).class_to_idx["FAIL"], 0)

    def test_fail_metrics_follow_explicit_label_mapping(self):
        metrics = summarize(2, 10, [[4, 1], [2, 3]], 0)
        self.assertAlmostEqual(metrics["fail_precision"], 4 / 6)
        self.assertAlmostEqual(metrics["fail_recall"], 4 / 5)
        self.assertIsNone(summarize(2, 2, [[2, 0], [0, 0]], 1)["fail_precision"])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (160, 100), "white")
        ImageDraw.Draw(self.image).rectangle((30, 35, 130, 65), fill="black")

    def test_exact_and_fractional_roi(self):
        crop = crop_roi(self.image, [0, 0, 160, 100])
        self.assertEqual(crop["status"], "OK")
        self.assertEqual(crop["roi"].size, (160, 100))
        crop = crop_roi(self.image, [10.2, 20.9, 90.1, 80.3])
        self.assertEqual(crop["raster_xyxy"], (10, 20, 91, 81))
        self.assertEqual(crop["roi"].size, (81, 61))

    def test_invalid_bbox_and_image_fail_safely(self):
        boxes = ([0, 0, 0, 10], [-1, 0, 5, 5], [0, 0, 161, 10], [4, 5, 3, 8],
                 [0, 0, float("nan"), 5], [False, 0, 5, 5], [1, 2], None, np.array(1))
        for box in boxes:
            with self.subTest(box=box):
                self.assertEqual(crop_roi(self.image, box)["status"], "INVALID_ROI")
        self.assertEqual(crop_roi(None, [0, 0, 1, 1])["status"], "INVALID_ROI")

    def test_pixel_geometry_known_rectangle(self):
        geometry = geometry_features(self.image)
        self.assertEqual(geometry["status"], "OK")
        self.assertAlmostEqual(geometry["features"]["length_px"], 100, places=4)
        self.assertAlmostEqual(geometry["features"]["width_px"], 30, places=4)
        self.assertEqual(geometry["units"], "pixels")
        self.assertEqual(geometry["calibration_status"], "NOT_CALIBRATED")
        self.assertEqual(len(segment_contours(self.image)["contours"]), 1)

    def test_empty_and_abnormal_geometry_fail_safely(self):
        for image in (None, np.empty((0, 0), np.uint8), np.full((10, 10), np.nan),
                      np.zeros((10, 10), np.float32), np.zeros((4, 4, 2), np.uint8), "invalid"):
            with self.subTest(image_type=type(image)):
                result = geometry_features(image)
                self.assertEqual(result["status"], "INPUT_ERROR")
                self.assertIsNone(result["features"])
        self.assertEqual(geometry_features(np.zeros((20, 20), np.uint8))["status"], "NO_CONTOUR")
        self.assertEqual(geometry_features(np.full((20, 20), 255, np.uint8))["status"], "NO_CONTOUR")

    def test_ambiguous_and_border_cases(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 25, 25), fill="black")
        draw.rectangle((60, 60, 80, 80), fill="black")
        self.assertEqual(geometry_features(image)["status"], "AMBIGUOUS_CONTOURS")
        self.assertEqual(geometry_features(self.image, foreground="light")["status"], "BORDER_TOUCHING")

    def test_invalid_segmentation_settings(self):
        for options in ({"foreground": "auto"}, {"threshold": -1}, {"threshold": float("nan")},
                        {"threshold": True}, {"min_area_px2": 0}):
            self.assertEqual(geometry_features(self.image, **options)["status"], "INPUT_ERROR")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        class Backend:
            class_to_idx = {"PASS": 1, "FAIL": 0}
            evidence_status = "TRAIN_VAL_ONLY"
            calls = 0
            def score_batch(self, images):
                self.calls += 1
                return [[0.25, 0.75] for _ in images]
        # A deterministic test double checks protocol routing; all model checks use real ResNet.
        self.backend = Backend()
        self.adapter = ClassificationAdapter(self.backend, [2] * 5, "PROVISIONAL_PENDING_PILOT")
        self.image = Image.new("RGB", (20, 20))

    def test_unsupported_counts_rejected_before_model_or_image(self):
        result = self.adapter.predict("missing.png", [1, 2, 2, 2, 2])
        self.assertEqual(result["status"], "UNSUPPORTED_COUNTS")
        self.assertIsNone(result["predicted_label"])
        self.assertIsNone(result["prediction_score"])
        self.assertEqual(self.backend.calls, 0)

    def test_bad_counts(self):
        for counts in ([True, 2, 2, 2, 2], [2, 2], {"1": 2}, [-1, 2, 2, 2, 2]):
            self.assertEqual(self.adapter.predict(self.image, counts)["status"], "INPUT_ERROR")

    def test_batch_preserves_order_and_errors(self):
        results = self.adapter.predict_batch([self.image, "missing.png", self.image], {str(i): 2 for i in range(1, 6)})
        self.assertEqual([r["status"] for r in results], ["OK", "INPUT_ERROR", "OK"])
        self.assertEqual(results[0]["predicted_label"], "PASS")
        self.assertEqual(results[0]["prediction_score"], 0.75)
        self.assertEqual(results[0]["score_type"], "uncalibrated_softmax")
        self.assertNotIn("objects", results[0])
        self.assertNotIn("observed_count", results[0])
        self.assertEqual(self.adapter.predict_batch([]), [])

    def test_smoke_status_and_invalid_backend(self):
        self.backend.evidence_status = "SMOKE_ONLY"
        self.assertEqual(self.adapter.predict(self.image)["status"], "SMOKE_ONLY")
        self.backend.score_batch = lambda images: [[float("nan"), 0]]
        with self.assertRaises(RuntimeError):
            self.adapter.predict(self.image)

    def test_shared_entry_point_does_not_invent_predictions(self):
        from members.member06_research.baseline2 import predict
        with patch.dict(os.environ, {}, clear=True):
            result = predict("not-read.png")
            self.assertEqual(result["status"], "MODEL_NOT_CONFIGURED")
            self.assertIsNone(result["predicted_label"])

    def test_adapter_runs_without_importing_torch(self):
        code = """
import sys
from PIL import Image
from members.member06_research.adapter import ClassificationAdapter
class Backend:
    class_to_idx = {'PASS': 0, 'FAIL': 1}
    evidence_status = 'SMOKE_ONLY'
    def score_batch(self, images):
        return [[0.6, 0.4] for _ in images]
adapter = ClassificationAdapter(Backend(), [2]*5, 'PROVISIONAL_PENDING_PILOT')
assert adapter.predict(Image.new('RGB', (5,5)))['status'] == 'SMOKE_ONLY'
assert 'torch' not in sys.modules and 'torchvision' not in sys.modules
"""
        completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = generate(Path(cls.tmp.name) / "fixtures")
        cls.config = Config.load(cls.directory / "smoke_config.json")
        cls.run_dir = Path(cls.tmp.name) / "run"
        cls.history = fit(cls.config, cls.run_dir)
        cls.checkpoint = cls.run_dir / "best.pt"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_actual_train_val_and_artifacts(self):
        self.assertEqual(len(self.history), 2)
        self.assertTrue(np.isfinite([r["train"]["loss"] for r in self.history]).all())
        for file in ("config.json", "run_config.json", "history.csv", "history.json", "best.pt",
                     "best_metrics.json", "best_val_predictions.json", "checkpoint_manifest.json"):
            self.assertTrue((self.run_dir / file).is_file(), file)
        self.assertEqual(self.history[0]["train"]["n"], 8)
        self.assertEqual(self.history[0]["val"]["n"], 4)

    def test_checkpoint_load_no_download_and_reproducible_inference(self):
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("must not download")):
            first = create_adapter(self.checkpoint, "cpu", allow_smoke=True)
            second = create_adapter(self.checkpoint, "cpu", allow_smoke=True)
        image = self.directory / "val/synthetic_0.png"
        self.assertEqual(first.predict(image), second.predict(image))
        self.assertEqual(first.predict(image), first.predict_batch([image])[0])
        self.assertEqual(first.predict(image)["status"], "SMOKE_ONLY")
        self.assertEqual(first.backend.model(torch.zeros(2, 3, 64, 64)).shape, (2, 2))

    def test_smoke_checkpoint_refused_by_default(self):
        with self.assertRaisesRegex(ValueError, "SMOKE_ONLY"):
            load_checkpoint(self.checkpoint)

    def test_same_seed_repeats_training(self):
        repeat = Path(self.tmp.name) / "repeat"
        history = fit(self.config, repeat)
        self.assertEqual(self.history, history)
        first, _, _, _ = load_checkpoint(self.checkpoint, allow_smoke=True)
        second, _, _, _ = load_checkpoint(repeat / "best.pt", allow_smoke=True)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_existing_output_refused(self):
        with self.assertRaisesRegex(ValueError, "overwriting"):
            fit(self.config, self.run_dir)

    def test_epoch_rejects_invalid_targets(self):
        model, _, device, _ = load_checkpoint(self.checkpoint, allow_smoke=True)
        for targets in (torch.tensor([2]), torch.tensor([0.0])):
            with self.assertRaises(ValueError):
                run_epoch(model, [(torch.zeros(1, 3, 64, 64), targets)], device, 1)

    def cli(self, module, *args, success=True):
        completed = subprocess.run([sys.executable, "-m", f"members.member06_research.{module}", *map(str, args)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=90,
                                   env={**os.environ, "MPLCONFIGDIR": str(Path(self.tmp.name) / "mpl")})
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0)
        return completed

    def test_readme_cli_integration(self):
        out = Path(self.tmp.name) / "cli"
        out.mkdir()
        self.cli("smoke_data", "--output", out / "data")
        self.cli("train", "--config", out / "data/smoke_config.json", "--output", out / "run")
        checkpoint = out / "run/best.pt"
        self.cli("validate", "--checkpoint", checkpoint, "--manifest", out / "data/val.csv", "--allow-smoke", "--output", out / "val.json")
        self.cli("inference", "--checkpoint", checkpoint, "--image", out / "data/val/synthetic_0.png", "--allow-smoke", "--output", out / "single.json")
        self.cli("inference", "--checkpoint", checkpoint, "--input-dir", out / "data/val", "--allow-smoke", "--output", out / "batch.json")
        single = json.loads((out / "single.json").read_text())["results"][0]
        batch = json.loads((out / "batch.json").read_text())["results"]
        self.assertEqual(single["predicted_label"], batch[0]["predicted_label"])
        self.assertAlmostEqual(single["prediction_score"], batch[0]["prediction_score"], places=6)
        counts_file = out / "counts.json"
        counts_file.write_text(json.dumps([{"image_id": f"synthetic_{i}.png", "expected_counts":
                                          {str(j): (1 if i == 1 and j == 1 else 2) for j in range(1, 6)}}
                                         for i in range(4)]))
        self.cli("inference", "--checkpoint", checkpoint, "--input-dir", out / "data/val", "--counts-file", counts_file,
                 "--allow-smoke", "--output", out / "counts_results.json", success=False)
        per_image = json.loads((out / "counts_results.json").read_text())["results"]
        self.assertEqual([r["status"] for r in per_image], ["SMOKE_ONLY", "UNSUPPORTED_COUNTS", "SMOKE_ONLY", "SMOKE_ONLY"])
        self.cli("inference", "--checkpoint", checkpoint, "--image", out / "data/val/synthetic_0.png", "--expected-counts", "[1,2,2,2,2]", "--allow-smoke", "--output", out / "unsupported.json", success=False)
        self.assertEqual(json.loads((out / "unsupported.json").read_text())["results"][0]["status"], "UNSUPPORTED_COUNTS")
        self.cli("research_demo", "--image", out / "data/geometry_rectangle.png", "--bbox", 0, 0, 160, 100,
                 "--checkpoint", checkpoint, "--allow-smoke", "--output", out / "research.json")
        self.cli("plot_history", "--run-dir", out / "run", "--output", out / "curves.png")
        research = json.loads((out / "research.json").read_text())
        self.assertEqual(research["geometry"]["status"], "OK")
        self.assertEqual(research["cnn"]["dimension"], 512)
        self.assertGreater((out / "curves.png").stat().st_size, 1000)
        self.cli("inference", "--checkpoint", checkpoint, "--image", "missing.png", "--output", out / "refused.json", success=False)


if __name__ == "__main__":
    unittest.main()
