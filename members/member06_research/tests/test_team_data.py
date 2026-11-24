"""Synthetic member03-format fixtures; never train or evaluate a real-data model."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from members.member06_research.config import Config
from members.member06_research.preflight import check_training_inputs
from members.member06_research.team_data import (
    approval_template, audit_dataset, classify_reviewed_gt, development_path,
    prepare_dataset, review_template, sha256_file,
)


class TeamDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "synthetic_team_pack"
        self.root.mkdir()
        self.catalogue_path = self.base / "catalogue.json"
        classes = [{"id": i, "name": f"class_{i}", "specification": f"spec_{i}"} for i in range(6)]
        self.write(self.catalogue_path, {"classes": classes, "reference_expected_counts": {str(i): 1 for i in range(1, 6)},
                                         "reference_counts_status": "PROPOSED_CONFIRM_AFTER_PILOT"})
        self.write(self.root / "统一GT/class_mapping.json", {"classes": [
            {**c, "catalogue_specification": c["specification"], "labelme_label": f"label_{c['id']}"} for c in classes]})
        self.rows = []
        rng = np.random.default_rng(61114)
        for split in ("train", "val"):
            for i in range(2):
                relative = f"{split}/synthetic_{split}_{i}.png"
                image = self.root / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)).save(image)
                ids = [1, 2, 3, 4, 5] if i == 0 else [1, 2, 3, 4]
                shapes, objects = [], []
                counts = {str(j): ids.count(j) for j in range(6)}
                for j, class_id in enumerate(ids):
                    box = [2 + j * 10, 5, 8 + j * 10, 20]
                    shapes.append({"label": f"label_{class_id}", "shape_type": "rectangle",
                                   "points": [box[:2], box[2:]], "flags": {"auto_label_review": True}})
                    objects.append({"class_id": class_id, "class_name": f"class_{class_id}", "bbox": box})
                raw_path = image.with_suffix(".json")
                self.write(raw_path, {"shapes": shapes, "imageWidth": 80, "imageHeight": 64, "flags": {}})
                self.write((self.root / "统一GT" / relative).with_suffix(".json"), {
                    "schema_version": "member03.gt.v1", "image_id": image.name, "split": split,
                    "image_relative_path": relative, "image_width": 80, "image_height": 64,
                    "bbox_convention": "original_pixel_xyxy_continuous_half_open",
                    "source_annotation_sha256": sha256_file(raw_path), "objects": objects, "actual_counts": counts})
                self.rows.append({"split": split, "category": "synthetic", "filename": image.name,
                                  "source": relative, "destination": relative, "sha256": sha256_file(image)})
        # These sentinels must never be decoded, parsed, or hashed by the adapter.
        self.write(self.root / "test/DO_NOT_READ.png", {"not": "a real image"})
        self.write(self.root / "统一GT/test/DO_NOT_READ.json", {"must_not": "be accessed"})
        self.rows.append({"split": "test", "category": "synthetic", "filename": "DO_NOT_READ.png",
                          "source": "test/DO_NOT_READ.png", "destination": "test/DO_NOT_READ.png", "sha256": "0" * 64})
        with (self.root / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.approval_path, self.reviews_path = self.base / "approval.json", self.base / "reviews.json"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def approved_fixture(self):
        # Fictional approvals for unit tests ONLY, not actual upstream decisions.
        audit = audit_dataset(self.root, self.catalogue_path)
        self.assertEqual(audit["errors"], [])
        approval = approval_template(audit)
        approval.update(status="APPROVED", approved_by="synthetic-test-only", decision_reference="synthetic-test-rule",
                        dataset_version="synthetic-test-v1", fixed_expected_counts=[1] * 5,
                        scene_split_verified=True, catalogue_specs_verified=True, dataset_usage_approved=True)
        reviews = review_template(audit)
        for row in reviews["images"]:
            row.update(scene_id=Path(row["image_path"]).stem, review_status="APPROVED", judgeable=True,
                       annotation_complete=True, physical_specifications_verified=True,
                       reviewer="synthetic-test-only", review_reference="synthetic-unit-test")
        self.write(self.approval_path, approval)
        self.write(self.reviews_path, reviews)
        return approval, reviews

    def prepare(self, output=None):
        output = output or self.base / "prepared"
        return prepare_dataset(self.root, self.catalogue_path, self.approval_path, self.reviews_path, output)

    def config(self):
        output = self.base / "prepared"
        return Config(data_kind="REAL", image_size=64, supported_counts=[1] * 5, counts_rule_status="CONFIRMED",
                      train_manifest=str(output / "train.csv"), val_manifest=str(output / "val.csv"),
                      data_provenance=str(output / "provenance.json"))

    def test_audit_never_reads_test_contents_and_does_not_generate_labels(self):
        original = Path.open
        accessed = []
        def guarded(path, *args, **kwargs):
            if path.is_relative_to(self.root / "test") or path.is_relative_to(self.root / "统一GT/test"):
                raise AssertionError(f"test content read: {path}")
            accessed.append(str(path))
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            report = audit_dataset(self.root, self.catalogue_path)
        self.assertEqual(report["split_counts"], {"train": 2, "val": 2, "test": 1})
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["test_content_read"])
        self.assertEqual(report["formal_labels_generated"], 0)
        self.assertEqual(report["per_split"]["train"]["raw_auto_review_pending_images"], 2)
        self.assertEqual(report["readiness"], "D")
        self.assertTrue(accessed)

    def test_pending_approval_is_rejected_before_output_or_review_file_access(self):
        report = audit_dataset(self.root, self.catalogue_path)
        self.write(self.approval_path, approval_template(report))
        with self.assertRaisesRegex(ValueError, "WAITING_UPSTREAM"):
            self.prepare()
        self.assertFalse((self.base / "prepared").exists())

    def test_approved_conversion_preserves_splits_and_lineage_then_loads_batch(self):
        self.approved_fixture()
        provenance = self.prepare()
        self.assertEqual(provenance["label_counts"], {"train": {"PASS": 1, "FAIL": 1}, "val": {"PASS": 1, "FAIL": 1}})
        self.assertEqual(provenance["scene_counts"], {"train": 2, "val": 2})
        self.assertFalse(provenance["test_content_read"])
        _, _, summary = check_training_inputs(self.config())
        self.assertEqual(summary["batch_shape"], [2, 3, 64, 64])
        self.assertFalse(summary["weights_updated"])
        self.assertFalse(summary["model_initialized"])

    def test_check_only_never_constructs_cnn_or_optimizer(self):
        self.approved_fixture()
        self.prepare()
        config_file = self.base / "config.json"
        self.write(config_file, self.config().to_dict())
        from members.member06_research import train
        with patch("sys.argv", ["train", "--config", str(config_file), "--check-only"]), \
             patch.object(train, "FrozenResNet18", side_effect=AssertionError("CNN created")), \
             patch.object(train.torch.optim, "Adam", side_effect=AssertionError("optimizer created")), \
             redirect_stdout(io.StringIO()) as stdout:
            train.main()
        self.assertEqual(json.loads(stdout.getvalue())["status"], "READY_TRAIN_VAL_ONLY")

    def test_real_training_refuses_provisional_counts_and_missing_provenance(self):
        with self.assertRaisesRegex(ValueError, "confirmed fixed counts"):
            check_training_inputs(Config())
        with self.assertRaisesRegex(ValueError, "approved data provenance"):
            check_training_inputs(Config(counts_rule_status="CONFIRMED"))

    def test_pending_check_only_reports_precheck_error_without_creating_model(self):
        from members.member06_research import train
        config_file = self.base / "pending_config.json"
        self.write(config_file, Config().to_dict())
        with patch("sys.argv", ["train", "--config", str(config_file), "--check-only"]), \
             patch.object(train, "FrozenResNet18", side_effect=AssertionError("CNN created")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as caught:
                train.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("PRECHECK_FAILED: WAITING_UPSTREAM", stderr.getvalue())

    def test_scene_overlap_is_rejected_without_changing_source_splits(self):
        _, reviews = self.approved_fixture()
        reviews["images"][2]["scene_id"] = reviews["images"][0]["scene_id"]
        self.write(self.reviews_path, reviews)
        original_manifest_hash = sha256_file(self.root / "split_manifest.csv")
        with self.assertRaisesRegex(ValueError, "scene_id crosses"):
            self.prepare()
        self.assertFalse((self.base / "prepared").exists())
        self.assertEqual(sha256_file(self.root / "split_manifest.csv"), original_manifest_hash)

    def test_single_class_is_rejected_instead_of_fabricating_fail(self):
        approval, _ = self.approved_fixture()
        approval["fixed_expected_counts"] = [5] * 5
        self.write(self.approval_path, approval)
        with self.assertRaisesRegex(ValueError, "both trusted PASS and FAIL"):
            self.prepare()
        self.assertFalse((self.base / "prepared").exists())

    def test_stale_review_and_changed_manifests_are_rejected(self):
        _, reviews = self.approved_fixture()
        reviews["images"][0]["gt_sha256"] = "1" * 64
        self.write(self.reviews_path, reviews)
        with self.assertRaisesRegex(ValueError, "stale per-image"):
            self.prepare()
        self.approved_fixture()
        self.prepare()
        with (self.base / "prepared/train.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            check_training_inputs(self.config())

    def test_tampering_pixels_or_count_rule_is_rejected(self):
        self.approved_fixture()
        self.prepare()
        config = self.config()
        config.supported_counts = [2] * 5
        with self.assertRaisesRegex(ValueError, "configured fixed counts differ"):
            check_training_inputs(config)
        Image.new("RGB", (80, 64)).save(self.root / "train/synthetic_train_0.png")
        with self.assertRaisesRegex(ValueError, "pixels changed"):
            check_training_inputs(self.config())

    def test_test_path_and_traversal_rejected_before_open(self):
        for split, relative in (("test", "test/a.png"), ("train", "../test/a.png"), ("val", "test/a.png")):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                development_path(self.root, relative, split)

    def test_corrupt_image_fails_audit_without_labels(self):
        (self.root / "val/synthetic_val_0.png").write_bytes(b"not-an-image")
        report = audit_dataset(self.root, self.catalogue_path)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["formal_labels_generated"], 0)

    def test_unknown_ignore_and_unjudgeable_are_not_guessed(self):
        review = {"review_status": "APPROVED", "judgeable": True,
                  "annotation_complete": True, "physical_specifications_verified": True}
        objects = [{"class_id": i} for i in range(1, 6)]
        self.assertEqual(classify_reviewed_gt({"objects": objects}, [1] * 5, review)["label"], "PASS")
        self.assertEqual(classify_reviewed_gt({"objects": objects + [{"class_id": 0}]}, [1] * 5, review)["label"], "FAIL")
        self.assertEqual(classify_reviewed_gt({"objects": objects[:-1]}, [1] * 5, review)["label"], "FAIL")
        for gt in ({"objects": objects, "ignore": True}, {"objects": objects, "judgeable": False},
                   {"objects": objects, "unknown_specification": True}):
            self.assertIsNone(classify_reviewed_gt(gt, [1] * 5, review)["label"])
        self.assertIsNone(classify_reviewed_gt({"objects": objects}, [1] * 5, {**review, "judgeable": None})["label"])
        with self.assertRaisesRegex(ValueError, "unknown class_id"):
            classify_reviewed_gt({"objects": [{"class_id": 6}]}, [1] * 5, review)


if __name__ == "__main__":
    unittest.main()
