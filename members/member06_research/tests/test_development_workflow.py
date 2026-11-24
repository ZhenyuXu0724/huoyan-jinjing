"""All approvals, labels, images and models below are isolated synthetic fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from members.member06_research.tests import test_team_data as legacy_fixture
from members.member06_research import development, experiments
from members.member06_research.batch_export import batch, requests
from members.member06_research.config import Config
from members.member06_research.evaluate_frozen import evaluate, EvaluationDataset
from members.member06_research.preflight import check_training_inputs
from members.member06_research.runtime import file_sha256
from members.member06_research.smoke_data import generate
from members.member06_research.team_data import read_json
from members.member06_research.train import fit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class DevelopmentTests(unittest.TestCase):
    write = staticmethod(write)

    def setUp(self):
        legacy_fixture.TeamDataTests.setUp(self)
        self.sources = self.base / "sources.json"
        self.extra = self.base / "extra"
        self.extra.mkdir()
        files = []
        for part, seed in (("train", 1), ("验证集_不要训练", 2)):
            image = self.extra / "package" / part / (part + ".png")
            image.parent.mkdir(parents=True)
            Image.fromarray(np.random.default_rng(seed).integers(0, 255, (32, 40, 3), dtype=np.uint8)).save(image)
            ids = [1, 2, 3, 4, 5] + ([0] if seed == 2 else [])
            write(image.with_suffix(".json"), {"imageWidth": 40, "imageHeight": 32, "shapes": [
                {"label": f"label_{i}", "shape_type": "rectangle", "points": [[1, 1], [5, 5]], "flags": {}}
                for i in ids]})
            for p in (image, image.with_suffix(".json")):
                files.append({"path": p.relative_to(self.extra).as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size})
        # Deliberately unreadable test sentinel. Its declared hash must not be read/verified.
        files.append({"path": "package/test/DO_NOT_READ.png", "sha256": "0" * 64, "bytes": 123})
        self.index = self.extra / "index.json"
        write(self.index, {"source_folder": "package", "class_ids": {str(i): f"label_{i}" for i in range(6)},
                          "splits": {p: {"images": 1} for p in ("train", "验证集_不要训练", "test")}, "files": files})
        write(self.sources, {"schema_version": "member06.sources.v1", "legacy_root": str(self.root),
            "catalogue": str(self.catalogue_path), "supplements": [{"root": str(self.extra), "manifest": str(self.index),
                "partitions": {"train": "train", "验证集_不要训练": "val", "test": "test"}}]})

    def approved(self):
        report = development.audit(self.sources)
        self.assertEqual(report["errors"], [])
        approval, reviews = development.templates(report)
        approval.update(status="APPROVED", approved_by="SYNTHETIC", decision_reference="UNIT_TEST_ONLY",
                        dataset_version="SYNTHETIC", fixed_expected_counts=[1]*5,
                        scene_split_verified=True, catalogue_specs_verified=True, dataset_usage_approved=True)
        for i, row in enumerate(reviews["images"]):
            row.update(scene_id=f"synthetic_scene_{i}", review_status="APPROVED", judgeable=True,
                       annotation_complete=True, physical_specifications_verified=True,
                       reviewer="SYNTHETIC", review_reference="UNIT_TEST_ONLY")
        write(self.approval_path, approval)
        write(self.reviews_path, reviews)
        return approval, reviews

    def prepare(self):
        return development.prepare(self.sources, self.approval_path, self.reviews_path, self.base / "prepared")

    def test_all_packages_and_no_test_content(self):
        original = Path.open
        def guarded(path, *a, **kw):
            if "test" in path.parts:
                raise AssertionError("test content accessed")
            return original(path, *a, **kw)
        with patch.object(Path, "open", guarded):
            report = development.audit(self.sources)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["per_split"]["train"]["images"], 3)
        self.assertEqual(report["per_split"]["val"]["reference_rule_mismatches_unreviewed"], 2)
        self.assertEqual(report["formal_labels_generated"], 0)

    def test_test_metadata_overlap_is_rejected_without_reading_test(self):
        index = read_json(self.index)
        index["files"][-1]["sha256"] = index["files"][0]["sha256"]
        write(self.index, index)
        report = development.audit(self.sources)
        self.assertTrue(any("test SHA256" in r["error"] for r in report["errors"]))
        self.assertFalse(report["test_content_read"])

    def test_pending_writes_no_training_manifest(self):
        report = development.audit(self.sources)
        approval, reviews = development.templates(report)
        write(self.approval_path, approval)
        write(self.reviews_path, reviews)
        with self.assertRaisesRegex(ValueError, "WAITING_UPSTREAM"):
            self.prepare()
        self.assertFalse((self.base / "prepared").exists())

    def test_prepared_roundtrip_and_batch_without_model(self):
        self.approved()
        record = self.prepare()
        config = Config.load(self.base / "prepared/config.json")
        config.image_size = 64
        _, _, summary = check_training_inputs(config)
        self.assertFalse(summary["model_initialized"])
        self.assertEqual(summary["train_images"], 3)
        self.assertEqual(record["schema_version"], development.SCHEMA)
        self.assertEqual(summary["val_labels"], {"PASS": 1, "FAIL": 2})

    def test_handoff_uses_only_approved_val_and_has_no_truth_in_requests(self):
        from members.member06_research.handoff import export
        self.approved()
        self.prepare()
        requests = export(self.base / "prepared/config.json", self.base, self.base / "handoff")
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(set(r) == {"image_id", "expected_counts"} for r in requests))
        with (self.base / "handoff/val_truth.csv").open() as stream:
            truth = list(csv.DictReader(stream))
        self.assertEqual(len(truth), 3)
        self.assertTrue(all("val" in r["image_id"] or "验证" in r["image_id"] for r in truth))

    def test_stale_sources_block_reuse(self):
        self.approved()
        self.prepare()
        with self.index.open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "source data/config changed"):
            check_training_inputs(Config.load(self.base / "prepared/config.json"))

    def test_source_tamper_and_unknown_label_fail_audit(self):
        index = read_json(self.index)
        annotation = self.extra / index["files"][1]["path"]
        raw = read_json(annotation)
        raw["shapes"][0]["label"] = "unknown"
        write(annotation, raw)
        self.assertTrue(development.audit(self.sources)["errors"])
        index["files"][1].update(sha256=file_sha256(annotation), bytes=annotation.stat().st_size)
        write(self.index, index)
        self.assertIn("unknown label", development.audit(self.sources)["errors"][0]["error"])

    def test_val_and_test_cannot_be_reassigned(self):
        for name in ("验证集_不要训练", "test"):
            source = read_json(self.sources)
            source["supplements"][0]["partitions"][name] = "train"
            path = self.base / f"bad_{name}.json"
            write(path, source)
            with self.assertRaisesRegex(ValueError, "cannot be reassigned"):
                development.audit(path)

    def test_scene_crossing_and_unverified_ids_block(self):
        for unverified in (False, True):
            _, reviews = self.approved()
            reviews["images"][2]["scene_id"] = "UNVERIFIED:filename" if unverified else reviews["images"][0]["scene_id"]
            write(self.reviews_path, reviews)
            with self.assertRaises(ValueError):
                self.prepare()

    def test_single_class_and_unreviewed_cannot_prepare(self):
        approval, reviews = self.approved()
        approval["fixed_expected_counts"] = [10]*5
        write(self.approval_path, approval)
        with self.assertRaisesRegex(ValueError, "PASS AND FAIL"):
            self.prepare()
        self.approved()
        reviews["images"][0]["judgeable"] = False
        write(self.reviews_path, reviews)
        with self.assertRaisesRegex(ValueError, "unresolved image review"):
            self.prepare()

    def test_audit_counts_tamper_cannot_corrupt_comparison_truth(self):
        self.approved()
        self.prepare()
        path = self.base / "prepared/provenance.json"
        record = read_json(path)
        record["audit"]["images"][0]["actual_counts"]["0"] = 99
        write(path, record)
        with self.assertRaisesRegex(ValueError, "source data/config changed"):
            check_training_inputs(Config.load(self.base / "prepared/config.json"))

    def test_manifest_and_lineage_tamper_rejected(self):
        self.approved()
        self.prepare()
        path = self.base / "prepared/train.csv"
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "prepared file changed"):
            check_training_inputs(Config.load(self.base / "prepared/config.json"))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        fixture = generate(cls.base / "fixtures")
        cls.plan_path = cls.base / "plan/plan.json"
        with patch("members.member06_research.train.FrozenResNet18", side_effect=AssertionError("plan must not create model")):
            record = experiments.plan(fixture / "smoke_config.json", cls.base / "plan", [0.001], [0.0], 1)
        candidate = record["candidates"][0]
        fit(Config.load(candidate["config"]), candidate["run"])
        cls.checkpoint = Path(candidate["run"]) / "best.pt"
        cls.selection = cls.base / "selection.json"
        experiments.select(cls.plan_path, cls.selection, allow_smoke=True)
        cls.frozen = cls.base / "freeze.json"
        experiments.freeze(cls.selection, cls.frozen, allow_smoke=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_completed_selection_and_freeze_match(self):
        selected = read_json(self.selection)["selected"]
        self.assertEqual(selected["checkpoint_sha256"], file_sha256(self.checkpoint))
        self.assertEqual(read_json(self.frozen)["checkpoint_sha256"], file_sha256(self.checkpoint))
        self.assertFalse(read_json(self.frozen)["test_used_for_selection"])

    def test_smoke_cannot_be_selected_as_real(self):
        with self.assertRaisesRegex(ValueError, "SMOKE_ONLY"):
            experiments.select(self.plan_path, self.base / "should_not_exist.json")

    def test_plan_code_change_and_incomplete_run_rejected(self):
        plan = read_json(self.plan_path)
        plan["source_sha256"] = {}
        altered = self.base / "altered.json"
        write(altered, plan)
        with self.assertRaisesRegex(ValueError, "code changed"):
            experiments.select(altered, self.base / "blocked.json", True)
        plan = read_json(self.plan_path)
        plan["candidates"][0]["run"] = str(self.base / "missing-run")
        write(altered, plan)
        with self.assertRaises(OSError):
            experiments.select(altered, self.base / "blocked.json", True)

    def test_batch_export_counts_errors_and_hog_consumer(self):
        from members.member04_hog.compare import compare
        counts = self.base / "counts.json"
        names = ["synthetic_0.png", "synthetic_1.png"]
        write(counts, [{"image_id": n, "expected_counts": {str(i): 2 for i in range(1, 6)}} for n in names])
        output = self.base / "exports"
        rows = batch(self.checkpoint, self.base / "fixtures/val", counts, output, allow_smoke=True)
        self.assertEqual([r["status"] for r in rows], ["SMOKE_ONLY", "SMOKE_ONLY"])
        self.assertTrue(all(r["non_catalogue_count"] is None for r in rows))
        # Actual member04 reader consumes the export; SMOKE_ONLY remains unjudged.
        truth = self.base / "synthetic_truth.csv"
        with truth.open("w", newline="") as stream:
            cols = ["image_id", "annotation_status", *[f"count_{i}" for i in range(1, 6)], "non_catalogue_count"]
            writer = csv.DictWriter(stream, fieldnames=cols); writer.writeheader()
            for n in names:
                writer.writerow(dict(zip(cols, [n, "VERIFIED", 2, 2, 2, 2, 2, 0])))
        comparison = self.base / "comparison"
        compare(truth, counts, [output / "images.csv"], comparison)
        self.assertEqual(read_json(comparison / "baseline2/metrics.json")["decided"], 0)
        write(counts, [{"image_id": "missing.png", "expected_counts": [3]*5},
                       {"image_id": "missing2.png", "expected_counts": [2]*5}])
        rows = batch(self.checkpoint, self.base / "fixtures/val", counts, self.base / "errors", allow_smoke=True)
        self.assertEqual([r["status"] for r in rows], ["UNSUPPORTED_COUNTS", "INPUT_ERROR"])

    def test_request_schema_refuses_duplicate_and_traversal(self):
        path = self.base / "bad_counts.json"
        for entries in ([{"image_id": "../a.png", "expected_counts": [2]*5}],
                        [{"image_id": "a.png", "expected_counts": [2]*5}]*2):
            write(path, entries)
            with self.assertRaises(ValueError):
                requests(path)

    def test_postfreeze_evaluation_synthetic_and_guards(self):
        image = self.base / "heldout.png"
        Image.fromarray(np.random.default_rng(938).integers(0, 255, (50, 70, 3), dtype=np.uint8)).save(image)
        manifest = self.base / "heldout.csv"
        row = {"image_path": str(image), "image_sha256": file_sha256(image), "label": "FAIL",
               "scene_id": "synthetic_heldout", "split": "test", "data_kind": "SMOKE_ONLY"}
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        approval = read_json(Path(__file__).parents[1] / "configs/evaluation_approval.template.json")
        approval.update(status="APPROVED", approved_by="SYNTHETIC", decision_reference="UNIT_TEST_ONLY",
                        dataset_version="SYNTHETIC", labels_reviewed=True, scene_split_verified=True,
                        freeze_sha256=file_sha256(self.frozen), manifest_sha256=file_sha256(manifest), expected_counts=[2]*5)
        path = self.base / "eval_approval.json"
        write(path, approval)
        with patch("members.member06_research.evaluate_frozen.load_checkpoint", side_effect=AssertionError("model initialized")):
            result = evaluate(self.frozen, manifest, path, self.base / "check.json", allow_smoke=True, check_only=True)
        self.assertNotIn("metrics", result)
        result = evaluate(self.frozen, manifest, path, self.base / "evaluated.json", allow_smoke=True)
        self.assertEqual(result["metrics"]["n"], 1)
        self.assertIsNone(result["metrics"]["false_fail_rate"])
        approval["freeze_sha256"] = "bad"
        write(path, approval)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            evaluate(self.frozen, manifest, path, self.base / "blocked_eval.json", allow_smoke=True)
        with self.assertRaisesRegex(ValueError, "overlap"):
            EvaluationDataset(manifest, Config(data_kind="SMOKE_ONLY"), approval,
                              [{"path": str(image), "scene_id": "another", "sha256_rgb": "other"}])


if __name__ == "__main__":
    unittest.main()
