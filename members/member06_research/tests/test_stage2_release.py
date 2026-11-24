"""Unit tests for Stage 2 model freeze, Member 7 live/batch adapter, and two-phase GT-isolated Test evaluation."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from members.member06_research.config import Config
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256, write_json
from members.member06_research.stage2_adapter import (
    DEFAULT_FREEZE_MANIFEST_REL,
    DEFAULT_FROZEN_WEIGHT_REL,
    Stage2ResNetAdapter,
    decode_rgb_image,
    predict as module_predict,
)
from members.member06_research.stage2_freeze import EXPECTED_SOURCE_SHA256
from members.member06_research.stage2_test_eval import evaluate_stage2_test, preflight_stage2_test
from members.member06_research.tests.test_final_training_v4 import _make_synthetic_v4_checkout

MEMBER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMBER_DIR.parents[1]


class Stage2ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)

    def test_official_freeze_artifacts_and_immutability(self) -> None:
        fm_path = REPO_ROOT / DEFAULT_FREEZE_MANIFEST_REL
        ckpt_path = REPO_ROOT / DEFAULT_FROZEN_WEIGHT_REL
        self.assertTrue(fm_path.is_file())
        self.assertTrue(ckpt_path.is_file())

        fm = json.loads(fm_path.read_text(encoding="utf-8"))
        self.assertEqual(fm["status"], "FROZEN_PRE_TEST")
        self.assertFalse(fm["test_used_for_training_or_selection"])
        self.assertEqual(fm["selected_model"]["experiment_id"], "V4-R2_RAW_WEIGHTED_CE_SEED42")
        self.assertEqual(fm["selected_model"]["selected_epoch"], 7)
        self.assertEqual(fm["selected_model"]["checkpoint_sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(file_sha256(ckpt_path), EXPECTED_SOURCE_SHA256)

    def test_adapter_rgb_bgr_bytes_contracts_and_member07_server_hook(self) -> None:
        adapter = Stage2ResNetAdapter(device="cpu", batch_size=4)
        # Construct deterministic non-symmetric color test image so RGB vs BGR swap would alter features
        arr_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        arr_rgb[:, :32, 0] = 240  # bright red left half
        arr_rgb[:, 32:, 2] = 220  # bright blue right half
        arr_rgb[10:38, 20:44, 1] = 180
        arr_bgr = np.ascontiguousarray(arr_rgb[..., ::-1])
        pil_img = Image.fromarray(arr_rgb, mode="RGB")

        with tempfile.TemporaryDirectory() as tmp:
            png_path = Path(tmp) / "color_probe.png"
            pil_img.save(png_path, format="PNG")
            png_bytes = png_path.read_bytes()

            r_path = adapter.predict(png_path)
            r_pil = adapter.predict(pil_img)
            r_bytes = adapter.predict(png_bytes)
            r_rgb = adapter.predict(arr_rgb, color_order="rgb")
            r_bgr = adapter.predict(arr_bgr, color_order="bgr")

            self.assertEqual(r_path["status"], "OK")
            for other in (r_pil, r_bytes, r_rgb, r_bgr):
                self.assertEqual(other["predicted_label"], r_path["predicted_label"])
                self.assertAlmostEqual(other["fail_probability"], r_path["fail_probability"], places=7)
                self.assertAlmostEqual(
                    other["fail_probability"] + other["pass_probability"], 1.0, places=6
                )
                self.assertIsNone(other["non_catalogue_count"])
                self.assertNotIn("objects", other)

            # Unsupported counts contract
            unsup = adapter.predict(png_path, expected_counts=[2, 2, 2, 3, 2])
            self.assertEqual(unsup["status"], "UNSUPPORTED_COUNTS")
            self.assertIsNone(unsup["predicted_label"])
            self.assertIsNone(unsup["prediction_score"])
            self.assertEqual(unsup["supported_counts"], [2, 2, 2, 2, 2])

            # Invalid image contract
            bad = adapter.predict(b"corrupted_bytes")
            self.assertEqual(bad["status"], "INPUT_ERROR")
            self.assertIsNone(bad["predicted_label"])
            self.assertIsNone(bad["prediction_score"])

            # Environment variable path configuration check
            with patch.dict(os.environ, {"MEMBER06_CHECKPOINT": str(REPO_ROOT / DEFAULT_FROZEN_WEIGHT_REL), "MEMBER06_DEVICE": "cpu"}):
                r_env = module_predict(png_path)
                self.assertEqual(r_env["predicted_label"], r_path["predicted_label"])

            # Member 7 server.py dynamic spec loader & validate_result check
            server_py = REPO_ROOT / "members/member07_app/server.py"
            s_spec = importlib.util.spec_from_file_location("m7_server_test", server_py)
            assert s_spec is not None and s_spec.loader is not None
            s_mod = importlib.util.module_from_spec(s_spec)
            s_spec.loader.exec_module(s_mod)

            m7_root = REPO_ROOT / "members/member07_app"
            m7_cfg = json.loads(
                (MEMBER_DIR / "stage2_release/member07_models_baseline2.example.json").read_text(encoding="utf-8")
            )["baseline2"]
            m7_cfg["device"] = "cpu"
            a_spec = importlib.util.spec_from_file_location(
                "adapter_baseline2", (m7_root / m7_cfg["adapter"]).resolve()
            )
            assert a_spec is not None and a_spec.loader is not None
            a_mod = importlib.util.module_from_spec(a_spec)
            a_spec.loader.exec_module(a_mod)
            live_fn = a_mod.load(m7_cfg, m7_root)

            validated = s_mod.validate_result(live_fn(png_bytes))
            self.assertEqual(validated["kind"], "image")
            self.assertEqual(validated["decision"], r_path["predicted_label"])
            self.assertEqual(validated["fixed_expected"], [2, 2, 2, 2, 2])
            self.assertAlmostEqual(validated["score"], round(r_path["fail_probability"], 6), places=6)

            with self.assertRaises(ValueError):
                live_fn(b"")

    def test_two_phase_gt_isolated_test_evaluator_on_synthetic_v4(self) -> None:
        """Verify stage2_test_eval on synthetic SMOKE_ONLY v4 without running real Test inference."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v4_dir, _ = _make_synthetic_v4_checkout(root)

            # Replace the poisoned training-isolation dummy test row with 4 valid synthetic test originals
            # covering all 4 primary scenario buckets: normal_pass, pure_missing, seen_distractor, unseen_mixed.
            (v4_dir / "images/test/FAIL/t_bad.jpg").unlink(missing_ok=True)
            with (v4_dir / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
                split_rows = [r for r in csv.DictReader(fh) if r["split"] != "test"]
                split_fields = list(split_rows[0].keys())

            synth_tests = [
                ("t_pass.png", "PASS", "11/23", "GT_P1", "none", "2|2|2|2|2", (0, 0, 0, 0, 0), (40, 200, 60)),
                ("t_miss.png", "FAIL", "11/11", "GT_F1", "missing", "2|1|2|2|2", (0, 0, 0, 0, 0), (200, 80, 40)),
                ("t_seen.png", "FAIL", "11/14", "GT_F2", "non_cat", "2|2|2|2|2", (0, 0, 1, 0, 0), (180, 50, 180)),
                ("t_unseen_mix.png", "FAIL", "11/20", "GT_F3", "mixed", "2|2|1|2|0", (1, 1, 0, 2, 0), (50, 80, 210)),
            ]
            test_csv_rows = []
            for fname, lbl, batch, scene, anom, t_cnt, (s_c, h_c, n_c, w_c, b_c), rgb in synth_tests:
                rel_dest = f"images/test/{lbl}/{fname}"
                abs_p = v4_dir / rel_dest
                Image.new("RGB", (48, 40), rgb).save(abs_p)
                sha = file_sha256(abs_p)
                row = dict(split_rows[0])
                row.update(
                    {
                        "filename": fname,
                        "split": "test",
                        "usage": "primary",
                        "label": lbl,
                        "tier": "original",
                        "batch": batch,
                        "scene_group": scene,
                        "parent": fname,
                        "anomaly": anom,
                        "target_counts": t_cnt,
                        "S": str(s_c),
                        "H": str(h_c),
                        "N": str(n_c),
                        "W": str(w_c),
                        "B": str(b_c),
                        "source_scene_group": scene,
                        "dest_image": rel_dest,
                        "image_sha256": sha,
                        "image_bytes": str(abs_p.stat().st_size),
                    }
                )
                split_rows.append(row)
                test_csv_rows.append(
                    {
                        "image_path": f"../../{rel_dest}",
                        "label": lbl,
                        "scene_id": scene,
                        "split": "test",
                        "data_kind": "SMOKE_ONLY",
                    }
                )

            with (v4_dir / "split_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=split_fields)
                writer.writeheader()
                writer.writerows(split_rows)

            with (v4_dir / "manifests/resnet/test.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "scene_id", "split", "data_kind"])
                writer.writeheader()
                writer.writerows(test_csv_rows)

            # Mark train/val manifests as SMOKE_ONLY so synthetic random-backbone checkpoint is allowed
            for split_name in ("train", "val"):
                mp = v4_dir / f"manifests/resnet/{split_name}.csv"
                mp.write_text(mp.read_text(encoding="utf-8").replace(",REAL\n", ",SMOKE_ONLY\n"), encoding="utf-8")

            cfg = Config(
                weights="RANDOM_SMOKE_ONLY",
                data_kind="SMOKE_ONLY",
                training_tier="FORMAL_READY",
                image_size=64,
                epochs=2,
                batch_size=2,
                seed=42,
            )
            model = FrozenResNet18("RANDOM_SMOKE_ONLY")
            ckpt_path = root / "synthetic_frozen.pt"
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "frozen_resnet18_linear",
                    "config": cfg.to_dict(),
                    "model_state": model.state_dict(),
                    "epoch": 1,
                    "evidence_status": cfg.evidence_status,
                },
                ckpt_path,
            )
            ckpt_sha = file_sha256(ckpt_path)

            fm_path = root / "synthetic_freeze_manifest.json"
            write_json(
                fm_path,
                {
                    "schema_version": "member06.stage2_freeze.v1",
                    "status": "FROZEN_PRE_TEST",
                    "test_used_for_training_or_selection": False,
                    "selected_model": {
                        "frozen_checkpoint_relpath": str(ckpt_path),
                        "checkpoint_sha256": ckpt_sha,
                    },
                    "dataset_and_git_identity": {
                        "upstream_v4_commit": "synthetic_commit",
                        "manifests_sha256": {
                            "train_csv": file_sha256(v4_dir / "manifests/resnet/train.csv"),
                            "val_csv": file_sha256(v4_dir / "manifests/resnet/val.csv"),
                            "test_csv": file_sha256(v4_dir / "manifests/resnet/test.csv"),
                            "split_manifest_csv": file_sha256(v4_dir / "split_manifest.csv"),
                        },
                    },
                },
            )

            pre = preflight_stage2_test(fm_path, v4_dir, allow_smoke=True)
            self.assertEqual(pre["status"], "STAGE2_TEST_READY_FOR_USER_EXECUTION")
            self.assertFalse(pre["inference_executed"])

            out_eval = root / "test_eval_out"
            summary = evaluate_stage2_test(
                out_eval,
                freeze_manifest_path=fm_path,
                v4_dir=v4_dir,
                device="cpu",
                batch_size=2,
                allow_smoke=True,
            )
            self.assertEqual(summary["status"], "STAGE2_TEST_EVALUATION_COMPLETED")
            for expected_file in (
                "images.csv",
                "run_config.json",
                "test_metrics.json",
                "per_image_predictions.csv",
                "grouped_metrics.csv",
                "failure_cases.csv",
                "STAGE2_FINAL_TEST_REPORT_CN.md",
            ):
                self.assertTrue((out_eval / expected_file).is_file(), expected_file)

            with (out_eval / "images.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), pre["n_test_images"])
            self.assertTrue(all(r["non_catalogue_count"] == "" for r in rows))


if __name__ == "__main__":
    unittest.main()
