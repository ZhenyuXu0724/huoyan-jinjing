"""Targeted tests for Dataset v2 Phase-2: 10/20-epoch dual budgets, seed42 reproduction, multi-seed comparison, R1 Val error analysis, and 384 config readiness."""
from __future__ import annotations

import copy
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from members.member06_research.config import Config
from members.member06_research.data import Preprocess
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256, load_checkpoint, write_json
from members.member06_research.tests.test_dataset_v2_phase1 import _make_synthetic_prepared_v2
from members.member06_research.v2_data import safe_dev_path
from members.member06_research.v2_error_analysis import analyze_r1_val_errors
from members.member06_research.v2_phase2_compare import compare_phase2_runs
from members.member06_research.v2_train import load_experiment_spec, run_v2_experiment

BASE_DIR = Path(__file__).resolve().parents[1]


def _verify_mini_fixture(pdir: str | Path):
    pdir = Path(pdir)
    prov = json.loads((pdir / "provenance.json").read_text(encoding="utf-8"))
    ds_r = Path(prov["dataset_v2_root"])
    manifests = {}
    for k, fn in (
        ("train_raw_294", "train_raw_294.csv"),
        ("train_full_438", "train_full_438.csv"),
        ("val_raw_74", "val_raw_74.csv"),
        ("val_aug_18", "val_aug_18.csv"),
    ):
        with (pdir / fn).open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["path"] = safe_dev_path(ds_r, r["v2_image_relpath"], r["split"])
            assert file_sha256(r["path"]) == r["image_sha256"]
        manifests[k] = rows
    r2_doc = json.loads((pdir / "train_r2_parents_294.json").read_text(encoding="utf-8"))
    for p in r2_doc["parents"]:
        for v in p["variants"]:
            v["path"] = safe_dev_path(ds_r, v["v2_image_relpath"], "train")
    return prov, manifests, r2_doc


class DatasetV2Phase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)

    def test_real_phase2_specs_and_384_resolution_support(self) -> None:
        for seed, exp_id in ((42, "R4_RAW_CE_SEED42"), (43, "R4_RAW_CE_SEED43"), (44, "R4_RAW_CE_SEED44")):
            spec, cfg = load_experiment_spec(BASE_DIR / f"configs/dataset_v2/r4_raw_ce_seed{seed}.json")
            self.assertEqual(spec["experiment_id"], exp_id)
            self.assertEqual(spec["budget_milestones"], [10, 20])
            self.assertEqual(cfg.seed, seed)
            self.assertEqual(cfg.epochs, 20)
            self.assertEqual(cfg.image_size, 224)
            self.assertEqual(cfg.learning_rate, 0.001)
            self.assertEqual(cfg.weight_decay, 0.0)
            self.assertEqual(cfg.batch_size, 16)
            self.assertEqual(cfg.train_class_weight, "none")

        # Verify 384 resolution works cleanly through real Config + Preprocess + FrozenResNet18 without code edits
        _, base_cfg = load_experiment_spec(BASE_DIR / "configs/dataset_v2/r4_raw_ce_seed42.json")
        real_cfg_384 = Config(**{**base_cfg.to_dict(), "image_size": 384})
        self.assertEqual(real_cfg_384.image_size, 384)
        smoke_cfg_384 = Config(**{
            **base_cfg.to_dict(),
            "image_size": 384,
            "weights": "RANDOM_SMOKE_ONLY",
            "data_kind": "SMOKE_ONLY",
            "training_tier": "FORMAL_READY",
        })
        from PIL import Image
        tf_384 = Preprocess(smoke_cfg_384)
        tensor_384 = tf_384(Image.new("RGB", (600, 800), (120, 130, 140)))
        self.assertEqual(tuple(tensor_384.shape), (3, 384, 384))
        model_384 = FrozenResNet18("RANDOM_SMOKE_ONLY")
        out_384 = model_384(tensor_384.unsqueeze(0))
        self.assertEqual(tuple(out_384.shape), (1, 2))

    def test_targeted_r1_val_error_analysis_without_test_access(self) -> None:
        report = analyze_r1_val_errors()
        self.assertFalse(report["test_content_read"])
        self.assertEqual(report["r1_false_pass_count"], 2)
        self.assertEqual(report["r1_false_fail_count"], 0)
        by_name = {c["image_filename"]: c for c in report["cases"]}
        self.assertEqual(set(by_name.keys()), {"IMG_20261114_160244.jpg", "IMG_7209.png"})
        self.assertAlmostEqual(by_name["IMG_20261114_160244.jpg"]["predicted_scores"]["R1_RAW_CE_fail_softmax"], 0.492379, places=5)
        self.assertEqual(by_name["IMG_20261114_160244.jpg"]["train_294_coverage"]["same_anomaly_category_in_train_294"], 5)
        self.assertAlmostEqual(by_name["IMG_7209.png"]["predicted_scores"]["R1_RAW_CE_fail_softmax"], 0.084753, places=5)
        self.assertEqual(by_name["IMG_7209.png"]["train_294_coverage"]["same_anomaly_category_in_train_294"], 1)
        self.assertEqual(by_name["IMG_7209.png"]["train_294_coverage"]["same_pure_missing_class_in_train_294"], 0)

    def test_20ep_budget_preserves_exact_10ep_trajectory_and_multi_seed_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_root, prep_dir = _make_synthetic_prepared_v2(root)
            cache_dir = root / "cache"

            def write_spec(path: Path, exp_id: str, seed: int, epochs: int, milestones=None, ref_dir=None) -> Path:
                write_json(path, {
                    "schema_version": "member06.v2_experiment.v1",
                    "experiment_id": exp_id,
                    "description": "synthetic phase2 test",
                    "train_mode": "raw_294",
                    "prepared_dir": str(prep_dir),
                    "use_feature_cache": True,
                    "cache_dir": str(cache_dir),
                    "online_augmentation": "none",
                    "lr_scheduler": "none",
                    "threshold_rule": "default_argmax_pass0_fail1_no_threshold_search",
                    "budget_milestones": milestones,
                    "reference_10ep_run_dir": str(ref_dir) if ref_dir else None,
                    "factor_role": "smoke",
                    "model_config": {
                        "class_to_idx": {"PASS": 0, "FAIL": 1},
                        "fail_label": "FAIL",
                        "image_size": 64,
                        "resize_mode": "letterbox",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "pad_rgb": [0, 0, 0],
                        "weights": "RANDOM_SMOKE_ONLY",
                        "data_kind": "SMOKE_ONLY",
                        "training_tier": "FORMAL_READY",
                        "supported_counts": [2, 2, 2, 2, 2],
                        "counts_rule_status": "PROVISIONAL_PENDING_PILOT",
                        "train_manifest": str(prep_dir / "train_raw_294.csv"),
                        "val_manifest": str(prep_dir / "val_raw_74.csv"),
                        "data_provenance": str(prep_dir / "provenance.json"),
                        "seed": seed,
                        "epochs": epochs,
                        "batch_size": 2,
                        "learning_rate": 0.005,
                        "train_class_weight": "none",
                        "weight_decay": 0.0,
                        "num_workers": 0,
                        "device": "cpu",
                        "cpu_threads": 2,
                    },
                })
                return path

            # Use a single fixed synthetic backbone state across seeds (mirroring fixed IMAGENET1K_V1 in real runs)
            fixed_backbone_state = copy.deepcopy(FrozenResNet18("RANDOM_SMOKE_ONLY").backbone.state_dict())
            orig_init = FrozenResNet18.__init__

            def _init_fixed_smoke_backbone(self_m, weights="IMAGENET1K_V1", *, initialize=True):
                orig_init(self_m, weights, initialize=initialize)
                if weights == "RANDOM_SMOKE_ONLY" and initialize:
                    self_m.backbone.load_state_dict(fixed_backbone_state)

            with (
                patch("members.member06_research.v2_train.verify_prepared_dataset_v2", side_effect=_verify_mini_fixture),
                patch.object(FrozenResNet18, "__init__", _init_fixed_smoke_backbone),
            ):
                # 1. Run standalone 10-epoch reference (seed=42)
                spec_r1_10 = write_spec(root / "r1_10.json", "R1_RAW_CE", 42, 10)
                run_v2_experiment(spec_r1_10, root / "r1_ref_10")

                # 2. Run 20-epoch Phase-2 runs for seeds 42, 43, 44 (seed=42 verifies reproduction against r1_ref_10)
                for s in (42, 43, 44):
                    sp = write_spec(
                        root / f"r4_s{s}.json",
                        f"R4_RAW_CE_SEED{s}",
                        s,
                        20,
                        milestones=[10, 20],
                        ref_dir=(root / "r1_ref_10") if s == 42 else None,
                    )
                    run_v2_experiment(sp, root / f"r4_s{s}")

                repro = json.loads((root / "r4_s42/reproduction_check_vs_r1.json").read_text(encoding="utf-8"))
                self.assertEqual(repro["status"], "REPRODUCED_WITHIN_TOLERANCE")
                self.assertEqual(repro["max_abs_train_loss_diff_epochs_1_10"], 0.0)
                self.assertEqual(repro["max_abs_val_loss_diff_epochs_1_10"], 0.0)
                self.assertEqual(repro["head_weight_max_abs_diff"], 0.0)
                self.assertEqual(repro["head_bias_max_abs_diff"], 0.0)
                self.assertTrue(repro["model_state_sha256_exact_match"])

                # Verify all 4 milestone checkpoints exist for each seed
                for s in (42, 43, 44):
                    for ck in ("best_within_10.pt", "best_within_20.pt", "epoch10.pt", "epoch20.pt"):
                        self.assertTrue((root / f"r4_s{s}" / ck).is_file())
                    rc = json.loads((root / f"r4_s{s}/run_config.json").read_text(encoding="utf-8"))
                    self.assertEqual(rc["feature_cache"]["cache_action"], "LOADED_VERIFIED")

                # 3. Verify tampering with reference history triggers REPRODUCTION_CHECK_FAILED (no silent pass)
                bad_ref = root / "bad_ref"
                bad_ref.mkdir()
                for fn in ("run_config.json", "best.pt", "checkpoint_manifest.json"):
                    (bad_ref / fn).write_bytes((root / "r1_ref_10" / fn).read_bytes())
                tampered_hist = json.loads((root / "r1_ref_10/history.json").read_text(encoding="utf-8"))
                tampered_hist[4]["val"]["loss"] += 0.05
                write_json(bad_ref / "history.json", tampered_hist)
                bad_spec = write_spec(root / "bad_s42.json", "R4_RAW_CE_SEED42", 42, 20, milestones=[10, 20], ref_dir=bad_ref)
                with self.assertRaisesRegex(RuntimeError, "REPRODUCTION_CHECK_FAILED"):
                    run_v2_experiment(bad_spec, root / "bad_out")

                # 4. Run Phase-2 3-seed comparison report generator
                comp_out = root / "phase2_comparison"
                compare_phase2_runs(
                    root / "r4_s42",
                    root / "r4_s43",
                    root / "r4_s44",
                    comp_out,
                    phase1_suite_dir=None,
                    dataset_v2_root=ds_root,
                )
                self.assertTrue((comp_out / "COMPARISON_PHASE2_CN.md").is_file())
                self.assertTrue((comp_out / "phase2_summary.json").is_file())
                self.assertTrue((comp_out / "phase2_per_seed_metrics.csv").is_file())
                self.assertTrue((comp_out / "phase2_val_74_prediction_changes.csv").is_file())
                self.assertTrue((comp_out / "curves_phase2_budgets.png").is_file())

    def test_phase2_runner_shell_syntax(self) -> None:
        script = BASE_DIR / "scripts/run_dataset_v2_phase2.sh"
        res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
