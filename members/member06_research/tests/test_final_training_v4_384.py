"""Targeted tests for V4-R4 (384x384) resolution comparison, cache isolation, same-seed sync, and v4_compare header fix."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from members.member06_research.config import Config
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256, load_checkpoint, write_json
from members.member06_research.tests.test_final_training_v4 import _make_synthetic_v4_checkout
from members.member06_research.v4_compare import compare_v4_suite
from members.member06_research.v4_data import load_verified_v4_manifests
from members.member06_research.v4_resolution_compare import compare_v4_224_vs_384
from members.member06_research.v4_train import load_v4_experiment_spec, run_v4_experiment

BASE_DIR = Path(__file__).resolve().parents[1]
SUITE_224_REAL = BASE_DIR / "runs/final_training_v4_20261124_192110_PJSHyF"


class FinalTrainingV4ResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)

    def test_real_v4_r2_checkpoints_used_exact_weights_and_384_configs_single_factor(self) -> None:
        # 1. Verify actual saved checkpoints from historical V4-R2 used exact [0.75, 1.5] weights
        for s in (42, 43, 44):
            ckpt_p = SUITE_224_REAL / f"V4-R2_RAW_WEIGHTED_CE_SEED{s}/model/best_within_20.pt"
            _, _, _, payload = load_checkpoint(ckpt_p, device="cpu")
            lp = payload["run_metadata"]["loss_protocol"]
            self.assertEqual(lp["train_class_weight"], "balanced")
            self.assertEqual(lp["weights"], [0.75, 1.5])
            self.assertEqual(lp["validation"], "unweighted CE mean")

            # 2. Verify 384 config strictly inherits V4-R2 with image_size=384 as sole model_config difference
            spec_224, cfg_224 = load_v4_experiment_spec(BASE_DIR / f"configs/final_training_v4/v4_r2_raw_weighted_ce_seed{s}.json")
            spec_384, cfg_384 = load_v4_experiment_spec(BASE_DIR / f"configs/final_training_v4/v4_r4_raw_weighted_ce_384_seed{s}.json")
            self.assertIsNone(spec_384["verified_donor_cache_root"])
            d224, d384 = cfg_224.to_dict(), cfg_384.to_dict()
            self.assertEqual(d224.pop("image_size"), 224)
            self.assertEqual(d384.pop("image_size"), 384)
            self.assertEqual(d224, d384)

    def test_v4_compare_placeholder_fixed_on_real_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "repaired_v4_report"
            run_dirs = {
                ("V4-R1_RAW_CE", s): SUITE_224_REAL / f"V4-R1_RAW_CE_SEED{s}/model"
                for s in (42, 43, 44)
            }
            run_dirs.update({
                ("V4-R2_RAW_WEIGHTED_CE", s): SUITE_224_REAL / f"V4-R2_RAW_WEIGHTED_CE_SEED{s}/model"
                for s in (42, 43, 44)
            })
            compare_v4_suite(run_dirs, out)
            md_text = (out / "COMPARISON_V4_CN.md").read_text(encoding="utf-8")
            self.assertIn("PASS=0.75, FAIL=1.50", md_text)
            self.assertNotIn("{w_r2", md_text)

    def test_paired_224_vs_384_cache_isolation_same_seed_sync_and_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v4_dir, intake_dir = _make_synthetic_v4_checkout(root)
            cache_dir = root / "shared_cache_root"

            for m_f in ("train.csv", "val.csv"):
                mp = v4_dir / "manifests/resnet" / m_f
                mp.write_text(mp.read_text(encoding="utf-8").replace(",REAL\n", ",SMOKE_ONLY\n"), encoding="utf-8")
            prov_p = intake_dir / "provenance.json"
            pdoc = json.loads(prov_p.read_text(encoding="utf-8"))
            pdoc["source_file_sha256"]["manifests/resnet/train.csv"] = file_sha256(v4_dir / "manifests/resnet/train.csv")
            pdoc["source_file_sha256"]["manifests/resnet/val.csv"] = file_sha256(v4_dir / "manifests/resnet/val.csv")
            write_json(prov_p, pdoc)

            fixed_backbone_state = copy.deepcopy(FrozenResNet18("RANDOM_SMOKE_ONLY").backbone.state_dict())
            orig_init = FrozenResNet18.__init__

            def _init_fixed_smoke(self_m, weights="IMAGENET1K_V1", *, initialize=True):
                orig_init(self_m, weights, initialize=initialize)
                if weights == "RANDOM_SMOKE_ONLY" and initialize:
                    self_m.backbone.load_state_dict(fixed_backbone_state)

            base_model_cfg = Config(
                weights="RANDOM_SMOKE_ONLY",
                data_kind="SMOKE_ONLY",
                training_tier="FORMAL_READY",
                image_size=224,
                train_manifest=str(v4_dir / "manifests/resnet/train.csv"),
                val_manifest=str(v4_dir / "manifests/resnet/val.csv"),
                data_provenance=str(intake_dir / "provenance.json"),
                epochs=20,
                batch_size=3,
                learning_rate=0.005,
                train_class_weight="balanced",
            ).to_dict()

            # Patch load_checkpoint in benchmark_offline_full_image_latency to allow SMOKE_ONLY in synthetic test
            orig_load_ckpt = load_checkpoint

            def _smoke_friendly_load(path, device="auto", *, allow_smoke=False, allow_dev_only=False):
                return orig_load_ckpt(path, device=device, allow_smoke=True, allow_dev_only=allow_dev_only)

            with (
                patch.object(FrozenResNet18, "__init__", _init_fixed_smoke),
                patch("members.member06_research.v4_resolution_compare.load_checkpoint", side_effect=_smoke_friendly_load),
            ):
                r224_dirs = {}
                r384_dirs = {}
                for img_sz, strat, target_map in (
                    (224, "V4-R2_RAW_WEIGHTED_CE", r224_dirs),
                    (384, "V4-R4_RAW_WEIGHTED_CE_384", r384_dirs),
                ):
                    for s in (42, 43, 44):
                        sp = root / f"{strat}_{s}.json"
                        write_json(sp, {
                            "schema_version": "member06.v4_experiment.v1",
                            "experiment_id": f"{strat}_SEED{s}",
                            "strategy_id": strat,
                            "description": "synthetic 224 vs 384 test",
                            "allow_optional_aug": False,
                            "use_feature_cache": True,
                            "cache_dir": str(cache_dir),
                            "verified_donor_cache_root": None,
                            "online_augmentation": "none",
                            "lr_scheduler": "none",
                            "threshold_rule": "default_argmax_pass0_fail1_no_threshold_search",
                            "budget_milestones": [10, 20],
                            "model_config": {
                                **base_model_cfg,
                                "image_size": img_sz,
                                "seed": s,
                            },
                        })
                        out_d = root / f"runs/{strat}_SEED{s}/model"
                        run_v4_experiment(sp, out_d)
                        target_map[s] = out_d

                # Verify 384 seed 42 built dedicated sz384 cache, and seeds 43 & 44 reused that exact sz384 cache
                rc_384_42 = json.loads((r384_dirs[42] / "run_config.json").read_text(encoding="utf-8"))
                rc_384_43 = json.loads((r384_dirs[43] / "run_config.json").read_text(encoding="utf-8"))
                rc_384_44 = json.loads((r384_dirs[44] / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(rc_384_42["feature_cache"]["cache_action"], "BUILT_V4_BOUND_CACHE")
                self.assertEqual(rc_384_43["feature_cache"]["cache_action"], "LOADED_V4_BOUND_CACHE")
                self.assertEqual(rc_384_44["feature_cache"]["cache_action"], "LOADED_V4_BOUND_CACHE")
                self.assertIn("v4_resnet18_sz384_", rc_384_42["feature_cache"]["cache_dir"])

                # Run paired 224 vs 384 comparison + offline full-image latency benchmark
                comp_dir = root / "final_res_comparison"
                compare_v4_224_vs_384(r224_dirs, r384_dirs, comp_dir, latency_device="cpu")
                report_md = comp_dir / "COMPARISON_V4_FINAL_RESOLUTION_CN.md"
                self.assertTrue(report_md.is_file())
                self.assertTrue((comp_dir / "v4_224_vs_384_summary.json").is_file())
                self.assertTrue((comp_dir / "v4_224_vs_384_per_seed_metrics.csv").is_file())
                self.assertTrue((comp_dir / "v4_224_vs_384_val_34_changes.csv").is_file())
                self.assertTrue((comp_dir / "curves_v4_224_vs_384.png").is_file())

                summary = json.loads((comp_dir / "v4_224_vs_384_summary.json").read_text(encoding="utf-8"))
                self.assertGreater(summary["offline_full_image_latency_reference_seed42"]["384"]["median_ms"], 0.0)
                self.assertIn("严禁冒充正式现场摄像头实时延迟", report_md.read_text(encoding="utf-8"))

    def test_runner_384_shell_syntax(self) -> None:
        script = BASE_DIR / "scripts/run_final_training_v4_384.sh"
        res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
