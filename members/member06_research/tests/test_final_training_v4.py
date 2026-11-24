"""Regression and safety tests for Baseline 2 final_training_v4 intake, v4-bound cache, 6-run budgets, and comparison."""
from __future__ import annotations

import copy
import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from members.member06_research.config import Config
from members.member06_research.data import Preprocess
from members.member06_research.inference import create_adapter
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256, load_checkpoint, write_json
from members.member06_research.train import class_weights
from members.member06_research.v2_train import build_or_load_feature_cache
from members.member06_research.v4_compare import compare_v4_suite
from members.member06_research.v4_data import (
    _resolve_safe_v4_image,
    load_verified_v4_manifests,
    prepare_v4_intake,
)
from members.member06_research.v4_train import (
    build_or_load_v4_feature_cache,
    load_v4_experiment_spec,
    run_v4_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parents[1]


def _make_synthetic_v4_checkout(root: Path) -> tuple[Path, Path]:
    """Build a miniature final_training_v4 directory structure for fast SMOKE_ONLY testing."""
    v4 = root / "final_training_v4"
    for rel_d in (
        "images/train/PASS",
        "images/train/FAIL",
        "images/train_augmentations/FAIL",
        "images/val/PASS",
        "images/val/FAIL",
        "images/test/PASS",
        "images/test/FAIL",
        "manifests/resnet",
    ):
        (v4 / rel_d).mkdir(parents=True, exist_ok=True)

    (v4 / "README.md").write_text("# synthetic v4\n", encoding="utf-8")
    write_json(v4 / "classes.json", {"classes": [], "expected_target_counts": [2, 2, 2, 2, 2]})

    # Poisoned test file: if any training/intake code tries to open/hash Test image content, we keep its metadata separate
    (v4 / "images/test/FAIL/t_bad.jpg").write_bytes(b"CORRUPT_TEST_BYTES_NEVER_DECODE")

    split_rows = []
    train_csv_rows = []
    val_csv_rows = []
    test_csv_rows = [
        {"image_path": "../../images/test/FAIL/t_bad.jpg", "label": "FAIL", "scene_id": "G_TEST_1", "split": "test", "data_kind": "REAL"}
    ]

    def add_img(fname: str, split: str, usage: str, tier: str, label: str, batch: str, scene: str, anomaly: str, t_cnt: str, shnwb: tuple[int, int, int, int, int], rgb: tuple[int, int, int]):
        folder = split if tier == "original" else "train_augmentations"
        dest_img = f"images/{folder}/{label}/{fname}"
        abs_p = v4 / dest_img
        Image.new("RGB", (44, 36), rgb).save(abs_p)
        sha = file_sha256(abs_p)
        size = abs_p.stat().st_size
        S, H, N, W, B = shnwb
        split_rows.append({
            "filename": fname,
            "split": split,
            "usage": usage,
            "label": label,
            "tier": tier,
            "batch": batch,
            "scene_group": scene,
            "parent": fname.split("__")[0] + Path(fname).suffix if "__" in fname else fname,
            "anomaly": anomaly,
            "target_counts": t_cnt,
            "S": S, "H": H, "N": N, "W": W, "B": B,
            "source_scene_group": scene,
            "condition": batch,
            "dest_image": dest_img,
            "dest_json": dest_img.rsplit(".", 1)[0] + ".json",
            "image_sha256": sha,
            "json_sha256": "0" * 64,
            "image_bytes": size,
            "json_bytes": 10,
            "yolo_label": f"labels/{folder}/{label}/{Path(fname).stem}.txt",
        })
        if usage == "primary" and split == "train":
            train_csv_rows.append({
                "image_path": f"../../{dest_img}",
                "label": label,
                "scene_id": scene,
                "split": "train",
                "data_kind": "REAL",
            })
        elif usage == "primary" and split == "val":
            val_csv_rows.append({
                "image_path": f"../../{dest_img}",
                "label": label,
                "scene_id": scene,
                "split": "val",
                "data_kind": "REAL",
            })

    # 4 Train primary (2 PASS, 2 FAIL -> wait: let's do 4 PASS, 2 FAIL = 6 so balanced weights are 0.75 and 1.50 just like 166/83!)
    add_img("tr_p1.png", "train", "primary", "original", "PASS", "11/11", "G_T1", "none", "2|2|2|2|2", (0,0,0,0,0), (30,110,200))
    add_img("tr_p2.png", "train", "primary", "original", "PASS", "11/11", "G_T1", "none", "2|2|2|2|2", (0,0,0,0,0), (40,120,210))
    add_img("tr_p3.jpg", "train", "primary", "original", "PASS", "11/14", "G_T2", "none", "2|2|2|2|2", (0,0,0,0,0), (50,140,180))
    add_img("tr_p4.jpg", "train", "primary", "original", "PASS", "11/23", "G_T3", "none", "2|2|2|2|2", (0,0,0,0,0), (70,160,150))
    add_img("tr_f1.jpg", "train", "primary", "original", "FAIL", "11/18", "G_T4", "non_catalogue_only_3", "2|2|2|2|2", (0,0,1,2,0), (200,50,40))
    add_img("tr_f2.jpg", "train", "primary", "original", "FAIL", "11/20", "G_T5", "non_catalogue_plus_missing", "2|2|2|2|0", (0,0,0,2,0), (180,90,40))
    # 1 optional train aug
    add_img("tr_f2__A01_lowlight.jpg", "train", "optional_train_aug", "augmented", "FAIL", "11/20", "G_T5", "non_catalogue_plus_missing", "2|2|2|2|0", (0,0,0,2,0), (150,70,30))

    # 4 Val primary (single-class batches: 11/11 has 2 PASS, 11/18 has 2 FAIL)
    add_img("vl_p1.png", "val", "primary", "original", "PASS", "11/11", "G_V1", "none", "2|2|2|2|2", (0,0,0,0,0), (35,115,205))
    add_img("vl_p2.png", "val", "primary", "original", "PASS", "11/11", "G_V1", "none", "2|2|2|2|2", (0,0,0,0,0), (45,125,215))
    add_img("vl_f1.jpg", "val", "primary", "original", "FAIL", "11/18", "G_V2", "non_catalogue_only_3", "2|2|2|2|2", (0,0,1,2,0), (205,55,45))
    add_img("vl_f2.jpg", "val", "primary", "original", "FAIL", "11/18", "G_V2", "pure_missing", "2|1|2|2|2", (0,0,0,0,0), (195,75,55))

    # Test row in split_manifest (must never be loaded)
    split_rows.append({
        "filename": "t_bad.jpg", "split": "test", "usage": "primary", "label": "FAIL", "tier": "original",
        "batch": "11/20", "scene_group": "G_TEST_1", "parent": "t_bad.jpg", "anomaly": "non_catalogue_plus_missing",
        "target_counts": "2|2|2|2|0", "S": 1, "H": 0, "N": 0, "W": 1, "B": 0,
        "source_scene_group": "G_TEST_1", "condition": "11/20",
        "dest_image": "images/test/FAIL/t_bad.jpg", "dest_json": "images/test/FAIL/t_bad.json",
        "image_sha256": "f" * 64, "json_sha256": "e" * 64, "image_bytes": 32, "json_bytes": 10,
        "yolo_label": "labels/test/FAIL/t_bad.txt",
    })

    with (v4 / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
        w.writeheader()
        w.writerows(split_rows)

    for m_name, m_rows in (("train.csv", train_csv_rows), ("val.csv", val_csv_rows), ("test.csv", test_csv_rows)):
        with (v4 / "manifests/resnet" / m_name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["image_path", "label", "scene_id", "split", "data_kind"])
            w.writeheader()
            w.writerows(m_rows)

    intake_dir = root / "v4_intake"
    prepare_v4_intake(v4, intake_dir, upstream_commit="synthetic_commit")
    return v4, intake_dir


class FinalTrainingV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)

    def test_real_v4_intake_weights_and_optional_351_capability(self) -> None:
        prov_path = REPO_ROOT / "members/member06_research/data/final_training_v4_intake/provenance.json"
        train_csv = REPO_ROOT / "members/member03_data/final_training_v4/manifests/resnet/train.csv"
        val_csv = REPO_ROOT / "members/member03_data/final_training_v4/manifests/resnet/val.csv"

        prov, tr, vl = load_verified_v4_manifests(train_csv, val_csv, prov_path)
        self.assertEqual(prov["upstream_commit"], "bde8ea7116761bc8ee7fca1d47c336126daa995f")
        self.assertFalse(prov["test_content_read"])
        self.assertEqual(len(tr), 249)
        self.assertEqual(sum(r["label"] == "PASS" for r in tr), 166)
        self.assertEqual(sum(r["label"] == "FAIL" for r in tr), 83)
        self.assertEqual(len(vl), 34)
        self.assertEqual(sum(r["label"] == "PASS" for r in vl), 20)
        self.assertEqual(sum(r["label"] == "FAIL" for r in vl), 14)

        # Verify exact V4-R2 balanced weights: PASS = 249/(2*166) = 0.75, FAIL = 249/(2*83) = 1.50
        _, cfg_r2 = load_v4_experiment_spec(BASE_DIR / "configs/final_training_v4/v4_r2_raw_weighted_ce_seed42.json")
        w = class_weights(tr, cfg_r2, "cpu")
        self.assertAlmostEqual(w[0].item(), 0.75, places=7)
        self.assertAlmostEqual(w[1].item(), 1.50, places=7)

        # Verify optional 351-image manifest (249 raw + 102 optional FAIL aug = 166 PASS + 185 FAIL) loads cleanly when explicitly enabled
        opt_csv = REPO_ROOT / "members/member06_research/data/final_training_v4_intake/train_with_optional_aug.csv"
        with self.assertRaisesRegex(ValueError, "disallowed usage"):
            load_verified_v4_manifests(opt_csv, val_csv, prov_path, allow_optional_aug=False)
        _, tr_351, _ = load_verified_v4_manifests(opt_csv, val_csv, prov_path, allow_optional_aug=True)
        self.assertEqual(len(tr_351), 351)
        self.assertEqual(sum(r["label"] == "PASS" for r in tr_351), 166)
        self.assertEqual(sum(r["label"] == "FAIL" for r in tr_351), 185)

        # Verify image_size=384 config readiness
        cfg_384 = Config(**{**cfg_r2.to_dict(), "image_size": 384})
        self.assertEqual(cfg_384.image_size, 384)

    def test_test_partition_and_cache_scope_strictly_blocked(self) -> None:
        v4_root = REPO_ROOT / "members/member03_data/final_training_v4"
        m_dir = v4_root / "manifests/resnet/train.csv"
        with self.assertRaisesRegex(ValueError, "FORBIDDEN_TEST_ACCESS"):
            _resolve_safe_v4_image(m_dir, "../../images/test/PASS/IMG_7206.png", v4_root, "train")
        with self.assertRaisesRegex(ValueError, "forbidden split"):
            _resolve_safe_v4_image(m_dir, "../../images/test/PASS/IMG_7206.png", v4_root, "test")

        c = Config(weights="RANDOM_SMOKE_ONLY", data_kind="SMOKE_ONLY", image_size=32)
        m = FrozenResNet18("RANDOM_SMOKE_ONLY")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "REFUSED_V4_CACHE_SCOPE"):
                build_or_load_v4_feature_cache(
                    tmp,
                    [{"split": "test", "dest_image": "images/test/x.jpg", "image_sha256": "a" * 64, "path": Path(tmp) / "x.jpg"}],
                    m,
                    c,
                    torch.device("cpu"),
                    None,
                    {"upstream_commit": "x"},
                )

    def test_v4_bound_cache_donor_verification_6runs_sync_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v4_dir, intake_dir = _make_synthetic_v4_checkout(root)
            donor_cache_dir = root / "donor_cache"
            v4_cache_dir = root / "v4_cache"

            # Fix synthetic backbone across seeds to mirror fixed ImageNet backbone
            fixed_backbone_state = copy.deepcopy(FrozenResNet18("RANDOM_SMOKE_ONLY").backbone.state_dict())
            orig_init = FrozenResNet18.__init__

            def _init_fixed_smoke(self_m, weights="IMAGENET1K_V1", *, initialize=True):
                orig_init(self_m, weights, initialize=initialize)
                if weights == "RANDOM_SMOKE_ONLY" and initialize:
                    self_m.backbone.load_state_dict(fixed_backbone_state)

            with patch.object(FrozenResNet18, "__init__", _init_fixed_smoke):
                prov, tr_rows, vl_rows = load_verified_v4_manifests(
                    v4_dir / "manifests/resnet/train.csv",
                    v4_dir / "manifests/resnet/val.csv",
                    intake_dir / "provenance.json",
                )
                # Populate a verified donor cache with half of Train images + 1 dummy non-v4 image
                base_cfg = Config(
                    weights="RANDOM_SMOKE_ONLY",
                    data_kind="SMOKE_ONLY",
                    training_tier="FORMAL_READY",
                    image_size=64,
                    train_manifest=str(v4_dir / "manifests/resnet/train.csv"),
                    val_manifest=str(v4_dir / "manifests/resnet/val.csv"),
                    data_provenance=str(intake_dir / "provenance.json"),
                )
                # Adjust manifest data_kind to SMOKE_ONLY for synthetic Config compatibility
                for m_f in ("train.csv", "val.csv"):
                    mp = v4_dir / "manifests/resnet" / m_f
                    txt = mp.read_text(encoding="utf-8").replace(",REAL\n", ",SMOKE_ONLY\n")
                    mp.write_text(txt, encoding="utf-8")
                # Update provenance hashes after synthetic SMOKE_ONLY adjustment
                prov_p = intake_dir / "provenance.json"
                pdoc = json.loads(prov_p.read_text(encoding="utf-8"))
                pdoc["source_file_sha256"]["manifests/resnet/train.csv"] = file_sha256(v4_dir / "manifests/resnet/train.csv")
                pdoc["source_file_sha256"]["manifests/resnet/val.csv"] = file_sha256(v4_dir / "manifests/resnet/val.csv")
                write_json(prov_p, pdoc)

                dummy_model = FrozenResNet18("RANDOM_SMOKE_ONLY")
                donor_subset = tr_rows[:3]
                build_or_load_feature_cache(donor_cache_dir, donor_subset, dummy_model, base_cfg, torch.device("cpu"), None)

                # Run all 6 experiments (V4-R1 & V4-R2 x seeds 42, 43, 44, 20 epochs with [10, 20] milestones)
                suite = root / "suite_v4"
                run_dirs = {}
                for strat, cw in (("V4-R1_RAW_CE", "none"), ("V4-R2_RAW_WEIGHTED_CE", "balanced")):
                    for s in (42, 43, 44):
                        sp_path = root / f"{strat}_{s}.json"
                        write_json(sp_path, {
                            "schema_version": "member06.v4_experiment.v1",
                            "experiment_id": f"{strat}_SEED{s}",
                            "strategy_id": strat,
                            "description": "synthetic v4 test",
                            "allow_optional_aug": False,
                            "use_feature_cache": True,
                            "cache_dir": str(v4_cache_dir),
                            "verified_donor_cache_root": str(donor_cache_dir),
                            "online_augmentation": "none",
                            "lr_scheduler": "none",
                            "threshold_rule": "default_argmax_pass0_fail1_no_threshold_search",
                            "budget_milestones": [10, 20],
                            "model_config": {
                                **base_cfg.to_dict(),
                                "seed": s,
                                "epochs": 20,
                                "batch_size": 3,
                                "learning_rate": 0.005,
                                "train_class_weight": cw,
                            },
                        })
                        out_d = suite / f"{strat}_SEED{s}/model"
                        run_v4_experiment(sp_path, out_d)
                        run_dirs[(strat, s)] = out_d

                # Verify first run built dedicated v4-bound cache (reusing 3 verified donor vectors + extracting 7 fresh = 10 dev items, 0 test)
                rc_first = json.loads((run_dirs[("V4-R1_RAW_CE", 42)] / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(rc_first["feature_cache"]["cache_action"], "BUILT_V4_BOUND_CACHE")
                self.assertEqual(rc_first["feature_cache"]["cached_images"], 10)
                self.assertEqual(rc_first["feature_cache"]["build_provenance"]["reused_verified_single_image_vectors"], 3)
                self.assertEqual(rc_first["feature_cache"]["build_provenance"]["freshly_extracted_vectors"], 7)
                self.assertLessEqual(rc_first["feature_cache"]["build_provenance"]["donor_spot_check_max_abs_diff"], 1e-6)

                # Subsequent 5 runs loaded strictly from the v4-bound cache
                rc_second = json.loads((run_dirs[("V4-R2_RAW_WEIGHTED_CE", 42)] / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(rc_second["feature_cache"]["cache_action"], "LOADED_V4_BOUND_CACHE")

                # Verify inference adapter loads saved checkpoint without needing cache
                adapter = create_adapter(run_dirs[("V4-R1_RAW_CE", 42)] / "best_within_20.pt", device="cpu", allow_smoke=True)
                res_pred = adapter.predict(vl_rows[0]["path"])
                self.assertIn(res_pred["predicted_label"], {"PASS", "FAIL"})

                # Generate unified comparison report across all 6 runs (checks same-seed head & order sync + single-class slice warnings)
                comp_out = suite / "comparison"
                compare_v4_suite(run_dirs, comp_out)
                self.assertTrue((comp_out / "COMPARISON_V4_CN.md").is_file())
                self.assertTrue((comp_out / "v4_summary.json").is_file())
                self.assertTrue((comp_out / "v4_per_run_budget_metrics.csv").is_file())
                self.assertTrue((comp_out / "v4_val_34_per_image_comparison.csv").is_file())
                self.assertTrue((comp_out / "v4_val_34_grouped_slices.csv").is_file())
                self.assertTrue((comp_out / "curves_v4_6runs.png").is_file())

                md_text = (comp_out / "COMPARISON_V4_CN.md").read_text(encoding="utf-8")
                self.assertIn("SINGLE_CLASS_SLICE", md_text)

                # Verify tampering with v4 cache manifest raises V4_FEATURE_CACHE_IDENTITY_MISMATCH
                v4_sub = list(v4_cache_dir.iterdir())[0]
                cm_p = v4_sub / "cache_manifest.json"
                cm_d = json.loads(cm_p.read_text(encoding="utf-8"))
                cm_d["features_file_sha256"] = "0" * 64
                cm_p.write_text(json.dumps(cm_d), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "V4_FEATURE_CACHE_IDENTITY_MISMATCH"):
                    build_or_load_v4_feature_cache(
                        v4_cache_dir, tr_rows + vl_rows, dummy_model, base_cfg, torch.device("cpu"), None, pdoc
                    )

    def test_v4_runner_shell_syntax(self) -> None:
        script = BASE_DIR / "scripts/run_final_training_v4.sh"
        res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
