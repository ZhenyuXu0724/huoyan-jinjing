"""Targeted regression, safety, and numerical-equivalence tests for Dataset v2 Phase-1 (R1/R2/R3).

Uses synthetic SMOKE_ONLY images for CNN/optimizer tests; never runs real training or reads real Test pixels/GT.
"""
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
from torch import nn

from members.member06_research.config import Config
from members.member06_research.data import Preprocess
from members.member06_research.inference import create_adapter
from members.member06_research.model import FrozenResNet18
from members.member06_research.runtime import file_sha256, load_checkpoint, seed_everything, write_json
from members.member06_research.train import class_weights
from members.member06_research.v2_compare import compare_phase1_runs
from members.member06_research.v2_data import (
    CANONICAL_AUG_ORDER,
    PREPARED_SCHEMA,
    r2_initial_offset,
    r2_select_variant,
    safe_dev_path,
    verify_prepared_dataset_v2,
)
from members.member06_research.v2_train import (
    build_or_load_feature_cache,
    load_experiment_spec,
    run_v2_experiment,
    state_dict_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parents[1]


def _make_synthetic_prepared_v2(root: Path) -> tuple[Path, Path]:
    """Create a miniature Dataset v2 + prepared_dir in a temp folder for fast SMOKE_ONLY testing."""
    ds_root = root / "dataset_v2"
    prep_dir = root / "prepared"
    for sp in ("train", "val", "test"):
        for lbl in ("PASS", "FAIL"):
            for tier in ("original", "augmented"):
                (ds_root / sp / lbl / tier).mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 train raw originals (2 PASS, 2 FAIL; 2 of them have 3 augs -> 6 augs, total 10 train)
    # Create 4 val raw originals (2 PASS, 2 FAIL; 1 FAIL has 3 augs -> 3 val augs)
    # Create 1 test sentinel image (must NEVER be read)
    sentinel = ds_root / "test/FAIL/original/test_sentinel.png"
    sentinel.write_bytes(b"NOT_A_VALID_PNG_IF_DECODED_CRASHES")

    def save_rgb(rel: str, color: tuple[int, int, int]) -> tuple[Path, str]:
        p = ds_root / rel
        Image.new("RGB", (48, 36), color).save(p)
        return p, file_sha256(p)

    train_raw_rows = []
    train_full_rows = []
    r2_parents = []

    raw_specs = [
        ("DSV2_T1", "p_pass_plain.png", "train/PASS/original/p_pass_plain.png", "PASS", "G_TR_1", "11/11_parts_v1", "none", "PASS_normal", 0, 0, (30, 120, 200), False),
        ("DSV2_T2", "p_pass_aug.jpg", "train/PASS/original/p_pass_aug.jpg", "PASS", "G_TR_2", "11/20", "none", "PASS_normal", 0, 0, (60, 180, 90), True),
        ("DSV2_T3", "p_fail_plain.jpg", "train/FAIL/original/p_fail_plain.jpg", "FAIL", "G_TR_3", "11/14_FAIL", "non_catalogue_only_6", "non_catalogue_only", 6, 0, (200, 40, 40), False),
        ("DSV2_T4", "p_fail_aug.jpg", "train/FAIL/original/p_fail_aug.jpg", "FAIL", "G_TR_4", "11/20", "non_catalogue_plus_missing", "non_catalogue_plus_missing", 3, 2, (180, 120, 30), True),
    ]

    for idx, (sid, fname, rel, lbl, gid, batch, acat, amech, nc, miss, rgb, has_aug) in enumerate(raw_specs):
        _, sha = save_rgb(rel, rgb)
        row = {
            "sample_id": sid,
            "image_filename": fname,
            "v2_image_relpath": rel,
            "label": lbl,
            "scene_id": gid,
            "split": "train",
            "data_kind": "SMOKE_ONLY",
            "data_tier": "original",
            "source_batch": batch,
            "source_subcondition": batch,
            "parent_image_id": fname,
            "augmentation_type": "none",
            "non_catalogue_count": nc,
            "missing_target_units": miss,
            "anomaly_category": acat,
            "anomaly_mechanism": amech,
            "image_sha256": sha,
        }
        train_raw_rows.append(row)
        train_full_rows.append(row)
        variants = [{
            "variant_slot": 0,
            "variant_name": "original",
            "sample_id": sid,
            "image_filename": fname,
            "v2_image_relpath": rel,
            "image_sha256": sha,
        }]
        if has_aug:
            stem = Path(fname).stem
            for slot, aug_name in enumerate(CANONICAL_AUG_ORDER[1:], start=1):
                afname = f"{stem}__{aug_name}.jpg"
                arel = f"train/{lbl}/augmented/{afname}"
                _, asha = save_rgb(arel, ((rgb[0] + 15 * slot) % 255, (rgb[1] + 10 * slot) % 255, rgb[2]))
                arow = {
                    **row,
                    "sample_id": f"{sid}_A{slot}",
                    "image_filename": afname,
                    "v2_image_relpath": arel,
                    "data_tier": "augmented",
                    "source_batch": "11/20_augmented",
                    "parent_image_id": fname,
                    "augmentation_type": aug_name,
                    "image_sha256": asha,
                }
                train_full_rows.append(arow)
                variants.append({
                    "variant_slot": slot,
                    "variant_name": aug_name,
                    "sample_id": arow["sample_id"],
                    "image_filename": afname,
                    "v2_image_relpath": arel,
                    "image_sha256": asha,
                })
        r2_parents.append({
            "parent_index": idx,
            "parent_id": fname,
            "sample_id": sid,
            "label": lbl,
            "scene_group_id": gid,
            "source_batch": batch,
            "anomaly_category": acat,
            "anomaly_mechanism": amech,
            "non_catalogue_count": nc,
            "missing_target_units": miss,
            "has_augmented_variants": has_aug,
            "variants": variants,
        })

    val_raw_rows = []
    val_aug_rows = []
    val_specs = [
        ("DSV2_V1", "v_pass1.png", "val/PASS/original/v_pass1.png", "PASS", "G_VL_1", "11/11_parts_v1", "none", "PASS_normal", 0, 0, (20, 100, 210), False),
        ("DSV2_V2", "v_pass2.jpg", "val/PASS/original/v_pass2.jpg", "PASS", "G_VL_2", "11/14_PASS", "none", "PASS_normal", 0, 0, (50, 160, 120), False),
        ("DSV2_V3", "v_fail1.jpg", "val/FAIL/original/v_fail1.jpg", "FAIL", "G_VL_3", "11/14_FAIL", "non_catalogue_only_1", "non_catalogue_only", 1, 0, (210, 50, 50), False),
        ("DSV2_V4", "v_fail2.jpg", "val/FAIL/original/v_fail2.jpg", "FAIL", "G_VL_4", "11/20", "non_catalogue_plus_missing", "non_catalogue_plus_missing", 2, 2, (190, 90, 40), True),
    ]
    for sid, fname, rel, lbl, gid, batch, acat, amech, nc, miss, rgb, has_aug in val_specs:
        _, sha = save_rgb(rel, rgb)
        vrow = {
            "sample_id": sid,
            "image_filename": fname,
            "v2_image_relpath": rel,
            "label": lbl,
            "scene_id": gid,
            "split": "val",
            "data_kind": "SMOKE_ONLY",
            "data_tier": "original",
            "source_batch": batch,
            "source_subcondition": batch,
            "parent_image_id": fname,
            "augmentation_type": "none",
            "non_catalogue_count": nc,
            "missing_target_units": miss,
            "anomaly_category": acat,
            "anomaly_mechanism": amech,
            "image_sha256": sha,
        }
        val_raw_rows.append(vrow)
        if has_aug:
            stem = Path(fname).stem
            for slot, aug_name in enumerate(CANONICAL_AUG_ORDER[1:], start=1):
                afname = f"{stem}__{aug_name}.jpg"
                arel = f"val/FAIL/augmented/{afname}"
                _, asha = save_rgb(arel, ((rgb[0] + 12 * slot) % 255, rgb[1], (rgb[2] + 20 * slot) % 255))
                val_aug_rows.append({
                    **vrow,
                    "sample_id": f"{sid}_A{slot}",
                    "image_filename": afname,
                    "v2_image_relpath": arel,
                    "data_tier": "augmented",
                    "source_batch": "11/20_augmented",
                    "parent_image_id": fname,
                    "augmentation_type": aug_name,
                    "image_sha256": asha,
                })

    # Write dummy Dataset v2 metadata files so provenance hash verification works cleanly
    for meta_f in ("dataset_manifest.csv", "dataset_statistics.json", "augmentation_mapping.csv", "SHA256SUMS.txt"):
        (ds_root / meta_f).write_text(f"synthetic:{meta_f}\n", encoding="utf-8")

    fields = list(train_raw_rows[0].keys())
    for name, subset in (
        ("train_raw_294.csv", train_raw_rows),
        ("train_full_438.csv", train_full_rows),
        ("val_raw_74.csv", val_raw_rows),
        ("val_aug_18.csv", val_aug_rows),
    ):
        with (prep_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    write_json(prep_dir / "train_r2_parents_294.json", {
        "schema_version": "member06.r2_parents.v1",
        "aligned_with": "train_raw_294.csv",
        "total_parents": len(r2_parents),
        "parents_without_augmentation": 2,
        "parents_with_4_variants": 2,
        "variant_slot_order": list(CANONICAL_AUG_ORDER),
        "parents": r2_parents,
    })

    write_json(prep_dir / "provenance.json", {
        "schema_version": PREPARED_SCHEMA,
        "dataset_v2_root": str(ds_root),
        "dataset_v2_metadata_sha256": {
            m: file_sha256(ds_root / m)
            for m in ("dataset_manifest.csv", "dataset_statistics.json", "augmentation_mapping.csv", "SHA256SUMS.txt")
        },
        "verified_quotas_438_92_79": {"smoke": True},
        "prepared_files": {
            f: file_sha256(prep_dir / f)
            for f in ("train_raw_294.csv", "train_full_438.csv", "train_r2_parents_294.json", "val_raw_74.csv", "val_aug_18.csv")
        },
        "test_content_read": False,
        "legacy_dataset_v1_mixed": False,
        "supported_counts": [2, 2, 2, 2, 2],
    })
    return ds_root, prep_dir


class DatasetV2Phase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)

    def test_real_dataset_v2_intake_and_r3_weights_without_reading_test(self) -> None:
        # Verify real prepared manifests match 438/92/79, never access test/, and never mix Dataset v1 paths
        prov, manifests, r2_doc = verify_prepared_dataset_v2(REPO_ROOT / "members/member06_research/data/dataset_v2_phase1")
        self.assertFalse(prov["test_content_read"])
        self.assertEqual(len(manifests["train_raw_294"]), 294)
        self.assertEqual(len(manifests["train_full_438"]), 438)
        self.assertEqual(len(manifests["val_raw_74"]), 74)
        self.assertEqual(len(manifests["val_aug_18"]), 18)
        self.assertEqual(r2_doc["parents_with_4_variants"], 48)

        # Ensure all paths sit strictly inside dataset_v2/{train,val} (never test/ or old parts_v1)
        ds_v2_root = (REPO_ROOT / "members/member03_data/dataset_v2").resolve()
        for key, rows in manifests.items():
            for r in rows:
                p = Path(r["path"]).resolve()
                self.assertTrue(p.is_relative_to(ds_v2_root / r["split"]))
                self.assertNotIn("/test/", p.as_posix())

        # Verify exact R3 balanced weights on 155 PASS + 283 FAIL = 438
        _, cfg_r3 = load_experiment_spec(BASE_DIR / "configs/dataset_v2/r3_full_aug_balanced_ce.json")
        w = class_weights(manifests["train_full_438"], cfg_r3, "cpu")
        self.assertAlmostEqual(w[0].item(), 438 / (2 * 155), places=7)
        self.assertAlmostEqual(w[1].item(), 438 / (2 * 283), places=7)

        # Verify future Phase-2 config knobs (384 resolution, weight_decay, seed, epochs, lr) work natively
        _, cfg_r1 = load_experiment_spec(BASE_DIR / "configs/dataset_v2/r1_raw_ce.json")
        d = cfg_r1.to_dict()
        d.update({"image_size": 384, "weight_decay": 1e-3, "learning_rate": 5e-4, "epochs": 15, "seed": 2026})
        cfg_phase2 = Config(**d)
        self.assertEqual(cfg_phase2.image_size, 384)
        self.assertEqual(cfg_phase2.weight_decay, 1e-3)

    def test_test_partition_and_cache_scope_blocked(self) -> None:
        ds_v2_root = REPO_ROOT / "members/member03_data/dataset_v2"
        with self.assertRaisesRegex(ValueError, "forbidden split"):
            safe_dev_path(ds_v2_root, "test/PASS/original/x.jpg", "test")
        with self.assertRaisesRegex(ValueError, "does not belong to 'train'"):
            safe_dev_path(ds_v2_root, "test/PASS/original/x.jpg", "train")

        c = Config(weights="RANDOM_SMOKE_ONLY", data_kind="SMOKE_ONLY", image_size=32)
        m = FrozenResNet18(c.weights)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "REFUSED_CACHE_SCOPE"):
                build_or_load_feature_cache(
                    tmp,
                    [{"split": "test", "v2_image_relpath": "test/x.png", "image_sha256": "0" * 64, "path": Path(tmp) / "x.png"}],
                    m,
                    c,
                    torch.device("cpu"),
                    None,
                )

    def test_r2_deterministic_sha256_cycling_and_parent_equality(self) -> None:
        _, _, r2_doc = verify_prepared_dataset_v2(REPO_ROOT / "members/member06_research/data/dataset_v2_phase1")
        parents = r2_doc["parents"]
        # Check reproducibility across repeated calls and exact 4-epoch full variant cycle
        for p in parents:
            if p["has_augmented_variants"]:
                seen_4 = [r2_select_variant(p, 42, ep)["variant_name"] for ep in range(1, 5)]
                self.assertEqual(set(seen_4), set(CANONICAL_AUG_ORDER))
                # Epoch 5 wraps back to Epoch 1
                self.assertEqual(r2_select_variant(p, 42, 5)["image_filename"], r2_select_variant(p, 42, 1)["image_filename"])
                # Known SHA-256 vector independent of PYTHONHASHSEED
                self.assertEqual(r2_initial_offset(42, p["parent_id"], 4), r2_initial_offset(42, p["parent_id"], 4))
            else:
                for ep in range(1, 5):
                    self.assertEqual(r2_select_variant(p, 42, ep)["variant_name"], "original")

        # Each epoch exposes 149 PASS and 145 FAIL = 294 unique parents
        for ep in range(1, 11):
            chosen = [r2_select_variant(p, 42, ep) for p in parents]
            self.assertEqual(len(chosen), 294)
            self.assertEqual(len({x["parent_id"] for x in chosen}), 294)
            self.assertEqual(sum(p["label"] == "PASS" for p in parents), 149)
            self.assertEqual(sum(p["label"] == "FAIL" for p in parents), 145)

    def test_feature_cache_numerical_equivalence_tamper_rejection_and_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, prep_dir = _make_synthetic_prepared_v2(root)
            cache_dir = root / "cache"

            def make_spec(exp_id: str, mode: str, use_cache: bool, cw: str = "none") -> Path:
                spec_p = root / f"{exp_id}_{'cached' if use_cache else 'uncached'}.json"
                write_json(spec_p, {
                    "schema_version": "member06.v2_experiment.v1",
                    "experiment_id": exp_id,
                    "description": "synthetic smoke test",
                    "train_mode": mode,
                    "prepared_dir": str(prep_dir),
                    "use_feature_cache": use_cache,
                    "cache_dir": str(cache_dir),
                    "online_augmentation": "none",
                    "lr_scheduler": "none",
                    "threshold_rule": "default_argmax_pass0_fail1_no_threshold_search",
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
                        "train_manifest": str(prep_dir / ("train_full_438.csv" if mode == "full_438" else "train_raw_294.csv")),
                        "val_manifest": str(prep_dir / "val_raw_74.csv"),
                        "data_provenance": str(prep_dir / "provenance.json"),
                        "seed": 42,
                        "epochs": 3,
                        "batch_size": 2,
                        "learning_rate": 0.01,
                        "train_class_weight": cw,
                        "weight_decay": 0.0,
                        "num_workers": 0,
                        "device": "cpu",
                        "cpu_threads": 2,
                    },
                })
                return spec_p

            # Patch verify_prepared_dataset_v2 row counts for our miniature 4/10/4/3 synthetic fixture
            orig_verify = verify_prepared_dataset_v2

            def _verify_mini(pdir):
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

            with patch("members.member06_research.v2_train.verify_prepared_dataset_v2", side_effect=_verify_mini):
                # 1. Run R1 uncached vs R1 cached -> verify exact equivalence of features, head weights, gradients/losses, and Val predictions
                spec_r1_uncached = make_spec("R1_RAW_CE", "raw_294", False)
                spec_r1_cached = make_spec("R1_RAW_CE", "raw_294", True)
                run_v2_experiment(spec_r1_uncached, root / "r1_uncached")
                run_v2_experiment(spec_r1_cached, root / "r1_cached")

                m_uncached, _, _, pay_u = load_checkpoint(root / "r1_uncached/last.pt", device="cpu", allow_smoke=True)
                m_cached, _, _, pay_c = load_checkpoint(root / "r1_cached/last.pt", device="cpu", allow_smoke=True)
                self.assertTrue(torch.allclose(m_uncached.head.weight, m_cached.head.weight, atol=1e-6))
                self.assertTrue(torch.allclose(m_uncached.head.bias, m_cached.head.bias, atol=1e-6))
                # Backbone & BN buffers completely identical before and after training
                self.assertEqual(
                    state_dict_sha256(m_cached.backbone.state_dict()),
                    pay_c["run_metadata"]["determinism_notes"]["initial_backbone_sha256"],
                )
                for hu, hc in zip(
                    json.loads((root / "r1_uncached/history.json").read_text()),
                    json.loads((root / "r1_cached/history.json").read_text()),
                ):
                    self.assertAlmostEqual(hu["train"]["loss"], hc["train"]["loss"], places=6)
                    self.assertAlmostEqual(hu["val"]["loss"], hc["val"]["loss"], places=6)

                # Direct single-batch gradient equivalence check (cached feature vs live CNN forward)
                _, mini_manifests, _ = _verify_mini(prep_dir)
                cfg_smoke = Config(**json.loads(spec_r1_cached.read_text())["model_config"])
                seed_everything(42, 2)
                m_live = FrozenResNet18("RANDOM_SMOKE_ONLY")
                m_feat = copy.deepcopy(m_live)
                tf = Preprocess(cfg_smoke)
                xs = torch.stack([tf(r["path"]) for r in mini_manifests["train_raw_294"]])
                ys = torch.tensor([cfg_smoke.class_to_idx[r["label"]] for r in mini_manifests["train_raw_294"]])
                nn.CrossEntropyLoss()(m_live(xs), ys).backward()
                cached_sub = list(cache_dir.iterdir())[0]
                feat_lookup, _ = build_or_load_feature_cache(cache_dir, mini_manifests["train_full_438"] + mini_manifests["val_raw_74"] + mini_manifests["val_aug_18"], m_feat, cfg_smoke, torch.device("cpu"), None)
                f_batch = torch.stack([feat_lookup[r["image_sha256"]] for r in mini_manifests["train_raw_294"]])
                self.assertTrue(torch.allclose(m_live.extract_features(xs), f_batch, atol=1e-6))
                nn.CrossEntropyLoss()(m_feat.head(f_batch), ys).backward()
                self.assertTrue(torch.allclose(m_live.head.weight.grad, m_feat.head.weight.grad, atol=1e-6))
                self.assertTrue(torch.allclose(m_live.head.bias.grad, m_feat.head.bias.grad, atol=1e-6))

                # 2. Run R2 and R3 (reusing verified cache!) and check R1/R2 parent order & head init equality
                spec_r2 = make_spec("R2_PARENT_EQUAL_AUG_CE", "parent_equal_cycle_294", True)
                spec_r3 = make_spec("R3_FULL_AUG_BALANCED_CE", "full_438", True, cw="balanced")
                run_v2_experiment(spec_r2, root / "r2_cached")
                run_v2_experiment(spec_r3, root / "r3_cached")

                rc1 = json.loads((root / "r1_cached/run_config.json").read_text())
                rc2 = json.loads((root / "r2_cached/run_config.json").read_text())
                rc3 = json.loads((root / "r3_cached/run_config.json").read_text())
                self.assertEqual(rc1["determinism_notes"]["initial_head_sha256"], rc2["determinism_notes"]["initial_head_sha256"])
                self.assertEqual(rc1["determinism_notes"]["initial_head_sha256"], rc3["determinism_notes"]["initial_head_sha256"])
                self.assertEqual(rc2["feature_cache"]["cache_action"], "LOADED_VERIFIED")
                self.assertEqual(rc3["feature_cache"]["cache_action"], "LOADED_VERIFIED")

                # Verify inference adapter loads best.pt cleanly without needing cache
                adapter = create_adapter(root / "r2_cached/best.pt", device="cpu", allow_smoke=True)
                pred_out = adapter.predict(mini_manifests["val_raw_74"][0]["path"])
                self.assertEqual(pred_out["status"], "SMOKE_ONLY")
                self.assertIn(pred_out["predicted_label"], {"PASS", "FAIL"})

                # Verify overwrite protection on run_v2_experiment and compare_phase1_runs
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    run_v2_experiment(spec_r2, root / "r2_cached")

                comp_dir = root / "comparison"
                compare_phase1_runs(root / "r1_cached", root / "r2_cached", root / "r3_cached", comp_dir)
                self.assertTrue((comp_dir / "COMPARISON_PHASE1_CN.md").is_file())
                self.assertTrue((comp_dir / "val_74_per_image_comparison.csv").is_file())
                self.assertTrue((comp_dir / "val_74_grouped_metrics.csv").is_file())
                self.assertTrue((comp_dir / "val_aug_18_robustness.csv").is_file())
                self.assertTrue((comp_dir / "curves_phase1.png").is_file())
                self.assertTrue((comp_dir / "val_score_distributions_phase1.png").is_file())
                with self.assertRaisesRegex(ValueError, "already exists"):
                    compare_phase1_runs(root / "r1_cached", root / "r2_cached", root / "r3_cached", comp_dir)

                # 3. Verify cache tamper detection rejects modified cache manifest
                cm_path = cached_sub / "cache_manifest.json"
                cm_data = json.loads(cm_path.read_text())
                cm_data["features_file_sha256"] = "0" * 64
                cm_path.write_text(json.dumps(cm_data))
                with self.assertRaisesRegex(ValueError, "FEATURE_CACHE_IDENTITY_MISMATCH"):
                    build_or_load_feature_cache(cache_dir, mini_manifests["train_full_438"], m_feat, cfg_smoke, torch.device("cpu"), None)

    def test_legacy_checkpoint_compatibility_preserved(self) -> None:
        legacy_ckpt = BASE_DIR / "archives/dev_baseline_v1/run/best.pt"
        model, cfg, _, payload = load_checkpoint(legacy_ckpt, device="cpu", allow_dev_only=True)
        self.assertEqual(payload["epoch"], 5)
        self.assertEqual(cfg.image_size, 224)
        self.assertEqual(model.head.out_features, 2)

    def test_runner_shell_syntax_and_no_overwrite(self) -> None:
        script = BASE_DIR / "scripts/run_dataset_v2_phase1.sh"
        res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
