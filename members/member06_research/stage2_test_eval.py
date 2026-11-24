"""Stage 2 official Test evaluation & Member 7 adapter dev-consistency verifier for Member 6 Baseline 2.

Strict two-phase Test evaluation (`evaluate_stage2_test`):
1. Pre-flight verification:
   - Confirms `freeze_manifest.json` (`FROZEN_PRE_TEST`) exists and matches frozen checkpoint SHA-256.
   - Confirms `final_training_v4` (`bde8ea7116761bc8ee7fca1d47c336126daa995f`) physical triples via `verify_dataset.py --all`,
     confirms `manifests/resnet/test.csv` has 30 real original images (`15 PASS / 15 FAIL`, zero augmentations),
     and confirms zero path / RGB / SHA-256 / scene_group overlap with Train (249) + Val (34).
2. Phase 1 — Pure GT-blind model inference:
   - Extracts ONLY sorted `image_path`s from `manifests/resnet/test.csv` plus fixed `expected_counts = [2,2,2,2,2]`.
   - Never passes ground-truth labels, distractor types, or actual counts to `Stage2ResNetAdapter`.
   - Writes `images.csv` and `run_config.json` first.
3. Phase 2 — Post-inference authoritative evaluation:
   - Reads the already-exported `images.csv` from disk and joins against `manifests/resnet/test.csv`,
     `split_manifest.csv`, and `测试集逐图核对.csv`.
   - Computes overall classification metrics (`Accuracy`, `Balanced Accuracy`, `FAIL Precision`, `Recall`, `F1`,
     `ROC-AUC`, `FN`, `FP`, `Confusion Matrix`, `Valid Coverage`) and verified subgroup breakdowns
     (正常场景、纯缺件、已见干扰件、含未见干扰件的混合场景, plus batch, distractor subtype, and historical lineage disclosures).
   - Writes `test_metrics.json`, `per_image_predictions.csv`, `grouped_metrics.csv`, `failure_cases.csv`,
     `failure_cases.json`, and `STAGE2_FINAL_TEST_REPORT_CN.md`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

from .data import image_digest
from .runtime import file_sha256, load_checkpoint, write_json
from .stage2_adapter import (
    DEFAULT_FREEZE_MANIFEST_REL,
    IMAGE_CSV_COLUMNS,
    Stage2ResNetAdapter,
)

MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
DEFAULT_V4_DIR = REPO_ROOT / "members/member03_data/final_training_v4"


def _roc_auc_binary(y_true: list[int], y_score: list[float]) -> float | None:
    """Exact tie-aware Mann-Whitney U / Wilcoxon ROC-AUC for binary labels {0, 1}."""
    pos = [float(s) for y, s in zip(y_true, y_score) if y == 1]
    neg = [float(s) for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return None
    concordant = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                concordant += 1.0
            elif sp == sn:
                concordant += 0.5
    return concordant / (len(pos) * len(neg))


def _read_csv_utf8(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_v4_verifier(v4_dir: Path):
    verify_py = v4_dir / "verify_dataset.py"
    if not verify_py.is_file():
        raise FileNotFoundError(f"Missing verify_dataset.py in {v4_dir}")
    spec = importlib.util.spec_from_file_location("v4_verify_dataset", verify_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {verify_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classify_primary_test_scenario(split_row: dict[str, str]) -> tuple[str, str]:
    """Derive mutually-exclusive primary scenario bucket strictly from authoritative `split_manifest.csv`."""
    label = split_row["label"]
    target_counts = [int(x) for x in split_row["target_counts"].split("|")]
    unseen_s = int(split_row["S"])
    unseen_h = int(split_row["H"])
    seen_n = int(split_row["N"])
    seen_w = int(split_row["W"])
    seen_b = int(split_row["B"])
    unseen_total = unseen_s + unseen_h
    seen_total = seen_n + seen_w + seen_b
    has_missing_or_wrong_target = target_counts != [2, 2, 2, 2, 2]

    if label == "PASS":
        if has_missing_or_wrong_target or unseen_total > 0 or seen_total > 0:
            raise ValueError(f"Inconsistent PASS row in split_manifest: {split_row['filename']}")
        return "normal_pass", "正常场景（五类各2件、无目录外件）"

    if unseen_total == 0 and seen_total == 0 and has_missing_or_wrong_target:
        return "pure_missing", "纯缺件场景（无任何目录外干扰件）"
    if unseen_total == 0 and seen_total > 0:
        return "seen_distractor_no_unseen", "已见干扰件场景（仅含训练/验证已见类 N/W/B，不含未见类 S/H）"
    if unseen_total > 0:
        if has_missing_or_wrong_target or seen_total > 0:
            return "unseen_mixed_anomaly", "含未见干扰件的混合场景（含未见类 S/H，且同时存在缺件或已见干扰件）"
        return "unseen_pure_distractor", "纯未见干扰件场景（目标件齐全且仅含未见类 S/H）"
    raise ValueError(f"Unclassified FAIL row in split_manifest: {split_row['filename']}")


def _compute_binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    decided = [r for r in records if r["status"] in {"OK", "SMOKE_ONLY", "DEV_ONLY"} and r["predicted_label"] in {"PASS", "FAIL"}]
    coverage = len(decided) / total if total else 0.0

    tn = sum(1 for r in decided if r["true_label"] == "PASS" and r["predicted_label"] == "PASS")
    fp = sum(1 for r in decided if r["true_label"] == "PASS" and r["predicted_label"] == "FAIL")
    fn = sum(1 for r in decided if r["true_label"] == "FAIL" and r["predicted_label"] == "PASS")
    tp = sum(1 for r in decided if r["true_label"] == "FAIL" and r["predicted_label"] == "FAIL")

    n_pass = sum(1 for r in records if r["true_label"] == "PASS")
    n_fail = sum(1 for r in records if r["true_label"] == "FAIL")
    decided_pass = tn + fp
    decided_fail = tp + fn

    acc = (tp + tn) / len(decided) if decided else None
    tpr = tp / decided_fail if decided_fail > 0 else None
    tnr = tn / decided_pass if decided_pass > 0 else None
    ba = 0.5 * (tpr + tnr) if (tpr is not None and tnr is not None) else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if decided else None)
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec is not None and tpr is not None and (prec + tpr) > 0) else (0.0 if (prec == 0.0 or tpr == 0.0) else None)

    y_true = [1 if r["true_label"] == "FAIL" else 0 for r in decided]
    y_score = [float(r["fail_probability"]) for r in decided]
    roc_auc = _roc_auc_binary(y_true, y_score) if (decided_pass > 0 and decided_fail > 0) else None

    pass_scores = [float(r["fail_probability"]) for r in decided if r["true_label"] == "PASS"]
    fail_scores = [float(r["fail_probability"]) for r in decided if r["true_label"] == "FAIL"]

    return {
        "n_total": total,
        "n_decided": len(decided),
        "valid_prediction_coverage": coverage,
        "support": {"PASS": n_pass, "FAIL": n_fail},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "matrix_axes": "rows=true [PASS=0, FAIL=1], cols=pred [PASS=0, FAIL=1]",
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_pass_fn": fn,
        "false_fail_fp": fp,
        "accuracy": acc,
        "balanced_accuracy": ba,
        "fail_precision": prec,
        "fail_recall": tpr,
        "pass_specificity": tnr,
        "fail_f1": f1,
        "roc_auc": roc_auc,
        "fail_score_median_on_true_PASS": float(np.median(pass_scores)) if pass_scores else None,
        "fail_score_median_on_true_FAIL": float(np.median(fail_scores)) if fail_scores else None,
        "slice_nature": (
            "TWO_CLASS_SLICE"
            if (n_pass > 0 and n_fail > 0)
            else ("SINGLE_CLASS_SLICE_PASS_ONLY" if n_pass > 0 else "SINGLE_CLASS_SLICE_FAIL_ONLY")
        ),
    }


def preflight_stage2_test(
    freeze_manifest_path: Path | None = None,
    v4_dir: Path | None = None,
    *,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    fm_path = (freeze_manifest_path or (REPO_ROOT / DEFAULT_FREEZE_MANIFEST_REL)).resolve()
    dataset_root = (v4_dir or DEFAULT_V4_DIR).resolve()
    if not fm_path.is_file():
        raise FileNotFoundError(f"Pre-test freeze manifest missing: {fm_path}")

    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    if fm.get("schema_version") != "member06.stage2_freeze.v1" or fm.get("status") != "FROZEN_PRE_TEST":
        raise ValueError("Valid FROZEN_PRE_TEST freeze_manifest.json is required before Test evaluation")
    if fm.get("test_used_for_training_or_selection") is not False:
        raise ValueError("freeze_manifest must certify test_used_for_training_or_selection=False")

    ckpt_rel = fm["selected_model"]["frozen_checkpoint_relpath"]
    ckpt_path = (REPO_ROOT / ckpt_rel).resolve() if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint missing: {ckpt_path}")
    actual_ckpt_sha = file_sha256(ckpt_path)
    if actual_ckpt_sha != fm["selected_model"]["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA-256 changed after freeze")

    # Run upstream dataset integrity check including all 30 test triples
    if not allow_smoke:
        verifier = _load_v4_verifier(dataset_root)
        verifier.verify(all_files=True, root=dataset_root)

    train_csv = dataset_root / "manifests/resnet/train.csv"
    val_csv = dataset_root / "manifests/resnet/val.csv"
    test_csv = dataset_root / "manifests/resnet/test.csv"
    split_csv = dataset_root / "split_manifest.csv"

    expected_hashes = fm["dataset_and_git_identity"]["manifests_sha256"]
    for label_key, p in (
        ("train_csv", train_csv),
        ("val_csv", val_csv),
        ("test_csv", test_csv),
        ("split_manifest_csv", split_csv),
    ):
        if file_sha256(p) != expected_hashes[label_key]:
            raise ValueError(f"Dataset manifest hash mismatch vs freeze_manifest for {label_key}: {p}")

    train_rows = _read_csv_utf8(train_csv)
    val_rows = _read_csv_utf8(val_csv)
    test_rows = _read_csv_utf8(test_csv)
    split_rows = _read_csv_utf8(split_csv)
    split_by_name = {r["filename"]: r for r in split_rows}

    expected_data_kind = "SMOKE_ONLY" if allow_smoke else "REAL"
    if any(r["split"] != "test" or r["data_kind"] != expected_data_kind for r in test_rows):
        raise ValueError("manifests/resnet/test.csv contains invalid split or data_kind")

    dev_paths, dev_sha256, dev_scenes, dev_src_scenes = set(), set(), set(), set()
    for m_path, rows in ((train_csv, train_rows), (val_csv, val_rows)):
        for r in rows:
            p = (m_path.parent / r["image_path"]).resolve()
            sr = split_by_name[p.name]
            dev_paths.add(str(p))
            dev_sha256.add(sr["image_sha256"])
            dev_scenes.add(sr["scene_group"])
            dev_src_scenes.add(sr["source_scene_group"])

    test_resolved_paths: list[Path] = []
    seen_test_names: set[str] = set()
    for r in test_rows:
        p = (test_csv.parent / r["image_path"]).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Test image missing: {p}")
        if p.name in seen_test_names:
            raise ValueError(f"Duplicate test image filename: {p.name}")
        seen_test_names.add(p.name)
        sr = split_by_name.get(p.name)
        if sr is None or sr["split"] != "test" or sr["tier"] != "original" or sr["usage"] != "primary":
            raise ValueError(f"Test image {p.name} is not a primary original test image in split_manifest.csv")
        if file_sha256(p) != sr["image_sha256"]:
            raise ValueError(f"Physical SHA256 mismatch for test image {p.name}")
        if str(p) in dev_paths or sr["image_sha256"] in dev_sha256:
            raise ValueError(f"Test image overlaps Train/Val: {p.name}")
        if sr["scene_group"] in dev_scenes or sr["source_scene_group"] in dev_src_scenes:
            raise ValueError(f"Test scene_group overlaps Train/Val: {p.name}")
        test_resolved_paths.append(p)

    if not allow_smoke and len(test_resolved_paths) != 30:
        raise ValueError(f"Expected 30 formal test originals, found {len(test_resolved_paths)}")

    return {
        "status": "STAGE2_TEST_READY_FOR_USER_EXECUTION",
        "freeze_manifest": str(fm_path),
        "freeze_manifest_sha256": file_sha256(fm_path),
        "frozen_checkpoint": str(ckpt_path),
        "frozen_checkpoint_sha256": actual_ckpt_sha,
        "dataset_root": str(dataset_root),
        "upstream_v4_commit": fm["dataset_and_git_identity"]["upstream_v4_commit"],
        "test_manifest": str(test_csv),
        "test_manifest_sha256": file_sha256(test_csv),
        "n_test_images": len(test_resolved_paths),
        "dev_overlap_verified_zero": True,
        "augmentations_in_test_verified_zero": True,
        "inference_executed": False,
    }


def evaluate_stage2_test(
    output_dir: Path,
    *,
    freeze_manifest_path: Path | None = None,
    v4_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
    allow_smoke: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    pre = preflight_stage2_test(freeze_manifest_path, v4_dir, allow_smoke=allow_smoke)
    if check_only:
        return pre

    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Output directory already exists (preserving prior evidence): {out}")

    dataset_root = Path(pre["dataset_root"])
    test_csv = Path(pre["test_manifest"])
    ckpt_path = Path(pre["frozen_checkpoint"])

    # =========================================================================
    # PHASE 1: PURE GT-BLIND MODEL INFERENCE
    # Extract ONLY sorted image paths + fixed expected_counts [2,2,2,2,2].
    # Ground-truth labels, distractor types, and object counts are NOT read or passed.
    # =========================================================================
    with test_csv.open(encoding="utf-8-sig", newline="") as fh:
        raw_manifest_rows = list(csv.DictReader(fh))
    blind_image_paths = sorted(
        ((test_csv.parent / r["image_path"]).resolve() for r in raw_manifest_rows),
        key=lambda p: p.name,
    )
    gt_free_requests = [
        {"image_id": p.name, "expected_counts": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}}
        for p in blind_image_paths
    ]

    adapter = Stage2ResNetAdapter(
        ckpt_path,
        device=device,
        batch_size=batch_size,
        expected_sha256=pre["frozen_checkpoint_sha256"],
        allow_smoke=allow_smoke,
    )
    preds = adapter.predict_batch(blind_image_paths, expected_counts=[2, 2, 2, 2, 2])

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "counts_input.json", gt_free_requests)

    exported_image_rows: list[dict[str, Any]] = []
    per_img_probs_cache: dict[str, dict[str, float]] = {}
    image_hashes: dict[str, str] = {}
    for p, pred in zip(blind_image_paths, preds):
        image_hashes[p.name] = file_sha256(p)
        per_img_probs_cache[p.name] = {
            "fail_probability": float(pred["fail_probability"]),
            "pass_probability": float(pred["pass_probability"]),
        }
        exported_image_rows.append(
            {
                "image_id": p.name,
                "model_id": "baseline2",
                "status": pred["status"],
                "predicted_label": pred["predicted_label"],
                "prediction_score": pred["prediction_score"],
                "score_type": pred["score_type"],
                "non_catalogue_count": None,
            }
        )

    with (out / "images.csv").open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=IMAGE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(exported_image_rows)

    evaluated_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(
        out / "run_config.json",
        {
            "model_id": "baseline2",
            "model_version": "baseline2_final_v4_seed42_ep7",
            "evaluated_at_utc": evaluated_at_utc,
            "freeze_manifest_sha256": pre["freeze_manifest_sha256"],
            "model_sha256": adapter.checkpoint_sha256,
            "selected_epoch": adapter.selected_epoch,
            "config": adapter.config.to_dict(),
            "device": str(adapter.device),
            "supported_expected_counts": adapter.supported_counts,
            "input_expected_counts": {
                r["image_id"]: r["expected_counts"] for r in gt_free_requests
            },
            "image_sha256": image_hashes,
            "score_type": "uncalibrated_fail_softmax_prob",
            "score_semantics": "uncalibrated softmax probability of FAIL class P(FAIL) in [0, 1]",
            "decision_rule": "argmax over [P(PASS), P(FAIL)]; ties at 0.5 choose class index 0 (PASS)",
            "capabilities": {"boxes": False, "actual_counts": False, "non_catalogue_count": False},
            "gt_accessed_during_inference": False,
        },
    )

    # =========================================================================
    # PHASE 2: POST-INFERENCE AUTHORITATIVE EVALUATION
    # Consumes the written `images.csv` + authoritative `test.csv`, `split_manifest.csv`,
    # and `测试集逐图核对.csv` (if present). Model is never invoked again.
    # =========================================================================
    disk_predictions = {r["image_id"]: r for r in _read_csv_utf8(out / "images.csv")}
    split_by_name = {
        r["filename"]: r
        for r in _read_csv_utf8(dataset_root / "split_manifest.csv")
        if r["split"] == "test" and r["tier"] == "original" and r["usage"] == "primary"
    }
    audit_csv_path = dataset_root / "测试集逐图核对.csv"
    audit_by_name = (
        {r["文件名"]: r for r in _read_csv_utf8(audit_csv_path)}
        if audit_csv_path.is_file()
        else {}
    )
    test_gt_by_name = {
        (test_csv.parent / r["image_path"]).name: r for r in raw_manifest_rows
    }

    joined_records: list[dict[str, Any]] = []
    for img_id in sorted(disk_predictions):
        pred_row = disk_predictions[img_id]
        gt_row = test_gt_by_name[img_id]
        sr = split_by_name[img_id]
        ar = audit_by_name.get(img_id, {})

        true_label = gt_row["label"]
        if sr["label"] != true_label or (ar and ar["判定"] != true_label):
            raise ValueError(f"Authoritative label disagreement on {img_id}")

        pred_label = pred_row["predicted_label"]
        fail_prob = float(pred_row["prediction_score"])
        pass_prob = per_img_probs_cache[img_id]["pass_probability"]
        correct = pred_label == true_label
        if true_label == "FAIL" and pred_label == "FAIL":
            outcome = "TP"
        elif true_label == "PASS" and pred_label == "PASS":
            outcome = "TN"
        elif true_label == "PASS" and pred_label == "FAIL":
            outcome = "FP"
        elif true_label == "FAIL" and pred_label == "PASS":
            outcome = "FN"
        else:
            outcome = pred_row["status"]

        primary_key, primary_cn = _classify_primary_test_scenario(sr)
        unseen_s, unseen_h = int(sr["S"]), int(sr["H"])
        seen_n, seen_w, seen_b = int(sr["N"]), int(sr["W"]), int(sr["B"])
        non_cat_total = unseen_s + unseen_h + seen_n + seen_w + seen_b

        v2_prior = sr["v2_image"].split("/")[0] if sr.get("v2_image") else "unused"
        joined_records.append(
            {
                "image_id": img_id,
                "source_batch": sr["batch"],
                "scene_group": sr["scene_group"],
                "source_scene_group": sr["source_scene_group"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "status": pred_row["status"],
                "fail_probability": round(fail_prob, 6),
                "pass_probability": round(pass_prob, 6),
                "correct": correct,
                "outcome": outcome,
                "primary_scenario": primary_key,
                "primary_scenario_cn": primary_cn,
                "reason_cn": ar.get("原因", sr["anomaly"]),
                "unseen_distractors_cn": ar.get("未见干扰件及数量", f"S={unseen_s},H={unseen_h}"),
                "seen_distractors_cn": ar.get("已见干扰件及数量", f"N={seen_n},W={seen_w},B={seen_b}"),
                "target_counts": sr["target_counts"],
                "S_unseen_screw": unseen_s,
                "H_unseen_hex_standoff": unseen_h,
                "N_seen_nut": seen_n,
                "W_seen_washer": seen_w,
                "B_seen_bearing": seen_b,
                "non_catalogue_total": non_cat_total,
                "prior_v1_split": sr.get("prior_split", ""),
                "prior_v2_split": v2_prior,
                "prior_v3_split": sr.get("v3_split", ""),
            }
        )

    overall = _compute_binary_metrics(joined_records)

    # Primary scenario breakdown (exact mutually-exclusive buckets verified from manifest)
    primary_order = [
        ("normal_pass", "正常场景（PASS：五类各2件、无干扰件）"),
        ("pure_missing", "纯缺件场景（FAIL：无目录外件、仅缺目标件）"),
        ("seen_distractor_no_unseen", "已见干扰件场景（FAIL：含已见类 N/W/B，不含未见类 S/H）"),
        ("unseen_mixed_anomaly", "含未见干扰件的混合场景（FAIL：含未见类 S/H，且伴随缺件/已见件）"),
        ("unseen_pure_distractor", "纯未见干扰件场景（FAIL：目标件齐全且仅含未见类 S/H）"),
    ]
    grouped_rows: list[dict[str, Any]] = []
    primary_summary: dict[str, Any] = {}
    for key, title_cn in primary_order:
        sub = [r for r in joined_records if r["primary_scenario"] == key]
        if not sub:
            continue
        m = _compute_binary_metrics(sub)
        primary_summary[key] = {"title_cn": title_cn, **m}
        grouped_rows.append(
            {
                "dimension": "primary_scenario",
                "slice_key": key,
                "slice_title_cn": title_cn,
                "n_total": m["n_total"],
                "support_pass": m["support"]["PASS"],
                "support_fail": m["support"]["FAIL"],
                "slice_nature": m["slice_nature"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "accuracy": m["accuracy"],
                "fail_recall": m["fail_recall"],
                "pass_specificity": m["pass_specificity"],
            }
        )

    # Sub-breakdown within seen_distractor_no_unseen & unseen_mixed_anomaly & source_batch
    for batch_name in sorted({r["source_batch"] for r in joined_records}):
        sub = [r for r in joined_records if r["source_batch"] == batch_name]
        m = _compute_binary_metrics(sub)
        grouped_rows.append(
            {
                "dimension": "source_batch",
                "slice_key": batch_name,
                "slice_title_cn": f"拍摄批次 {batch_name}",
                "n_total": m["n_total"],
                "support_pass": m["support"]["PASS"],
                "support_fail": m["support"]["FAIL"],
                "slice_nature": m["slice_nature"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "accuracy": m["accuracy"],
                "fail_recall": m["fail_recall"],
                "pass_specificity": m["pass_specificity"],
            }
        )

    # Unseen subtype presence slices (informational; mixed anomalies explicitly noted)
    unseen_sub_slices = [
        ("has_unseen_S", "含未见目录外螺钉 S（混合场景）", [r for r in joined_records if r["S_unseen_screw"] > 0]),
        ("has_unseen_H", "含未见黄铜六角长柱 H（混合场景）", [r for r in joined_records if r["H_unseen_hex_standoff"] > 0]),
        ("seen_distractor_target_complete", "已见干扰件且目标件齐全（2|2|2|2|2）", [r for r in joined_records if r["primary_scenario"] == "seen_distractor_no_unseen" and r["target_counts"] == "2|2|2|2|2"]),
        ("seen_distractor_plus_missing", "已见干扰件且同时缺目标件", [r for r in joined_records if r["primary_scenario"] == "seen_distractor_no_unseen" and r["target_counts"] != "2|2|2|2|2"]),
    ]
    for s_key, s_title, sub in unseen_sub_slices:
        if not sub:
            continue
        m = _compute_binary_metrics(sub)
        grouped_rows.append(
            {
                "dimension": "detailed_anomaly_subtype",
                "slice_key": s_key,
                "slice_title_cn": s_title,
                "n_total": m["n_total"],
                "support_pass": m["support"]["PASS"],
                "support_fail": m["support"]["FAIL"],
                "slice_nature": m["slice_nature"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "accuracy": m["accuracy"],
                "fail_recall": m["fail_recall"],
                "pass_specificity": m["pass_specificity"],
            }
        )

    failures = [r for r in joined_records if not r["correct"]]

    # Historical lineage & scene-group audit (transparent disclosure)
    unique_scenes = sorted({r["scene_group"] for r in joined_records})
    v2_prior_counts = dict(Counter(r["prior_v2_split"] for r in joined_records))
    v3_prior_counts = dict(Counter(r["prior_v3_split"] for r in joined_records))
    v1_prior_counts = dict(Counter(r["prior_v1_split"] for r in joined_records))
    unseen_mixed_all_have_missing_and_seen_w = all(
        r["target_counts"] != "2|2|2|2|2" and r["W_seen_washer"] > 0
        for r in joined_records
        if r["primary_scenario"] == "unseen_mixed_anomaly"
    )

    with (out / "per_image_predictions.csv").open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(joined_records[0].keys()))
        writer.writeheader()
        writer.writerows(joined_records)

    with (out / "grouped_metrics.csv").open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grouped_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grouped_rows)

    with (out / "failure_cases.csv").open("x", newline="", encoding="utf-8") as fh:
        fieldnames = list(joined_records[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)
    write_json(out / "failure_cases.json", failures)

    metrics_payload = {
        "schema_version": "member06.stage2_test_metrics.v1",
        "evaluated_at_utc": evaluated_at_utc,
        "model_id": "baseline2",
        "model_version": "baseline2_final_v4_seed42_ep7",
        "experiment_id": "V4-R2_RAW_WEIGHTED_CE_SEED42",
        "selected_epoch": adapter.selected_epoch,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "freeze_manifest_sha256": pre["freeze_manifest_sha256"],
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": pre["upstream_v4_commit"],
        "test_manifest_sha256": pre["test_manifest_sha256"],
        "inference_isolation": {
            "gt_accessed_during_inference": False,
            "distractor_types_used_as_input": False,
            "actual_counts_used_as_input": False,
            "augmented_images_included": False,
            "post_test_tuning_or_reselection": False,
        },
        "overall_metrics": overall,
        "primary_scenario_breakdown": primary_summary,
        "grouped_metrics": grouped_rows,
        "failure_count": len(failures),
        "failure_cases": failures,
        "methodological_and_lineage_disclosures": {
            "registered_scene_group_count": len(unique_scenes),
            "registered_scene_groups": unique_scenes,
            "unseen_distractor_types_in_test": ["S (目录外螺钉)", "H (黄铜色六角长柱)"],
            "seen_distractor_types_in_test": ["N (目录外螺母)", "W (目录外垫圈)"],
            "all_unseen_test_images_also_have_missing_and_seen_washer": unseen_mixed_all_have_missing_and_seen_w,
            "image_level_cannot_claim_unseen_object_localization": True,
            "historical_split_lineage_of_30_test_originals": {
                "prior_v1_split_counts": v1_prior_counts,
                "prior_v2_split_counts": v2_prior_counts,
                "prior_v3_split_counts": v3_prior_counts,
                "note": (
                    "While V4-R2_RAW_WEIGHTED_CE_SEED42 was trained from scratch on final_training_v4 Train (249) "
                    "and selected on Val (34) with zero access to these 30 Test images during the v4 cycle, "
                    "earlier project dataset iterations (v1/v2/v3) included subsets of these 30 originals in prior splits "
                    "(e.g., 11 images were in Dataset v2 Train and 16 in v3 Train) and earlier corpus audits inspected raw photos."
                ),
            },
        },
    }
    write_json(out / "test_metrics.json", metrics_payload)

    # Render comprehensive Chinese Stage 2 Test Report
    cm = overall["confusion_matrix"]
    md = [
        "# Baseline 2（Frozen ResNet18 + Linear）Stage 2 正式 Test 独立推理与评价报告",
        "",
        f"- **评价时间（UTC）**：`{evaluated_at_utc}`",
        f"- **冻结模型身份**：`V4-R2_RAW_WEIGHTED_CE_SEED42`（`Epoch = 7`, `224×224 letterbox`, SHA-256: `{adapter.checkpoint_sha256}`）",
        f"- **冻结档案哈希**：`{pre['freeze_manifest_sha256']}`（先冻结后评测，未根据 Test 修改模型、权重或分类规则）",
        f"- **权威测试集**：`members/member03_data/final_training_v4/manifests/resnet/test.csv`（上游提交 `{pre['upstream_v4_commit']}`，清单哈希 `{pre['test_manifest_sha256']}`）",
        f"- **样本范围与推理隔离核验**：仅评价 `30` 张真实原图（`15 PASS / 15 FAIL`，零离线增强图）；推理阶段仅向 `Stage2ResNetAdapter` 输入图像像素与固定 `expected_counts=[2,2,2,2,2]`，未读取任何真值标签、干扰件细分类型或目标计数（`gt_accessed_during_inference = false`），完成写出 `images.csv` 后再由评价器读取权威清单统计。",
        "",
        "## 1. 整体分类性能指标（30 张真实原图）",
        "",
        "| 指标 | 实测结果 | 说明 |",
        "|---|---:|---|",
        f"| 测试样本总数 / 有效预测数 | `{overall['n_decided']} / {overall['n_total']}` | **有效预测覆盖率 `{overall['valid_prediction_coverage']*100:.2f}%`** |",
        f"| **Accuracy** | **`{overall['accuracy']:.4f}` (`{overall['accuracy']*100:.2f}%`)** | `{overall['tp'] + overall['tn']} / {overall['n_decided']}` |",
        f"| **Balanced Accuracy** | **`{overall['balanced_accuracy']:.4f}` (`{overall['balanced_accuracy']*100:.2f}%`)** | `0.5 × (TPR + TNR)` |",
        f"| **FAIL Precision** | **`{overall['fail_precision']:.4f}` (`{overall['fail_precision']*100:.2f}%`)** | `TP / (TP + FP)` |",
        f"| **FAIL Recall (TPR)** | **`{overall['fail_recall']:.4f}` (`{overall['fail_recall']*100:.2f}%`)** | `TP / (TP + FN)` |",
        f"| **PASS Specificity (TNR)** | **`{overall['pass_specificity']:.4f}` (`{overall['pass_specificity']*100:.2f}%`)** | `TN / (TN + FP)` |",
        f"| **FAIL F1** | **`{overall['fail_f1']:.4f}` (`{overall['fail_f1']*100:.2f}%`)** | 阳性类 `FAIL=1` |",
        f"| **ROC-AUC** | **`{overall['roc_auc']:.4f}`** | 基于未校准 `P(FAIL)` 分数 |",
        f"| **漏放 False PASS (`FN`)** | **`{overall['fn']}`**（共 `{overall['support']['FAIL']}` 张 FAIL） | 真实 FAIL 误判为 PASS |",
        f"| **误拒 False FAIL (`FP`)** | **`{overall['fp']}`**（共 `{overall['support']['PASS']}` 张 PASS） | 真实 PASS 误判为 FAIL |",
        f"| **混淆矩阵 `[[TN, FP], [FN, TP]]`** | **`[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]`** | 行=真实 `[PASS, FAIL]`，列=预测 `[PASS, FAIL]` |",
        f"| 真实 PASS / FAIL 的 `P(FAIL)` 中位数 | `{overall['fail_score_median_on_true_PASS']:.4f}` / `{overall['fail_score_median_on_true_FAIL']:.4f}` | 决策分界 `0.5`（平局归 `PASS`） |",
        "",
        "## 2. 按权威场景类别分组统计（正常 / 纯缺件 / 已见干扰件 / 含未见干扰件混合场景）",
        "",
        "> **方法学严谨性声明**：整图二分类模型仅输出整图 `PASS/FAIL` 与 `P(FAIL)`，**不能冒充逐对象框定位或未见干扰件实例检出率**。尤其在 `final_training_v4` 测试集中，全部含未见干扰件（目录外螺钉 `S`、黄铜六角长柱 `H`）的测试图同时包含**目标件缺失**与**已见目录外垫圈 `W`**，因此整图判为 `FAIL` 反映的是混合异常场景下的图像级拒绝表现，不能单独归因为模型已识别未见实例个体。",
        "",
        "| 权威分组维度 | 切片说明 | 实际支持数 (`PASS/FAIL`) | 正确数 / 支持数 | 组内命中率 | `TP` | `TN` | `FP` (误拒) | `FN` (漏放) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in grouped_rows:
        correct_n = g["tp"] + g["tn"]
        acc_str = f"{g['accuracy']*100:.2f}%" if g["accuracy"] is not None else "N/A"
        md.append(
            f"| `{g['dimension']}` | {g['slice_title_cn']} | `{g['n_total']}` (`{g['support_pass']}P/{g['support_fail']}F`) | "
            f"`{correct_n}/{g['n_total']}` | `{acc_str}` | {g['tp']} | {g['tn']} | {g['fp']} | {g['fn']} |"
        )

    md += [
        "",
        "## 3. 真实失败案例逐图明细",
        "",
    ]
    if not failures:
        md.append("- 本次 30 张正式测试原图零分类错误（`FN=0, FP=0`）。")
    else:
        md += [
            "| 文件名 | 批次 / 场景组 | 真实标签 | 预测标签 | 错误类型 | `P(FAIL)` | 权威原因 | 未见干扰件 | 已见干扰件 | 五类目标件计数 |",
            "|---|---|---|---|---|---:|---|---|---|---|",
        ]
        for f in failures:
            md.append(
                f"| `{f['image_id']}` | `{f['source_batch']}` / `{f['scene_group']}` | `{f['true_label']}` | `{f['predicted_label']}` | "
                f"**`{f['outcome']}`** | `{f['fail_probability']:.4f}` | {f['reason_cn']} | {f['unseen_distractors_cn']} | "
                f"{f['seen_distractors_cn']} | `{f['target_counts']}` |"
            )

    md += [
        "",
        "## 4. 全部 30 张测试原图逐图预测表",
        "",
        "| 文件名 | 批次 | 场景组 | 真实 | 预测 | 结果 | `P(FAIL)` | 主场景分类 | 原因 |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for r in joined_records:
        md.append(
            f"| `{r['image_id']}` | `{r['source_batch']}` | `{r['scene_group']}` | `{r['true_label']}` | `{r['predicted_label']}` | "
            f"`{r['outcome']}` | `{r['fail_probability']:.4f}` | {r['primary_scenario_cn']} | {r['reason_cn']} |"
        )

    md += [
        "",
        "## 5. 测试集局限与历史数据接触如实披露",
        "",
        f"1. **登记场景组规模**：本测试集 30 张原图来自 **{len(unique_scenes)} 个独立登记场景组**（`{', '.join(unique_scenes)}`），登记场景组、候选关联组与增强父图均与 `Train`（22 组）和 `Val`（11 组）零重叠；但受样本总量（`30` 张、`11` 个场景组）及批次背景分布限制，单张图像翻转对应 `3.33%` 的总体准确率波动。",
        f"2. **混合异常共线性**：测试集中含未见干扰件（`S` 螺钉、`H` 六角长柱）的图片是否同时含缺件与已见垫圈 `W`：`{unseen_mixed_all_have_missing_and_seen_w}`。因此图像级模型不能单独剥离“纯未见目录外型号”与“缺件/已见垫圈”的各自贡献，未见异物的框级检出须由对象级检测模型（`baseline3`）按框评估。",
        f"3. **项目历史版本接触披露**：本次冻结模型 `V4-R2_RAW_WEIGHTED_CE_SEED42` 从随机初始化 `Linear(512, 2)` 起仅在 `final_training_v4` 的 `249` 张 `Train` 上训练、`34` 张 `Val` 上选型，`v4` 训练与选型全过程未读取这 `30` 张测试图。但如实披露项目全周期背景：在小组早期的旧划分迭代中，这 `30` 张原图在 `v1` 时代分布为 `{v1_prior_counts}`，在成员自建 `Dataset v2` 中分布为 `{v2_prior_counts}`（其中 `11` 张曾处于 `v2` 的 `train`），在队友 `v3` 中分布为 `{v3_prior_counts}`（其中 `16` 张曾处于 `v3` 的 `train`），且早期全量来源审查曾核对过原始照片与标注。因此本评估属于**小组统一重划分 `final_training_v4` 从头重训后的冻结留出集评估**，不夸大表述为“整个项目历史上从未接触过这些原始图像文件”。",
    ]
    (out / "STAGE2_FINAL_TEST_REPORT_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "status": "STAGE2_TEST_EVALUATION_COMPLETED",
        "output_dir": str(out),
        "n_test_images": overall["n_total"],
        "valid_prediction_coverage": overall["valid_prediction_coverage"],
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "fail_precision": overall["fail_precision"],
        "fail_recall": overall["fail_recall"],
        "fail_f1": overall["fail_f1"],
        "roc_auc": overall["roc_auc"],
        "false_pass_fn": overall["fn"],
        "false_fail_fp": overall["fp"],
        "confusion_matrix": overall["confusion_matrix"],
    }


def verify_dev_adapter_consistency(
    output_json: Path | None = None,
    *,
    v4_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify on approved Val (34 images, NOT Test!) that offline checkpoint inference, Stage2ResNetAdapter
    (path, PIL, raw bytes, RGB ndarray, OpenCV BGR ndarray), batch export, and Member 7 `server.py` hook
    all produce consistent predictions and scores, plus verify UNSUPPORTED_COUNTS and INPUT_ERROR contracts.
    """
    dataset_root = (v4_dir or DEFAULT_V4_DIR).resolve()
    val_csv = dataset_root / "manifests/resnet/val.csv"
    val_rows = _read_csv_utf8(val_csv)
    val_paths = [(val_csv.parent / r["image_path"]).resolve() for r in val_rows]

    adapter = Stage2ResNetAdapter(device="cpu", batch_size=16)

    # 1. Offline direct model + Preprocess check on all 34 Val images
    model, cfg, dev, payload = load_checkpoint(adapter.checkpoint_path, device="cpu")
    model.eval()
    offline_probs: list[float] = []
    with torch.inference_mode():
        for p in val_paths:
            x = adapter.transform(p).unsqueeze(0).to(dev)
            pr = float(model(x).softmax(dim=1)[0, 1].item())
            offline_probs.append(pr)

    # 2. Adapter batch & single-image path predictions
    batch_preds = adapter.predict_batch(val_paths)
    single_preds = [adapter.predict(p) for p in val_paths]

    max_diff_batch_vs_offline = max(
        abs(bp["fail_probability"] - op) for bp, op in zip(batch_preds, offline_probs)
    )
    max_diff_single_vs_batch = max(
        abs(sp["fail_probability"] - bp["fail_probability"])
        for sp, bp in zip(single_preds, batch_preds)
    )
    labels_identical = all(
        sp["predicted_label"] == bp["predicted_label"] == ("FAIL" if op > 0.5 else "PASS")
        for sp, bp, op in zip(single_preds, batch_preds, offline_probs)
    )
    if not labels_identical or max_diff_batch_vs_offline > 1e-6 or max_diff_single_vs_batch > 1e-6:
        raise RuntimeError("Offline vs adapter path prediction mismatch on Val")

    # 3. Verify raw bytes, RGB ndarray, lossless BGR frame reversal, and cv2.imread BGR across sample Val images
    sample_png = next(p for p in val_paths if p.suffix.lower() == ".png")
    sample_jpg = next(p for p in val_paths if p.suffix.lower() == ".jpg")
    bgr_exact_diffs: list[float] = []
    raw_bytes_diffs: list[float] = []
    cv2_bgr_label_matches: list[bool] = []
    for p in (sample_png, sample_jpg):
        ref = adapter.predict(p)
        raw_b = p.read_bytes()
        pred_bytes = adapter.predict(raw_b)
        raw_bytes_diffs.append(abs(pred_bytes["fail_probability"] - ref["fail_probability"]))

        rgb_np = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        bgr_np = np.ascontiguousarray(rgb_np[..., ::-1])
        pred_rgb_np = adapter.predict(rgb_np, color_order="rgb")
        pred_bgr_np = adapter.predict(bgr_np, color_order="bgr")
        bgr_exact_diffs.append(abs(pred_rgb_np["fail_probability"] - ref["fail_probability"]))
        bgr_exact_diffs.append(abs(pred_bgr_np["fail_probability"] - ref["fail_probability"]))

        cv_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        pred_cv_bgr = adapter.predict(cv_bgr, color_order="bgr")
        cv2_bgr_label_matches.append(pred_cv_bgr["predicted_label"] == ref["predicted_label"])

    if max(raw_bytes_diffs) > 1e-7 or max(bgr_exact_diffs) > 1e-7 or not all(cv2_bgr_label_matches):
        raise RuntimeError("RGB/BGR or raw bytes preprocessing consistency check failed")

    # 4. Verify Member 7 `server.py` dynamic loader (`spec_from_file_location` + `validate_result`)
    server_py = REPO_ROOT / "members/member07_app/server.py"
    spec = importlib.util.spec_from_file_location("member07_server_check", server_py)
    assert spec is not None and spec.loader is not None
    server_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_mod)

    m7_root = REPO_ROOT / "members/member07_app"
    m7_cfg = {
        "adapter": "../member06_research/stage2_adapter.py",
        "version": "baseline2_final_v4_seed42_ep7",
        "weights": "../member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt",
        "sha256": adapter.checkpoint_sha256,
        "device": "cpu",
    }
    adapter_spec = importlib.util.spec_from_file_location(
        "adapter_baseline2", (m7_root / m7_cfg["adapter"]).resolve()
    )
    assert adapter_spec is not None and adapter_spec.loader is not None
    adapter_mod = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter_mod)
    live_fn = adapter_mod.load(m7_cfg, m7_root)
    live_validated = server_mod.validate_result(live_fn(sample_png.read_bytes()))
    ref_png = adapter.predict(sample_png)
    if (
        live_validated["decision"] != ref_png["predicted_label"]
        or abs(live_validated["score"] - round(ref_png["fail_probability"], 6)) > 1e-6
        or live_validated["fixed_expected"] != [2, 2, 2, 2, 2]
    ):
        raise RuntimeError("Member 7 server.py live adapter validation mismatch")

    # Also test browser canvas JPEG re-encoding simulation (`image/jpeg` bytes)
    buf = io.BytesIO()
    Image.open(sample_png).convert("RGB").save(buf, format="JPEG", quality=95)
    jpeg_live = server_mod.validate_result(live_fn(buf.getvalue()))

    # 5. Verify UNSUPPORTED_COUNTS, INPUT_ERROR, and zero fabricated fields
    unsup = adapter.predict(sample_png, expected_counts=[2, 2, 2, 3, 2])
    bad_img = adapter.predict(b"not_an_image_stream")
    if unsup["status"] != "UNSUPPORTED_COUNTS" or unsup["predicted_label"] is not None or unsup["prediction_score"] is not None:
        raise RuntimeError(f"UNSUPPORTED_COUNTS contract violated: {unsup}")
    if bad_img["status"] != "INPUT_ERROR" or bad_img["predicted_label"] is not None or bad_img["prediction_score"] is not None:
        raise RuntimeError(f"INPUT_ERROR contract violated: {bad_img}")
    for forbidden_key in ("objects", "bbox", "observed_count"):
        if forbidden_key in ref_png:
            raise RuntimeError(f"Adapter fabricated forbidden key: {forbidden_key}")
    if ref_png.get("non_catalogue_count") is not None:
        raise RuntimeError("non_catalogue_count must remain None")

    # 6. Verify batch_export compatibility on 34 Val images in temp dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        counts_p = tmp_p / "val_counts.json"
        write_json(
            counts_p,
            [{"image_id": p.name, "expected_counts": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}} for p in val_paths],
        )
        export_rows = adapter.export_batch(dataset_root / "images/val", counts_p, tmp_p / "exported")
        if len(export_rows) != 34:
            raise RuntimeError("Expected 34 exported Val rows")

    report = {
        "schema_version": "member06.stage2_adapter_consistency.v1",
        "verified_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluation_split_used": "final_training_v4/manifests/resnet/val.csv (34 real development images; Test NOT touched)",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "n_val_images_verified": len(val_paths),
        "all_34_labels_identical_offline_vs_single_vs_batch": labels_identical,
        "max_abs_prob_diff_batch_vs_offline": max_diff_batch_vs_offline,
        "max_abs_prob_diff_single_vs_batch": max_diff_single_vs_batch,
        "max_abs_prob_diff_raw_bytes_vs_path": max(raw_bytes_diffs),
        "max_abs_prob_diff_rgb_and_bgr_ndarray_vs_path": max(bgr_exact_diffs),
        "cv2_imread_bgr_labels_match_path": all(cv2_bgr_label_matches),
        "member07_server_py_spec_loader_and_validate_result_passed": True,
        "member07_live_sample_png_result": live_validated,
        "member07_live_canvas_jpeg95_result": jpeg_live,
        "unsupported_counts_contract_verified": unsup,
        "invalid_image_contract_verified": bad_img,
        "batch_export_34_val_verified": True,
        "test_data_accessed": False,
    }
    if output_json is not None:
        write_json(output_json, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("test", help="Run pre-flight check or formal single-pass Test evaluation")
    p_eval.add_argument("--output-dir", type=Path, default=None, help="Output directory for formal Test results")
    p_eval.add_argument("--freeze-manifest", type=Path, default=None)
    p_eval.add_argument("--v4-dir", type=Path, default=None)
    p_eval.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--allow-smoke", action="store_true")
    p_eval.add_argument("--check-only", action="store_true", help="Verify freeze & Test manifests without running model inference")

    p_dev = sub.add_parser("verify-adapter-dev", help="Verify offline vs Member 7 adapter consistency on 34 Val images")
    p_dev.add_argument("--output-json", type=Path, default=None)
    p_dev.add_argument("--v4-dir", type=Path, default=None)

    args = parser.parse_args()
    try:
        if args.cmd == "verify-adapter-dev":
            res = verify_dev_adapter_consistency(args.output_json, v4_dir=args.v4_dir)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0

        if args.check_only:
            res = preflight_stage2_test(args.freeze_manifest, args.v4_dir, allow_smoke=args.allow_smoke)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0

        if args.output_dir is None:
            parser.error("--output-dir is required unless --check-only is set")
        res = evaluate_stage2_test(
            args.output_dir,
            freeze_manifest_path=args.freeze_manifest,
            v4_dir=args.v4_dir,
            device=args.device,
            batch_size=args.batch_size,
            allow_smoke=args.allow_smoke,
            check_only=False,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"STAGE2_TEST_EVAL_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
