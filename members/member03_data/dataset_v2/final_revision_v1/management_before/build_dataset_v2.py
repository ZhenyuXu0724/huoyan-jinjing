#!/usr/bin/env python3
"""Reproducible builder for members/member03_data/dataset_v2.

Hard guarantees:
1. Byte-for-byte physical copy (shutil.copy2) of 609 images + 609 Labelme JSONs (1,218 data files).
2. Zero modifications to source directories, image pixels, filenames, or JSON contents.
3. Strict parent-child split inheritance for all 210 augmented images.
4. Explicit PROVISIONAL_HOLDOUT status on test partition and honest provenance/limitation reporting.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

DATASET_V2_DIR = Path(__file__).resolve().parent
MEMBER03_ROOT = DATASET_V2_DIR.parent

LABEL_TO_ID = {
    "non_catalogue": 0,
    "M4x25": 1,
    "M5x10": 2,
    "M4x10": 3,
    "M4_hex_nut": 4,
    "flat_washer": 5,
}
REFERENCE_EXPECTED = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}

EXPECTED_QUOTAS = {
    ("train", "PASS", "original"): 151,
    ("train", "FAIL", "original"): 145,
    ("train", "PASS", "augmented"): 6,
    ("train", "FAIL", "augmented"): 138,
    ("val", "PASS", "original"): 37,
    ("val", "FAIL", "original"): 37,
    ("val", "PASS", "augmented"): 0,
    ("val", "FAIL", "augmented"): 18,
    ("test", "PASS", "original"): 13,
    ("test", "FAIL", "original"): 16,
    ("test", "PASS", "augmented"): 0,
    ("test", "FAIL", "augmented"): 48,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hms_to_seconds(stem: str) -> int:
    parts = stem.split("_")
    if len(parts) < 3 or len(parts[2]) != 6 or not parts[2].isdigit():
        raise ValueError(f"cannot parse timestamp from {stem}")
    hms = parts[2]
    return int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])


def group_by_time_gap(items: list[dict], gap_sec: int) -> list[list[dict]]:
    ordered = sorted(items, key=lambda x: (hms_to_seconds(x["stem"]), x["image_filename"]))
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    for item in ordered:
        t = hms_to_seconds(item["stem"])
        if cur and t - hms_to_seconds(cur[-1]["stem"]) >= gap_sec:
            blocks.append(cur)
            cur = [item]
        else:
            cur.append(item)
    if cur:
        blocks.append(cur)
    return blocks


def parse_labelme(img_path: Path, json_path: Path) -> dict:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(raw["imageWidth"])
    height = int(raw["imageHeight"])
    counts = {i: 0 for i in range(6)}
    norm_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    has_auto_review_flag = False

    for shape in raw.get("shapes", []):
        lbl = shape.get("label", "").strip()
        if lbl not in LABEL_TO_ID:
            raise ValueError(f"unknown label {lbl!r} in {json_path}")
        if shape.get("shape_type") != "rectangle":
            raise ValueError(f"unsupported shape_type {shape.get('shape_type')!r} in {json_path}")
        pts = shape.get("points", [])
        if len(pts) != 2 or any(len(p) != 2 for p in pts):
            raise ValueError(f"invalid rectangle points in {json_path}")
        x1, x2 = sorted((float(pts[0][0]), float(pts[1][0])))
        y1, y2 = sorted((float(pts[0][1]), float(pts[1][1])))
        if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
            raise ValueError(f"out-of-bounds box in {json_path}: {(x1, y1, x2, y2)} vs {(width, height)}")
        counts[LABEL_TO_ID[lbl]] += 1
        norm_boxes.append((lbl, (x1 / width, y1 / height, x2 / width, y2 / height)))
        if shape.get("flags", {}).get("auto_label_review") or "human review required" in str(shape.get("description", "")).lower():
            has_auto_review_flag = True

    is_pass = counts[0] == 0 and all(counts[i] == REFERENCE_EXPECTED[i] for i in range(1, 6))
    label = "PASS" if is_pass else "FAIL"
    total_objects = sum(counts.values())

    if is_pass:
        anomaly_category = "none"
    elif total_objects == 0:
        anomaly_category = "empty_scene"
    else:
        has_nc = counts[0] > 0
        has_missing = any(counts[i] < REFERENCE_EXPECTED[i] for i in range(1, 6))
        has_extra = any(counts[i] > REFERENCE_EXPECTED[i] for i in range(1, 6))
        if has_nc and has_missing and not has_extra:
            anomaly_category = "non_catalogue_plus_missing"
        elif has_nc and not has_missing and not has_extra:
            anomaly_category = f"non_catalogue_only_{counts[0]}"
        elif not has_nc and has_missing and not has_extra:
            anomaly_category = "pure_missing"
        else:
            parts = []
            if has_nc:
                parts.append(f"nc_{counts[0]}")
            if has_missing:
                parts.append("missing")
            if has_extra:
                parts.append("extra")
            anomaly_category = "+".join(parts) or "other_fail"

    return {
        "stem": img_path.stem,
        "image_filename": img_path.name,
        "json_filename": json_path.name,
        "src_img": img_path,
        "src_json": json_path,
        "width": width,
        "height": height,
        "counts": counts,
        "total_objects": total_objects,
        "label": label,
        "anomaly_category": anomaly_category,
        "norm_boxes": norm_boxes,
        "has_auto_review_flag": has_auto_review_flag,
    }


def rot_box(b: tuple[float, float, float, float], k: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    if k == 0:
        return (x1, y1, x2, y2)
    if k == 1:
        return (y1, 1.0 - x2, y2, 1.0 - x1)
    if k == 2:
        return (1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1)
    if k == 3:
        return (1.0 - y2, x1, 1.0 - y1, x2)
    raise ValueError(k)


def box_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def rot_invariant_layout_sim(
    bx1: list[tuple[str, tuple[float, float, float, float]]],
    bx2: list[tuple[str, tuple[float, float, float, float]]],
) -> float:
    # One-to-one matching prevents one object being reused for multiple boxes.
    # Empty annotations carry no evidence of independence (audit handles as N/A).
    from scene_audit import layout_similarity
    score = layout_similarity(bx1, bx2)
    return score[0] if score is not None else 0.0


def collect_and_assign() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    scene_groups: list[dict] = []

    # 1. 11/11 parts_v1 (160 originals across 4 conditions)
    p11_root = MEMBER03_ROOT / "full_dataset_upload/datasets/parts_v1"
    cond_items: dict[str, list[dict]] = collections.defaultdict(list)
    for old_split in ("train", "val", "test"):
        sp_dir = p11_root / old_split
        for cond_dir in sorted(p for p in sp_dir.iterdir() if p.is_dir()):
            cond_name = cond_dir.name
            for img_path in sorted(cond_dir.glob("*.png")):
                json_path = img_path.with_suffix(".json")
                info = parse_labelme(img_path, json_path)
                info["source_batch"] = "11/11_parts_v1"
                info["source_subcondition"] = f"{cond_name}_legacy_{old_split}"
                info["cond_name"] = cond_name
                cond_items[cond_name].append(info)

    for cond_name, items in sorted(cond_items.items()):
        items.sort(key=lambda x: x["image_filename"])
        if len(items) != 40:
            raise RuntimeError(f"expected 40 images in 11/11 {cond_name}, got {len(items)}")
        train_cut = 31 if cond_name == "标准光正视" else 32
        for idx, item in enumerate(items):
            assigned_split = "train" if idx < train_cut else "val"
            gid = f"G_1111_{cond_name}_{assigned_split.upper()}"
            notes = []
            if item["has_auto_review_flag"]:
                notes.append("upstream_auto_label_review_flag")
            if item["stem"] == "IMG_7209":
                notes.append("legacy_test_missing_1_M5x10_pending_manual_recheck")
            item.update({
                "split": assigned_split,
                "data_tier": "original",
                "parent_image_id": "",
                "augmentation_type": "none",
                "scene_group_id": gid,
                "scene_independence_status": "CANDIDATE_SEQUENTIAL_SEGMENT_UNVERIFIED",
                "quality_notes": ";".join(notes),
            })
            records.append(item)
        for sp_name, sub_items in (("train", items[:train_cut]), ("val", items[train_cut:])):
            scene_groups.append({
                "scene_group_id": f"G_1111_{cond_name}_{sp_name.upper()}",
                "source_batch": "11/11_parts_v1",
                "subcondition": cond_name,
                "assigned_split": sp_name,
                "grouping_basis": f"ordered_index_cut_{train_cut}_{40 - train_cut}",
                "independence_status": "CANDIDATE_SEQUENTIAL_SEGMENT_UNVERIFIED",
                "original_count": len(sub_items),
                "augmented_count": 0,
                "original_images": [x["image_filename"] for x in sub_items],
                "augmented_images": [],
            })

    # 2. 11/14 PASS (21 originals: first 16 train, last 5 val)
    p14_pass_dir = MEMBER03_ROOT / "11.14训练补强＋干扰验证/补强数据(训练）"
    p14_pass_items = [
        parse_labelme(p, p.with_suffix(".json"))
        for p in sorted(p14_pass_dir.glob("*.jpg"))
    ]
    if len(p14_pass_items) != 21:
        raise RuntimeError(f"expected 21 in 11/14 PASS, got {len(p14_pass_items)}")
    for idx, item in enumerate(p14_pass_items):
        assigned_split = "train" if idx < 16 else "val"
        gid = f"G_1114_PASS_{assigned_split.upper()}"
        notes = ["upstream_auto_label_review_flag"] if item["has_auto_review_flag"] else []
        item.update({
            "source_batch": "11/14_PASS",
            "source_subcondition": "补强数据(训练）",
            "split": assigned_split,
            "data_tier": "original",
            "parent_image_id": "",
            "augmentation_type": "none",
            "scene_group_id": gid,
            "scene_independence_status": "CANDIDATE_TIME_SEGMENT_UNVERIFIED",
            "quality_notes": ";".join(notes),
        })
        records.append(item)
    for sp_name, sub_items in (("train", p14_pass_items[:16]), ("val", p14_pass_items[16:])):
        scene_groups.append({
            "scene_group_id": f"G_1114_PASS_{sp_name.upper()}",
            "source_batch": "11/14_PASS",
            "subcondition": "补强数据(训练）",
            "assigned_split": sp_name,
            "grouping_basis": "chronological_16_train_5_val_preserving_early_high_iou_pairs",
            "independence_status": "CANDIDATE_TIME_SEGMENT_UNVERIFIED",
            "original_count": len(sub_items),
            "augmented_count": 0,
            "original_images": [x["image_filename"] for x in sub_items],
            "augmented_images": [],
        })

    # 3. 11/14 FAIL (77 originals -> 11 blocks at gap >= 30s; B01, B03, B08, B11 -> val [18], rest -> train [59])
    p14_fail_dir = MEMBER03_ROOT / "11.14训练补强＋干扰验证/验证集_不要训练"
    p14_fail_items = [
        parse_labelme(p, p.with_suffix(".json"))
        for p in sorted(p14_fail_dir.glob("*.jpg"))
    ]
    if len(p14_fail_items) != 77:
        raise RuntimeError(f"expected 77 in 11/14 FAIL, got {len(p14_fail_items)}")
    p14_blocks = group_by_time_gap(p14_fail_items, gap_sec=30)
    if len(p14_blocks) != 11:
        raise RuntimeError(f"expected 11 time blocks in 11/14 FAIL, got {len(p14_blocks)}")
    val_block_indices_1114 = {1, 3, 8, 11}
    for b_idx, block in enumerate(p14_blocks, 1):
        assigned_split = "val" if b_idx in val_block_indices_1114 else "train"
        gid = f"G_1114_FAIL_B{b_idx:02d}"
        for item in block:
            notes = ["reassigned_from_legacy_nov14_val_only_pending_team_signoff"]
            if item["has_auto_review_flag"]:
                notes.append("upstream_auto_label_review_flag")
            if item["stem"] == "IMG_20261114_162432":
                notes.append("upstream_commit_16d9239_label_swap_corrected")
            item.update({
                "source_batch": "11/14_FAIL",
                "source_subcondition": f"验证集_不要训练_B{b_idx:02d}",
                "split": assigned_split,
                "data_tier": "original",
                "parent_image_id": "",
                "augmentation_type": "none",
                "scene_group_id": gid,
                "scene_independence_status": "CANDIDATE_TIME_BLOCK_30S_UNVERIFIED",
                "quality_notes": ";".join(notes),
            })
            records.append(item)
        scene_groups.append({
            "scene_group_id": gid,
            "source_batch": "11/14_FAIL",
            "subcondition": f"time_block_{b_idx:02d}",
            "assigned_split": assigned_split,
            "grouping_basis": "timestamp_gap_ge_30s",
            "independence_status": "CANDIDATE_TIME_BLOCK_30S_UNVERIFIED",
            "original_count": len(block),
            "augmented_count": 0,
            "original_images": [x["image_filename"] for x in block],
            "augmented_images": [],
        })

    # 4. 11/18 FAIL (52 originals -> 5 blocks at gap >= 25s; B03 -> val [12], B01/B02/B04/B05 -> train [40])
    p18_dir = MEMBER03_ROOT / "数据11.18_补含干扰件训练集"
    p18_items = [
        parse_labelme(p, p.with_suffix(".json"))
        for p in sorted(p18_dir.glob("*.jpg"))
    ]
    if len(p18_items) != 52:
        raise RuntimeError(f"expected 52 in 11/18 FAIL, got {len(p18_items)}")
    p18_blocks = group_by_time_gap(p18_items, gap_sec=25)
    if len(p18_blocks) != 5:
        raise RuntimeError(f"expected 5 time blocks in 11/18 FAIL, got {len(p18_blocks)}")
    for b_idx, block in enumerate(p18_blocks, 1):
        assigned_split = "val" if b_idx == 3 else "train"
        gid = f"G_1118_FAIL_B{b_idx:02d}"
        for item in block:
            notes = []
            if item["has_auto_review_flag"]:
                notes.append("upstream_auto_label_review_flag")
            item.update({
                "source_batch": "11/18_FAIL",
                "source_subcondition": f"数据11.18_B{b_idx:02d}",
                "split": assigned_split,
                "data_tier": "original",
                "parent_image_id": "",
                "augmentation_type": "none",
                "scene_group_id": gid,
                "scene_independence_status": "CANDIDATE_TIME_BLOCK_25S_UNVERIFIED",
                "quality_notes": ";".join(notes),
            })
            records.append(item)
        scene_groups.append({
            "scene_group_id": gid,
            "source_batch": "11/18_FAIL",
            "subcondition": f"time_block_{b_idx:02d}",
            "assigned_split": assigned_split,
            "grouping_basis": "timestamp_gap_ge_25s",
            "independence_status": "CANDIDATE_TIME_BLOCK_25S_UNVERIFIED",
            "original_count": len(block),
            "augmented_count": 0,
            "original_images": [x["image_filename"] for x in block],
            "augmented_images": [],
        })

    # 5. 11/20 Originals (70 originals -> 11 blocks at gap >= 60s) + 210 Augmented derivatives
    p20_items: list[dict] = []
    for sub in ("11.20/11.20 fail", "11.20/11.20 pass"):
        sub_dir = MEMBER03_ROOT / sub
        for img_path in sorted(sub_dir.glob("*.jpg")):
            info = parse_labelme(img_path, img_path.with_suffix(".json"))
            info["orig_subfolder"] = sub
            p20_items.append(info)
    if len(p20_items) != 70:
        raise RuntimeError(f"expected 70 originals in 11/20, got {len(p20_items)}")
    p20_blocks = group_by_time_gap(p20_items, gap_sec=60)
    if len(p20_blocks) != 11:
        raise RuntimeError(f"expected 11 time blocks in 11/20, got {len(p20_blocks)}")

    parent_lookup: dict[str, dict] = {}
    group_lookup_1120: dict[str, dict] = {}
    for b_idx, block in enumerate(p20_blocks, 1):
        if b_idx in {1, 2, 3, 4, 5, 6, 8}:
            assigned_split = "train"
        elif b_idx == 7:
            assigned_split = "val"
        else:
            assigned_split = "test"
        gid = f"G_1120_B{b_idx:02d}"
        g_entry = {
            "scene_group_id": gid,
            "source_batch": "11/20",
            "subcondition": f"time_block_{b_idx:02d}",
            "assigned_split": assigned_split,
            "grouping_basis": "timestamp_gap_ge_60s",
            "independence_status": "CANDIDATE_TIME_BLOCK_60S_UNVERIFIED",
            "original_count": len(block),
            "augmented_count": 0,
            "original_images": [x["image_filename"] for x in block],
            "augmented_images": [],
        }
        scene_groups.append(g_entry)
        group_lookup_1120[gid] = g_entry
        for item in block:
            notes = []
            if item["stem"] == "MVIMG_20261120_172557":
                notes.append("visually_verified_empty_tabletop_zero_objects_valid_FAIL")
            if item["has_auto_review_flag"]:
                notes.append("upstream_auto_label_review_flag")
            item.update({
                "source_batch": "11/20",
                "source_subcondition": f"{item['orig_subfolder']}_B{b_idx:02d}",
                "split": assigned_split,
                "data_tier": "original",
                "parent_image_id": "",
                "augmentation_type": "none",
                "scene_group_id": gid,
                "scene_independence_status": "CANDIDATE_TIME_BLOCK_60S_UNVERIFIED",
                "quality_notes": ";".join(notes),
            })
            records.append(item)
            parent_lookup[item["stem"]] = item

    aug_count = 0
    for sub in ("11.20_泛化增强/11.20 fail", "11.20_泛化增强/11.20 pass"):
        sub_dir = MEMBER03_ROOT / sub
        for img_path in sorted(sub_dir.glob("*.jpg")):
            info = parse_labelme(img_path, img_path.with_suffix(".json"))
            if "__" not in info["stem"]:
                raise RuntimeError(f"unexpected augmented filename without '__': {img_path}")
            parent_stem, aug_suffix = info["stem"].split("__", 1)
            if parent_stem not in parent_lookup:
                raise RuntimeError(f"augmented image {img_path} has no parent {parent_stem}")
            parent = parent_lookup[parent_stem]
            if info["label"] != parent["label"]:
                raise RuntimeError(
                    f"label mismatch between parent {parent['image_filename']} ({parent['label']}) "
                    f"and augmented {info['image_filename']} ({info['label']})"
                )
            notes = ["derived_offline_augmentation_not_independent_scene"]
            if parent_stem == "MVIMG_20261120_172557":
                notes.append("derived_from_visually_verified_empty_tabletop")
            info.update({
                "source_batch": "11/20_augmented",
                "source_subcondition": f"{sub}_{aug_suffix}",
                "split": parent["split"],
                "data_tier": "augmented",
                "parent_image_id": parent["image_filename"],
                "augmentation_type": aug_suffix,
                "scene_group_id": parent["scene_group_id"],
                "scene_independence_status": "DERIVED_AUGMENTATION_BOUND_TO_PARENT",
                "quality_notes": ";".join(notes),
            })
            records.append(info)
            g_entry = group_lookup_1120[parent["scene_group_id"]]
            g_entry["augmented_count"] += 1
            g_entry["augmented_images"].append(info["image_filename"])
            aug_count += 1
    if aug_count != 210:
        raise RuntimeError(f"expected 210 augmented images, got {aug_count}")

    # 6. 11/23 PASS (19 originals -> 2 blocks at gap >= 60s; B01 [6] -> train, B02 [13] -> test)
    p23_dir = MEMBER03_ROOT / "1123pass/11.23补pass"
    p23_items = [
        parse_labelme(p, p.with_suffix(".json"))
        for p in sorted(p23_dir.glob("*.jpg"))
    ]
    if len(p23_items) != 19:
        raise RuntimeError(f"expected 19 in 11/23 PASS, got {len(p23_items)}")
    p23_blocks = group_by_time_gap(p23_items, gap_sec=60)
    if len(p23_blocks) != 2 or len(p23_blocks[0]) != 6 or len(p23_blocks[1]) != 13:
        raise RuntimeError(f"unexpected 11/23 blocks: {[len(b) for b in p23_blocks]}")
    for b_idx, block in enumerate(p23_blocks, 1):
        assigned_split = "train" if b_idx == 1 else "test"
        gid = f"G_1123_PASS_B{b_idx:02d}"
        for item in block:
            notes = ["shared_nov23_paper_session_background"]
            if item["has_auto_review_flag"]:
                notes.append("upstream_auto_label_review_flag")
            item.update({
                "source_batch": "11/23_PASS",
                "source_subcondition": f"11.23补pass_B{b_idx:02d}",
                "split": assigned_split,
                "data_tier": "original",
                "parent_image_id": "",
                "augmentation_type": "none",
                "scene_group_id": gid,
                "scene_independence_status": "CANDIDATE_TIME_BLOCK_60S_SAME_PAPER_SESSION",
                "quality_notes": ";".join(notes),
            })
            records.append(item)
        scene_groups.append({
            "scene_group_id": gid,
            "source_batch": "11/23_PASS",
            "subcondition": f"time_block_{b_idx:02d}",
            "assigned_split": assigned_split,
            "grouping_basis": "timestamp_gap_ge_60s_distinct_component_layouts_shared_paper_session",
            "independence_status": "CANDIDATE_TIME_BLOCK_60S_SAME_PAPER_SESSION",
            "original_count": len(block),
            "augmented_count": 0,
            "original_images": [x["image_filename"] for x in block],
            "augmented_images": [],
        })

    return records, scene_groups


def check_cross_split_similarity(records: list[dict]) -> dict:
    originals = [r for r in records if r["data_tier"] == "original"]
    by_batch: dict[str, list[dict]] = collections.defaultdict(list)
    for r in originals:
        by_batch[r["source_batch"]].append(r)

    # Also screen all sources jointly; same-batch-only checks miss cross-source links.
    by_batch["__ALL_BATCHES_SCREENING_ONLY__"] = originals
    summary = {}
    for batch, items in sorted(by_batch.items()):
        max_same_split = 0.0
        max_cross_split = 0.0
        cross_pairs_gt_035 = []
        same_pairs_gt_035 = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = rot_invariant_layout_sim(items[i]["norm_boxes"], items[j]["norm_boxes"])
                if items[i]["split"] == items[j]["split"]:
                    if sim > max_same_split:
                        max_same_split = sim
                    if sim > 0.35:
                        same_pairs_gt_035 += 1
                else:
                    if sim > max_cross_split:
                        max_cross_split = sim
                    if sim > 0.35:
                        cross_pairs_gt_035.append({
                            "image_a": items[i]["image_filename"],
                            "split_a": items[i]["split"],
                            "group_a": items[i]["scene_group_id"],
                            "image_b": items[j]["image_filename"],
                            "split_b": items[j]["split"],
                            "group_b": items[j]["scene_group_id"],
                            "rot_invariant_box_iou": round(sim, 4),
                        })
        summary[batch] = {
            "original_images": len(items),
            "same_split_pairs_gt_035": same_pairs_gt_035,
            "max_same_split_rot_inv_iou": round(max_same_split, 4),
            "cross_split_pairs_gt_035_count": len(cross_pairs_gt_035),
            "max_cross_split_rot_inv_iou": round(max_cross_split, 4),
            "cross_split_pairs_gt_035": cross_pairs_gt_035,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned Dataset v2 planner; existing data is never removed")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='default: verify sources and approved assignments; do not copy')
    modes.add_argument('--build', action='store_true', help='create only if all managed data and metadata paths are absent')
    args = parser.parse_args()
    from build_safety import validate_plan, refuse_existing
    if args.build:
        refuse_existing(DATASET_V2_DIR)
    records, scene_groups = collect_and_assign()
    validate_plan(records, MEMBER03_ROOT, Path(__file__).with_name('source_snapshot_lock.json'))
    if not args.build:
        print(json.dumps({'status':'PLAN_VERIFIED_NO_WRITE','records':len(records),'approved_commit':'755e038cc8f05b9701cceb28e3381d5fa3bc6a81','test_status':'PROVISIONAL_HOLDOUT'}))
        return
    if len(records) != 609:
        raise RuntimeError(f"expected 609 total records, got {len(records)}")

    # Verify quotas before copying anything
    actual_quotas = collections.Counter((r["split"], r["label"], r["data_tier"]) for r in records)
    for key, expected_n in EXPECTED_QUOTAS.items():
        got_n = actual_quotas.get(key, 0)
        if got_n != expected_n:
            raise RuntimeError(f"quota mismatch for {key}: expected {expected_n}, got {got_n}")

    # Verify global filename uniqueness
    img_names = [r["image_filename"] for r in records]
    if len(set(img_names)) != len(img_names):
        raise RuntimeError("duplicate image filename detected; separate source subdirs would be required")

    # Create only absent data directories; interrupted builds fail closed on rerun.
    for split in ("train", "val", "test"):
        sp_dir = DATASET_V2_DIR / split
        for label in ("PASS", "FAIL"):
            for tier in ("original", "augmented"):
                (DATASET_V2_DIR / split / label / tier).mkdir(parents=True, exist_ok=True)

    # Physical byte-for-byte copy + immediate SHA256 verification
    records.sort(key=lambda r: (
        {"train": 0, "val": 1, "test": 2}[r["split"]],
        {"PASS": 0, "FAIL": 1}[r["label"]],
        {"original": 0, "augmented": 1}[r["data_tier"]],
        r["image_filename"],
    ))

    for idx, r in enumerate(records, 1):
        dst_dir = DATASET_V2_DIR / r["split"] / r["label"] / r["data_tier"]
        dst_img = dst_dir / r["image_filename"]
        dst_json = dst_dir / r["json_filename"]
        if dst_img.exists() or dst_json.exists():
            raise RuntimeError(f"destination collision: {dst_img}")

        src_img_sha = sha256_file(r["src_img"])
        src_json_sha = sha256_file(r["src_json"])

        shutil.copy2(r["src_img"], dst_img)
        shutil.copy2(r["src_json"], dst_json)

        # Ensure neither hardlink nor byte drift occurred
        if os.path.samefile(r["src_img"], dst_img) or os.path.samefile(r["src_json"], dst_json):
            raise RuntimeError("hardlink/symlink detected; physical copy required")
        dst_img_sha = sha256_file(dst_img)
        dst_json_sha = sha256_file(dst_json)
        if dst_img_sha != src_img_sha or dst_json_sha != src_json_sha:
            raise RuntimeError(f"SHA256 mismatch after copy for {r['image_filename']}")

        r["sample_id"] = f"DSV2_{idx:04d}"
        r["partition_status"] = (
            "PROVISIONAL_HOLDOUT" if r["split"] == "test"
            else ("DEVELOPMENT_TRAIN" if r["split"] == "train" else "DEVELOPMENT_VAL")
        )
        r["original_relpath"] = r["src_img"].relative_to(MEMBER03_ROOT).as_posix()
        r["original_json_relpath"] = r["src_json"].relative_to(MEMBER03_ROOT).as_posix()
        r["v2_image_relpath"] = dst_img.relative_to(DATASET_V2_DIR).as_posix()
        r["v2_json_relpath"] = dst_json.relative_to(DATASET_V2_DIR).as_posix()
        r["image_sha256"] = dst_img_sha
        r["json_sha256"] = dst_json_sha

    sim_summary = check_cross_split_similarity(records)

    # Write dataset_manifest.csv
    manifest_fields = [
        "sample_id", "image_filename", "json_filename", "split", "partition_status",
        "label", "data_tier", "source_batch", "source_subcondition",
        "original_relpath", "original_json_relpath", "v2_image_relpath", "v2_json_relpath",
        "scene_group_id", "scene_independence_status", "parent_image_id", "augmentation_type",
        "width", "height",
        "count_1_M4x25", "count_2_M5x10", "count_3_M4x10", "count_4_M4_hex_nut",
        "count_5_flat_washer", "count_0_non_catalogue", "total_objects",
        "anomaly_category", "quality_notes", "image_sha256", "json_sha256",
    ]
    manifest_path = DATASET_V2_DIR / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_fields)
        writer.writeheader()
        for r in records:
            c = r["counts"]
            writer.writerow({
                "sample_id": r["sample_id"],
                "image_filename": r["image_filename"],
                "json_filename": r["json_filename"],
                "split": r["split"],
                "partition_status": r["partition_status"],
                "label": r["label"],
                "data_tier": r["data_tier"],
                "source_batch": r["source_batch"],
                "source_subcondition": r["source_subcondition"],
                "original_relpath": r["original_relpath"],
                "original_json_relpath": r["original_json_relpath"],
                "v2_image_relpath": r["v2_image_relpath"],
                "v2_json_relpath": r["v2_json_relpath"],
                "scene_group_id": r["scene_group_id"],
                "scene_independence_status": r["scene_independence_status"],
                "parent_image_id": r["parent_image_id"],
                "augmentation_type": r["augmentation_type"],
                "width": r["width"],
                "height": r["height"],
                "count_1_M4x25": c[1],
                "count_2_M5x10": c[2],
                "count_3_M4x10": c[3],
                "count_4_M4_hex_nut": c[4],
                "count_5_flat_washer": c[5],
                "count_0_non_catalogue": c[0],
                "total_objects": r["total_objects"],
                "anomaly_category": r["anomaly_category"],
                "quality_notes": r["quality_notes"],
                "image_sha256": r["image_sha256"],
                "json_sha256": r["json_sha256"],
            })

    # Write augmentation_mapping.csv
    orig_by_name = {r["image_filename"]: r for r in records if r["data_tier"] == "original"}
    aug_fields = [
        "aug_sample_id", "aug_filename", "augmentation_type",
        "split", "partition_status", "label",
        "aug_v2_image_relpath", "aug_v2_json_relpath", "aug_image_sha256", "aug_json_sha256",
        "parent_sample_id", "parent_filename", "parent_split", "parent_label",
        "parent_v2_image_relpath", "parent_image_sha256", "scene_group_id",
        "same_split_verified", "same_label_verified",
    ]
    aug_map_path = DATASET_V2_DIR / "augmentation_mapping.csv"
    with aug_map_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=aug_fields)
        writer.writeheader()
        for r in records:
            if r["data_tier"] != "augmented":
                continue
            p = orig_by_name[r["parent_image_id"]]
            writer.writerow({
                "aug_sample_id": r["sample_id"],
                "aug_filename": r["image_filename"],
                "augmentation_type": r["augmentation_type"],
                "split": r["split"],
                "partition_status": r["partition_status"],
                "label": r["label"],
                "aug_v2_image_relpath": r["v2_image_relpath"],
                "aug_v2_json_relpath": r["v2_json_relpath"],
                "aug_image_sha256": r["image_sha256"],
                "aug_json_sha256": r["json_sha256"],
                "parent_sample_id": p["sample_id"],
                "parent_filename": p["image_filename"],
                "parent_split": p["split"],
                "parent_label": p["label"],
                "parent_v2_image_relpath": p["v2_image_relpath"],
                "parent_image_sha256": p["image_sha256"],
                "scene_group_id": r["scene_group_id"],
                "same_split_verified": r["split"] == p["split"],
                "same_label_verified": r["label"] == p["label"],
            })

    # Write scene_groups.json
    scene_payload = {
        "schema_version": "dataset_v2.scene_groups.v1",
        "disclaimer": (
            "Scene groups in dataset_v2 are candidate groupings derived from capture timestamps, "
            "ordered sequence blocks, and parent-child augmentation links, supplemented by rotation-invariant "
            "normalized bounding-box IoU checks. Low bounding-box overlap does not establish physical scene independence; it only describes overlap "
            "within image coordinates, NOT certified physical rig/session independence."
        ),
        "total_candidate_groups": len(scene_groups),
        "cross_split_layout_similarity_audit": sim_summary,
        "groups": scene_groups,
    }
    (DATASET_V2_DIR / "scene_groups.json").write_text(
        json.dumps(scene_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Compute dataset_statistics.json
    by_split_summary = {}
    for sp in ("train", "val", "test"):
        sp_rows = [r for r in records if r["split"] == sp]
        orig_rows = [r for r in sp_rows if r["data_tier"] == "original"]
        aug_rows = [r for r in sp_rows if r["data_tier"] == "augmented"]
        obj_orig = {str(k): sum(r["counts"][k] for r in orig_rows) for k in range(6)}
        obj_all = {str(k): sum(r["counts"][k] for r in sp_rows) for k in range(6)}
        by_split_summary[sp] = {
            "partition_status": "PROVISIONAL_HOLDOUT" if sp == "test" else f"DEVELOPMENT_{sp.upper()}",
            "total_images": len(sp_rows),
            "original_images": len(orig_rows),
            "augmented_images": len(aug_rows),
            "original_PASS": sum(r["label"] == "PASS" for r in orig_rows),
            "original_FAIL": sum(r["label"] == "FAIL" for r in orig_rows),
            "augmented_PASS": sum(r["label"] == "PASS" for r in aug_rows),
            "augmented_FAIL": sum(r["label"] == "FAIL" for r in aug_rows),
            "total_PASS": sum(r["label"] == "PASS" for r in sp_rows),
            "total_FAIL": sum(r["label"] == "FAIL" for r in sp_rows),
            "original_by_source_batch": dict(collections.Counter(r["source_batch"] for r in orig_rows)),
            "original_anomaly_categories": dict(collections.Counter(r["anomaly_category"] for r in orig_rows)),
            "augmented_by_type": dict(collections.Counter(r["augmentation_type"] for r in aug_rows)),
            "object_instances_original_only": obj_orig,
            "object_instances_including_augmented": obj_all,
        }

    source_matrix = {}
    for batch in ("11/11_parts_v1", "11/14_PASS", "11/14_FAIL", "11/18_FAIL", "11/20", "11/23_PASS"):
        b_rows = [r for r in records if r["source_batch"] == batch]
        source_matrix[batch] = {
            "train_PASS": sum(r["split"] == "train" and r["label"] == "PASS" for r in b_rows),
            "train_FAIL": sum(r["split"] == "train" and r["label"] == "FAIL" for r in b_rows),
            "val_PASS": sum(r["split"] == "val" and r["label"] == "PASS" for r in b_rows),
            "val_FAIL": sum(r["split"] == "val" and r["label"] == "FAIL" for r in b_rows),
            "test_PASS": sum(r["split"] == "test" and r["label"] == "PASS" for r in b_rows),
            "test_FAIL": sum(r["split"] == "test" and r["label"] == "FAIL" for r in b_rows),
            "total_originals": len(b_rows),
        }

    stats_payload = {
        "schema_version": "dataset_v2.statistics.v1",
        "reference_expected_counts": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "0_non_catalogue": 0},
        "totals": {
            "all_images": len(records),
            "original_images": sum(r["data_tier"] == "original" for r in records),
            "augmented_images": sum(r["data_tier"] == "augmented" for r in records),
            "total_copied_files_images_plus_jsons": len(records) * 2,
            "original_PASS": sum(r["data_tier"] == "original" and r["label"] == "PASS" for r in records),
            "original_FAIL": sum(r["data_tier"] == "original" and r["label"] == "FAIL" for r in records),
            "augmented_PASS": sum(r["data_tier"] == "augmented" and r["label"] == "PASS" for r in records),
            "augmented_FAIL": sum(r["data_tier"] == "augmented" and r["label"] == "FAIL" for r in records),
        },
        "splits": by_split_summary,
        "original_source_allocation_matrix": source_matrix,
        "integrity_summary": {
            "byte_identical_copies_verified": len(records) * 2,
            "unique_image_sha256_count": len({r["image_sha256"] for r in records}),
            "parent_child_split_mismatches": 0,
            "cross_split_layout_similarity_audit": sim_summary,
        },
        "quality_and_provenance_notes": {
            "MVIMG_20261120_172557.jpg": "Visually inspected; genuine empty white tabletop with zero components. Shapes=[] is valid FAIL (plus 3 derived augmented copies in test/FAIL/augmented).",
            "IMG_20261114_162432.json": "Uses upstream commit 16d9239 corrected Labelme JSON (sha256 recorded in dataset_manifest.csv and SHA256SUMS.txt; legacy data_20261114_manifest.json left untouched).",
            "IMG_7209.png": "Legacy 11/11 test sample with 1 M5x10 missing; placed in val/FAIL/original, pending manual ground-truth confirmation.",
            "test_partition_status": "PROVISIONAL_HOLDOUT (29 real originals: 13 PASS from 11/23 Block 02 + 16 FAIL from 11/20 Blocks 09-11; plus 48 derived FAIL augmentations). Does NOT yet meet the >=30 independent physical scene / >=15 PASS formal blind test target.",
        },
    }
    (DATASET_V2_DIR / "dataset_statistics.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Write SHA256SUMS.txt covering all 1,218 copied data files in deterministic order
    sha_lines = []
    for r in records:
        sha_lines.append(f"{r['image_sha256']}  {r['v2_image_relpath']}")
        sha_lines.append(f"{r['json_sha256']}  {r['v2_json_relpath']}")
    (DATASET_V2_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "BUILT_OK",
        "copied_images": len(records),
        "copied_jsons": len(records),
        "total_data_files": len(records) * 2,
        "cross_split_audit": {
            k: {
                "cross_gt_035": v["cross_split_pairs_gt_035_count"],
                "max_cross_iou": v["max_cross_split_rot_inv_iou"],
            }
            for k, v in sim_summary.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
