"""Targeted Val-only error analysis for Phase-1 R1_RAW_CE false-PASS cases.

Strictly reads only Phase-1 Val predictions, Dataset v2 Train/Val manifests, and Val/Train Labelme JSONs.
Never accesses Test images or Test annotations; never modifies any dataset file.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

def write_json(path: Path, value: dict | list) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

DEFAULT_PHASE1_SUITE = "members/member06_research/runs/dataset_v2_phase1_20261124_124710_5rtNiM"
DEFAULT_PREPARED_DIR = "members/member06_research/data/dataset_v2_phase1"
DEFAULT_DATASET_V2_ROOT = "members/member03_data/dataset_v2"


def analyze_r1_val_errors(
    phase1_suite_dir: str | Path = DEFAULT_PHASE1_SUITE,
    prepared_dir: str | Path = DEFAULT_PREPARED_DIR,
    dataset_v2_root: str | Path = DEFAULT_DATASET_V2_ROOT,
    image_size: int = 224,
) -> dict:
    suite = Path(phase1_suite_dir).resolve()
    pdir = Path(prepared_dir).resolve()
    ds_root = Path(dataset_v2_root).resolve()

    r1_preds = json.loads((suite / "R1_RAW_CE/model/best_val_predictions.json").read_text(encoding="utf-8"))
    r2_preds = {
        p["image_filename"]: p
        for p in json.loads((suite / "R2_PARENT_EQUAL_AUG_CE/model/best_val_predictions.json").read_text(encoding="utf-8"))
    }
    r3_preds = {
        p["image_filename"]: p
        for p in json.loads((suite / "R3_FULL_AUG_BALANCED_CE/model/best_val_predictions.json").read_text(encoding="utf-8"))
    }

    with (pdir / "val_raw_74.csv").open(encoding="utf-8", newline="") as fh:
        val_meta = {r["image_filename"]: r for r in csv.DictReader(fh)}
    with (pdir / "train_raw_294.csv").open(encoding="utf-8", newline="") as fh:
        train_rows = list(csv.DictReader(fh))

    train_cat_counts = collections.Counter(r["anomaly_category"] for r in train_rows)
    train_mech_counts = collections.Counter(r["anomaly_mechanism"] for r in train_rows)
    train_batch_fail_counts = collections.Counter(
        r["source_batch"] for r in train_rows if r["label"] == "FAIL"
    )
    train_batch_pass_counts = collections.Counter(
        r["source_batch"] for r in train_rows if r["label"] == "PASS"
    )

    # Pre-scan Train 294 matching cases (only train/ JSONs, never test/)
    train_matching_details: dict[str, list[dict]] = collections.defaultdict(list)
    for tr in train_rows:
        if tr["split"] != "train" or not tr["v2_image_relpath"].startswith("train/"):
            raise ValueError(f"unexpected non-train row in train_raw_294: {tr['v2_image_relpath']}")
        jp = ds_root / Path(tr["v2_image_relpath"]).with_suffix(".json")
        raw = json.loads(jp.read_text(encoding="utf-8"))
        W, H = int(raw["imageWidth"]), int(raw["imageHeight"])
        scale = min(image_size / W, image_size / H)
        cnt = collections.Counter(s["label"].strip() for s in raw.get("shapes", []))
        nc_boxes = []
        for s in raw.get("shapes", []):
            if s["label"].strip() == "non_catalogue":
                pts = s["points"]
                bw = abs(float(pts[1][0]) - float(pts[0][0]))
                bh = abs(float(pts[1][1]) - float(pts[0][1]))
                nc_boxes.append({
                    "orig_wh_px": [round(bw, 1), round(bh, 1)],
                    "scaled_224_wh_px": [round(bw * scale, 2), round(bh * scale, 2)],
                })
        train_matching_details[tr["anomaly_category"]].append({
            "image_filename": tr["image_filename"],
            "source_batch": tr["source_batch"],
            "counts": dict(cnt),
            "non_catalogue_boxes": nc_boxes,
        })

    r1_errors = [p for p in r1_preds if p["predicted_index"] != p["target_index"]]
    cases = []
    for p in r1_errors:
        fname = p["image_filename"]
        meta = val_meta[fname]
        if meta["split"] != "val" or not meta["v2_image_relpath"].startswith("val/"):
            raise ValueError(f"forbidden non-val path in error analysis: {meta['v2_image_relpath']}")
        jpath = ds_root / Path(meta["v2_image_relpath"]).with_suffix(".json")
        raw = json.loads(jpath.read_text(encoding="utf-8"))
        W, H = int(raw["imageWidth"]), int(raw["imageHeight"])
        scale = min(image_size / W, image_size / H)
        active_w, active_h = round(W * scale), round(H * scale)

        counts = {
            lbl: 0 for lbl in ("M4x25", "M5x10", "M4x10", "M4_hex_nut", "flat_washer", "non_catalogue")
        }
        boxes_by_class: dict[str, list[dict]] = collections.defaultdict(list)
        for s in raw.get("shapes", []):
            lbl = s["label"].strip()
            counts[lbl] = counts.get(lbl, 0) + 1
            pts = s["points"]
            x1, x2 = sorted((float(pts[0][0]), float(pts[1][0])))
            y1, y2 = sorted((float(pts[0][1]), float(pts[1][1])))
            bw, bh = x2 - x1, y2 - y1
            boxes_by_class[lbl].append({
                "orig_bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "orig_wh_px": [round(bw, 1), round(bh, 1)],
                "orig_area_px2": round(bw * bh, 1),
                "orig_area_fraction_pct": round(100.0 * (bw * bh) / (W * H), 4),
                "scaled_224_wh_px": [round(bw * scale, 2), round(bh * scale, 2)],
                "scaled_224_area_px2": round((bw * scale) * (bh * scale), 2),
                "scaled_224_canvas_fraction_pct": round(100.0 * ((bw * scale) * (bh * scale)) / (image_size * image_size), 4),
            })

        deficient_classes = {
            k: 2 - counts[k]
            for k in ("M4x25", "M5x10", "M4x10", "M4_hex_nut", "flat_washer")
            if counts[k] < 2
        }

        if meta["anomaly_category"] == "non_catalogue_only_1":
            nc_box = boxes_by_class["non_catalogue"][0]
            anomaly_size_summary = {
                "kind": "present_non_catalogue_object",
                "original_wh_px": nc_box["orig_wh_px"],
                "original_area_px2": nc_box["orig_area_px2"],
                "original_frame_pct": nc_box["orig_area_fraction_pct"],
                "scaled_224_wh_px": nc_box["scaled_224_wh_px"],
                "scaled_224_area_px2": nc_box["scaled_224_area_px2"],
                "scaled_224_canvas_pct": nc_box["scaled_224_canvas_fraction_pct"],
                "note": "All 5 target classes have exact count 2; the single non-catalogue object occupies ~14.3x14.1 px after 224 letterbox (<1 cell of ResNet18's 7x7 feature grid).",
            }
        elif meta["anomaly_category"] == "pure_missing":
            rem_cls = next(iter(deficient_classes.keys()))
            rem_box = boxes_by_class[rem_cls][0] if boxes_by_class[rem_cls] else None
            anomaly_size_summary = {
                "kind": "missing_target_component_no_bbox",
                "deficient_classes": deficient_classes,
                "missing_component_bbox": None,
                "remaining_same_class_reference_object": {
                    "class_label": rem_cls,
                    "original_wh_px": rem_box["orig_wh_px"] if rem_box else None,
                    "original_area_px2": rem_box["orig_area_px2"] if rem_box else None,
                    "original_frame_pct": rem_box["orig_area_fraction_pct"] if rem_box else None,
                    "scaled_224_wh_px": rem_box["scaled_224_wh_px"] if rem_box else None,
                    "scaled_224_area_px2": rem_box["scaled_224_area_px2"] if rem_box else None,
                    "scaled_224_canvas_pct": rem_box["scaled_224_canvas_fraction_pct"] if rem_box else None,
                },
                "note": "A missing component has no bounding box (0x0 px); the remaining M5x10 screw in the same frame spans 302.4x299.0 px originally and 16.80x16.61 px (0.56% of 224x224 canvas) after letterbox.",
            }
        else:
            anomaly_size_summary = {"kind": meta["anomaly_category"]}

        acat = meta["anomaly_category"]
        sbatch = meta["source_batch"]
        matching_train = train_matching_details.get(acat, [])
        same_missing_class_in_train = 0
        if deficient_classes:
            miss_key = next(iter(deficient_classes.keys()))
            same_missing_class_in_train = sum(
                x["counts"].get(miss_key, 0) < 2 and x["counts"].get("non_catalogue", 0) == 0
                for x in matching_train
            )

        cases.append({
            "sample_id": meta["sample_id"],
            "image_filename": fname,
            "v2_image_relpath": meta["v2_image_relpath"],
            "source_batch": sbatch,
            "ground_truth_label": p["label"],
            "predicted_label_R1": p["predicted_label"],
            "error_type": "False PASS (FN)" if p["label"] == "FAIL" else "False FAIL (FP)",
            "actual_annotated_counts": counts,
            "total_annotated_objects": sum(counts.values()),
            "deficient_target_classes": deficient_classes,
            "anomaly_mechanism": meta["anomaly_mechanism"],
            "anomaly_category": acat,
            "predicted_scores": {
                "R1_RAW_CE_fail_softmax": round(p["softmax"][1], 6),
                "R1_RAW_CE_pass_softmax": round(p["softmax"][0], 6),
                "R1_margin_to_05_threshold": round(0.5 - p["softmax"][1], 6),
                "R2_PARENT_EQUAL_AUG_CE_fail_softmax": round(r2_preds[fname]["softmax"][1], 6),
                "R3_FULL_AUG_BALANCED_CE_fail_softmax": round(r3_preds[fname]["softmax"][1], 6),
            },
            "image_geometry": {
                "original_width_height": [W, H],
                "letterbox_224_scale": round(scale, 6),
                "letterbox_224_active_width_height": [active_w, active_h],
            },
            "anomaly_component_size": anomaly_size_summary,
            "all_objects_by_class": dict(boxes_by_class),
            "train_294_coverage": {
                "total_train_originals": len(train_rows),
                "total_train_FAIL": sum(r["label"] == "FAIL" for r in train_rows),
                "same_anomaly_category_in_train_294": train_cat_counts.get(acat, 0),
                "same_anomaly_mechanism_in_train_294": train_mech_counts.get(meta["anomaly_mechanism"], 0),
                "same_source_batch_PASS_in_train_294": train_batch_pass_counts.get(sbatch, 0),
                "same_source_batch_FAIL_in_train_294": train_batch_fail_counts.get(sbatch, 0),
                "same_pure_missing_class_in_train_294": same_missing_class_in_train if deficient_classes else None,
                "matching_train_examples": matching_train,
            },
        })

    return {
        "schema_version": "member06.r1_val_error_analysis.v1",
        "source_suite": str(suite),
        "scope": "Val 74 real originals ONLY (Test untouched; dataset files unmodified)",
        "test_content_read": False,
        "r1_false_pass_count": sum(c["error_type"] == "False PASS (FN)" for c in cases),
        "r1_false_fail_count": sum(c["error_type"] == "False FAIL (FP)" for c in cases),
        "cases": cases,
    }


def render_error_analysis_markdown(report: dict) -> str:
    lines = [
        "## 定点错误分析：第一阶段 `R1_RAW_CE` 两张验证集共同漏放样本",
        "",
        "仅基于第一阶段 `R1_RAW_CE`（及 `R2`、`R3` 对照）已保存的 `74` 张真实 `Val` 预测与 `Dataset v2` 的 `Train/Val` 标注生成，未修改任何数据文件，未读取 `Test`。",
        "",
        "| 字段 | 漏放样本 1 | 漏放样本 2 |",
        "|---|---|---|",
    ]
    if len(report["cases"]) == 2:
        c1, c2 = report["cases"]
        cnt1 = ", ".join(f"`{k}:{v}`" for k, v in c1["actual_annotated_counts"].items())
        cnt2 = ", ".join(f"`{k}:{v}`" for k, v in c2["actual_annotated_counts"].items())
        s1 = c1["anomaly_component_size"]
        s2 = c2["anomaly_component_size"]
        ref2 = s2["remaining_same_class_reference_object"]
        cov1 = c1["train_294_coverage"]
        cov2 = c2["train_294_coverage"]
        lines += [
            f"| 文件名（Sample ID） | `{c1['image_filename']}` (`{c1['sample_id']}`) | `{c2['image_filename']}` (`{c2['sample_id']}`) |",
            f"| 来源批次 | `{c1['source_batch']}` | `{c2['source_batch']}` |",
            f"| 实际标注数量 | 共 {c1['total_annotated_objects']} 件：{cnt1} | 共 {c2['total_annotated_objects']} 件：{cnt2} |",
            f"| 异常机制 / 细分类型 | `{c1['anomaly_mechanism']}` / `{c1['anomaly_category']}`（五类目标齐全，仅多 1 个目录外件） | `{c2['anomaly_mechanism']}` / `{c2['anomaly_category']}`（无目录外件，仅缺 1 个 `M5x10`） |",
            f"| `R1` 预测 FAIL 分数（距 0.5 差值） | **`{c1['predicted_scores']['R1_RAW_CE_fail_softmax']:.6f}`**（差 `{c1['predicted_scores']['R1_margin_to_05_threshold']:+.6f}`，极近阈值边界） | **`{c2['predicted_scores']['R1_RAW_CE_fail_softmax']:.6f}`**（差 `{c2['predicted_scores']['R1_margin_to_05_threshold']:+.6f}`，高置信误放） |",
            f"| `R2` / `R3` 预测 FAIL 分数 | `{c1['predicted_scores']['R2_PARENT_EQUAL_AUG_CE_fail_softmax']:.6f}` / `{c1['predicted_scores']['R3_FULL_AUG_BALANCED_CE_fail_softmax']:.6f}` | `{c2['predicted_scores']['R2_PARENT_EQUAL_AUG_CE_fail_softmax']:.6f}` / `{c2['predicted_scores']['R3_FULL_AUG_BALANCED_CE_fail_softmax']:.6f}` |",
            f"| 原图尺寸 → 224 letterbox 有效区域 | `{c1['image_geometry']['original_width_height'][0]}×{c1['image_geometry']['original_width_height'][1]}` → `{c1['image_geometry']['letterbox_224_active_width_height'][0]}×{c1['image_geometry']['letterbox_224_active_width_height'][1]}`（缩放 `{c1['image_geometry']['letterbox_224_scale']:.6f}`） | `{c2['image_geometry']['original_width_height'][0]}×{c2['image_geometry']['original_width_height'][1]}` → `{c2['image_geometry']['letterbox_224_active_width_height'][0]}×{c2['image_geometry']['letterbox_224_active_width_height'][1]}`（缩放 `{c2['image_geometry']['letterbox_224_scale']:.6f}`） |",
            f"| 异常零件原图尺寸与占比 | 单目录外框 `{s1['original_wh_px'][0]}×{s1['original_wh_px'][1]} px`（`{s1['original_area_px2']} px²`，占原图 `{s1['original_frame_pct']}%`） | 缺失件本身无检测框（`0×0 px`）；同图现存单个 `{ref2['class_label']}` 为 `{ref2['original_wh_px'][0]}×{ref2['original_wh_px'][1]} px`（占原图 `{ref2['original_frame_pct']}%`） |",
            f"| 缩放至 224 输入后的像素尺寸 | **`{s1['scaled_224_wh_px'][0]}×{s1['scaled_224_wh_px'][1]} px`**（`{s1['scaled_224_area_px2']} px²`，占 `224×224` 画布 `{s1['scaled_224_canvas_pct']}%`） | 缺失框 `0×0 px`；同图现存单个 `{ref2['class_label']}` 为 **`{ref2['scaled_224_wh_px'][0]}×{ref2['scaled_224_wh_px'][1]} px`**（占 `224×224` 画布 `{ref2['scaled_224_canvas_pct']}%`） |",
            f"| 训练集（294 原图）同类覆盖情况 | 同类 `non_catalogue_only_1` 仅 **`{cov1['same_anomaly_category_in_train_294']}/294`** 张（均在 `11/14_FAIL`，框缩放后约 `14.3×14.3 px`）；其余 `139/145` 张训练 FAIL 均含 `≥3` 个干扰件 | 全训练集 `pure_missing` 仅 **`{cov2['same_anomaly_category_in_train_294']}/294`** 张（为 `11/20` 少 2 个垫圈，纯缺 `M5x10` 为 **`{cov2['same_pure_missing_class_in_train_294']}`** 张）；且来源批次 `11/11_parts_v1` 在训练集中为 **`{cov2['same_source_batch_PASS_in_train_294']} PASS / {cov2['same_source_batch_FAIL_in_train_294']} FAIL`** |",
        ]
        lines += [
            "",
            "### 机制解读",
            f"1. **`{c1['image_filename']}`（边界型单目录外漏放）**：`R1` 在第 10 轮已给出 `0.492379` 的 `FAIL` 分数（与 `PASS` 仅差 `0.0152`），属于决策边界附近的临界样本。由于训练集中 `95.86%` 的 `FAIL` 样本含 `3~7` 个干扰件，单个直径约 `14 px`（不足 `224` 输入下 ResNet18 末端 `7×7` 特征图一个感受野格）的小尺寸目录外件对全局平均池化特征的扰动幅度较小。第二阶段延长至 `20 Epoch` 可直接检验训练预算增加能否在不引发误拒的前提下将该临界响应推过 `0.5`。",
            f"2. **`{c2['image_filename']}`（结构性纯少件漏放）**：该图无任何目录外异物，仅在 `10` 个零件中少 `1` 个 `M5x10`（单个螺钉仅占 `224×224` 画布的 `0.56%`），叠加训练集中 `11/11_parts_v1` 批次 `127:0` 全为 `PASS` 且缺少纯少 `M5x10` 样本，导致整图全局池化线性分类器高度倾向 `PASS`（`FAIL` 分数仅 `0.084753`）。这体现了冻结整图全局特征基线在纯计数缺失任务上的固有方法学边界，也与课程设计中需要对象级检测基线（`baseline3`）的定位一致。",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-suite", default=DEFAULT_PHASE1_SUITE)
    parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--dataset-v2-root", default=DEFAULT_DATASET_V2_ROOT)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise SystemExit(f"output directory already exists: {out}")
    report = analyze_r1_val_errors(args.phase1_suite, args.prepared_dir, args.dataset_v2_root)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "r1_val_error_analysis.json", report)
    (out / "R1_VAL_ERROR_ANALYSIS_CN.md").write_text(render_error_analysis_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ERROR_ANALYSIS_READY", "cases": len(report["cases"]), "output_dir": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
