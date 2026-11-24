"""Read-only Phase-1 comparison across R1_RAW_CE, R2_PARENT_EQUAL_AUG_CE, and R3_FULL_AUG_BALANCED_CE.

Outputs:
- COMPARISON_PHASE1_CN.md
- comparison_summary.json
- val_74_per_image_comparison.csv
- val_74_grouped_metrics.csv
- val_aug_18_robustness.csv
- curves_phase1.png
- val_score_distributions_phase1.png
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

from .runtime import file_sha256, load_checkpoint, write_json
from .v2_train import compute_rich_metrics

REQUIRED_FILES = (
    "experiment_spec.json",
    "config.json",
    "run_config.json",
    "history.json",
    "best.pt",
    "last.pt",
    "checkpoint_manifest.json",
    "best_metrics.json",
    "best_val_predictions.json",
    "eval_train_raw_294.json",
    "eval_train_supplementary.json",
    "eval_val_raw_74.json",
    "eval_val_aug_18_diagnostics.json",
)


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_v2_run(model_dir: str | Path, expected_id: str) -> dict:
    d = Path(model_dir).resolve()
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    if missing:
        raise ValueError(f"incomplete run {d} (missing {missing})")

    spec = _read_json(d / "experiment_spec.json")
    cfg = _read_json(d / "config.json")
    run_cfg = _read_json(d / "run_config.json")
    history = _read_json(d / "history.json")
    ckpt_man = _read_json(d / "checkpoint_manifest.json")
    best_metrics = _read_json(d / "best_metrics.json")
    val_preds = _read_json(d / "best_val_predictions.json")
    train_294 = _read_json(d / "eval_train_raw_294.json")
    train_supp = _read_json(d / "eval_train_supplementary.json")
    val_74 = _read_json(d / "eval_val_raw_74.json")
    val_aug_18 = _read_json(d / "eval_val_aug_18_diagnostics.json")

    if spec.get("experiment_id") != expected_id or ckpt_man.get("experiment_id") != expected_id:
        raise ValueError(f"expected experiment_id={expected_id}, got {spec.get('experiment_id')}")
    if run_cfg["selection_protocol"].get("test_data_used") is not False or ckpt_man.get("test_data_used") is not False:
        raise ValueError(f"Test exclusion violated in {d}")
    if run_cfg["selection_protocol"].get("val_aug_18_used_for_selection") is not False:
        raise ValueError(f"val_aug_18 isolation violated in {d}")

    if file_sha256(d / "best.pt") != ckpt_man["best"]["sha256"]:
        raise ValueError(f"best.pt SHA256 mismatch in {d}")
    if file_sha256(d / "last.pt") != ckpt_man["last"]["sha256"]:
        raise ValueError(f"last.pt SHA256 mismatch in {d}")

    # Verify checkpoint loads cleanly via standard runtime.load_checkpoint
    _, loaded_cfg, _, payload = load_checkpoint(
        d / "best.pt",
        device="cpu",
        allow_smoke=(cfg["data_kind"] == "SMOKE_ONLY"),
        allow_dev_only=(cfg["training_tier"] == "DEV_ONLY"),
    )
    if loaded_cfg.to_dict() != cfg:
        raise ValueError(f"checkpoint config mismatch in {d}")
    if [h["epoch"] for h in history] != list(range(1, cfg["epochs"] + 1)):
        raise ValueError(f"incomplete epoch history in {d}")

    # Verify best_epoch is minimum unweighted Val loss (earliest epoch wins ties)
    selected = min(history, key=lambda h: h["val"]["loss"])
    if payload["epoch"] != selected["epoch"] or ckpt_man["best"]["epoch"] != selected["epoch"]:
        raise ValueError(f"best_epoch selection rule mismatch in {d}")

    # Recompute unweighted CE from saved 74 Val predictions
    recomputed_ce = sum(-math.log(max(p["softmax"][p["target_index"]], 1e-300)) for p in val_preds) / len(val_preds)
    if abs(recomputed_ce - best_metrics["loss"]) > 1e-5:
        raise ValueError(f"saved Val predictions do not match unweighted Val loss in {d}")

    return {
        "dir": str(d),
        "spec": spec,
        "config": cfg,
        "run_config": run_cfg,
        "history": history,
        "ckpt_manifest": ckpt_man,
        "best_metrics": best_metrics,
        "val_preds": val_preds,
        "train_294": train_294,
        "train_supp": train_supp,
        "val_74": val_74,
        "val_aug_18": val_aug_18,
    }


def _fmt(v: float | int | None, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def compare_phase1_runs(r1_dir: str | Path, r2_dir: str | Path, r3_dir: str | Path, output_dir: str | Path) -> Path:
    out = Path(output_dir).resolve()
    if out.exists():
        raise ValueError(f"comparison output directory already exists (refusing overwrite): {out}")

    runs = {
        "R1": inspect_v2_run(r1_dir, "R1_RAW_CE"),
        "R2": inspect_v2_run(r2_dir, "R2_PARENT_EQUAL_AUG_CE"),
        "R3": inspect_v2_run(r3_dir, "R3_FULL_AUG_BALANCED_CE"),
    }

    # Verify all 3 runs share identical Dataset v2 provenance, initial head/backbone weights, seed, and identical 74 Val set
    prov_hashes = {k: r["run_config"]["data_provenance_sha256"] for k, r in runs.items()}
    if len(set(prov_hashes.values())) != 1:
        raise ValueError(f"data_provenance_sha256 differs across runs: {prov_hashes}")

    init_heads = {k: r["run_config"]["determinism_notes"]["initial_head_sha256"] for k, r in runs.items()}
    if len(set(init_heads.values())) != 1:
        raise ValueError(f"initial Linear head weights differ across runs: {init_heads}")

    init_backbones = {k: r["run_config"]["determinism_notes"]["initial_backbone_sha256"] for k, r in runs.items()}
    if len(set(init_backbones.values())) != 1:
        raise ValueError(f"initial frozen backbone weights differ across runs: {init_backbones}")

    val_relpaths_r1 = [p["v2_image_relpath"] for p in runs["R1"]["val_preds"]]
    for k in ("R2", "R3"):
        if [p["v2_image_relpath"] for p in runs[k]["val_preds"]] != val_relpaths_r1:
            raise ValueError(f"Val 74 image ordering/identity mismatch between R1 and {k}")

    # Load rich Val 74 row metadata from prepared val_raw_74.csv
    val_manifest_path = Path(runs["R1"]["config"]["val_manifest"]).resolve()
    with val_manifest_path.open(encoding="utf-8", newline="") as fh:
        val_meta_rows = list(csv.DictReader(fh))
    val_meta_by_rel = {r["v2_image_relpath"]: r for r in val_meta_rows}

    out.mkdir(parents=True, exist_ok=False)

    # 1. Per-image Val 74 CSV
    per_img_rows = []
    for i, rel in enumerate(val_relpaths_r1):
        meta = val_meta_by_rel[rel]
        p1 = runs["R1"]["val_preds"][i]
        p2 = runs["R2"]["val_preds"][i]
        p3 = runs["R3"]["val_preds"][i]
        preds_set = {p1["predicted_label"], p2["predicted_label"], p3["predicted_label"]}
        per_img_rows.append({
            "sample_id": meta["sample_id"],
            "image_filename": meta["image_filename"],
            "label": meta["label"],
            "source_batch": meta["source_batch"],
            "anomaly_mechanism": meta["anomaly_mechanism"],
            "anomaly_category": meta["anomaly_category"],
            "non_catalogue_count": int(meta["non_catalogue_count"]),
            "missing_target_units": int(meta["missing_target_units"]),
            "R1_pred": p1["predicted_label"],
            "R1_correct": p1["predicted_label"] == meta["label"],
            "R1_fail_softmax": round(p1["softmax"][1], 6),
            "R2_pred": p2["predicted_label"],
            "R2_correct": p2["predicted_label"] == meta["label"],
            "R2_fail_softmax": round(p2["softmax"][1], 6),
            "R3_pred": p3["predicted_label"],
            "R3_correct": p3["predicted_label"] == meta["label"],
            "R3_fail_softmax": round(p3["softmax"][1], 6),
            "models_disagree": len(preds_set) > 1,
        })

    with (out / "val_74_per_image_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_img_rows[0].keys()))
        w.writeheader()
        w.writerows(per_img_rows)

    # 2. Grouped metrics on Val 74 (by source_batch, anomaly_mechanism, non_catalogue_count, anomaly_category)
    group_fields = ("source_batch", "anomaly_mechanism", "non_catalogue_count", "anomaly_category")
    grouped_rows = []
    for gf in group_fields:
        distinct_vals = sorted({str(val_meta_by_rel[rel][gf]) for rel in val_relpaths_r1})
        for gv in distinct_vals:
            idxs = [i for i, rel in enumerate(val_relpaths_r1) if str(val_meta_by_rel[rel][gf]) == gv]
            for run_key in ("R1", "R2", "R3"):
                sub_preds = [runs[run_key]["val_preds"][i] for i in idxs]
                sub_ce = sum(-math.log(max(p["softmax"][p["target_index"]], 1e-300)) for p in sub_preds) / len(sub_preds)
                gm = compute_rich_metrics(sub_preds, sub_ce)
                grouped_rows.append({
                    "group_field": gf,
                    "group_value": gv,
                    "run": run_key,
                    "n": gm["n"],
                    "support_PASS": gm["support"]["PASS"],
                    "support_FAIL": gm["support"]["FAIL"],
                    "unweighted_loss": round(gm["loss"], 6),
                    "accuracy": round(gm["accuracy"], 6),
                    "balanced_accuracy": round(gm["balanced_accuracy"], 6) if gm["balanced_accuracy"] is not None else None,
                    "fail_precision": round(gm["fail_precision"], 6) if gm["fail_precision"] is not None else None,
                    "fail_recall": round(gm["fail_recall"], 6) if gm["fail_recall"] is not None else None,
                    "fail_f1": round(gm["fail_f1"], 6) if gm["fail_f1"] is not None else None,
                    "false_pass_FN": gm["false_pass"],
                    "false_fail_FP": gm["false_fail"],
                    "roc_auc": round(gm["roc_auc"], 6) if gm["roc_auc"] is not None else None,
                })

    with (out / "val_74_grouped_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grouped_rows[0].keys()))
        w.writeheader()
        w.writerows(grouped_rows)

    # 3. Val Aug 18 side-by-side CSV
    aug_pairs_r1 = runs["R1"]["val_aug_18"]["pairs"]
    aug_pairs_r2 = {x["aug_filename"]: x for x in runs["R2"]["val_aug_18"]["pairs"]}
    aug_pairs_r3 = {x["aug_filename"]: x for x in runs["R3"]["val_aug_18"]["pairs"]}
    aug_csv_rows = []
    for a1 in aug_pairs_r1:
        fn = a1["aug_filename"]
        a2 = aug_pairs_r2[fn]
        a3 = aug_pairs_r3[fn]
        aug_csv_rows.append({
            "aug_filename": fn,
            "augmentation_type": a1["augmentation_type"],
            "parent_filename": a1["parent_filename"],
            "R1_parent_pred": a1["parent_predicted_label"],
            "R1_aug_pred": a1["aug_predicted_label"],
            "R1_score_delta": round(a1["fail_softmax_delta"], 6),
            "R1_flipped": a1["prediction_flipped"],
            "R2_parent_pred": a2["parent_predicted_label"],
            "R2_aug_pred": a2["aug_predicted_label"],
            "R2_score_delta": round(a2["fail_softmax_delta"], 6),
            "R2_flipped": a2["prediction_flipped"],
            "R3_parent_pred": a3["parent_predicted_label"],
            "R3_aug_pred": a3["aug_predicted_label"],
            "R3_score_delta": round(a3["fail_softmax_delta"], 6),
            "R3_flipped": a3["prediction_flipped"],
        })

    with (out / "val_aug_18_robustness.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(aug_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(aug_csv_rows)

    # 4. Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    series_labels = [("R1_RAW_CE", runs["R1"]), ("R2_PARENT_EQUAL_AUG_CE", runs["R2"]), ("R3_FULL_AUG_BALANCED_CE", runs["R3"])]
    for title_lbl, r in series_labels:
        eps = [h["epoch"] for h in r["history"]]
        axes[0, 0].plot(eps, [h["train"]["loss"] for h in r["history"]], marker="o", label=title_lbl)
        axes[0, 1].plot(eps, [h["val"]["loss"] for h in r["history"]], marker="o", label=title_lbl)
        axes[1, 0].plot(eps, [h["val"]["balanced_accuracy"] for h in r["history"]], marker="o", label=title_lbl)
        axes[1, 1].plot(eps, [h["val"]["roc_auc"] for h in r["history"]], marker="o", label=title_lbl)

    axes[0, 0].set_title("Train CE Loss (Note: R3 uses balanced weights)")
    axes[0, 1].set_title("Primary Val (74 Real) Unweighted CE Loss")
    axes[1, 0].set_title("Primary Val (74 Real) Balanced Accuracy")
    axes[1, 1].set_title("Primary Val (74 Real) ROC-AUC")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves_phase1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for ax, (title_lbl, r) in zip(axes, series_labels):
        pass_scores = [p["softmax"][1] for p in r["val_preds"] if p["target_index"] == 0]
        fail_scores = [p["softmax"][1] for p in r["val_preds"] if p["target_index"] == 1]
        ax.hist(pass_scores, bins=bins, alpha=0.55, label="PASS (n=37)")
        ax.hist(fail_scores, bins=bins, alpha=0.55, label="FAIL (n=37)")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="argmax=0.5")
        ax.set_title(f"{title_lbl} (best ep={r['best_metrics']['epoch']})")
        ax.set_xlabel("Uncalibrated FAIL Softmax")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Val Image Count")
    fig.tight_layout()
    fig.savefig(out / "val_score_distributions_phase1.png", dpi=150)
    plt.close(fig)

    # 5. Summary JSON + Chinese Markdown Report
    summary = {
        "schema_version": "member06.dataset_v2_phase1_comparison.v1",
        "dataset_version": "dataset_v2 (438/92/79 freeze_v1)",
        "data_provenance_sha256": prov_hashes["R1"],
        "initial_head_sha256": init_heads["R1"],
        "initial_backbone_sha256": init_backbones["R1"],
        "test_data_used": False,
        "legacy_dataset_v1_compared_as_baseline": False,
        "runs": {
            k: {
                "experiment_id": r["spec"]["experiment_id"],
                "train_mode": r["spec"]["train_mode"],
                "train_class_weight": r["config"]["train_class_weight"],
                "class_weights_pass0_fail1": r["run_config"]["loss_protocol"]["weights"],
                "best_epoch": r["best_metrics"]["epoch"],
                "exposure_accounting": r["ckpt_manifest"]["exposure_accounting"],
                "timing_seconds": r["ckpt_manifest"]["timing_seconds"],
                "feature_cache": r["run_config"]["feature_cache"],
                "primary_val_74_metrics": r["val_74"]["metrics"],
                "unified_train_raw_294_metrics": r["train_294"]["metrics"],
                "val_aug_18_diagnostics": {
                    "overall": r["val_aug_18"]["overall_18_augmented_FAIL"],
                    "parent_6_original_FAIL_detected": r["val_aug_18"]["parent_6_original_FAIL_detected"],
                    "total_flips_across_18": r["val_aug_18"]["total_flips_across_18"],
                    "by_augmentation_type": r["val_aug_18"]["by_augmentation_type"],
                },
            }
            for k, r in runs.items()
        },
        "pairwise_val_74_disagreements_count": sum(row["models_disagree"] for row in per_img_rows),
    }
    write_json(out / "comparison_summary.json", summary)

    m1, m2, m3 = runs["R1"]["val_74"]["metrics"], runs["R2"]["val_74"]["metrics"], runs["R3"]["val_74"]["metrics"]
    t1, t2, t3 = runs["R1"]["train_294"]["metrics"], runs["R2"]["train_294"]["metrics"], runs["R3"]["train_294"]["metrics"]
    e1, e2, e3 = (runs[k]["ckpt_manifest"]["exposure_accounting"] for k in ("R1", "R2", "R3"))
    tm1, tm2, tm3 = (runs[k]["ckpt_manifest"]["timing_seconds"] for k in ("R1", "R2", "R3"))
    w3 = runs["R3"]["run_config"]["loss_protocol"]["weights"]

    md = [
        "# Dataset v2 第一阶段核心实验比较报告（R1 / R2 / R3）",
        "",
        "> **边界与性质声明**：本报告仅基于 `Dataset v2`（`438/92/79` 冻结版）的 `Train` 与 `74` 张真实 `Val` 原图生成（`DEV_ONLY` 阶段开发对比）。`18` 张 `Val` 离线增强图仅用于训练后鲁棒性诊断，不参与最佳 Epoch 选择或主选型；`Test`（`31` 原图 + `48` 增强 = `79` 张）全程未读取（`test_data_used = false`）。旧 `Dataset v1` 验证集构成与分布已变，不以旧验证集分数冒充新版基准。",
        "",
        "## 1. 三组实验配置差异与受控关系",
        "",
        "| 项目 | `R1_RAW_CE` | `R2_PARENT_EQUAL_AUG_CE` | `R3_FULL_AUG_BALANCED_CE` |",
        "|---|---|---|---|",
        "| 训练数据源 | 294 张真实原图 | 同 294 个父原图（48 个含增强父图按 4 版本确定性循环） | 全部 438 张训练文件（294 原图 + 144 增强） |",
        "| 每轮类别曝光数 | 149 PASS / 145 FAIL（294） | 149 PASS / 145 FAIL（294） | 155 PASS / 283 FAIL（438） |",
        f"| 10 轮总曝光数 | {e1['total_exposures_10_epochs']} | {e2['total_exposures_10_epochs']} | {e3['total_exposures_10_epochs']} |",
        f"| 每轮 / 10轮优化器更新步数 | {e1['optimizer_steps_per_epoch']} / {e1['total_optimizer_steps_10_epochs']} | {e2['optimizer_steps_per_epoch']} / {e2['total_optimizer_steps_10_epochs']} | {e3['optimizer_steps_per_epoch']} / {e3['total_optimizer_steps_10_epochs']} |",
        f"| 训练损失权重 `[PASS, FAIL]` | `none` (`[1, 1]`) | `none` (`[1, 1]`) | `balanced` (`[{w3[0]:.6f}, {w3[1]:.6f}]`) |",
        "| 主验证集与选模准则 | 74 张真实 Val，最小无权 CE（平局取最早轮） | 同左 | 同左 |",
        "| 模型 / 分辨率 / 优化器 | Frozen ResNet18 + Linear(512,2)，224 letterbox，Adam lr=0.001，wd=0，seed=42 | 完全一致（与 R1 同初始头权重、同每轮父图顺序） | 完全一致（同初始头权重） |",
        "| 实验科学解释角色 | 真实原图基线对照 | **相对 R1 的受控离线增强替换实验**（仅替换 48 个 `11/20` 父图的单次曝光版本） | **多因素完整策略对照**（同时改变增强纳入、`11/20` 4 倍场景曝光、类别权重及每轮更新步数，非单因素实验） |",
        "",
        "## 2. 主验证集（74 张真实原图：37 PASS + 37 FAIL）核心指标",
        "",
        "| 指标 | `R1_RAW_CE` | `R2_PARENT_EQUAL_AUG_CE` | `R3_FULL_AUG_BALANCED_CE` | `R2 - R1` | `R3 - R1` |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 最佳 Epoch | {runs['R1']['best_metrics']['epoch']} | {runs['R2']['best_metrics']['epoch']} | {runs['R3']['best_metrics']['epoch']} | {runs['R2']['best_metrics']['epoch'] - runs['R1']['best_metrics']['epoch']:+d} | {runs['R3']['best_metrics']['epoch'] - runs['R1']['best_metrics']['epoch']:+d} |",
        f"| 无权 Val Loss | {_fmt(m1['loss'], 6)} | {_fmt(m2['loss'], 6)} | {_fmt(m3['loss'], 6)} | {m2['loss'] - m1['loss']:+.6f} | {m3['loss'] - m1['loss']:+.6f} |",
        f"| Accuracy | {_fmt(m1['accuracy'])} | {_fmt(m2['accuracy'])} | {_fmt(m3['accuracy'])} | {m2['accuracy'] - m1['accuracy']:+.4f} | {m3['accuracy'] - m1['accuracy']:+.4f} |",
        f"| Balanced Accuracy | {_fmt(m1['balanced_accuracy'])} | {_fmt(m2['balanced_accuracy'])} | {_fmt(m3['balanced_accuracy'])} | {m2['balanced_accuracy'] - m1['balanced_accuracy']:+.4f} | {m3['balanced_accuracy'] - m1['balanced_accuracy']:+.4f} |",
        f"| FAIL Precision | {_fmt(m1['fail_precision'])} | {_fmt(m2['fail_precision'])} | {_fmt(m3['fail_precision'])} | {(m2['fail_precision'] - m1['fail_precision']) if (m1['fail_precision'] is not None and m2['fail_precision'] is not None) else 0.0:+.4f} | {(m3['fail_precision'] - m1['fail_precision']) if (m1['fail_precision'] is not None and m3['fail_precision'] is not None) else 0.0:+.4f} |",
        f"| FAIL Recall | {_fmt(m1['fail_recall'])} | {_fmt(m2['fail_recall'])} | {_fmt(m3['fail_recall'])} | {m2['fail_recall'] - m1['fail_recall']:+.4f} | {m3['fail_recall'] - m1['fail_recall']:+.4f} |",
        f"| FAIL F1 | {_fmt(m1['fail_f1'])} | {_fmt(m2['fail_f1'])} | {_fmt(m3['fail_f1'])} | {(m2['fail_f1'] - m1['fail_f1']) if (m1['fail_f1'] is not None and m2['fail_f1'] is not None) else 0.0:+.4f} | {(m3['fail_f1'] - m1['fail_f1']) if (m1['fail_f1'] is not None and m3['fail_f1'] is not None) else 0.0:+.4f} |",
        f"| False PASS / FN（共 37 FAIL） | {m1['false_pass']} | {m2['false_pass']} | {m3['false_pass']} | {m2['false_pass'] - m1['false_pass']:+d} | {m3['false_pass'] - m1['false_pass']:+d} |",
        f"| False FAIL / FP（共 37 PASS） | {m1['false_fail']} | {m2['false_fail']} | {m3['false_fail']} | {m2['false_fail'] - m1['false_fail']:+d} | {m3['false_fail'] - m1['false_fail']:+d} |",
        f"| ROC-AUC | {_fmt(m1['roc_auc'])} | {_fmt(m2['roc_auc'])} | {_fmt(m3['roc_auc'])} | {m2['roc_auc'] - m1['roc_auc']:+.4f} | {m3['roc_auc'] - m1['roc_auc']:+.4f} |",
        f"| PASS / FAIL 分数中位数 | {_fmt(m1['score_median_PASS'])} / {_fmt(m1['score_median_FAIL'])} | {_fmt(m2['score_median_PASS'])} / {_fmt(m2['score_median_FAIL'])} | {_fmt(m3['score_median_PASS'])} / {_fmt(m3['score_median_FAIL'])} | — | — |",
        "",
        "## 3. 固定最佳权重的训练内统一参考指标（同 294 张真实原图）",
        "",
        "| 指标（固定 `best.pt` 重评估） | `R1_RAW_CE` | `R2_PARENT_EQUAL_AUG_CE` | `R3_FULL_AUG_BALANCED_CE` |",
        "|---|---:|---:|---:|",
        f"| 294 原图无权 CE Loss | {_fmt(t1['loss'], 6)} | {_fmt(t2['loss'], 6)} | {_fmt(t3['loss'], 6)} |",
        f"| 294 原图 Accuracy | {_fmt(t1['accuracy'])} | {_fmt(t2['accuracy'])} | {_fmt(t3['accuracy'])} |",
        f"| 294 原图 FAIL Recall / FN / FP | {_fmt(t1['fail_recall'])} / {t1['false_pass']} / {t1['false_fail']} | {_fmt(t2['fail_recall'])} / {t2['false_pass']} / {t2['false_fail']} | {_fmt(t3['fail_recall'])} / {t3['false_pass']} / {t3['false_fail']} |",
        f"| 294 原图 ROC-AUC | {_fmt(t1['roc_auc'])} | {_fmt(t2['roc_auc'])} | {_fmt(t3['roc_auc'])} |",
        "",
        "## 4. 主验证集（74 张）按来源、异常机制与目录外数量分组",
        "",
        "完整明细见 `val_74_grouped_metrics.csv`，逐图对比见 `val_74_per_image_comparison.csv`（三模型预测存在分歧的图片共 "
        f"`{summary['pairwise_val_74_disagreements_count']}/74` 张）。",
        "",
        "| 分组维度 | 组别 | 支持数 (PASS/FAIL) | R1 Acc (FN/FP) | R2 Acc (FN/FP) | R3 Acc (FN/FP) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for gf in ("source_batch", "anomaly_mechanism", "non_catalogue_count"):
        sub = [g for g in grouped_rows if g["group_field"] == gf]
        for gv in sorted({g["group_value"] for g in sub}):
            g1 = next(g for g in sub if g["group_value"] == gv and g["run"] == "R1")
            g2 = next(g for g in sub if g["group_value"] == gv and g["run"] == "R2")
            g3 = next(g for g in sub if g["group_value"] == gv and g["run"] == "R3")
            md.append(
                f"| `{gf}` | `{gv}` | {g1['n']} ({g1['support_PASS']}/{g1['support_FAIL']}) | "
                f"{g1['accuracy']:.4f} ({g1['false_pass_FN']}/{g1['false_fail_FP']}) | "
                f"{g2['accuracy']:.4f} ({g2['false_pass_FN']}/{g2['false_fail_FP']}) | "
                f"{g3['accuracy']:.4f} ({g3['false_pass_FN']}/{g3['false_fail_FP']}) |"
            )

    md += [
        "",
        "## 5. Val 增强图（18 张 `11/20` FAIL 派生图）训练后鲁棒性诊断",
        "",
        "对应 `val_raw_74` 中的 6 张 `11/20` FAIL 父原图（每张派生 `lowlight` / `warm` / `softnoise` 各 1 张），逐对明细见 `val_aug_18_robustness.csv`：",
        "",
        "| 模型 | 6 张父原图 FAIL 检出 | 18 张增强 FAIL 检出 (Recall) | 相对父原图分类翻转数 (/18) | `lowlight` 均分差 (翻转) | `warm` 均分差 (翻转) | `softnoise` 均分差 (翻转) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rk in ("R1", "R2", "R3"):
        va = runs[rk]["val_aug_18"]
        ov = va["overall_18_augmented_FAIL"]
        by_t = va["by_augmentation_type"]
        md.append(
            f"| `{runs[rk]['spec']['experiment_id']}` | {va['parent_6_original_FAIL_detected']}/6 | "
            f"{ov['support']['FAIL'] - ov['false_pass']}/18 ({ov['fail_recall']:.4f}) | {va['total_flips_across_18']}/18 | "
            f"{by_t['A01_lowlight']['mean_score_delta_vs_parent']:+.4f} ({by_t['A01_lowlight']['prediction_flips']}) | "
            f"{by_t['A02_warm']['mean_score_delta_vs_parent']:+.4f} ({by_t['A02_warm']['prediction_flips']}) | "
            f"{by_t['A03_softnoise']['mean_score_delta_vs_parent']:+.4f} ({by_t['A03_softnoise']['prediction_flips']}) |"
        )

    md += [
        "",
        "## 6. 实际训练耗时与特征缓存记录",
        "",
        "| 实验 | 缓存状态 | 缓存构建/校验耗时 (s) | 10 轮训练+验证耗时 (s) | 实验总耗时 (s) | 设备 |",
        "|---|---|---:|---:|---:|---|",
        f"| `R1_RAW_CE` | `{runs['R1']['run_config']['feature_cache']['cache_action']}` | {tm1['feature_cache_seconds']:.2f} | {tm1['training_10_epochs_seconds']:.2f} | {tm1['total_experiment_seconds']:.2f} | `{runs['R1']['run_config']['environment']['device']}` |",
        f"| `R2_PARENT_EQUAL_AUG_CE` | `{runs['R2']['run_config']['feature_cache']['cache_action']}` | {tm2['feature_cache_seconds']:.2f} | {tm2['training_10_epochs_seconds']:.2f} | {tm2['total_experiment_seconds']:.2f} | `{runs['R2']['run_config']['environment']['device']}` |",
        f"| `R3_FULL_AUG_BALANCED_CE` | `{runs['R3']['run_config']['feature_cache']['cache_action']}` | {tm3['feature_cache_seconds']:.2f} | {tm3['training_10_epochs_seconds']:.2f} | {tm3['total_experiment_seconds']:.2f} | `{runs['R3']['run_config']['environment']['device']}` |",
        "",
        "## 7. 图表与工件索引",
        "",
        "- 训练/验证损失与指标曲线：`curves_phase1.png`",
        "- 74 张真实 Val 的 PASS/FAIL 分数分布：`val_score_distributions_phase1.png`",
        "- 74 张真实 Val 逐图预测对比表：`val_74_per_image_comparison.csv`",
        "- 来源/机制/干扰件数量分组表：`val_74_grouped_metrics.csv`",
        "- 18 张 Val 增强图鲁棒性与翻转表：`val_aug_18_robustness.csv`",
    ]
    (out / "COMPARISON_PHASE1_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", required=True, help="path to R1_RAW_CE model directory")
    parser.add_argument("--r2", required=True, help="path to R2_PARENT_EQUAL_AUG_CE model directory")
    parser.add_argument("--r3", required=True, help="path to R3_FULL_AUG_BALANCED_CE model directory")
    parser.add_argument("--output", required=True, help="new output directory for comparison report")
    args = parser.parse_args()
    try:
        out = compare_phase1_runs(args.r1, args.r2, args.r3, args.output)
        print(json.dumps({"status": "COMPARISON_GENERATED", "output_dir": str(out)}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"V2_COMPARISON_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
