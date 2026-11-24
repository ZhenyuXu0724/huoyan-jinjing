"""Read-only Phase-2 comparison: 10-epoch vs 20-epoch budget & multi-seed stability (seeds 42/43/44).

Generates:
- COMPARISON_PHASE2_CN.md (paired 10ep vs 20ep per-seed & mean/std tables, per-seed Val prediction changes,
  seed42-vs-R1 10-epoch reproduction audit, targeted R1 Val error analysis, and pre-committed selection rules)
- phase2_summary.json
- phase2_per_seed_metrics.csv
- phase2_val_74_prediction_changes.csv
- curves_phase2_budgets.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .runtime import file_sha256, load_checkpoint, write_json
from .v2_error_analysis import DEFAULT_DATASET_V2_ROOT, DEFAULT_PHASE1_SUITE, analyze_r1_val_errors, render_error_analysis_markdown

REQUIRED_PHASE2_FILES = (
    "experiment_spec.json",
    "config.json",
    "run_config.json",
    "history.json",
    "epoch_sample_orders.json",
    "best_within_10.pt",
    "best_within_20.pt",
    "epoch10.pt",
    "epoch20.pt",
    "best_within_10_val_predictions.json",
    "best_within_20_val_predictions.json",
    "budget_evaluations.json",
    "checkpoint_manifest.json",
)


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float | int]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "pop_std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(x) for x in values],
    }


def inspect_phase2_run(model_dir: str | Path, expected_id: str, expected_seed: int) -> dict:
    d = Path(model_dir).resolve()
    missing = [f for f in REQUIRED_PHASE2_FILES if not (d / f).is_file()]
    if missing:
        raise ValueError(f"incomplete Phase-2 run {d} (missing {missing})")

    spec = _read_json(d / "experiment_spec.json")
    cfg = _read_json(d / "config.json")
    run_cfg = _read_json(d / "run_config.json")
    history = _read_json(d / "history.json")
    budget_evals = _read_json(d / "budget_evaluations.json")
    ckpt_man = _read_json(d / "checkpoint_manifest.json")
    preds_10 = _read_json(d / "best_within_10_val_predictions.json")
    preds_20 = _read_json(d / "best_within_20_val_predictions.json")

    if spec.get("experiment_id") != expected_id or cfg.get("seed") != expected_seed:
        raise ValueError(f"expected ({expected_id}, seed={expected_seed}), got ({spec.get('experiment_id')}, seed={cfg.get('seed')})")
    if cfg.get("epochs") != 20 or spec.get("budget_milestones") != [10, 20]:
        raise ValueError(f"expected epochs=20 and budget_milestones=[10,20] in {d}")
    if run_cfg["selection_protocol"].get("test_data_used") is not False or ckpt_man.get("test_data_used") is not False:
        raise ValueError(f"Test exclusion violated in {d}")
    if [h["epoch"] for h in history] != list(range(1, 21)):
        raise ValueError(f"history must contain epochs 1..20 in {d}")

    allow_smoke = cfg["data_kind"] == "SMOKE_ONLY"
    allow_dev = cfg["training_tier"] == "DEV_ONLY"

    # Verify all 4 budget/milestone checkpoints load cleanly and match recorded SHA256 + selection rules
    for m_str, m_int in (("10", 10), ("20", 20)):
        m_info = ckpt_man["milestones"][m_str]
        b_file = d / m_info["best_within_budget"]["file"]
        e_file = d / m_info["epoch_end"]["file"]
        if file_sha256(b_file) != m_info["best_within_budget"]["sha256"]:
            raise ValueError(f"{b_file} SHA256 mismatch")
        if file_sha256(e_file) != m_info["epoch_end"]["sha256"]:
            raise ValueError(f"{e_file} SHA256 mismatch")
        _, _, _, b_pay = load_checkpoint(b_file, device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev)
        _, _, _, e_pay = load_checkpoint(e_file, device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev)
        expected_best = min(history[:m_int], key=lambda h: h["val"]["loss"])
        if b_pay["epoch"] != expected_best["epoch"]:
            raise ValueError(f"best_within_{m_int} epoch mismatch: got {b_pay['epoch']}, expected {expected_best['epoch']}")
        if e_pay["epoch"] != m_int:
            raise ValueError(f"epoch{m_int}.pt epoch mismatch: got {e_pay['epoch']}")

    repro = _read_json(d / "reproduction_check_vs_r1.json") if (d / "reproduction_check_vs_r1.json").is_file() else None
    if expected_seed == 42 and spec.get("reference_10ep_run_dir") and not repro:
        raise ValueError("seed=42 missing reproduction_check_vs_r1.json")

    return {
        "dir": str(d),
        "seed": expected_seed,
        "spec": spec,
        "config": cfg,
        "run_config": run_cfg,
        "history": history,
        "budgets": budget_evals["budgets"],
        "preds_10": preds_10,
        "preds_20": preds_20,
        "ckpt_manifest": ckpt_man,
        "reproduction": repro,
    }


def _compute_seed_prediction_changes(preds_10: list[dict], preds_20: list[dict], val_meta_by_rel: dict[str, dict]) -> dict:
    if [p["v2_image_relpath"] for p in preds_10] != [p["v2_image_relpath"] for p in preds_20]:
        raise ValueError("Val 74 ordering mismatch between 10ep and 20ep predictions")
    newly_correct = []
    newly_wrong = []
    shared_errors = []
    unchanged_correct = 0
    rows = []

    for p10, p20 in zip(preds_10, preds_20):
        rel = p10["v2_image_relpath"]
        meta = val_meta_by_rel[rel]
        c10 = p10["predicted_label"] == meta["label"]
        c20 = p20["predicted_label"] == meta["label"]
        s10 = p10["softmax"][1]
        s20 = p20["softmax"][1]

        if not c10 and c20:
            change_type = "newly_correct_at_20"
            newly_correct.append({
                "image_filename": meta["image_filename"],
                "label": meta["label"],
                "anomaly_category": meta["anomaly_category"],
                "pred_10": p10["predicted_label"],
                "fail_score_10": round(s10, 6),
                "pred_20": p20["predicted_label"],
                "fail_score_20": round(s20, 6),
                "score_delta_20_minus_10": round(s20 - s10, 6),
            })
        elif c10 and not c20:
            change_type = "newly_wrong_at_20"
            newly_wrong.append({
                "image_filename": meta["image_filename"],
                "label": meta["label"],
                "anomaly_category": meta["anomaly_category"],
                "pred_10": p10["predicted_label"],
                "fail_score_10": round(s10, 6),
                "pred_20": p20["predicted_label"],
                "fail_score_20": round(s20, 6),
                "score_delta_20_minus_10": round(s20 - s10, 6),
            })
        elif not c10 and not c20:
            change_type = "shared_error"
            shared_errors.append({
                "image_filename": meta["image_filename"],
                "label": meta["label"],
                "anomaly_category": meta["anomaly_category"],
                "pred_10": p10["predicted_label"],
                "fail_score_10": round(s10, 6),
                "pred_20": p20["predicted_label"],
                "fail_score_20": round(s20, 6),
                "score_delta_20_minus_10": round(s20 - s10, 6),
            })
        else:
            change_type = "both_correct"
            unchanged_correct += 1

        rows.append({
            "sample_id": meta["sample_id"],
            "image_filename": meta["image_filename"],
            "label": meta["label"],
            "source_batch": meta["source_batch"],
            "anomaly_category": meta["anomaly_category"],
            "pred_10": p10["predicted_label"],
            "fail_score_10": round(s10, 6),
            "pred_20": p20["predicted_label"],
            "fail_score_20": round(s20, 6),
            "score_delta": round(s20 - s10, 6),
            "change_type": change_type,
        })

    return {
        "both_correct_count": unchanged_correct,
        "newly_correct_count": len(newly_correct),
        "newly_wrong_count": len(newly_wrong),
        "shared_error_count": len(shared_errors),
        "newly_correct": newly_correct,
        "newly_wrong": newly_wrong,
        "shared_errors": shared_errors,
        "per_image_rows": rows,
    }


def compare_phase2_runs(
    s42_dir: str | Path,
    s43_dir: str | Path,
    s44_dir: str | Path,
    output_dir: str | Path,
    *,
    phase1_suite_dir: str | Path | None = DEFAULT_PHASE1_SUITE,
    dataset_v2_root: str | Path = DEFAULT_DATASET_V2_ROOT,
) -> Path:
    out = Path(output_dir).resolve()
    if out.exists():
        raise ValueError(f"comparison output directory already exists (refusing overwrite): {out}")

    runs = {
        42: inspect_phase2_run(s42_dir, "R4_RAW_CE_SEED42", 42),
        43: inspect_phase2_run(s43_dir, "R4_RAW_CE_SEED43", 43),
        44: inspect_phase2_run(s44_dir, "R4_RAW_CE_SEED44", 44),
    }

    prov_hashes = {s: r["run_config"]["data_provenance_sha256"] for s, r in runs.items()}
    if len(set(prov_hashes.values())) != 1:
        raise ValueError(f"data_provenance_sha256 mismatch across seeds: {prov_hashes}")

    backbone_hashes = {s: r["run_config"]["determinism_notes"]["initial_backbone_sha256"] for s, r in runs.items()}
    if len(set(backbone_hashes.values())) != 1:
        raise ValueError("initial frozen backbone differs across seeds")

    # Distinct seeds must produce distinct initial Linear head weights
    head_hashes = {s: r["run_config"]["determinism_notes"]["initial_head_sha256"] for s, r in runs.items()}
    if len(set(head_hashes.values())) != 3:
        raise ValueError(f"seeds 42/43/44 did not produce distinct initial head weights: {head_hashes}")

    val_manifest_path = Path(runs[42]["config"]["val_manifest"]).resolve()
    with val_manifest_path.open(encoding="utf-8", newline="") as fh:
        val_meta_rows = list(csv.DictReader(fh))
    val_meta_by_rel = {r["v2_image_relpath"]: r for r in val_meta_rows}

    seed_changes = {
        s: _compute_seed_prediction_changes(runs[s]["preds_10"], runs[s]["preds_20"], val_meta_by_rel)
        for s in (42, 43, 44)
    }

    # Optional targeted error analysis of Phase-1 R1 if Phase-1 suite exists
    error_report = None
    if phase1_suite_dir and (Path(phase1_suite_dir) / "R1_RAW_CE/model/best_val_predictions.json").is_file():
        error_report = analyze_r1_val_errors(
            phase1_suite_dir=phase1_suite_dir,
            prepared_dir=runs[42]["spec"]["prepared_dir"],
            dataset_v2_root=dataset_v2_root,
            image_size=runs[42]["config"]["image_size"],
        )

    out.mkdir(parents=True, exist_ok=False)

    # 1. Per-seed x budget CSV
    per_seed_csv_rows = []
    metric_specs = [
        ("selected_epoch", lambda b: b["selected_epoch"]),
        ("val_loss", lambda b: b["primary_val_74_metrics"]["loss"]),
        ("val_accuracy", lambda b: b["primary_val_74_metrics"]["accuracy"]),
        ("val_balanced_accuracy", lambda b: b["primary_val_74_metrics"]["balanced_accuracy"]),
        ("val_fail_recall", lambda b: b["primary_val_74_metrics"]["fail_recall"]),
        ("val_fail_precision", lambda b: b["primary_val_74_metrics"]["fail_precision"]),
        ("val_fail_f1", lambda b: b["primary_val_74_metrics"]["fail_f1"]),
        ("val_false_pass_FN", lambda b: b["primary_val_74_metrics"]["false_pass"]),
        ("val_false_fail_FP", lambda b: b["primary_val_74_metrics"]["false_fail"]),
        ("val_roc_auc", lambda b: b["primary_val_74_metrics"]["roc_auc"]),
        ("train294_fixed_loss", lambda b: b["unified_train_raw_294_metrics"]["loss"]),
        ("train294_fixed_accuracy", lambda b: b["unified_train_raw_294_metrics"]["accuracy"]),
        ("train294_fixed_FN", lambda b: b["unified_train_raw_294_metrics"]["false_pass"]),
        ("train294_fixed_FP", lambda b: b["unified_train_raw_294_metrics"]["false_fail"]),
        ("train294_fixed_roc_auc", lambda b: b["unified_train_raw_294_metrics"]["roc_auc"]),
        ("val_aug18_fail_recall", lambda b: b["val_aug_18_diagnostics"]["overall_18_augmented_FAIL"]["fail_recall"]),
        ("val_aug18_flips", lambda b: b["val_aug_18_diagnostics"]["total_flips_across_18"]),
    ]

    for s in (42, 43, 44):
        for b_key in ("10", "20"):
            b = runs[s]["budgets"][b_key]
            row = {
                "seed": s,
                "experiment_id": runs[s]["spec"]["experiment_id"],
                "budget_epochs": int(b_key),
                "checkpoint": b["checkpoint"],
                "exposures_within_budget": b["exposures_within_budget"],
                "optimizer_steps_within_budget": b["optimizer_steps_within_budget"],
            }
            for m_name, fn in metric_specs:
                row[m_name] = fn(b)
            per_seed_csv_rows.append(row)

    with (out / "phase2_per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(per_seed_csv_rows)

    # 2. Combined 74-image prediction changes CSV across all 3 seeds
    combined_val_rows = []
    for idx in range(len(runs[42]["preds_10"])):
        base = seed_changes[42]["per_image_rows"][idx]
        crow = {
            "sample_id": base["sample_id"],
            "image_filename": base["image_filename"],
            "label": base["label"],
            "source_batch": base["source_batch"],
            "anomaly_category": base["anomaly_category"],
        }
        for s in (42, 43, 44):
            sr = seed_changes[s]["per_image_rows"][idx]
            crow[f"s{s}_10ep_pred"] = sr["pred_10"]
            crow[f"s{s}_10ep_score"] = sr["fail_score_10"]
            crow[f"s{s}_20ep_pred"] = sr["pred_20"]
            crow[f"s{s}_20ep_score"] = sr["fail_score_20"]
            crow[f"s{s}_change"] = sr["change_type"]
        combined_val_rows.append(crow)

    with (out / "phase2_val_74_prediction_changes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined_val_rows[0].keys()))
        w.writeheader()
        w.writerows(combined_val_rows)

    # 3. Aggregate stats (10ep mean/std, 20ep mean/std, paired delta 20ep - 10ep mean/std)
    agg_stats = {}
    for m_name, fn in metric_specs:
        v10 = [fn(runs[s]["budgets"]["10"]) for s in (42, 43, 44)]
        v20 = [fn(runs[s]["budgets"]["20"]) for s in (42, 43, 44)]
        deltas = [b - a for a, b in zip(v10, v20)]
        agg_stats[m_name] = {
            "budget_10": _mean_std(v10),
            "budget_20": _mean_std(v20),
            "paired_delta_20_minus_10": _mean_std(deltas),
        }

    # 4. Plots (epochs 1..20 for all 3 seeds, vertical boundary at epoch 10, markers on selected best_within_10 & best_within_20)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
    for s in (42, 43, 44):
        hist = runs[s]["history"]
        eps = [h["epoch"] for h in hist]
        c = colors[s]
        ep10_sel = runs[s]["budgets"]["10"]["selected_epoch"]
        ep20_sel = runs[s]["budgets"]["20"]["selected_epoch"]

        axes[0, 0].plot(eps, [h["train"]["loss"] for h in hist], color=c, marker=".", label=f"seed={s}")
        axes[0, 1].plot(eps, [h["val"]["loss"] for h in hist], color=c, marker=".", label=f"seed={s} (b10=ep{ep10_sel}, b20=ep{ep20_sel})")
        axes[0, 1].scatter([ep10_sel], [hist[ep10_sel - 1]["val"]["loss"]], color=c, marker="o", s=55, zorder=5)
        axes[0, 1].scatter([ep20_sel], [hist[ep20_sel - 1]["val"]["loss"]], color=c, marker="*", s=95, zorder=6)

        axes[1, 0].plot(eps, [h["val"]["balanced_accuracy"] for h in hist], color=c, marker=".", label=f"seed={s}")
        axes[1, 1].plot(eps, [h["val"]["roc_auc"] for h in hist], color=c, marker=".", label=f"seed={s}")

    for ax in axes.flat:
        ax.axvline(10, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Epoch (vertical dashed line = 10-epoch budget boundary)")
        ax.set_xticks(range(2, 21, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0, 0].set_title("Online Train CE Loss (Epochs 1..20)")
    axes[0, 1].set_title("Primary Val (74 Real) Unweighted CE Loss (o=best_10, *=best_20)")
    axes[1, 0].set_title("Primary Val (74 Real) Balanced Accuracy")
    axes[1, 1].set_title("Primary Val (74 Real) ROC-AUC")
    fig.tight_layout()
    fig.savefig(out / "curves_phase2_budgets.png", dpi=150)
    plt.close(fig)

    summary_payload = {
        "schema_version": "member06.dataset_v2_phase2_comparison.v1",
        "precommitted_representative_seed": 42,
        "final_freeze_decision_made": False,
        "test_data_used": False,
        "note_on_metrics": "All reported checkpoint metrics use fixed-weight eval() re-evaluations of best_within_10.pt and best_within_20.pt; online epoch-20 train metrics are never substituted.",
        "seed42_reproduction_vs_phase1_r1": runs[42]["reproduction"],
        "per_seed_budgets": {str(s): runs[s]["budgets"] for s in (42, 43, 44)},
        "aggregate_3seed_statistics": agg_stats,
        "per_seed_val_74_prediction_changes": {
            str(s): {k: v for k, v in seed_changes[s].items() if k != "per_image_rows"}
            for s in (42, 43, 44)
        },
        "timing_and_cache": {
            str(s): {
                "cache_action": runs[s]["run_config"]["feature_cache"]["cache_action"],
                "timing_seconds": runs[s]["ckpt_manifest"]["timing_seconds"],
            }
            for s in (42, 43, 44)
        },
        "phase1_r1_targeted_val_error_analysis": error_report,
    }
    write_json(out / "phase2_summary.json", summary_payload)
    if error_report:
        write_json(out / "r1_val_error_analysis.json", error_report)

    # 5. Render comprehensive Chinese Markdown Report
    md = [
        "# Dataset v2 第二阶段：训练预算（10轮 vs 20轮）与多随机种子稳定性比较报告",
        "",
        "> **性质与边界声明**：本报告严格继承第一阶段优选方案 `R1_RAW_CE`（`294` 张真实训练原图、`74` 张真实验证原图、冻结 `ResNet18 + Linear(512,2)`、`224 letterbox`、无权 CE、`Adam lr=0.001`、默认 `argmax`），仅研究 **训练预算（10 Epoch vs 20 Epoch）** 与 **随机种子稳定性（`seed=42, 43, 44`）**。所有指标均来自已保存的固定最佳权重 `best_within_10.pt` 与 `best_within_20.pt` 的 `eval()` 重评估，**绝不使用第 20 轮在线训练指标冒充最佳模型表现**。全程未读取 `Test`（`test_data_used = false`），**本轮不自动作出最终冻结裁决**（最终交付代表种子已预先固定为 `seed=42`）。",
        "",
        "## 1. 实验设置、缓存复用与 `seed=42` 前 10 轮历史复现核验",
        "",
        "| 实验 ID | Seed | 10轮预算曝光/步数 | 20轮预算曝光/步数 | 特征缓存状态 | 20轮训练耗时 (s) | 总耗时 (s) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for s in (42, 43, 44):
        tm = runs[s]["ckpt_manifest"]["timing_seconds"]
        ca = runs[s]["run_config"]["feature_cache"]["cache_action"]
        md.append(
            f"| `{runs[s]['spec']['experiment_id']}` | {s} | 2940 / 190 | 5880 / 380 | `{ca}` | "
            f"{tm['training_epochs_seconds']:.2f} | {tm['total_experiment_seconds']:.2f} |"
        )

    rep = runs[42]["reproduction"]
    if rep:
        md += [
            "",
            "### `R4_RAW_CE_SEED42` 前 10 轮相对第一阶段 `R1_RAW_CE` 复现核验",
            f"- **核验状态**：`{rep['status']}`（容差 `tol = {rep['tolerance']}`）",
            f"- **初始 Backbone / Linear Head 参数哈希一致**：`{rep['initial_backbone_sha256_match']}` / `{rep['initial_head_sha256_match']}`",
            f"- **第 1–10 轮每轮实际打乱样本顺序 SHA-256 逐轮完全一致**：`{rep['all_10_epochs_sample_order_exact_match']}`",
            f"- **第 1–10 轮最大 Train Loss / Val Loss 绝对误差**：`{rep['max_abs_train_loss_diff_epochs_1_10']:.3e}` / `{rep['max_abs_val_loss_diff_epochs_1_10']:.3e}`",
            f"- **`best_within_10.pt`（Epoch {rep['best_epoch_within_10']}）相对历史 `R1/best.pt`（Epoch {rep['reference_best_epoch']}）模型参数最大绝对误差**：`head.weight = {rep['head_weight_max_abs_diff']:.3e}`，`head.bias = {rep['head_bias_max_abs_diff']:.3e}`，`backbone = {rep['backbone_max_abs_diff']:.3e}`（`state_dict` 字节哈希完全一致 = `{rep['model_state_sha256_exact_match']}`；注：因 `.pt` 内含 `epochs=20` 配置及时间戳元数据，不以文件级哈希替代张量核验）。",
        ]

    md += [
        "",
        "## 2. 逐 Seed 的 10 轮预算 vs 20 轮预算核心指标明细",
        "",
        "| Seed | 预算 | 最佳Epoch | Val74 Loss | Val Acc | Val BA | FAIL Recall | FAIL Prec | FAIL F1 | FN / FP | Val AUC | Train294固定Loss / Acc / AUC | Val增强18 Recall (翻转) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for s in (42, 43, 44):
        for bk, blabel in (("10", "10轮 (`best_within_10`)"), ("20", "20轮 (`best_within_20`)")):
            b = runs[s]["budgets"][bk]
            vm = b["primary_val_74_metrics"]
            tm = b["unified_train_raw_294_metrics"]
            am = b["val_aug_18_diagnostics"]
            md.append(
                f"| {s} | {blabel} | {b['selected_epoch']} | {vm['loss']:.6f} | {vm['accuracy']:.4f} | {vm['balanced_accuracy']:.4f} | "
                f"{vm['fail_recall']:.4f} | {vm['fail_precision']:.4f} | {vm['fail_f1']:.4f} | {vm['false_pass']} / {vm['false_fail']} | "
                f"{vm['roc_auc']:.4f} | {tm['loss']:.4f} / {tm['accuracy']:.4f} / {tm['roc_auc']:.4f} | "
                f"{am['overall_18_augmented_FAIL']['fail_recall']:.4f} ({am['total_flips_across_18']}/18) |"
            )

    md += [
        "",
        "## 3. 三组随机种子（42 / 43 / 44）均值 ± 样本标准差（`n=3, ddof=1`）与配对差值",
        "",
        "| 指标 | 10 轮预算 (`Mean ± SD`) | 20 轮预算 (`Mean ± SD`) | 配对差 `20轮 - 10轮` (`Mean ± SD`) | 三个 Seed 差值明细 `(s42, s43, s44)` |",
        "|---|---|---|---|---|",
    ]
    display_labels = [
        ("selected_epoch", "最佳 Epoch", 2),
        ("val_loss", "主验证集无权 Val Loss", 6),
        ("val_accuracy", "Val Accuracy", 4),
        ("val_balanced_accuracy", "Val Balanced Accuracy", 4),
        ("val_fail_recall", "Val FAIL Recall", 4),
        ("val_fail_precision", "Val FAIL Precision", 4),
        ("val_fail_f1", "Val FAIL F1", 4),
        ("val_false_pass_FN", "Val False PASS (FN)", 2),
        ("val_false_fail_FP", "Val False FAIL (FP)", 2),
        ("val_roc_auc", "Val ROC-AUC", 4),
        ("train294_fixed_loss", "Train 294 固定重评 Loss", 6),
        ("train294_fixed_accuracy", "Train 294 固定重评 Acc", 4),
        ("train294_fixed_roc_auc", "Train 294 固定重评 AUC", 4),
        ("val_aug18_fail_recall", "Val 增强 18 FAIL Recall", 4),
        ("val_aug18_flips", "Val 增强 18 分类翻转数", 2),
    ]
    for key, label_cn, dec in display_labels:
        st = agg_stats[key]
        b10, b20, dlt = st["budget_10"], st["budget_20"], st["paired_delta_20_minus_10"]
        vals_str = ", ".join(f"{v:+.{dec}f}" for v in dlt["values"])
        md.append(
            f"| {label_cn} | `{b10['mean']:.{dec}f} ± {b10['sample_std']:.{dec}f}` | "
            f"`{b20['mean']:.{dec}f} ± {b20['sample_std']:.{dec}f}` | "
            f"`{dlt['mean']:+.{dec}f} ± {dlt['sample_std']:.{dec}f}` | `({vals_str})` |"
        )

    md += [
        "",
        "## 4. 逐 Seed 的 10 轮候选 vs 20 轮候选逐图预测变化（74 张真实 Val）",
        "",
        "| Seed | 两预算均正确 | **20轮新增正确** (`10错→20对`) | **20轮新增错误** (`10对→20错`) | **两预算共同错误** | 具体变化与共同错误明细（`10轮分数 → 20轮分数`） |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for s in (42, 43, 44):
        ch = seed_changes[s]
        notes = []
        for x in ch["newly_correct"]:
            notes.append(f"新增正确 `{x['image_filename']}` ({x['label']}, `{x['fail_score_10']:.4f}→{x['fail_score_20']:.4f}`)")
        for x in ch["newly_wrong"]:
            notes.append(f"新增错误 `{x['image_filename']}` ({x['label']}, `{x['fail_score_10']:.4f}→{x['fail_score_20']:.4f}`)")
        for x in ch["shared_errors"]:
            notes.append(f"共同错误 `{x['image_filename']}` ({x['label']}, `{x['fail_score_10']:.4f}→{x['fail_score_20']:.4f}`)")
        md.append(
            f"| {s} | {ch['both_correct_count']} | **{ch['newly_correct_count']}** | **{ch['newly_wrong_count']}** | "
            f"**{ch['shared_error_count']}** | {'；'.join(notes) if notes else '无'} |"
        )

    if error_report:
        md += ["", render_error_analysis_markdown(error_report)]

    md += [
        "",
        "## 5. 预先约定的模型选择原则（本报告不自动作出最终冻结裁决）",
        "",
        "1. **预算升级条件**：`20 Epoch` 方案须在三个随机种子上呈现可解释的平均验证收益（如无权 `Val Loss`、`ROC-AUC`、`Balanced Accuracy` 一致改善），且 `FN`（漏放）与 `FP`（误拒）无不可接受恶化，才值得替换更简单的 `10 Epoch` 方案；若仅为微小偶然波动、跨种子方向不一致或无实质收益，优先保留 `10 Epoch` 预算。",
        "2. **代表种子纪律**：最终交付的代表模型种子已预先固定为 **`seed = 42`**（`R4_RAW_CE_SEED42` 的 `best_within_10.pt` 或 `best_within_20.pt`），严禁根据 `seed=42/43/44` 中谁在验证集最高分而事后挑种子。",
        "3. **后续决策交接**：请结合本报告曲线（`curves_phase2_budgets.png`）、逐图变化表（`phase2_val_74_prediction_changes.csv`）及两张漏放图的像素尺度分析，决定是否直接选定 `224` 预算方案或进一步启动 `384` 分辨率受控对照。",
    ]

    (out / "COMPARISON_PHASE2_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42", required=True, help="path to R4_RAW_CE_SEED42 model directory")
    parser.add_argument("--seed43", required=True, help="path to R4_RAW_CE_SEED43 model directory")
    parser.add_argument("--seed44", required=True, help="path to R4_RAW_CE_SEED44 model directory")
    parser.add_argument("--phase1-suite", default=DEFAULT_PHASE1_SUITE, help="optional Phase-1 suite path for R1 targeted error analysis")
    parser.add_argument("--output", required=True, help="new output directory for Phase-2 comparison report")
    args = parser.parse_args()
    try:
        out = compare_phase2_runs(
            args.seed42,
            args.seed43,
            args.seed44,
            args.output,
            phase1_suite_dir=args.phase1_suite,
        )
        print(json.dumps({"status": "PHASE2_COMPARISON_GENERATED", "output_dir": str(out)}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"V2_PHASE2_COMPARISON_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
