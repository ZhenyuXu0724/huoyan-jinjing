"""Unified read-only comparison for final_training_v4 6 core runs (V4-R1 & V4-R2 x seeds 42/43/44 x 10/20 budgets).

Outputs:
- COMPARISON_V4_CN.md
- v4_summary.json
- v4_per_run_budget_metrics.csv
- v4_val_34_per_image_comparison.csv
- v4_val_34_grouped_slices.csv
- curves_v4_6runs.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .runtime import file_sha256, load_checkpoint, write_json
from .v2_train import compute_rich_metrics
from .v4_data import load_verified_v4_manifests

REQUIRED_V4_FILES = (
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

SEEDS = (42, 43, 44)
STRATEGIES = ("V4-R1_RAW_CE", "V4-R2_RAW_WEIGHTED_CE")


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float | int | None]) -> dict | None:
    if any(v is None for v in values):
        return None
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "pop_std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(x) for x in values],
    }


def _fmt_ms(ms: dict | None, dec: int = 4) -> str:
    if ms is None:
        return "—"
    return f"{ms['mean']:.{dec}f} ± {ms['sample_std']:.{dec}f}"


def inspect_v4_run(model_dir: str | Path, expected_strategy: str, expected_seed: int) -> dict:
    d = Path(model_dir).resolve()
    missing = [f for f in REQUIRED_V4_FILES if not (d / f).is_file()]
    if missing:
        raise ValueError(f"incomplete v4 run {d} (missing {missing})")

    spec = _read_json(d / "experiment_spec.json")
    cfg = _read_json(d / "config.json")
    run_cfg = _read_json(d / "run_config.json")
    history = _read_json(d / "history.json")
    orders = _read_json(d / "epoch_sample_orders.json")
    budget_evals = _read_json(d / "budget_evaluations.json")
    ckpt_man = _read_json(d / "checkpoint_manifest.json")
    preds_10 = _read_json(d / "best_within_10_val_predictions.json")
    preds_20 = _read_json(d / "best_within_20_val_predictions.json")

    expected_exp_id = f"{expected_strategy}_SEED{expected_seed}"
    if spec.get("experiment_id") != expected_exp_id or cfg.get("seed") != expected_seed:
        raise ValueError(f"expected {expected_exp_id} (seed={expected_seed}), got {spec.get('experiment_id')} (seed={cfg.get('seed')})")
    if run_cfg.get("dataset_id") != "final_training_v4":
        raise ValueError(f"run {d} is not bound to final_training_v4")
    if run_cfg["selection_protocol"].get("test_data_used") is not False or ckpt_man.get("test_data_used") is not False:
        raise ValueError(f"Test exclusion violated in {d}")

    allow_smoke = cfg["data_kind"] == "SMOKE_ONLY"
    allow_dev = cfg["training_tier"] == "DEV_ONLY"

    for m_str, m_int in (("10", 10), ("20", 20)):
        m_info = ckpt_man["milestones"][m_str]
        b_file = d / m_info["best_within_budget"]["file"]
        e_file = d / m_info["epoch_end"]["file"]
        if file_sha256(b_file) != m_info["best_within_budget"]["sha256"]:
            raise ValueError(f"SHA256 mismatch on {b_file}")
        if file_sha256(e_file) != m_info["epoch_end"]["sha256"]:
            raise ValueError(f"SHA256 mismatch on {e_file}")
        _, _, _, b_pay = load_checkpoint(b_file, device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev)
        _, _, _, e_pay = load_checkpoint(e_file, device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev)
        expected_best = min(history[:m_int], key=lambda h: h["val"]["loss"])
        if b_pay["epoch"] != expected_best["epoch"]:
            raise ValueError(f"best_within_{m_int} epoch selection mismatch in {d}")
        if e_pay["epoch"] != m_int:
            raise ValueError(f"epoch{m_int}.pt epoch mismatch in {d}")

    return {
        "dir": str(d),
        "strategy": expected_strategy,
        "seed": expected_seed,
        "spec": spec,
        "config": cfg,
        "run_config": run_cfg,
        "history": history,
        "orders": orders,
        "budgets": budget_evals["budgets"],
        "preds": {"10": preds_10, "20": preds_20},
        "ckpt_manifest": ckpt_man,
    }


def compare_v4_suite(
    run_dirs: dict[tuple[str, int], str | Path],
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir).resolve()
    if out.exists():
        raise ValueError(f"comparison output directory already exists (refusing overwrite): {out}")

    runs: dict[tuple[str, int], dict] = {}
    for strat in STRATEGIES:
        for s in SEEDS:
            runs[(strat, s)] = inspect_v4_run(run_dirs[(strat, s)], strat, s)

    # 1. Verify global dataset/manifest/backbone identity across all 6 runs
    ref_rc = runs[("V4-R1_RAW_CE", 42)]["run_config"]
    for key, r in runs.items():
        if r["run_config"]["manifests"] != ref_rc["manifests"]:
            raise ValueError(f"manifest hash mismatch in {key}")
        if r["run_config"]["determinism_notes"]["initial_backbone_sha256"] != ref_rc["determinism_notes"]["initial_backbone_sha256"]:
            raise ValueError(f"initial frozen backbone mismatch in {key}")

    # 2. Verify same-seed synchronization between V4-R1 and V4-R2 (identical initial head AND identical per-epoch sample order!)
    same_seed_sync = {}
    for s in SEEDS:
        r1 = runs[("V4-R1_RAW_CE", s)]
        r2 = runs[("V4-R2_RAW_WEIGHTED_CE", s)]
        h1 = r1["run_config"]["determinism_notes"]["initial_head_sha256"]
        h2 = r2["run_config"]["determinism_notes"]["initial_head_sha256"]
        if h1 != h2:
            raise ValueError(f"seed={s} initial Linear head differs between V4-R1 and V4-R2!")
        if r1["orders"] != r2["orders"]:
            raise ValueError(f"seed={s} per-epoch sample order differs between V4-R1 and V4-R2!")
        if r1["run_config"]["loss_protocol"]["train_class_weight"] != "none":
            raise ValueError("V4-R1 must use unweighted CE")
        if r2["run_config"]["loss_protocol"]["train_class_weight"] != "balanced":
            raise ValueError("V4-R2 must use balanced CE")
        same_seed_sync[str(s)] = {
            "initial_head_sha256": h1,
            "all_epochs_sample_order_identical": True,
            "r2_weights_pass0_fail1": r2["run_config"]["loss_protocol"]["weights"],
        }

    if len({same_seed_sync[str(s)]["initial_head_sha256"] for s in SEEDS}) != len(SEEDS):
        raise ValueError("seeds 42/43/44 did not produce distinct initial Linear heads")

    # Load verified Val metadata for rich slice grouping
    ref_cfg = runs[("V4-R1_RAW_CE", 42)]["config"]
    prov, train_rows, val_rows = load_verified_v4_manifests(
        ref_cfg["train_manifest"], ref_cfg["val_manifest"], ref_cfg["data_provenance"]
    )
    val_meta_by_name = {r["image_filename"]: r for r in val_rows}

    out.mkdir(parents=True, exist_ok=False)

    # 3. Per-run x budget CSV (12 rows: 2 strategies x 3 seeds x 2 budgets)
    metric_extractors = [
        ("selected_epoch", lambda b: b["selected_epoch"]),
        ("val_loss", lambda b: b["val_metrics"]["loss"]),
        ("val_accuracy", lambda b: b["val_metrics"]["accuracy"]),
        ("val_balanced_accuracy", lambda b: b["val_metrics"]["balanced_accuracy"]),
        ("val_fail_precision", lambda b: b["val_metrics"]["fail_precision"]),
        ("val_fail_recall", lambda b: b["val_metrics"]["fail_recall"]),
        ("val_fail_f1", lambda b: b["val_metrics"]["fail_f1"]),
        ("val_false_pass_FN", lambda b: b["val_metrics"]["false_pass"]),
        ("val_false_fail_FP", lambda b: b["val_metrics"]["false_fail"]),
        ("val_TN", lambda b: b["val_metrics"]["confusion_matrix"][0][0]),
        ("val_TP", lambda b: b["val_metrics"]["confusion_matrix"][1][1]),
        ("val_roc_auc", lambda b: b["val_metrics"]["roc_auc"]),
        ("train_fixed_loss", lambda b: b["unified_primary_train_metrics"]["loss"]),
        ("train_fixed_accuracy", lambda b: b["unified_primary_train_metrics"]["accuracy"]),
        ("train_fixed_balanced_accuracy", lambda b: b["unified_primary_train_metrics"]["balanced_accuracy"]),
        ("train_fixed_FN", lambda b: b["unified_primary_train_metrics"]["false_pass"]),
        ("train_fixed_FP", lambda b: b["unified_primary_train_metrics"]["false_fail"]),
        ("train_fixed_roc_auc", lambda b: b["unified_primary_train_metrics"]["roc_auc"]),
    ]

    per_run_rows = []
    for strat in STRATEGIES:
        for b_key in ("10", "20"):
            for s in SEEDS:
                r = runs[(strat, s)]
                b = r["budgets"][b_key]
                row = {
                    "strategy": strat,
                    "budget_epochs": int(b_key),
                    "seed": s,
                    "experiment_id": r["spec"]["experiment_id"],
                    "checkpoint": b["checkpoint"],
                    "exposures_within_budget": b["exposures_within_budget"],
                    "optimizer_steps_within_budget": b["optimizer_steps_within_budget"],
                    "training_20ep_wall_seconds": r["ckpt_manifest"]["timing_seconds"]["training_epochs_seconds"],
                    "total_run_wall_seconds": r["ckpt_manifest"]["timing_seconds"]["total_experiment_seconds"],
                }
                for m_name, fn in metric_extractors:
                    row[m_name] = fn(b)
                per_run_rows.append(row)

    with (out / "v4_per_run_budget_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)

    # 4. Aggregate 3-seed statistics per (strategy, budget) + paired comparisons
    candidates = [
        ("R1_10", "V4-R1_RAW_CE", "10"),
        ("R1_20", "V4-R1_RAW_CE", "20"),
        ("R2_10", "V4-R2_RAW_WEIGHTED_CE", "10"),
        ("R2_20", "V4-R2_RAW_WEIGHTED_CE", "20"),
    ]
    agg_by_candidate: dict[str, dict] = {}
    for cid, strat, b_key in candidates:
        agg_by_candidate[cid] = {
            m_name: _mean_std([fn(runs[(strat, s)]["budgets"][b_key]) for s in SEEDS])
            for m_name, fn in metric_extractors
        }

    # 5. Val per-image comparison CSV across all 12 (strategy, budget, seed) checkpoints
    val_filenames = [p["image_filename"] for p in runs[("V4-R1_RAW_CE", 42)]["preds"]["10"]]
    per_img_rows = []
    for idx, fname in enumerate(val_filenames):
        meta = val_meta_by_name[fname]
        crow = {
            "image_filename": fname,
            "label": meta["label"],
            "scene_id": meta["scene_id"],
            "source_batch": meta["source_batch"],
            "anomaly_mechanism": meta["anomaly_mechanism"],
            "anomaly_category": meta["anomaly_category"],
            "non_catalogue_count": meta["non_catalogue_count"],
        }
        all_preds = set()
        for cid, strat, b_key in candidates:
            for s in SEEDS:
                p = runs[(strat, s)]["preds"][b_key][idx]
                if p["image_filename"] != fname:
                    raise ValueError(f"Val image ordering mismatch on {fname}")
                crow[f"{cid}_s{s}_pred"] = p["predicted_label"]
                crow[f"{cid}_s{s}_score"] = round(p["softmax"][1], 6)
                all_preds.add(p["predicted_label"])
        crow["any_disagreement"] = len(all_preds) > 1
        per_img_rows.append(crow)

    with (out / "v4_val_34_per_image_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_img_rows[0].keys()))
        w.writeheader()
        w.writerows(per_img_rows)

    # 6. Grouped Val slice statistics (source_batch, anomaly_mechanism, non_catalogue_count) with explicit single-class warnings
    group_fields = ("source_batch", "anomaly_mechanism", "non_catalogue_count")
    slice_rows = []
    for gf in group_fields:
        vals = sorted({str(val_meta_by_name[fn][gf]) for fn in val_filenames})
        for gv in vals:
            idxs = [i for i, fn in enumerate(val_filenames) if str(val_meta_by_name[fn][gf]) == gv]
            n_pass = sum(val_meta_by_name[val_filenames[i]]["label"] == "PASS" for i in idxs)
            n_fail = sum(val_meta_by_name[val_filenames[i]]["label"] == "FAIL" for i in idxs)
            single_class = (n_pass == 0 or n_fail == 0)
            slice_note = (
                "SINGLE_CLASS_SLICE_PASS_ONLY" if n_fail == 0
                else ("SINGLE_CLASS_SLICE_FAIL_ONLY" if n_pass == 0 else "TWO_CLASS_SLICE")
            )
            for cid, strat, b_key in candidates:
                accs, fns, fps = [], [], []
                for s in SEEDS:
                    sub_p = [runs[(strat, s)]["preds"][b_key][i] for i in idxs]
                    sub_ce = sum(-math.log(max(p["softmax"][p["target_index"]], 1e-300)) for p in sub_p) / len(sub_p)
                    gm = compute_rich_metrics(sub_p, sub_ce)
                    accs.append(gm["accuracy"])
                    fns.append(gm["false_pass"])
                    fps.append(gm["false_fail"])
                slice_rows.append({
                    "group_field": gf,
                    "group_value": gv,
                    "candidate": cid,
                    "n_total": len(idxs),
                    "support_PASS": n_pass,
                    "support_FAIL": n_fail,
                    "slice_coverage_type": slice_note,
                    "single_class_warning": single_class,
                    "mean_accuracy_3seeds": round(float(np.mean(accs)), 6),
                    "sd_accuracy_3seeds": round(float(np.std(accs, ddof=1)), 6),
                    "mean_FN_3seeds": round(float(np.mean(fns)), 4),
                    "mean_FP_3seeds": round(float(np.mean(fps)), 4),
                    "seed_FN_list": "|".join(str(x) for x in fns),
                    "seed_FP_list": "|".join(str(x) for x in fps),
                })

    with (out / "v4_val_34_grouped_slices.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(slice_rows[0].keys()))
        w.writeheader()
        w.writerows(slice_rows)

    # 7. Training & Validation curves across all 6 runs (epochs 1..20)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    seed_colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
    strat_styles = {"V4-R1_RAW_CE": "-", "V4-R2_RAW_WEIGHTED_CE": "--"}

    for strat in STRATEGIES:
        short = "R1(raw)" if strat == "V4-R1_RAW_CE" else "R2(weighted)"
        for s in SEEDS:
            hist = runs[(strat, s)]["history"]
            eps = [h["epoch"] for h in hist]
            lbl = f"{short} s={s}"
            kw = {"color": seed_colors[s], "linestyle": strat_styles[strat], "alpha": 0.85}
            axes[0, 0].plot(eps, [h["train"]["loss"] for h in hist], label=lbl, **kw)
            axes[0, 1].plot(eps, [h["val"]["loss"] for h in hist], label=lbl, **kw)
            axes[1, 0].plot(eps, [h["val"]["balanced_accuracy"] for h in hist], label=lbl, **kw)
            axes[1, 1].plot(eps, [h["val"]["roc_auc"] for h in hist], label=lbl, **kw)

    for ax in axes.flat:
        ax.axvline(10, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Epoch (dotted line = 10-epoch budget)")
        ax.set_xticks(range(2, 21, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)

    axes[0, 0].set_title("Online Train Loss (Solid=Unweighted CE, Dashed=Weighted CE)")
    axes[0, 1].set_title(f"Primary Val ({len(val_rows)} Real) Unweighted CE Loss")
    axes[1, 0].set_title(f"Primary Val ({len(val_rows)} Real) Balanced Accuracy")
    axes[1, 1].set_title(f"Primary Val ({len(val_rows)} Real) ROC-AUC")
    fig.tight_layout()
    fig.savefig(out / "curves_v4_6runs.png", dpi=150)
    plt.close(fig)

    summary_payload = {
        "schema_version": "member06.final_training_v4_comparison.v1",
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": prov["upstream_commit"],
        "manifests": ref_rc["manifests"],
        "dataset_counts": {
            "train_primary": len(train_rows),
            "val_primary": len(val_rows),
        },
        "same_seed_head_and_order_sync_verified": same_seed_sync,
        "test_data_used": False,
        "final_freeze_selected_automatically": False,
        "cross_dataset_accuracy_comparison_forbidden": True,
        "aggregate_3seed_candidates": agg_by_candidate,
        "per_run_budget_rows": per_run_rows,
    }
    write_json(out / "v4_summary.json", summary_payload)

    # 8. Render comprehensive Chinese Markdown Report
    w_r2 = same_seed_sync["42"]["r2_weights_pass0_fail1"]
    md = [
        "# Baseline 2：`final_training_v4` 核心实验统一比较报告（V4-R1 vs V4-R2 × 10/20轮 × 3 Seeds）",
        "",
        f"- **权威数据版本**：`members/member03_data/final_training_v4/`（上游主分支提交 `{prov['upstream_commit']}`）",
        f"- **清单哈希**：`train.csv = {ref_rc['manifests']['train_manifest_sha256'][:16]}...`（`{len(train_rows)}` 张原图：`166 PASS / 83 FAIL`），`val.csv = {ref_rc['manifests']['val_manifest_sha256'][:16]}...`（`{len(val_rows)}` 张原图：`20 PASS / 14 FAIL`）",
        f"- **方法与隔离声明**：Backbone（`IMAGENET1K_V1`）与 BatchNorm 完全冻结，每组重新初始化 `Linear(512, 2)`，禁止继承任何历史训练头权重；`Weighted CE`（`PASS={w_r2[0]:.2f}, FAIL={w_r2[1]:.2f}`）仅用于 `V4-R2` 训练损失，主验证集与固定权重重评估始终采用**无权 CE**。全程未读取任何 `Test` 图像或标注（`test_data_used = false`）。**不直接与此前不同数据集（Dataset v1/v2）的验证正确率跨集比较以宣称提升；本报告暂不自动选定最终冻结模型。**",
        "",
        "## 1. 六次真实训练运行概览与同 Seed 同步核验",
        "",
        "| 实验 ID | 策略 | Seed | 每轮曝光 / 步数 | 10轮 / 20轮总步数 | 同 Seed 初始头与20轮顺序同步 | v4 专属缓存动作 | 缓存耗时 (s) | 20轮训练耗时 (s) | 总耗时 (s) |",
        "|---|---|---:|---|---:|---|---|---:|---:|---:|",
    ]
    for strat in STRATEGIES:
        for s in SEEDS:
            r = runs[(strat, s)]
            ea = r["ckpt_manifest"]["exposure_accounting"]
            tm = r["ckpt_manifest"]["timing_seconds"]
            ca = r["run_config"]["feature_cache"]["cache_action"]
            md.append(
                f"| `{r['spec']['experiment_id']}` | `{strat}` | {s} | {ea['exposures_per_epoch']} / {ea['optimizer_steps_per_epoch']} | "
                f"{ea['optimizer_steps_per_epoch']*10} / {ea['total_optimizer_steps']} | 已核验一致 (`{same_seed_sync[str(s)]['initial_head_sha256'][:10]}...`) | "
                f"`{ca}` | {tm['feature_cache_seconds']:.2f} | {tm['training_epochs_seconds']:.2f} | {tm['total_experiment_seconds']:.2f} |"
            )

    md += [
        "",
        "## 2. 逐 Seed 结果明细（34 张真实 Val + 统一 249 张真实 Train 固定最佳权重重评）",
        "",
        "> 混淆矩阵格式为 `[[TN, FP], [FN, TP]]`（行=真实 `[PASS, FAIL]`，列=预测 `[PASS, FAIL]`）。",
        "",
        "| 策略 | 预算 | Seed | 最佳Epoch | 无权Val Loss | Val Acc | Val BA | FAIL Prec | FAIL Recall | FAIL F1 | FN / FP | 混淆矩阵 `[[TN,FP],[FN,TP]]` | Val AUC | Train249固定 Loss / Acc / AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for strat in STRATEGIES:
        for b_key in ("10", "20"):
            for s in SEEDS:
                b = runs[(strat, s)]["budgets"][b_key]
                vm = b["val_metrics"]
                tm = b["unified_primary_train_metrics"]
                cm = vm["confusion_matrix"]
                prec_s = f"{vm['fail_precision']:.4f}" if vm["fail_precision"] is not None else "—"
                f1_s = f"{vm['fail_f1']:.4f}" if vm["fail_f1"] is not None else "—"
                md.append(
                    f"| `{strat}` | {b_key}轮 | {s} | {b['selected_epoch']} | {vm['loss']:.6f} | {vm['accuracy']:.4f} | "
                    f"{vm['balanced_accuracy']:.4f} | {prec_s} | {vm['fail_recall']:.4f} | {f1_s} | "
                    f"{vm['false_pass']} / {vm['false_fail']} | `[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]` | "
                    f"{vm['roc_auc']:.4f} | {tm['loss']:.4f} / {tm['accuracy']:.4f} / {tm['roc_auc']:.4f} |"
                )

    md += [
        "",
        "## 3. 三组随机种子（42 / 43 / 44）均值 ± 样本标准差（`n=3, ddof=1`）汇总对比",
        "",
        "| 指标 | `V4-R1` (10轮) | `V4-R1` (20轮) | `V4-R2 Weighted` (10轮) | `V4-R2 Weighted` (20轮) |",
        "|---|---|---|---|---|",
    ]
    summary_rows_spec = [
        ("selected_epoch", "最佳 Epoch", 2),
        ("val_loss", "无权 Val Loss", 6),
        ("val_accuracy", "Val Accuracy", 4),
        ("val_balanced_accuracy", "Val Balanced Accuracy", 4),
        ("val_fail_precision", "Val FAIL Precision", 4),
        ("val_fail_recall", "Val FAIL Recall", 4),
        ("val_fail_f1", "Val FAIL F1", 4),
        ("val_false_pass_FN", "False PASS (FN, /14)", 2),
        ("val_false_fail_FP", "False FAIL (FP, /20)", 2),
        ("val_roc_auc", "Val ROC-AUC", 4),
        ("train_fixed_loss", "Train 249 固定无权 Loss", 6),
        ("train_fixed_accuracy", "Train 249 固定 Accuracy", 4),
        ("train_fixed_balanced_accuracy", "Train 249 固定 Balanced Acc", 4),
        ("train_fixed_roc_auc", "Train 249 固定 ROC-AUC", 4),
    ]
    for m_key, m_lbl, dec in summary_rows_spec:
        md.append(
            f"| {m_lbl} | `{_fmt_ms(agg_by_candidate['R1_10'][m_key], dec)}` | "
            f"`{_fmt_ms(agg_by_candidate['R1_20'][m_key], dec)}` | "
            f"`{_fmt_ms(agg_by_candidate['R2_10'][m_key], dec)}` | "
            f"`{_fmt_ms(agg_by_candidate['R2_20'][m_key], dec)}` |"
        )

    md += [
        "",
        "## 4. 34 张真实 Val 按来源批次、异常机制及干扰件数量分组统计",
        "",
        "> **重要统计警示**：`final_training_v4` 验证集中各拍摄批次（`11/11`、`11/14`、`11/23` 全为 `PASS`；`11/18`、`11/20` 全为 `FAIL`）以及各非零干扰件切片均属于**单一类别切片（`SINGLE_CLASS_SLICE`）**。单一类别切片没有跨类判别基数（`Balanced Accuracy` 与 `ROC-AUC` 无定义），下表仅报告各切片支持数与 3-seed 平均命中/误判计数，**严禁将单一类别切片的高准确率单独引申为该批次下的完整两类泛化证据**。",
        "",
        "| 分组维度 | 切片值 | 支持数 (`PASS / FAIL`) | 切片性质 | `R1_10` 均Acc (均FN/均FP) | `R1_20` 均Acc (均FN/均FP) | `R2_10` 均Acc (均FN/均FP) | `R2_20` 均Acc (均FN/均FP) |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for gf in group_fields:
        sub_s = [r for r in slice_rows if r["group_field"] == gf]
        for gv in sorted({r["group_value"] for r in sub_s}):
            by_c = {r["candidate"]: r for r in sub_s if r["group_value"] == gv}
            r0 = by_c["R1_10"]
            md.append(
                f"| `{gf}` | `{gv}` | {r0['n_total']} (`{r0['support_PASS']}P / {r0['support_FAIL']}F`) | `{r0['slice_coverage_type']}` | "
                f"{by_c['R1_10']['mean_accuracy_3seeds']:.4f} ({by_c['R1_10']['mean_FN_3seeds']:.2f}/{by_c['R1_10']['mean_FP_3seeds']:.2f}) | "
                f"{by_c['R1_20']['mean_accuracy_3seeds']:.4f} ({by_c['R1_20']['mean_FN_3seeds']:.2f}/{by_c['R1_20']['mean_FP_3seeds']:.2f}) | "
                f"{by_c['R2_10']['mean_accuracy_3seeds']:.4f} ({by_c['R2_10']['mean_FN_3seeds']:.2f}/{by_c['R2_10']['mean_FP_3seeds']:.2f}) | "
                f"{by_c['R2_20']['mean_accuracy_3seeds']:.4f} ({by_c['R2_20']['mean_FN_3seeds']:.2f}/{by_c['R2_20']['mean_FP_3seeds']:.2f}) |"
            )

    disagree_imgs = [r["image_filename"] for r in per_img_rows if r["any_disagreement"]]
    md += [
        "",
        "## 5. 34 张真实 Val 逐图分歧与后续候选说明",
        "",
        f"- **逐图对比表**：见 `v4_val_34_per_image_comparison.csv`。在全部 12 个候选检查点（2 策略 × 2 预算 × 3 种子）之间存在预测分歧的验证图片共 **`{len(disagree_imgs)} / {len(val_rows)}`** 张"
        + (f"：`{', '.join(disagree_imgs)}`" if disagree_imgs else "。")
        + "",
        "- **保留的后续候选能力（本轮未启动）**：",
        "  1. `351` 张含可选离线增强清单已预备于 `members/member06_research/data/final_training_v4_intake/train_with_optional_aug.csv`（配置模板 `configs/final_training_v4/v4_optional_aug351_template.json`）；",
        "  2. `image_size = 384` 分辨率参数在 `v4_train.py` 与独立专属缓存命名中原生支持。",
        "- **冻结决策待定**：请根据本报告的 6 组真实训练指标、误放/误拒权衡及训练曲线 `curves_v4_6runs.png`，共同裁决是否直接从本次候选中冻结最终 `baseline2` 模型或追加单组受控对照。",
    ]

    (out / "COMPARISON_V4_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", help="suite directory containing V4-R1_RAW_CE_SEED{42,43,44} and V4-R2_RAW_WEIGHTED_CE_SEED{42,43,44}")
    parser.add_argument("--r1-s42")
    parser.add_argument("--r1-s43")
    parser.add_argument("--r1-s44")
    parser.add_argument("--r2-s42")
    parser.add_argument("--r2-s43")
    parser.add_argument("--r2-s44")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.suite_dir:
            sd = Path(args.suite_dir).resolve()
            run_dirs = {
                ("V4-R1_RAW_CE", s): sd / f"V4-R1_RAW_CE_SEED{s}/model"
                for s in SEEDS
            }
            run_dirs.update({
                ("V4-R2_RAW_WEIGHTED_CE", s): sd / f"V4-R2_RAW_WEIGHTED_CE_SEED{s}/model"
                for s in SEEDS
            })
        else:
            req = [args.r1_s42, args.r1_s43, args.r1_s44, args.r2_s42, args.r2_s43, args.r2_s44]
            if any(x is None for x in req):
                parser.error("provide either --suite-dir or all six --r1-s42..--r2-s44 paths")
            run_dirs = {
                ("V4-R1_RAW_CE", 42): args.r1_s42,
                ("V4-R1_RAW_CE", 43): args.r1_s43,
                ("V4-R1_RAW_CE", 44): args.r1_s44,
                ("V4-R2_RAW_WEIGHTED_CE", 42): args.r2_s42,
                ("V4-R2_RAW_WEIGHTED_CE", 43): args.r2_s43,
                ("V4-R2_RAW_WEIGHTED_CE", 44): args.r2_s44,
            }
        out = compare_v4_suite(run_dirs, args.output)
        print(json.dumps({"status": "V4_COMPARISON_GENERATED", "output_dir": str(out)}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"V4_COMPARISON_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
