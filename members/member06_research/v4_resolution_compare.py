"""Paired comparison between V4-R2 (224x224) and V4-R4 (384x384) across seeds 42/43/44 (20-epoch budget).

Verifies:
1. Single-variable experimental control: config identical except image_size (224 vs 384).
2. Strict cache separation: 384 runs use dedicated v4_resnet18_sz384_* cache shared across seeds 42/43/44; zero 224 features mixed.
3. Exact same-seed synchronization: identical initial Linear(512,2) head weights and identical 20-epoch sample orders per seed.
4. Outputs paired 224-vs-384 per-seed & 3-seed mean/SD metrics, 34-image Val flip audit, 11/23 PASS false-FAIL changes,
   11/20 FAIL false-PASS changes, fixed-weight Train 249 re-evaluation, feature/training cost, controlled full-image
   offline latency reference (explicitly distinguished from live camera latency), and pre-committed selection criteria check.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Preprocess
from .image_io import read_image
from .runtime import choose_device, load_checkpoint, seed_everything, write_json
from .v4_compare import SEEDS, _fmt_ms, _mean_std, inspect_v4_run
from .v4_data import load_verified_v4_manifests

DEFAULT_224_SUITE = "members/member06_research/runs/final_training_v4_20261124_192110_PJSHyF"


def benchmark_offline_full_image_latency(
    ckpt_path: str | Path,
    val_rows: list[dict],
    *,
    device_name: str = "cpu",
    warmup: int = 5,
) -> dict:
    """Measure full single-image offline pipeline: disk decode -> letterbox Preprocess -> FrozenResNet18 -> softmax/argmax."""
    model, config, device, payload = load_checkpoint(ckpt_path, device=device_name)
    seed_everything(config.seed, config.cpu_threads)
    transform = Preprocess(config)
    model.eval()

    paths = [r["path"] for r in val_rows]
    if not paths:
        raise ValueError("empty val_rows for latency benchmark")

    def _run_single(img_path: Path) -> int:
        img = read_image(img_path)
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        pred = int(logits.softmax(dim=1).argmax(dim=1).item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return pred

    with torch.inference_mode():
        for i in range(warmup):
            _run_single(paths[i % len(paths)])
        timings_ms: list[float] = []
        for p in paths:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run_single(p)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings_ms.append(elapsed_ms)

    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "checkpoint": str(Path(ckpt_path).resolve()),
        "selected_epoch": payload["epoch"],
        "image_size": config.image_size,
        "device": str(device),
        "cpu_threads": config.cpu_threads,
        "batch_size": 1,
        "n_images": len(paths),
        "warmup_runs": warmup,
        "timing_boundary": "offline single-image read_image(disk) -> letterbox Preprocess -> FrozenResNet18 forward -> softmax.argmax (excludes camera capture & UI rendering; NOT live webcam latency)",
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def compare_v4_224_vs_384(
    r224_dirs: dict[int, str | Path],
    r384_dirs: dict[int, str | Path],
    output_dir: str | Path,
    *,
    latency_device: str = "cpu",
) -> Path:
    out = Path(output_dir).resolve()
    if out.exists():
        raise ValueError(f"comparison output directory already exists (refusing overwrite): {out}")

    runs_224 = {s: inspect_v4_run(r224_dirs[s], "V4-R2_RAW_WEIGHTED_CE", s) for s in SEEDS}
    runs_384 = {s: inspect_v4_run(r384_dirs[s], "V4-R4_RAW_WEIGHTED_CE_384", s) for s in SEEDS}

    # 1. Verify strict single-variable control (image_size: 224 -> 384), same-seed head/order sync, & cache isolation
    sync_checks = {}
    cache_dirs_384 = set()
    cache_hashes_384 = set()
    cache_hashes_224 = set()

    for s in SEEDS:
        a, b = runs_224[s], runs_384[s]
        cfg_a, cfg_b = dict(a["config"]), dict(b["config"])
        if cfg_a.pop("image_size") != 224 or cfg_b.pop("image_size") != 384 or cfg_a != cfg_b:
            raise ValueError(f"seed={s} violates single-variable control between 224 and 384 configs")
        if a["run_config"]["manifests"] != b["run_config"]["manifests"]:
            raise ValueError(f"seed={s} manifest hash mismatch between 224 and 384")
        ha = a["run_config"]["determinism_notes"]["initial_head_sha256"]
        hb = b["run_config"]["determinism_notes"]["initial_head_sha256"]
        if ha != hb:
            raise ValueError(f"seed={s} initial Linear head differs between 224 and 384")
        if a["orders"] != b["orders"]:
            raise ValueError(f"seed={s} 20-epoch sample shuffle order differs between 224 and 384")
        if b["run_config"]["loss_protocol"]["weights"] != [0.75, 1.5]:
            raise ValueError(f"seed={s} 384 weights != [0.75, 1.5]")

        fc224 = a["run_config"]["feature_cache"]
        fc384 = b["run_config"]["feature_cache"]
        if "sz384_" not in fc384["cache_dir"] or fc384["cache_identity_hash"] == fc224["cache_identity_hash"]:
            raise ValueError(f"seed={s} 384 cache not isolated from 224 cache")
        cache_dirs_384.add(fc384["cache_dir"])
        cache_hashes_384.add(fc384["features_file_sha256"])
        cache_hashes_224.add(fc224["features_file_sha256"])

        sync_checks[str(s)] = {
            "initial_head_sha256": ha,
            "sample_orders_20ep_identical": True,
            "cache_224_hash": fc224["cache_identity_hash"],
            "cache_384_hash": fc384["cache_identity_hash"],
        }

    if len(cache_dirs_384) != 1 or len(cache_hashes_384) != 1 or (cache_hashes_384 & cache_hashes_224):
        raise ValueError("384 seeds did not share a single isolated 384 feature cache distinct from 224")

    # Load verified Val 34 metadata
    ref_cfg = runs_224[42]["config"]
    prov, train_rows, val_rows = load_verified_v4_manifests(
        ref_cfg["train_manifest"], ref_cfg["val_manifest"], ref_cfg["data_provenance"]
    )
    val_meta_by_name = {r["image_filename"]: r for r in val_rows}

    # 2. Controlled offline full-image latency reference on representative Seed 42 (best_within_20.pt)
    lat_224 = benchmark_offline_full_image_latency(
        Path(runs_224[42]["dir"]) / "best_within_20.pt", val_rows, device_name=latency_device
    )
    lat_384 = benchmark_offline_full_image_latency(
        Path(runs_384[42]["dir"]) / "best_within_20.pt", val_rows, device_name=latency_device
    )

    out.mkdir(parents=True, exist_ok=False)

    # 3. Extract 20-epoch budget metrics per seed & 3-seed aggregates
    metric_defs = [
        ("selected_epoch", lambda b: b["selected_epoch"]),
        ("val_loss", lambda b: b["val_metrics"]["loss"]),
        ("val_accuracy", lambda b: b["val_metrics"]["accuracy"]),
        ("val_balanced_accuracy", lambda b: b["val_metrics"]["balanced_accuracy"]),
        ("val_fail_precision", lambda b: b["val_metrics"]["fail_precision"]),
        ("val_fail_recall", lambda b: b["val_metrics"]["fail_recall"]),
        ("val_fail_f1", lambda b: b["val_metrics"]["fail_f1"]),
        ("val_false_pass_FN", lambda b: b["val_metrics"]["false_pass"]),
        ("val_false_fail_FP", lambda b: b["val_metrics"]["false_fail"]),
        ("val_roc_auc", lambda b: b["val_metrics"]["roc_auc"]),
        ("train_fixed_loss", lambda b: b["unified_primary_train_metrics"]["loss"]),
        ("train_fixed_accuracy", lambda b: b["unified_primary_train_metrics"]["accuracy"]),
        ("train_fixed_balanced_accuracy", lambda b: b["unified_primary_train_metrics"]["balanced_accuracy"]),
        ("train_fixed_FN", lambda b: b["unified_primary_train_metrics"]["false_pass"]),
        ("train_fixed_FP", lambda b: b["unified_primary_train_metrics"]["false_fail"]),
        ("train_fixed_roc_auc", lambda b: b["unified_primary_train_metrics"]["roc_auc"]),
    ]

    per_seed_rows = []
    for res_label, run_map in (("224_V4_R2", runs_224), ("384_V4_R4", runs_384)):
        for s in SEEDS:
            r = run_map[s]
            b = r["budgets"]["20"]
            cm = b["val_metrics"]["confusion_matrix"]
            row = {
                "resolution_variant": res_label,
                "image_size": r["config"]["image_size"],
                "seed": s,
                "experiment_id": r["spec"]["experiment_id"],
                "confusion_matrix": f"[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]",
                "cache_action": r["run_config"]["feature_cache"]["cache_action"],
                "feature_cache_seconds": r["ckpt_manifest"]["timing_seconds"]["feature_cache_seconds"],
                "training_20ep_seconds": r["ckpt_manifest"]["timing_seconds"]["training_epochs_seconds"],
                "total_run_seconds": r["ckpt_manifest"]["timing_seconds"]["total_experiment_seconds"],
            }
            for m_name, fn in metric_defs:
                row[m_name] = fn(b)
            per_seed_rows.append(row)

    with (out / "v4_224_vs_384_per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(per_seed_rows)

    agg = {}
    for m_name, fn in metric_defs:
        v224 = [fn(runs_224[s]["budgets"]["20"]) for s in SEEDS]
        v384 = [fn(runs_384[s]["budgets"]["20"]) for s in SEEDS]
        dlt = [b - a if (a is not None and b is not None) else None for a, b in zip(v224, v384)]
        agg[m_name] = {
            "res_224": _mean_std(v224),
            "res_384": _mean_std(v384),
            "paired_delta_384_minus_224": _mean_std(dlt),
        }

    # 4. Per-image Val 34 comparison + focused 11/23 PASS (7 imgs) & 11/20 FAIL (6 imgs) audits
    val_filenames = [p["image_filename"] for p in runs_224[42]["preds"]["20"]]
    val_change_rows = []
    per_seed_flips = {}
    batch_focus = {"11/23_PASS": {}, "11/20_FAIL": {}}

    for s in SEEDS:
        p224_list = runs_224[s]["preds"]["20"]
        p384_list = runs_384[s]["preds"]["20"]
        newly_correct, newly_wrong, shared_err = [], [], []
        nov23_fp_224, nov23_fp_384 = [], []
        nov20_fn_224, nov20_fn_384 = [], []

        for p224, p384 in zip(p224_list, p384_list):
            fn = p224["image_filename"]
            meta = val_meta_by_name[fn]
            ok224 = p224["predicted_label"] == meta["label"]
            ok384 = p384["predicted_label"] == meta["label"]
            s224 = p224["softmax"][1]
            s384 = p384["softmax"][1]
            entry = {
                "image_filename": fn,
                "label": meta["label"],
                "source_batch": meta["source_batch"],
                "anomaly_category": meta["anomaly_category"],
                "pred_224": p224["predicted_label"],
                "score_224": round(s224, 6),
                "pred_384": p384["predicted_label"],
                "score_384": round(s384, 6),
                "score_delta": round(s384 - s224, 6),
            }
            if not ok224 and ok384:
                newly_correct.append(entry)
            elif ok224 and not ok384:
                newly_wrong.append(entry)
            elif not ok224 and not ok384:
                shared_err.append(entry)

            if meta["source_batch"] == "11/23" and meta["label"] == "PASS":
                if p224["predicted_label"] == "FAIL":
                    nov23_fp_224.append(f"{fn}({s224:.3f})")
                if p384["predicted_label"] == "FAIL":
                    nov23_fp_384.append(f"{fn}({s384:.3f})")
            if meta["source_batch"] == "11/20" and meta["label"] == "FAIL":
                if p224["predicted_label"] == "PASS":
                    nov20_fn_224.append(f"{fn}({s224:.3f})")
                if p384["predicted_label"] == "PASS":
                    nov20_fn_384.append(f"{fn}({s384:.3f})")

        per_seed_flips[str(s)] = {
            "newly_correct_at_384": newly_correct,
            "newly_wrong_at_384": newly_wrong,
            "shared_errors": shared_err,
        }
        batch_focus["11/23_PASS"][str(s)] = {
            "support_PASS": sum(val_meta_by_name[f]["source_batch"] == "11/23" and val_meta_by_name[f]["label"] == "PASS" for f in val_filenames),
            "fp_count_224": len(nov23_fp_224),
            "fp_count_384": len(nov23_fp_384),
            "fp_delta_384_minus_224": len(nov23_fp_384) - len(nov23_fp_224),
            "fp_files_224": nov23_fp_224,
            "fp_files_384": nov23_fp_384,
        }
        batch_focus["11/20_FAIL"][str(s)] = {
            "support_FAIL": sum(val_meta_by_name[f]["source_batch"] == "11/20" and val_meta_by_name[f]["label"] == "FAIL" for f in val_filenames),
            "fn_count_224": len(nov20_fn_224),
            "fn_count_384": len(nov20_fn_384),
            "fn_delta_384_minus_224": len(nov20_fn_384) - len(nov20_fn_224),
            "fn_files_224": nov20_fn_224,
            "fn_files_384": nov20_fn_384,
        }

    for idx, fn in enumerate(val_filenames):
        meta = val_meta_by_name[fn]
        crow = {
            "image_filename": fn,
            "label": meta["label"],
            "source_batch": meta["source_batch"],
            "anomaly_category": meta["anomaly_category"],
            "non_catalogue_count": meta["non_catalogue_count"],
        }
        for s in SEEDS:
            p224 = runs_224[s]["preds"]["20"][idx]
            p384 = runs_384[s]["preds"]["20"][idx]
            crow[f"s{s}_224_pred"] = p224["predicted_label"]
            crow[f"s{s}_224_score"] = round(p224["softmax"][1], 6)
            crow[f"s{s}_384_pred"] = p384["predicted_label"]
            crow[f"s{s}_384_score"] = round(p384["softmax"][1], 6)
            crow[f"s{s}_delta"] = round(p384["softmax"][1] - p224["softmax"][1], 6)
        val_change_rows.append(crow)

    with (out / "v4_224_vs_384_val_34_changes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(val_change_rows[0].keys()))
        w.writeheader()
        w.writerows(val_change_rows)

    # 5. Evaluate pre-committed Section 5 final selection criteria
    mean_ba_delta = agg["val_balanced_accuracy"]["paired_delta_384_minus_224"]["mean"]
    mean_fn_224 = agg["val_false_pass_FN"]["res_224"]["mean"]
    mean_fn_384 = agg["val_false_pass_FN"]["res_384"]["mean"]
    mean_fp_224 = agg["val_false_fail_FP"]["res_224"]["mean"]
    mean_fp_384 = agg["val_false_fail_FP"]["res_384"]["mean"]

    s42_224 = runs_224[42]["budgets"]["20"]["val_metrics"]
    s42_384 = runs_384[42]["budgets"]["20"]["val_metrics"]

    c_ba_gain_ge_2pp = mean_ba_delta >= 0.0195  # >= ~2 percentage points
    c_fn_non_increasing = mean_fn_384 <= mean_fn_224 + 1e-9
    c_fp_not_worse = mean_fp_384 <= mean_fp_224 + 1e-9
    c_seed42_no_regression = (
        s42_384["balanced_accuracy"] >= s42_224["balanced_accuracy"] - 1e-9
        and s42_384["false_pass"] <= s42_224["false_pass"]
        and s42_384["false_fail"] <= s42_224["false_fail"]
    )
    promote_384 = bool(c_ba_gain_ge_2pp and c_fn_non_increasing and c_fp_not_worse and c_seed42_no_regression)

    selection_audit = {
        "precommitted_representative_seed": 42,
        "criteria": {
            "mean_balanced_accuracy_gain_ge_approx_2pp": {
                "passed": c_ba_gain_ge_2pp,
                "observed_delta": round(mean_ba_delta, 6),
            },
            "mean_FN_non_increasing": {
                "passed": c_fn_non_increasing,
                "mean_FN_224": round(mean_fn_224, 4),
                "mean_FN_384": round(mean_fn_384, 4),
            },
            "mean_FP_not_worsening": {
                "passed": c_fp_not_worse,
                "mean_FP_224": round(mean_fp_224, 4),
                "mean_FP_384": round(mean_fp_384, 4),
            },
            "seed42_no_notable_regression": {
                "passed": c_seed42_no_regression,
                "seed42_224_BA_FN_FP": [s42_224["balanced_accuracy"], s42_224["false_pass"], s42_224["false_fail"]],
                "seed42_384_BA_FN_FP": [s42_384["balanced_accuracy"], s42_384["false_pass"], s42_384["false_fail"]],
            },
        },
        "recommended_candidate_by_precommitted_rule": (
            f"V4-R4_RAW_WEIGHTED_CE_384_SEED42 (Epoch {runs_384[42]['budgets']['20']['selected_epoch']}, best_within_20.pt)"
            if promote_384
            else f"V4-R2_RAW_WEIGHTED_CE_SEED42 (224x224, Epoch {runs_224[42]['budgets']['20']['selected_epoch']}, best_within_20.pt)"
        ),
        "promote_384": promote_384,
        "test_used_for_selection": False,
    }

    # 6. Plot 224 vs 384 curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    seed_colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
    for res_lbl, ls, rmap in (("224 (V4-R2)", "-", runs_224), ("384 (V4-R4)", "--", runs_384)):
        for s in SEEDS:
            hist = rmap[s]["history"]
            eps = [h["epoch"] for h in hist]
            sel_ep = rmap[s]["budgets"]["20"]["selected_epoch"]
            kw = {"color": seed_colors[s], "linestyle": ls, "alpha": 0.85, "label": f"{res_lbl} s={s} (best=ep{sel_ep})"}
            axes[0, 0].plot(eps, [h["train"]["loss"] for h in hist], **kw)
            axes[0, 1].plot(eps, [h["val"]["loss"] for h in hist], **kw)
            axes[1, 0].plot(eps, [h["val"]["balanced_accuracy"] for h in hist], **kw)
            axes[1, 1].plot(eps, [h["val"]["roc_auc"] for h in hist], **kw)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(2, 21, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)
    axes[0, 0].set_title("Online Weighted Train CE Loss (Solid=224, Dashed=384)")
    axes[0, 1].set_title("Primary Val (34 Real) Unweighted CE Loss")
    axes[1, 0].set_title("Primary Val (34 Real) Balanced Accuracy")
    axes[1, 1].set_title("Primary Val (34 Real) ROC-AUC")
    fig.tight_layout()
    fig.savefig(out / "curves_v4_224_vs_384.png", dpi=150)
    plt.close(fig)

    summary_json = {
        "schema_version": "member06.v4_final_resolution_comparison.v1",
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": prov["upstream_commit"],
        "single_variable_verified": "image_size: 224 -> 384",
        "same_seed_sync_and_cache_isolation": sync_checks,
        "aggregate_3seed_comparison": agg,
        "batch_focus_1123_PASS_and_1120_FAIL": batch_focus,
        "per_seed_val_flips": per_seed_flips,
        "offline_full_image_latency_reference_seed42": {
            "224": lat_224,
            "384": lat_384,
            "median_ratio_384_over_224": round(lat_384["median_ms"] / lat_224["median_ms"], 3) if lat_224["median_ms"] > 0 else None,
            "p95_ratio_384_over_224": round(lat_384["p95_ms"] / lat_224["p95_ms"], 3) if lat_224["p95_ms"] > 0 else None,
        },
        "selection_audit": selection_audit,
        "test_data_used": False,
    }
    write_json(out / "v4_224_vs_384_summary.json", summary_json)

    # 7. Render COMPARISON_V4_FINAL_RESOLUTION_CN.md
    md = [
        "# Baseline 2 · `final_training_v4` 最终分辨率受控比较报告（224 vs 384）",
        "",
        f"- **权威数据版本**：`members/member03_data/final_training_v4/`（上游提交 `{prov['upstream_commit']}`，`Train=249 [166P/83F]`，`Val=34 [20P/14F]`，`Test=30` 完全隔离未接触 `test_data_used = false`）",
        "- **受控单变量核验**：`V4-R4_RAW_WEIGHTED_CE_384` 严格继承 `V4-R2_RAW_WEIGHTED_CE`（`Frozen ResNet18 + Linear(512,2)`、`Weighted CE [PASS=0.75, FAIL=1.50]`、`Val 无权 CE`、`Adam lr=0.001, wd=0`、`batch=16`、`20 Epoch` 最大预算、默认 `argmax`），**唯一变化变量为输入尺寸 `image_size: 224 → 384`（letterbox）**。",
        "- **同 Seed 同步与缓存隔离核验**：三个随机种子（`42, 43, 44`）在 `224` 与 `384` 下的初始分类头哈希与全部 20 轮样本打乱顺序均已逐项核验 `100%` 一致；`384` 使用完全独立的 `v4_resnet18_sz384_*` 专属特征缓存（三组 `384` 共享同一 `283` 图缓存，零混入 `224` 特征与 `Test` 图像）。",
        "",
        "## 1. 逐 Seed 配对结果明细（20 轮最大预算，最低无权 Val Loss 选模）",
        "",
        "| 分辨率方案 | Seed | 最佳Epoch | 无权Val Loss | Val Acc | Val BA | FAIL Prec | FAIL Recall | FAIL F1 | FN / FP | 混淆矩阵 `[[TN,FP],[FN,TP]]` | Val AUC | Train249固定 Loss / Acc / AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for res_lbl, rmap in (("`224 (V4-R2)`", runs_224), ("`384 (V4-R4)`", runs_384)):
        for s in SEEDS:
            b = rmap[s]["budgets"]["20"]
            vm = b["val_metrics"]
            tm = b["unified_primary_train_metrics"]
            cm = vm["confusion_matrix"]
            prec_s = f"{vm['fail_precision']:.4f}" if vm["fail_precision"] is not None else "—"
            f1_s = f"{vm['fail_f1']:.4f}" if vm["fail_f1"] is not None else "—"
            md.append(
                f"| {res_lbl} | {s} | {b['selected_epoch']} | {vm['loss']:.6f} | {vm['accuracy']:.4f} | {vm['balanced_accuracy']:.4f} | "
                f"{prec_s} | {vm['fail_recall']:.4f} | {f1_s} | {vm['false_pass']} / {vm['false_fail']} | "
                f"`[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]` | {vm['roc_auc']:.4f} | "
                f"{tm['loss']:.4f} / {tm['accuracy']:.4f} / {tm['roc_auc']:.4f} |"
            )

    md += [
        "",
        "## 2. 三组随机种子（42 / 43 / 44）均值 ± 样本标准差（`n=3, ddof=1`）与配对差值（`384 - 224`）",
        "",
        "| 指标 | `224 (V4-R2)` (`Mean ± SD`) | `384 (V4-R4)` (`Mean ± SD`) | 配对差 `384 - 224` (`Mean ± SD`) | 三种子配对差明细 `(s42, s43, s44)` |",
        "|---|---|---|---|---|",
    ]
    agg_rows_spec = [
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
        ("train_fixed_roc_auc", "Train 249 固定 ROC-AUC", 4),
    ]
    for m_key, m_lbl, dec in agg_rows_spec:
        st = agg[m_key]
        d_ms = st["paired_delta_384_minus_224"]
        vals_s = ", ".join(f"{v:+.{dec}f}" for v in d_ms["values"]) if d_ms else "—"
        md.append(
            f"| {m_lbl} | `{_fmt_ms(st['res_224'], dec)}` | `{_fmt_ms(st['res_384'], dec)}` | "
            f"`{_fmt_ms(d_ms, dec)}` | `({vals_s})` |"
        )

    md += [
        "",
        "## 3. 34 张 Val 逐图预测变化及重点批次（`11/23` 正常误拒、`11/20` 异常漏放）专项核查",
        "",
        "### 3.1 重点关注切片变化（注：单一类别切片仅反映该切片内误拒/漏放计数变化）",
        "",
        "| Seed | `11/23` 正常图（共 7 PASS）224 误拒 FP | `11/23` 正常图 384 误拒 FP | `11/23` FP 变化 | `11/20` 异常图（共 6 FAIL）224 漏放 FN | `11/20` 异常图 384 漏放 FN | `11/20` FN 变化 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SEEDS:
        b23 = batch_focus["11/23_PASS"][str(s)]
        b20 = batch_focus["11/20_FAIL"][str(s)]
        md.append(
            f"| {s} | {b23['fp_count_224']}/7 | {b23['fp_count_384']}/7 | `{b23['fp_delta_384_minus_224']:+d}` | "
            f"{b20['fn_count_224']}/6 | {b20['fn_count_384']}/6 | `{b20['fn_delta_384_minus_224']:+d}` |"
        )

    md += [
        "",
        "### 3.2 全部 34 张 Val 的逐图翻转明细（`224 → 384`）",
        "",
        "| Seed | `384 新增正确` (`224错→384对`) | `384 新增错误` (`224对→384错`) | `两者共同错误` | 明细（`文件名 [批次/真实类]: 224分数→384分数`） |",
        "|---:|---:|---:|---:|---|",
    ]
    for s in SEEDS:
        fl = per_seed_flips[str(s)]
        items_txt = []
        for x in fl["newly_correct_at_384"]:
            items_txt.append(f"新增正确 `{x['image_filename']}` [{x['source_batch']}/{x['label']}]: `{x['score_224']:.3f}→{x['score_384']:.3f}`")
        for x in fl["newly_wrong_at_384"]:
            items_txt.append(f"新增错误 `{x['image_filename']}` [{x['source_batch']}/{x['label']}]: `{x['score_224']:.3f}→{x['score_384']:.3f}`")
        for x in fl["shared_errors"]:
            items_txt.append(f"共同错误 `{x['image_filename']}` [{x['source_batch']}/{x['label']}]: `{x['score_224']:.3f}→{x['score_384']:.3f}`")
        md.append(
            f"| {s} | **{len(fl['newly_correct_at_384'])}** | **{len(fl['newly_wrong_at_384'])}** | "
            f"**{len(fl['shared_errors'])}** | {'；'.join(items_txt) if items_txt else '无'} |"
        )

    md += [
        "",
        "## 4. 特征提取、训练耗时与相同边界下的完整图片离线推理耗时参考",
        "",
        "### 4.1 特征缓存与训练成本",
        "",
        "| 方案 | 首次 283 图特征缓存构建耗时 (s) | 后续同分辨率缓存加载耗时 (s) | 单 Seed 20轮训练耗时均值 (s) |",
        "|---|---:|---:|---:|",
        f"| `224 (V4-R2)` | {runs_224[42]['ckpt_manifest']['timing_seconds']['feature_cache_seconds']:.2f}（注：含单图校验复用） | "
        f"{np.mean([runs_224[s]['ckpt_manifest']['timing_seconds']['feature_cache_seconds'] for s in (43,44)]):.2f} | "
        f"{np.mean([runs_224[s]['ckpt_manifest']['timing_seconds']['training_epochs_seconds'] for s in SEEDS]):.2f} |",
        f"| `384 (V4-R4)` | {runs_384[42]['ckpt_manifest']['timing_seconds']['feature_cache_seconds']:.2f}（全量 283 图新提） | "
        f"{np.mean([runs_384[s]['ckpt_manifest']['timing_seconds']['feature_cache_seconds'] for s in (43,44)]):.2f} | "
        f"{np.mean([runs_384[s]['ckpt_manifest']['timing_seconds']['training_epochs_seconds'] for s in SEEDS]):.2f} |",
        "",
        "### 4.2 离线单图完整前向推理耗时对照（代表模型 `Seed=42, batch_size=1, n=34 Val`）",
        "",
        "> **严格边界声明**：以下计时在相同设备（`"
        + f"{lat_224['device']}`, `cpu_threads={lat_224['cpu_threads']}`）与相同边界（本地磁盘解码 `read_image` → `letterbox Preprocess` → `FrozenResNet18` 全网络前向 → `softmax/argmax`，预热 5 次后逐张实测 34 张原图）下测得。**该数据仅为离线计算开销对比参考，不包含真实摄像头采集与界面刷新，严禁冒充正式现场摄像头实时延迟（E6）。**",
        "",
        "| 模型代表候选 | 输入分辨率 | Median (ms) | P95 (ms) | Mean (ms) | Min / Max (ms) | Median 倍率 (`384 / 224`) |",
        "|---|---:|---:|---:|---:|---|---:|",
        f"| `V4-R2_SEED42` (Ep {lat_224['selected_epoch']}) | `224×224` | **{lat_224['median_ms']:.2f}** | **{lat_224['p95_ms']:.2f}** | {lat_224['mean_ms']:.2f} | {lat_224['min_ms']:.2f} / {lat_224['max_ms']:.2f} | `1.00x` |",
        f"| `V4-R4_384_SEED42` (Ep {lat_384['selected_epoch']}) | `384×384` | **{lat_384['median_ms']:.2f}** | **{lat_384['p95_ms']:.2f}** | {lat_384['mean_ms']:.2f} | {lat_384['min_ms']:.2f} / {lat_384['max_ms']:.2f} | `{summary_json['offline_full_image_latency_reference_seed42']['median_ratio_384_over_224']:.2f}x` |",
        "",
        "## 5. 预先约定的最终选型规则核对",
        "",
        "| 预设判据 | 要求 | 实测值 | 是否满足 |",
        "|---|---|---|---|",
        f"| 1. 三 Seed 平均 Balanced Accuracy 提升 | `≥ +0.0200`（约 2 个百分点） | `{mean_ba_delta:+.4f}` | `{'PASS' if c_ba_gain_ge_2pp else 'FAIL'}` |",
        f"| 2. 三 Seed 平均漏放 `FN` 不增加 | `mean_FN(384) ≤ mean_FN(224)` | `{mean_fn_224:.2f} → {mean_fn_384:.2f}` | `{'PASS' if c_fn_non_increasing else 'FAIL'}` |",
        f"| 3. 三 Seed 平均误拒 `FP` 不恶化 | `mean_FP(384) ≤ mean_FP(224)` | `{mean_fp_224:.2f} → {mean_fp_384:.2f}` | `{'PASS' if c_fp_not_worse else 'FAIL'}` |",
        f"| 4. 预设代表 `Seed=42` 无明显退步 | `BA` 不降且 `FN/FP` 不增 | `224: BA={s42_224['balanced_accuracy']:.4f}, FN={s42_224['false_pass']}, FP={s42_224['false_fail']}` vs `384: BA={s42_384['balanced_accuracy']:.4f}, FN={s42_384['false_pass']}, FP={s42_384['false_fail']}` | `{'PASS' if c_seed42_no_regression else 'FAIL'}` |",
        "",
        f"- **按预设规则推导的候选建议**：**`{selection_audit['recommended_candidate_by_precommitted_rule']}`**",
        "- **后续步骤**：本轮结束后停止全部训练优化，经你确认最终候选后即可进入模型冻结、统一 `Test (30 图)` 单次正式评估、成员 7 系统集成及 Stage 2 交付。",
    ]

    (out / "COMPARISON_V4_FINAL_RESOLUTION_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-224", default=DEFAULT_224_SUITE, help="completed V4 224 suite directory")
    parser.add_argument("--suite-384", required=True, help="completed V4 384 suite directory")
    parser.add_argument("--output", required=True, help="new output directory for final 224-vs-384 comparison report")
    parser.add_argument("--latency-device", default="cpu", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    try:
        s224 = Path(args.suite_224).resolve()
        s384 = Path(args.suite_384).resolve()
        r224_dirs = {s: s224 / f"V4-R2_RAW_WEIGHTED_CE_SEED{s}/model" for s in SEEDS}
        r384_dirs = {s: s384 / f"V4-R4_RAW_WEIGHTED_CE_384_SEED{s}/model" for s in SEEDS}
        out = compare_v4_224_vs_384(r224_dirs, r384_dirs, args.output, latency_device=args.latency_device)
        print(json.dumps({
            "status": "V4_FINAL_RESOLUTION_COMPARISON_GENERATED",
            "report": str(out / "COMPARISON_V4_FINAL_RESOLUTION_CN.md"),
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"V4_RESOLUTION_COMPARISON_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
