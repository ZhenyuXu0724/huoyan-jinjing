"""Traceable v4 report: 75 runs, ablation, freeze, one-time evaluation, stability and cost."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (
    AUDIT, CONFIG, DATA, DIAGNOSIS, EVALUATION, EXPERIMENTS, METHOD_SUFFIX, OUT, PROTOCOL,
    REFINEMENT, REPORTS, ROOT, SEEDS, code_hashes, ensure, now, read, record_stage, sha256,
    update_state, write, write_csv,
)

REPORT_CODE = (ROOT / "members/member04_hog/final_v4/report.py",)


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def mean_sd(values, percentage=False):
    array = np.asarray(values, dtype=float) * (100 if percentage else 1)
    return f"{array.mean():.4f} ± {array.std(ddof=0):.4f}" + ("%" if percentage else "")


def rows_for_groups(summaries):
    rows, means = [], []
    for method, summary in summaries.items():
        for group, metrics in summary["groups"].items():
            for metric in metrics:
                rows.append({
                    "method": method, "group": group, "seed": metric["seed"],
                    "threshold": metric["threshold"], "epoch": metric["epoch"],
                    "fn": metric["fn"], "fp": metric["fp"],
                    "tp": metric["tp"], "tn": metric["tn"],
                    "f1": metric["f1"], "balanced_accuracy": metric["balanced_accuracy"],
                })
            means.append({
                "method": method, "group": group,
                "mean_fn": float(np.mean([item["fn"] for item in metrics])),
                "sd_fn": float(np.std([item["fn"] for item in metrics], ddof=0)),
                "mean_fp": float(np.mean([item["fp"] for item in metrics])),
                "sd_fp": float(np.std([item["fp"] for item in metrics], ddof=0)),
                "mean_f1": float(np.mean([item["f1"] for item in metrics])),
                "sd_f1": float(np.std([item["f1"] for item in metrics], ddof=0)),
                "mean_balanced_accuracy": float(np.mean(
                    [item["balanced_accuracy"] for item in metrics])),
                "sd_balanced_accuracy": float(np.std(
                    [item["balanced_accuracy"] for item in metrics], ddof=0)),
                "views": metrics[0]["views"],
            })
    return rows, means


def report():
    ensure(REPORTS)
    ae = read(EXPERIMENTS / "summary_ae.json")
    supervised = read(EXPERIMENTS / "summary_supervised.json")
    refinement = read(REFINEMENT / "summary.json")
    freeze = read(OUT / "freeze.json")
    evaluation = read(EVALUATION / "summary.json")
    diagnosis = read(DIAGNOSIS / "summary.json")
    development = read(AUDIT / "dataset_audit_development.json")
    full = read(AUDIT / "dataset_audit_full.json")
    protocol = read(PROTOCOL)
    summaries = {"ae": ae, "supervised": supervised}
    group_rows, group_means = rows_for_groups(summaries)
    write_csv(REPORTS / "group_results.csv", group_rows)
    write_csv(REPORTS / "group_means.csv", group_means)
    comparison = []
    for group in ae["groups"]:
        left = next(row for row in group_means if row["method"] == "ae" and row["group"] == group)
        right = next(row for row in group_means
                     if row["method"] == "supervised" and row["group"] == group)
        comparison.append({
            "group": group, "ae_mean_fn": left["mean_fn"], "ae_mean_fp": left["mean_fp"],
            "supervised_mean_fn": right["mean_fn"], "supervised_mean_fp": right["mean_fp"],
            "ae_mean_f1": left["mean_f1"], "supervised_mean_f1": right["mean_f1"],
        })
    write_csv(REPORTS / "method_comparison.csv", comparison)
    ablation = []
    for method, summary in summaries.items():
        base = next(row for row in group_means if row["method"] == method and row["group"] == "A0")
        for row in group_means:
            if row["method"] != method:
                continue
            ablation.append({
                "method": method, "group": row["group"],
                "delta_mean_fn_vs_A0": row["mean_fn"] - base["mean_fn"],
                "delta_mean_fp_vs_A0": row["mean_fp"] - base["mean_fp"],
                "delta_mean_f1_vs_A0": row["mean_f1"] - base["mean_f1"],
            })
    write_csv(REPORTS / "ablation_table.csv", ablation)
    equivalence = []
    for method in ("ae", "supervised"):
        for item in read(EXPERIMENTS / f"summary_{method}.json")["equivalence"]:
            equivalence.append({"method": method, "pair": " vs ".join(item["pair"]),
                                "identical": item["identical"]})
    write_csv(REPORTS / "equivalence.csv", equivalence)
    refinement_rows, refinement_means = [], []
    for config, metrics in refinement["groups"].items():
        for item in metrics:
            refinement_rows.append({
                "config": config, "seed": item["seed"],
                "best_epoch": item["best"]["epoch"], "best_fn": item["best"]["fn"],
                "best_fp": item["best"]["fp"], "last_fn": item["last"]["fn"],
                "last_fp": item["last"]["fp"],
                "best_threshold": item["best"]["threshold"],
                "last_threshold": item["last"]["threshold"],
            })
        refinement_means.append({
            "config": config,
            "mean_best_fn": float(np.mean([item["best"]["fn"] for item in metrics])),
            "mean_best_fp": float(np.mean([item["best"]["fp"] for item in metrics])),
            "mean_last_fn": float(np.mean([item["last"]["fn"] for item in metrics])),
            "mean_last_fp": float(np.mean([item["last"]["fp"] for item in metrics])),
            "parameters": refinement["configs"][config]["parameters"],
            "alpha": refinement["configs"][config]["alpha"],
        })
    write_csv(REPORTS / "refinement_table.csv", refinement_rows)
    write_csv(REPORTS / "refinement_means.csv", refinement_means)
    candidates = []
    for row in freeze["candidates"]:
        candidates.append({**row, "selected_default": row["method"] == freeze["selected_default"]})
    write_csv(REPORTS / "candidates.csv", candidates)
    split_rows = []
    for key, metrics in evaluation["split_metrics"].items():
        split, method = key.split("/")
        split_rows.append({"split": split, "model": method, "tp": metrics["tp"],
                           "fp": metrics["fp"], "fn": metrics["fn"], "tn": metrics["tn"],
                           "f1": metrics["f1"], "balanced_accuracy": metrics["balanced_accuracy"],
                           "coverage": metrics["coverage"]})
    write_csv(REPORTS / "split_metrics.csv", split_rows)
    constant_rows = [{"split": split, **values}
                     for split, values in evaluation["constant_baselines"].items()]
    write_csv(REPORTS / "constant_baselines.csv", constant_rows)
    if evaluation["error_ids"]:
        write_csv(REPORTS / "error_ids.csv", evaluation["error_ids"])
    stability_rows = []
    for key, result in diagnosis["results"].items():
        split, model = key.split("_", 1)
        stability_rows.append({"split": split, "model": model, "variant": "ANY",
                               "fn": None, "fp": None, "flips": result["any_flip"],
                               "correct_to_wrong": None, "wrong_to_correct": None})
        for variant, metrics in result["variants"].items():
            stability_rows.append({
                "split": split, "model": model, "variant": variant,
                "fn": metrics["fn"], "fp": metrics["fp"], "flips": metrics["flips"],
                "correct_to_wrong": metrics["correct_to_wrong"],
                "wrong_to_correct": metrics["wrong_to_correct"],
            })
    write_csv(REPORTS / "stability_summary.csv", stability_rows)
    cost_rows = [{"model": method, **values} for method, values in evaluation["cost"].items()]
    write_csv(REPORTS / "cost.csv", cost_rows)
    write(REPORTS / "report_manifest.json", {
        "created_at": now(),
        "code_sha256": code_hashes(REPORT_CODE),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (
            PROTOCOL, OUT / "freeze.json", EVALUATION / "summary.json",
            DIAGNOSIS / "summary.json", AUDIT / "dataset_audit_development.json",
            AUDIT / "dataset_audit_full.json")},
    })

    lines = [
        "# Baseline1_HOGMLP_v4_FinalDataset_Rebuild",
        "",
        "成员4在固定数据版本 `final_training_v4` 上完整重做的 HOG+MLP 实验基线。"
        "全部模型从随机初始化学习，旧权重、旧阈值和旧默认入口不参与本轮。"
        "test 只在冻结后一次性评价，本轮成绩属于开发证据，不构成现场性能保证。",
        "",
        "## 数据与规则",
        "",
        f"- 数据来源 commit `{protocol['source']['commit']}`，目录 tree "
        f"`{protocol['source']['dataset_tree']}`。",
        f"- 开发审计：{development['summary']['files']} 个文件；全量审计："
        f"{full['summary']['files']} 个文件；test 期间只使用冻结后的三个代表模型。",
        "- 训练 249 张原图（166 PASS、83 FAIL）；验证 34 张（20 PASS、14 FAIL）；"
        "test 30 张（15/15）。AE 只学习 166 张 PASS 原图。",
        "- 预处理 256×256、灰度、等比例缩放白边；HOG 9 方向/16×16/2×2/L2-Hys=8100 维；"
        "每轮每图一个随机视图；主实验固定第 200 轮。",
        "- 阈值在 34 张验证图上选：误拒≤4/20，先漏放少、再误拒少、再取较大规范阈值；"
        "全 PASS 与全 FAIL 都在候选内。",
        "",
        "## 66 次主实验",
        "",
        "| 方法 | 入选组 | 平均漏放 | 平均误拒 | 平均 FAIL F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        selected = summary["selected_group"]
        row = next(item for item in group_means
                   if item["method"] == method and item["group"] == selected)
        lines.append(f"| {method} | {selected} | {row['mean_fn']:.2f} | "
                     f"{row['mean_fp']:.2f} | {pct(row['mean_f1'])} |")
    lines += ["", "| 方法 | 组 | 平均漏放±sd | 平均误拒±sd | 平均 FAIL F1±sd | 视图数 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in group_means:
        lines.append(f"| {row['method']} | {row['group']} | "
                     f"{row['mean_fn']:.2f} ± {row['sd_fn']:.2f} | "
                     f"{row['mean_fp']:.2f} ± {row['sd_fp']:.2f} | "
                     f"{pct(row['mean_f1'])} | {row['views']} |")
    lines += ["", "标准差为三个种子的总体标准差（ddof=0），不是泛化置信区间。", ""]
    for method in ("ae", "supervised"):
        checks = read(EXPERIMENTS / f"summary_{method}.json")["equivalence"]
        detail = "; ".join(
            f"{' vs '.join(item['pair'])}={'一致' if item['identical'] else '不一致'}"
            for item in checks)
        lines.append(f"- {method} 等价组：{detail}。")
    lines += ["", "## 9 次有限改进", "",
              "| 配置 | 平均最佳漏放 | 平均最佳误拒 | 平均末轮漏放 | 平均末轮误拒 | 参数量 | L2 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in refinement_means:
        lines.append(f"| {row['config']} | {row['mean_best_fn']:.2f} | "
                     f"{row['mean_best_fp']:.2f} | {row['mean_last_fn']:.2f} | "
                     f"{row['mean_last_fp']:.2f} | {row['parameters']} | {row['alpha']} |")
    lines += ["", f"入选配置 **{refinement['selected_config']}**；R0 三个种子第 200 轮"
                  "精确复现主实验入选组，未用验证结果改变梯度或增强采样。", ""]
    lines += ["## 代表模型与默认选择", "",
              "| 方法 | 组/配置 | 平均漏放 | 平均误拒 | 参数量 | 训练视图 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in freeze["candidates"]:
        lines.append(f"| {row['method']} | {row['group']} | {row['mean_fn']:.2f} | "
                     f"{row['mean_fp']:.2f} | {row['parameters']} | {row['training_views']} |")
    lines += ["", f"本轮默认选中 **{freeze['selected_default']}** seed42；"
                  f"阈值 `{freeze['representatives'][freeze['selected_default']]['threshold']}`，"
                  f"版本 `{freeze['representatives'][freeze['selected_default']]['version']}`。"
                  "冻结发生在任何 test 解码之前。", ""]
    lines += ["## 冻结后逐分区评价", "",
              "| 分区/模型 | TP | FP误拒 | FN漏放 | TN | FAIL F1 | 平衡准确率 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in split_rows:
        lines.append(f"| {row['split']}/{row['model']} | {row['tp']} | {row['fp']} | "
                     f"{row['fn']} | {row['tn']} | {pct(row['f1'])} | "
                     f"{pct(row['balanced_accuracy'])} |")
    lines += ["", "| 分区 | 全PASS准确率 | 全FAIL准确率 | 全FAIL F1 |",
              "|---|---:|---:|---:|"]
    for row in constant_rows:
        lines.append(f"| {row['split']} | {pct(row['all_pass_accuracy'])} | "
                     f"{pct(row['all_fail_accuracy'])} | {pct(row['all_fail_f1'])} |")
    lines += ["", "AE 的 train 评价包含 83 张它从未学习的 FAIL 图，应与“AE 实际学习 166 张 PASS”"
                  "分开理解。", ""]
    lines += ["## 未见类型与错误", ""]
    for method, subset in evaluation["unseen_types"].items():
        unseen = subset["unseen"]
        lines.append(f"- {method}：test 含 S/H 未见件的 {subset['unseen_types_images']} 张图中，"
                     f"图级漏放 {unseen['fn']}、误拒 {unseen['fp']}（仅图像级判定，不是定位或分类成功率）。")
    lines += ["", f"三方法共有/修复/新增错误 ID 见 `error_ids.csv`；共 {len(evaluation['error_ids'])} "
                  "条涉及至少一个方法的错误。", ""]
    lines += ["## 稳定性诊断", "",
              "每图 21 设置：原图＋{240,224}×{白/边缘灰度5%补白}×{居中,左,右,上,下}。"
              "原图对缩小居中同时改变尺度、重采样与补白，不能当作纯位置实验；零翻转不自动等于鲁棒。",
              "", "| 分区/模型 | 至少一种变体翻转/原图数 | 原图漏放 | 原图误拒 |",
              "|---|---:|---:|---:|"]
    for key, result in diagnosis["results"].items():
        base = result["variants"]["original"]
        lines.append(f"| {key} | {result['any_flip']}/{result['n']} | {base['fn']} | {base['fp']} |")
    lines += ["", "## 离线成本", "",
              "| 模型 | Median ms | P95 ms | 模型字节 | 训练秒 |",
              "|---|---:|---:|---:|---:|"]
    for method, values in evaluation["cost"].items():
        lines.append(f"| {method} | {values['median_ms']:.2f} | {values['p95_ms']:.2f} | "
                     f"{values['model_bytes']} | {values['training_seconds']:.1f} |")
    lines += ["", "单线程、预加载、20 张固定验证图、每模型预热 2 图、三遍交错，含读图与预处理；"
                  "不含摄像头、界面和端到端部署开销。", ""]
    lines += ["## 解释与限制", "",
              "- 验证集仅 34 张，每次漏放变化约为 1/14=7.14 个百分点，误拒变化约为 1/20=5 个百分点；"
              "75 次训练、轮次和阈值选择会带来开发集乐观性。",
              "- AE 与监督同时改变学习方法与参与学习的样本，不能把差异单独归因于网络结构。",
              "- test 含历史接触和 11 张未见件混缺件图像；HOG 只输出整图判定，不输出框或实际数量。",
              "- 场景组字段互不重叠，但物理场景独立性未独立证实；来源、批次、异常类型与标签仍可能混杂。",
              "- 本轮没有因为 test 成绩回头改阈值、换种子或重训。", ""]
    lines += ["## 复现与证据", "",
              "```powershell",
              "python -m members.member04_hog.final_v4.prepare --source-commit "
              + protocol["source"]["commit"],
              "python -m members.member04_hog.final_v4.audit --scope development",
              "python -m members.member04_hog.final_v4.train --method ae "
              "--protocol members/member04_hog/configs/final_v4/protocol.json",
              "python -m members.member04_hog.final_v4.train --method supervised "
              "--protocol members/member04_hog/configs/final_v4/protocol.json",
              "python -m members.member04_hog.final_v4.refine "
              "--protocol members/member04_hog/configs/final_v4/protocol.json",
              "python -m members.member04_hog.final_v4.freeze",
              "python -m members.member04_hog.final_v4.diagnose --splits train val",
              "python -m members.member04_hog.final_v4.prepare --scope full",
              "python -m members.member04_hog.final_v4.audit --scope full",
              "python -m members.member04_hog.final_v4.evaluate --frozen",
              "python -m members.member04_hog.final_v4.report",
              "python -m members.member04_hog.final_v4.verify",
              "```",
              "",
              "产物在 `outputs/member04_hog/final_v4/`，权重在 `weights/member04_hog/final_v4/`；"
              "原始图片不进入交付包。"]
    text = "\n".join(lines) + "\n"
    (REPORTS / "REPORT.md").write_text(text, encoding="utf-8")
    docs = ROOT / "members/member04_hog/docs"
    ensure(docs)
    (docs / "RESULTS.md").write_text(text, encoding="utf-8")
    record_stage("report", "completed", "REPORT.md and docs/RESULTS.md generated")
    update_state(next_command="python -m members.member04_hog.final_v4.verify")
    print("Report complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        report()
    except Exception as error:
        record_stage("report", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
