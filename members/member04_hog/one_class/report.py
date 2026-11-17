"""Produce reproducible validation and provided-test reports for the delivered model."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import joblib

from ..data import sha256
from ..evaluate import write_json


def report(results, weights, data, destination, test_results=None):
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    selected = summary["selected_group"]
    source = weights / selected / "seed42/model.joblib"
    chosen = next(run for run in summary["groups"][selected]["runs"] if run["seed"] == 42)
    if sha256(source) != chosen["model_sha256"]:
        raise ValueError("Selected model changed after validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    bundle = joblib.load(destination)
    handoff = results / "handoff"
    handoff.mkdir(exist_ok=True)
    fields = ["image_id", "model_id", "status", "predicted_label",
              "prediction_score", "score_type", "non_catalogue_count"]
    with (results / selected / "seed42/valid_predictions.csv").open(
            encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    sources = {Path(row["image_path"]).name: row for row in provenance["sources"]
               if row["split"] == "valid"}
    with (results / "valid_error_analysis.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["image_id", "label", "predicted_label", "prediction_score",
                   "error_type", "source_condition", "image_path"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            if row["predicted_label"] == row["label"] and row["status"] == "OK":
                continue
            source_record = sources[row["image_id"]]
            writer.writerow({**{key: row[key] for key in columns[:4]},
                             "error_type": "false_pass" if row["label"] == "FAIL" else "false_fail",
                             "source_condition": source_record["source_session"],
                             "image_path": Path(os.path.relpath(data / source_record["image_path"], results)).as_posix()})
    with (handoff / "images.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: ("baseline1" if key == "model_id" else row.get(key))
                             for key in fields})
    catalogue = json.loads((Path(__file__).resolve().parents[3] /
                            "shared/catalogue.json").read_text(encoding="utf-8"))
    write_json(handoff / "run_config.json", {
        "model_id": "baseline1", "method": summary["method"], "role": bundle["role"],
        "model_sha256": chosen["model_sha256"], "class_catalogue": catalogue["classes"],
        "preprocessing": bundle["config"], "dependencies": bundle["dependencies"],
        "capabilities": bundle["capabilities"], "score_type": "hog_reconstruction_mse",
        "supported_expected_counts": bundle["config"]["expected_counts"],
        "input_expected_counts": {row["image_id"]: bundle["config"]["expected_counts"]
                                  for row in predictions},
        "dataset_role": "validation"})
    aggregate_rows = []
    for name, group in summary["groups"].items():
        runs = group["runs"]
        aggregate_rows.append({
            "group": name, "mean_f1": group["mean_f1"], "stdev_f1": group["stdev_f1"],
            "mean_precision": sum(row["precision"] for row in runs) / len(runs),
            "mean_recall": sum(row["recall"] for row in runs) / len(runs),
            "mean_coverage": sum(row["coverage"] for row in runs) / len(runs),
            "mean_balanced_accuracy": sum(row["balanced_accuracy"] for row in runs) / len(runs),
            "mean_false_pass": group["mean_false_pass"],
            "mean_false_fail": group["mean_false_fail"],
            "mean_seconds": sum(row["seconds"] for row in runs) / len(runs)})
    with (results / "ablation_table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    write_json(results / "selection.json", {
        "purpose": "validation_selection_before_test",
        "method": summary["method"], "source_commit": provenance["commit"],
        "selected_group": selected, "integration_seed": 42,
        "model_path": Path(os.path.relpath(destination, results)).as_posix(),
        "model_sha256": sha256(destination), "summary_sha256": sha256(results / "summary.json"),
        "data_provenance_sha256": sha256(data / "provenance.json"),
        "train_manifest_sha256": sha256(data / "train.csv"),
        "valid_manifest_sha256": sha256(data / "valid.csv"),
        "scene_independence": provenance["scene_independence"],
        "selection": "mean validation FAIL F1, simpler setting on tie; seed42 predetermined"})
    support = provenance["support"]
    lines = [
        "# HOG+MLP 自编码器训练、验证与测试报告", "",
        "## 数据与训练规则", "",
        f"数据固定至上游 {provenance['commit']}。"
        f"训练使用 {support['train'].get('PASS', 0)} 张 PASS，其中旧 train 112 张、新补强 21 张。"
        f"验证使用 {support['valid'].get('PASS', 0)} 张 PASS 和 "
        f"{support['valid'].get('FAIL', 0)} 张 FAIL。验证目录中的图片只参与阈值和方案选择。", "",
        "采用 256×256 等比例补白灰度图、9 方向 HOG、16×16 单元、2×2 单元块及 L2-Hys。"
        "MLP 自编码器为 64–16–64 隐藏层，Adam 学习率 0.001，L2 系数 0.0001。"
        "标准化器和权重只学习训练 PASS。每组使用种子 42、43、44，固定 200 轮，"
        "保存最后一轮。验证阈值先最大化 FAIL F1，再比较平衡准确率、误拒数和阈值大小。", "",
        "## 单项与消融结果", "",
        "| 组 | FAIL F1 均值 | 标准差 | FAIL 召回率 | 平衡准确率 | 错误放行 | 错误拒绝 | 单次运行秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        lines.append(f"| {row['group']} | {row['mean_f1']:.4f} | {row['stdev_f1']:.4f} | "
                     f"{row['mean_recall']:.4f} | {row['mean_balanced_accuracy']:.4f} | {row['mean_false_pass']:.2f} | "
                     f"{row['mean_false_fail']:.2f} | {row['mean_seconds']:.2f} |")
    lines.extend([
        "", "A0 为原图；A1/A2 为左右和上下镜像；A3 为直角旋转；"
        "A4 为双轴镜像组合；A5 为八种唯一方向变换；B1 为 Gamma；B2 为 CLAHE。"
        "各组呈现相同数量的训练视图及更新预算。", "",
        f"按预定验证均值与简单性规则选择 {selected}，集成种子固定为 42。"
        "各组都使用同一验证集选阈值，这些结果属于开发证据。"
        "标准差描述三个随机种子的波动，不是场景总体性能的置信区间。", ""])
    lines.append("各组完整 Precision、Recall、覆盖率和运行成本见 ablation_table.csv；"
                 "逐种子的混淆矩阵和错误图片 ID 见各 seed 的 metrics.json。"
                 "运行成本包括特征准备、训练及验证，不是纯优化器耗时。")
    combined = [name for name in summary["groups"] if name.startswith("C")]
    lines.append("已运行 C0 组合与逐项移除对照，启用项见各组 config.json。"
                 if combined else "没有两项同时达到组合条件，按预定规则未运行 C 组合。")
    if "C0" in summary["groups"]:
        full = summary["groups"]["C0"]["mean_f1"]
        for name in combined:
            if name.startswith("C_without_"):
                delta = full - summary["groups"][name]["mean_f1"]
                lines.extend(["", f"移除 {name.removeprefix('C_without_')} 后，C0 的平均 F1 相对该移除组为 "
                              f"{delta:+.4f}。正值表示该项在组合中有开发收益，负值表示移除更好；"
                              "不把很小的三种子差异解读为已证明的普遍改善。"])
    baseline = summary["groups"]["A0"]
    improved = summary["groups"][selected]
    lines.extend([
        "", f"相对 A0，所选组的验证平均 F1 差值为 "
        f"{improved['mean_f1'] - baseline['mean_f1']:+.4f}，"
        f"错误放行差值为 {improved['mean_false_pass'] - baseline['mean_false_pass']:+.2f}，"
        f"错误拒绝差值为 {improved['mean_false_fail'] - baseline['mean_false_fail']:+.2f}。"
        "误拒与漏放需同时解读。", "",
        f"所选 seed42 的验证结果为 TP={chosen['tp']}、FP={chosen['fp']}、"
        f"FN={chosen['fn']}、TN={chosen['tn']}，覆盖率 {chosen['coverage']:.4f}。"
        f"PASS 平均重建误差 {chosen['valid_pass_error_mean']:.6g}，"
        f"FAIL 平均重建误差 {chosen['valid_fail_error_mean']:.6g}。", "",
        "## 冻结后测试", ""])
    if test_results is None:
        lines.append("尚未读取本轮 test。先固定所选组、A0、seed42、权重校验值、阈值及推理代码，"
                     "再接入旧包的 test 分区。")
    else:
        lines.extend([
            "测试使用原数据包提供的公开 test 分区。模型和阈值在首次接入其内容前已固定。"
            "测试结果只用于本节汇报，未用于重新训练、改阈值或挑选组别。", "",
            "| 模型组 | PASS 支持数 | FAIL 支持数 | TP | FP | FN | TN | FAIL F1 | 覆盖率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        tests = {}
        for path in sorted(test_results.glob("*/seed42/metrics.json")):
            scores = json.loads(path.read_text(encoding="utf-8"))
            group = path.parent.parent.name
            tests[group] = scores
            lines.append(f"| {group} | {scores['support']['PASS']} | {scores['support']['FAIL']} | "
                         f"{scores['tp']} | {scores['fp']} | {scores['fn']} | {scores['tn']} | "
                         f"{scores['f1']:.4f} | {scores['coverage']:.4f} |")
        if not tests:
            raise ValueError("No frozen test metrics found")
        support = next(iter(tests.values()))["support"]
        lines.extend(["", f"本分区只有 {support['FAIL']} 张 FAIL 和 {support['PASS']} 张 PASS。"
                      "样本少的类别不能给出稳定性能结论；若某类为零，其 Recall/F1 的数值约定不代表测得能力。"
                      "完整错误 ID 保留在 assessment/test 的逐图记录与 metrics.json。"])
        test_sources = json.loads((data / "test_provenance.json").read_text(encoding="utf-8"))["sources"]
        for row in test_sources:
            if row["label"] == "FAIL":
                lines.extend(["", f"按交付标注，{Path(row['image_path']).name} 为 FAIL，"
                              f"类别 0–5 的计数为 {[row['counts'][str(i)] for i in range(6)]}。"
                              "这是标注派生标签，仍需成员 3 完成人工复核。"])
        if selected in tests and tests[selected]["false_pass"]:
            lines.extend(["", "所选模型仍有测试漏放，不能据验证高 F1 宣称具备可靠现场检测能力。"
                          "验证异常主要是目录外干扰，而测试失败类型及正常图像条件不同，"
                          "这一差异提示泛化不足；尚不能仅凭本轮结果确定原因。"
                          "保留当前失败记录，不切回测试表现更好的对照、不调阈值。"])
        write_json(results / "provided_test_summary.json", tests)
    lines.extend([
        "", "## 局限与工程解释", "",
        "旧数据按文件名分段，实物场景独立性未核实。"
        "同一摆放的连拍或光照变体可能跨分区，验证和测试成绩可能偏乐观。"
        "已有 test 已公开，测试结果应称为提供分区评测。"
        "它不能代替后续按独立实物场景采集的新测试。", "",
        "新增验证 FAIL 主要由目标数量正确但混入目录外件构成。"
        "损伤、真实少件、多出合规件、相似错规格等能力，应按各自的真实支持数解释。"
        "来源许可、垫圈规格和预标注最终人工审核仍沿用成员 3 的待确认状态。", "",
        "自编码器学习正常图像的重建，HOG 保留位置与方向信息。"
        "重建误差可能同时受背景、摆放和拍摄条件影响。"
        "高误差不能直接定位或说明是哪颗零件出错。", "",
        "## 交接", "",
        "baseline1.joblib 保存标准化器、自编码器、阈值、数量范围及环境版本。"
        "公共 predict 输出 hog_reconstruction_mse，较高误差支持 FAIL。"
        "批量推理输出 images.csv 与 run_config.json。"
        "同图更换不支持的数量返回 UNSUPPORTED_COUNTS。", "",
        "全部训练历史、原始验证预测、逐组指标、源数据校验索引、冻结与测试访问记录均保留。"
        "CPU 离线耗时记录不能替代摄像头端到端 Median/P95；摄像头和 UI 联调由成员 7 完成。", "",
        "本轮使用成员 3 交付的训练、验证及测试分区，按各自用途保持隔离。"])
    (results / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Selected {selected} seed42; saved {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("results", "weights", "data", "destination"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--test-results", type=Path)
    args = parser.parse_args()
    report(args.results, args.weights, args.data, args.destination, args.test_results)


if __name__ == "__main__":
    main()
