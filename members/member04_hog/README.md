# 成员 4 · HOG 基线交付指南

本模块交付 baseline1 的整图 PASS/FAIL 判定、训练消融和共同图像级评价。当前活动版本为 **Baseline1_HOGMLP_v4_FinalDataset_Rebuild**：在固定数据版本 `final_training_v4` 上完整重做 HOG + MLP 的三个方法（仅 PASS 自编码器、监督 MLP、监督容量/L2/选轮次），75 次正式训练全部从随机初始化学习，旧权重、旧阈值和旧默认入口不参与。

## 从哪里开始

| 需要了解的内容 | 说明文件 |
|---|---|
| 数据来源、恢复范围与审计 | [数据审计](docs/DATA.md) |
| 预处理、11 组、阈值与选型规则 | [方法与实验协议](docs/PROTOCOL.md) |
| 75 次完整结果、消融与改进 | [结果报告](docs/RESULTS.md) |
| 三方法对照与默认选择 | [方法对照](docs/COMPARISON.md) |
| train/val/test 逐分区混淆 | [分区结果](docs/SPLIT_RESULTS.md) |
| 错误 ID、S/H 未见件与分组 | [错误分析](docs/ERROR_ANALYSIS.md) |
| 21 变体稳定性与离线成本 | [稳定性与成本](docs/STABILITY_COST.md) |
| 复现、续跑、验证与打包 | [复现与交接](docs/WORKFLOW.md) |
| 版本命名 | [版本说明](docs/VERSIONS.md) |
| 旧版归档位置与映射 | [归档映射](docs/ARCHIVE_MAPPING.md) |
| 本轮实际状态与待办 | [进度与交接](progress.md) |

## 当前结论

- 验证集 34 张（20 PASS、14 FAIL）。默认改进 R0 seed42 在 val 上漏放 0、误拒 4；监督主实验 A2 seed42 漏放 1、误拒 1；AE A1 seed42 漏放 2、误拒 4。
- 冻结后一次性 test 30 张（15/15）：AE 漏放 4、误拒 7，FAIL F1 0.6667；监督主实验漏放 10、误拒 3，FAIL F1 0.4348；默认 R0 漏放 6、误拒 4，FAIL F1 0.6429。
- 默认按“验证集平均漏放优先”的规则选中 R0，不代表现场合格；test 仍是历史接触分区，11 张未见件图混有缺件，HOG 不输出框或实际数量。
- 旧 v1/v2/v3 已移入仓库同级归档，旧成绩不进入本报告，也不能与 v4 混表或当作性能增量。

## 文件与产物位置

```text
members/member04_hog/
  model.py                  # 公共单图入口，读取 configs/active_model.json
  inference.py              # 批量入口（image_id + expected_counts）
  train.py                  # final_v4.train 兼容包装
  evaluate.py               # 通用指标/JSON/CSV
  compare.py                # 共同图像级评价器
  delivery.py               # 便携包校验与打包委托
  features.py / data.py     # HOG 预处理、清单与重复检查
  final_v4/                 # 数据恢复、审计、66+9 训练、冻结、诊断、评价、报告、验证、打包
  configs/final_v4/         # 来源锁与 75 次任务单
  configs/active_model.json # 冻结后的默认模型指针
  docs/                     # 当前版本文档
  plans/                    # v4 重建计划与规划审计
data/member04_hog/final_training_v4/       # 本地忽略目录，按固定 commit 恢复
weights/member04_hog/final_v4/              # 84 份检查点 + 三个代表/默认副本
outputs/member04_hog/final_v4/              # 审计、运行记录、冻结、诊断、评价、报告、验证
```

`data/`、`weights/`、`outputs/` 在 Git 忽略范围内，克隆不会自动附带；迁移时重新恢复数据并生成模型，或使用独立交付包。

## 安装与单图预测

从仓库根目录执行，Python 3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r members/member04_hog/requirements.txt
python -m members.member04_hog.final_v4.prepare --source-commit bde8ea7116761bc8ee7fca1d47c336126daa995f
python -m members.member04_hog.final_v4.audit --scope development
```

然后用完整流程训练/冻结/评价，或直接使用随交付包提供的默认权重：

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.evaluate predict --model weights/member04_hog/final_v4/representatives/default.joblib --image your_image.jpg
```

程序也可调用 `from members.member04_hog.model import predict`，再执行 `predict(image_path)`。返回 `status`、`predicted_label`、`prediction_score`、`score_type`：

- 正常为 `OK`；坏图/缺图为 `INPUT_ERROR`；默认权重缺失、摘要不符或版本不匹配为 `MODEL_UNAVAILABLE`。
- 默认模型是未校准 FAIL 输出，`score_type=uncalibrated_fail_probability`；AE 代表为 `hog_reconstruction_mse`。两者数值不可直接比较。
- 判定固定为 `score >= threshold → FAIL`。阈值、轮次、版本和摘要记录在 `configs/active_model.json` 与 `freeze.json`。

## 批量入口与数量规则

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.inference --input_dir IMAGES --counts_file counts.json --output_dir RESULT
```

`counts.json` 是非空列表，每项只有 `image_id` 与 `expected_counts`；只接受五类各 2。数量不支持返回 `UNSUPPORTED_COUNTS`，坏图/缺图返回 `INPUT_ERROR`，均不输出判定和分数；缺权重时启动失败且不创建输出目录。批量输出 `images.csv` 与 `run_config.json`，不伪造对象、框或实际数量。

## 数量与协作边界

当前模型只支持类别 1–5 各 2 个的候选规则；最终规格与数量仍以组长确认为准。单图入口默认使用模型内置数量；不输出零件框或实际数量，不能把空值当作“没有目录外零件”。

成员3负责场景、规格与标注审核；成员4负责基线与图像评价；成员6对齐共同指标；成员7接入摄像头并测端到端耗时。公共接口及其他成员目录由组长整合，本轮未修改共享注册表、其他成员目录或公共检查规则。

## 版本与归档

统一名称规则见 [版本说明](docs/VERSIONS.md)。旧 v1/v2/v3 的代码、权重、运行记录和文档保存在仓库同级 `Project/delivery/member04_history_before_v4_20261124/`，映射见 [归档映射](docs/ARCHIVE_MAPPING.md)。旧 Release 的标签、附件名、下载地址与摘要不变。

本轮 GitHub 提交：[PR #14](https://github.com/ZhenyuXu0724/huoyan-jinjing/pull/14) 已合并；预发布 [Baseline1_HOGMLP_v4_FinalDataset_Rebuild](https://github.com/ZhenyuXu0724/huoyan-jinjing/releases/tag/Baseline1_HOGMLP_v4_FinalDataset_Rebuild) 含 ZIP、三个代表权重及验证/测试附件。上传记录与限制见 [progress.md](progress.md)。
