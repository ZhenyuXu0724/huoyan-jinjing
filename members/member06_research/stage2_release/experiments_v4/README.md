# `final_training_v4` 九次正式训练实验完整归档

本目录归档了在全组统一权威数据集 `members/member03_data/final_training_v4/`（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`：`Train = 249 [166P/83F]`、`Val = 34 [20P/14F]`）上实际完成的全部 **9 次**可控训练实验产物、两份统一对比报告及逐轮曲线。

> **归档范围与轻量化原则**：本目录保留全部配置、运行身份、预检记录、逐轮历史、10/20 轮预算结果、逐图验证预测与对比报告，**不包含** `*.pt` 中间检查点或特征缓存目录。唯一入选冻结模型权重见 `../weights/baseline2_v4_seed42_best_within_20.pt`（SHA-256: `c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`）。提交 Git 时，本地可视化曲线图 `*.png` 按仓库 `.gitignore` 规则不纳入提交白名单。

---

## 1. 九次真实训练运行总览（20 轮最大预算，最低无权 Val Loss 选模）

| 实验 ID | 策略 | 输入尺寸 | 训练 CE 权重 `[PASS, FAIL]` | Seed | 最佳 Epoch | 无权 Val Loss | Val Acc | Val BA | FN / FP | Val ROC-AUC | 原运行目录来源 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `V4-R1_RAW_CE_SEED42` | `V4-R1_RAW_CE` | 224 | `[1.00, 1.00]` | 42 | 10 | 0.322921 | 85.29% | 86.43% | 1 / 4 | 0.9643 | `final_training_v4_20261124_192110_PJSHyF` |
| `V4-R1_RAW_CE_SEED43` | `V4-R1_RAW_CE` | 224 | `[1.00, 1.00]` | 43 | 14 | 0.336127 | 82.35% | 82.86% | 2 / 4 | 0.9286 | `final_training_v4_20261124_192110_PJSHyF` |
| `V4-R1_RAW_CE_SEED44` | `V4-R1_RAW_CE` | 224 | `[1.00, 1.00]` | 44 | 13 | 0.322473 | 88.24% | 90.00% | 0 / 4 | 0.9429 | `final_training_v4_20261124_192110_PJSHyF` |
| **`V4-R2_RAW_WEIGHTED_CE_SEED42`** ★ | **`V4-R2_RAW_WEIGHTED_CE`** | **224** | **`[0.75, 1.50]`** | **42** | **7** | **0.353494** | **88.24%** | **90.00%** | **0 / 4** | **0.9679** | `final_training_v4_20261124_192110_PJSHyF` |
| `V4-R2_RAW_WEIGHTED_CE_SEED43` | `V4-R2_RAW_WEIGHTED_CE` | 224 | `[0.75, 1.50]` | 43 | 14 | 0.352471 | 85.29% | 86.43% | 1 / 4 | 0.9071 | `final_training_v4_20261124_192110_PJSHyF` |
| `V4-R2_RAW_WEIGHTED_CE_SEED44` | `V4-R2_RAW_WEIGHTED_CE` | 224 | `[0.75, 1.50]` | 44 | 10 | 0.329175 | 88.24% | 90.00% | 0 / 4 | 0.9393 | `final_training_v4_20261124_192110_PJSHyF` |
| `V4-R4_RAW_WEIGHTED_CE_384_SEED42` | `V4-R4_RAW_WEIGHTED_CE_384` | 384 | `[0.75, 1.50]` | 42 | 16 | 0.375946 | 82.35% | 83.93% | 1 / 5 | 0.9714 | `final_training_v4_384_20261124_195746_Rg1Mug` |
| `V4-R4_RAW_WEIGHTED_CE_384_SEED43` | `V4-R4_RAW_WEIGHTED_CE_384` | 384 | `[0.75, 1.50]` | 43 | 14 | 0.314301 | 85.29% | 86.43% | 1 / 4 | 0.9714 | `final_training_v4_384_20261124_195746_Rg1Mug` |
| `V4-R4_RAW_WEIGHTED_CE_384_SEED44` | `V4-R4_RAW_WEIGHTED_CE_384` | 384 | `[0.75, 1.50]` | 44 | 10 | 0.340479 | 82.35% | 83.93% | 1 / 5 | 0.9643 | `final_training_v4_384_20261124_195746_Rg1Mug` |

★ `V4-R2_RAW_WEIGHTED_CE_SEED42`（预设代表种子 `42`，`Epoch = 7`）为最终冻结模型。

---

## 2. 三种子（42 / 43 / 44）汇总均值 ± 样本标准差（`n=3, ddof=1`）

| 策略（20 轮预算） | 无权 Val Loss | Val Accuracy | Val Balanced Acc | FAIL Recall | FN /14 | FP /20 | Val ROC-AUC |
|---|---|---|---|---|---|---|---|
| `V4-R1_RAW_CE` (224) | `0.3272 ± 0.0078` | `85.29% ± 2.94%` | `86.43% ± 3.57%` | `92.86% ± 7.14%` | `1.00 ± 1.00` | `4.00 ± 0.00` | `0.9452 ± 0.0180` |
| **`V4-R2_RAW_WEIGHTED_CE` (224)** | `0.3450 ± 0.0138` | **`87.25% ± 1.70%`** | **`88.81% ± 2.06%`** | **`97.62% ± 4.12%`** | **`0.33 ± 0.58`** | **`4.00 ± 0.00`** | `0.9381 ± 0.0304` |
| `V4-R4_RAW_WEIGHTED_CE_384` (384) | `0.3436 ± 0.0309` | `83.33% ± 1.70%` | `84.76% ± 1.44%` | `92.86% ± 0.00%` | `1.00 ± 0.00` | `4.67 ± 0.58` | `0.9690 ± 0.0041` |

---

## 3. 归档目录结构

- [comparison_224_r1_vs_r2/COMPARISON_V4_CN.md](comparison_224_r1_vs_r2/COMPARISON_V4_CN.md)：`V4-R1` vs `V4-R2` × `10/20` 轮 × `3 Seeds` 统一比较报告（含占位符修正）、`v4_summary.json`、`v4_per_run_budget_metrics.csv`、`v4_val_34_grouped_slices.csv`、`v4_val_34_per_image_comparison.csv`、`curves_v4_6runs.png`
- [comparison_224_vs_384/COMPARISON_V4_FINAL_RESOLUTION_CN.md](comparison_224_vs_384/COMPARISON_V4_FINAL_RESOLUTION_CN.md)：`224` vs `384` 单变量分辨率对照报告（含四项预设选型门槛核对及相同边界下离线推理延迟参考）、`v4_224_vs_384_summary.json`、`v4_224_vs_384_per_seed_metrics.csv`、`v4_224_vs_384_val_34_changes.csv`、`curves_v4_224_vs_384.png`
- `runs/<EXPERIMENT_ID>/`：每次运行的 `preflight.json`、`experiment_spec.json`、`config.json`、`run_config.json`、`checkpoint_manifest.json`、`best_metrics.json`、`budget_evaluations.json`、`history.csv`、`history.json`、`best_val_predictions.json`、`best_within_20_val_predictions.json`
