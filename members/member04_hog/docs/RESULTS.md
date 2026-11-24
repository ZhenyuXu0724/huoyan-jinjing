# Baseline1_HOGMLP_v4_FinalDataset_Rebuild

成员4在固定数据版本 `final_training_v4` 上完整重做的 HOG+MLP 实验基线。全部模型从随机初始化学习，旧权重、旧阈值和旧默认入口不参与本轮。test 只在冻结后一次性评价，本轮成绩属于开发证据，不构成现场性能保证。

## 数据与规则

- 数据来源 commit `bde8ea7116761bc8ee7fca1d47c336126daa995f`，目录 tree `c6882fd823b1f458105b47fea2f52af0b357ef9c`。
- 开发审计：862 个文件；全量审计：1260 个文件；test 期间只使用冻结后的三个代表模型。
- 训练 249 张原图（166 PASS、83 FAIL）；验证 34 张（20 PASS、14 FAIL）；test 30 张（15/15）。AE 只学习 166 张 PASS 原图。
- 预处理 256×256、灰度、等比例缩放白边；HOG 9 方向/16×16/2×2/L2-Hys=8100 维；每轮每图一个随机视图；主实验固定第 200 轮。
- 阈值在 34 张验证图上选：误拒≤4/20，先漏放少、再误拒少、再取较大规范阈值；全 PASS 与全 FAIL 都在候选内。

## 66 次主实验

| 方法 | 入选组 | 平均漏放 | 平均误拒 | 平均 FAIL F1 |
|---|---|---:|---:|---:|
| ae | A1 | 2.33 | 4.00 | 78.62% |
| supervised | A2 | 0.33 | 2.67 | 90.23% |

| 方法 | 组 | 平均漏放±sd | 平均误拒±sd | 平均 FAIL F1±sd | 视图数 |
|---|---|---:|---:|---:|---:|
| ae | A0 | 3.00 ± 0.82 | 4.00 ± 0.00 | 75.76% | 1 |
| ae | A1 | 2.33 ± 0.47 | 4.00 ± 0.00 | 78.62% | 2 |
| ae | A2 | 4.00 ± 1.41 | 4.00 ± 0.00 | 71.09% | 2 |
| ae | A3 | 7.33 ± 0.94 | 3.67 ± 0.47 | 54.57% | 4 |
| ae | A4 | 4.67 ± 1.70 | 3.67 ± 0.47 | 68.54% | 4 |
| ae | A5 | 10.00 ± 0.82 | 3.33 ± 0.47 | 37.19% | 8 |
| ae | B1 | 3.00 ± 0.82 | 3.67 ± 0.47 | 76.67% | 3 |
| ae | B2 | 6.67 ± 0.47 | 3.67 ± 0.47 | 58.62% | 1 |
| ae | C0 | 7.00 ± 2.16 | 3.00 ± 0.82 | 57.07% | 2 |
| ae | C_without_geometry | 6.67 ± 0.47 | 3.67 ± 0.47 | 58.62% | 1 |
| ae | C_without_clahe | 4.00 ± 1.41 | 4.00 ± 0.00 | 71.09% | 2 |
| supervised | A0 | 1.00 ± 0.00 | 2.67 ± 1.25 | 87.80% | 1 |
| supervised | A1 | 0.67 ± 0.94 | 2.00 ± 1.41 | 91.03% | 2 |
| supervised | A2 | 0.33 ± 0.47 | 2.67 ± 1.25 | 90.23% | 2 |
| supervised | A3 | 3.33 ± 0.47 | 3.67 ± 0.47 | 75.29% | 4 |
| supervised | A4 | 2.00 ± 1.41 | 2.00 ± 1.63 | 85.62% | 4 |
| supervised | A5 | 1.33 ± 0.47 | 2.33 ± 1.25 | 87.50% | 8 |
| supervised | B1 | 1.33 ± 0.94 | 1.33 ± 0.47 | 90.38% | 3 |
| supervised | B2 | 1.67 ± 0.47 | 3.67 ± 0.47 | 82.22% | 1 |
| supervised | C0 | 2.33 ± 1.25 | 2.67 ± 1.25 | 82.28% | 2 |
| supervised | C_without_geometry | 1.67 ± 0.47 | 3.67 ± 0.47 | 82.22% | 1 |
| supervised | C_without_clahe | 0.33 ± 0.47 | 2.67 ± 1.25 | 90.23% | 2 |

标准差为三个种子的总体标准差（ddof=0），不是泛化置信区间。

- ae 等价组：B2 vs C_without_geometry=一致; A2 vs C_without_clahe=一致。
- supervised 等价组：B2 vs C_without_geometry=一致; A2 vs C_without_clahe=一致。

## 9 次有限改进

| 配置 | 平均最佳漏放 | 平均最佳误拒 | 平均末轮漏放 | 平均末轮误拒 | 参数量 | L2 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.00 | 3.33 | 0.33 | 2.67 | 519538 | 0.0001 |
| R1 | 0.33 | 3.00 | 0.67 | 3.67 | 259514 | 0.0001 |
| R2 | 0.00 | 3.33 | 0.33 | 3.00 | 519538 | 0.01 |

入选配置 **R0**；R0 三个种子第 200 轮精确复现主实验入选组，未用验证结果改变梯度或增强采样。

## 代表模型与默认选择

| 方法 | 组/配置 | 平均漏放 | 平均误拒 | 参数量 | 训练视图 |
|---|---|---:|---:|---:|---:|
| ae | A1 | 2.33 | 4.00 | 1047092 | 2 |
| supervised | A2 | 0.33 | 2.67 | 519538 | 2 |
| refinement | R0 | 0.00 | 3.33 | 519538 | 2 |

本轮默认选中 **refinement** seed42；阈值 `0.9850894510746002`，版本 `Baseline1_HOGMLP_v4_FinalDataset_Supervised_Capacity_L2_EpochSelection`。冻结发生在任何 test 解码之前。

## 冻结后逐分区评价

| 分区/模型 | TP | FP误拒 | FN漏放 | TN | FAIL F1 | 平衡准确率 |
|---|---:|---:|---:|---:|---:|---:|
| train/ae | 47 | 0 | 36 | 166 | 72.31% | 78.31% |
| train/supervised | 82 | 0 | 1 | 166 | 99.39% | 99.40% |
| train/refinement | 71 | 0 | 12 | 166 | 92.21% | 92.77% |
| val/ae | 12 | 4 | 2 | 16 | 80.00% | 82.86% |
| val/supervised | 13 | 1 | 1 | 19 | 92.86% | 93.93% |
| val/refinement | 14 | 4 | 0 | 16 | 87.50% | 90.00% |
| test/ae | 11 | 7 | 4 | 8 | 66.67% | 63.33% |
| test/supervised | 5 | 3 | 10 | 12 | 43.48% | 56.67% |
| test/refinement | 9 | 4 | 6 | 11 | 64.29% | 66.67% |

| 分区 | 全PASS准确率 | 全FAIL准确率 | 全FAIL F1 |
|---|---:|---:|---:|
| train | 66.67% | 33.33% | 50.00% |
| val | 58.82% | 41.18% | 58.33% |
| test | 50.00% | 50.00% | 66.67% |

AE 的 train 评价包含 83 张它从未学习的 FAIL 图，应与“AE 实际学习 166 张 PASS”分开理解。

## 未见类型与错误

- ae：test 含 S/H 未见件的 11 张图中，图级漏放 2、误拒 0（仅图像级判定，不是定位或分类成功率）。
- supervised：test 含 S/H 未见件的 11 张图中，图级漏放 7、误拒 0（仅图像级判定，不是定位或分类成功率）。
- refinement：test 含 S/H 未见件的 11 张图中，图级漏放 4、误拒 0（仅图像级判定，不是定位或分类成功率）。

三方法共有/修复/新增错误 ID 见 `error_ids.csv`；共 63 条涉及至少一个方法的错误。

## 稳定性诊断

每图 21 设置：原图＋{240,224}×{白/边缘灰度5%补白}×{居中,左,右,上,下}。原图对缩小居中同时改变尺度、重采样与补白，不能当作纯位置实验；零翻转不自动等于鲁棒。

| 分区/模型 | 至少一种变体翻转/原图数 | 原图漏放 | 原图误拒 |
|---|---:|---:|---:|
| train_ae | 202/249 | 36 | 0 |
| train_supervised | 162/249 | 1 | 0 |
| train_refinement | 201/249 | 12 | 0 |
| val_ae | 18/34 | 2 | 4 |
| val_supervised | 24/34 | 1 | 1 |
| val_refinement | 22/34 | 0 | 4 |

## 离线成本

| 模型 | Median ms | P95 ms | 模型字节 | 训练秒 |
|---|---:|---:|---:|---:|
| ae | 93.28 | 229.30 | 16963178 | 40.8 |
| supervised | 94.07 | 227.11 | 8522699 | 32.3 |
| refinement | 94.80 | 216.11 | 8518762 | 22.0 |

单线程、预加载、20 张固定验证图、每模型预热 2 图、三遍交错，含读图与预处理；不含摄像头、界面和端到端部署开销。

## 解释与限制

- 验证集仅 34 张，每次漏放变化约为 1/14=7.14 个百分点，误拒变化约为 1/20=5 个百分点；75 次训练、轮次和阈值选择会带来开发集乐观性。
- AE 与监督同时改变学习方法与参与学习的样本，不能把差异单独归因于网络结构。
- test 含历史接触和 11 张未见件混缺件图像；HOG 只输出整图判定，不输出框或实际数量。
- 场景组字段互不重叠，但物理场景独立性未独立证实；来源、批次、异常类型与标签仍可能混杂。
- 本轮没有因为 test 成绩回头改阈值、换种子或重训。

## 复现与证据

```powershell
python -m members.member04_hog.final_v4.prepare --source-commit bde8ea7116761bc8ee7fca1d47c336126daa995f
python -m members.member04_hog.final_v4.audit --scope development
python -m members.member04_hog.final_v4.train --method ae --protocol members/member04_hog/configs/final_v4/protocol.json
python -m members.member04_hog.final_v4.train --method supervised --protocol members/member04_hog/configs/final_v4/protocol.json
python -m members.member04_hog.final_v4.refine --protocol members/member04_hog/configs/final_v4/protocol.json
python -m members.member04_hog.final_v4.freeze
python -m members.member04_hog.final_v4.diagnose --splits train val
python -m members.member04_hog.final_v4.prepare --scope full
python -m members.member04_hog.final_v4.audit --scope full
python -m members.member04_hog.final_v4.evaluate --frozen
python -m members.member04_hog.final_v4.report
python -m members.member04_hog.final_v4.verify
```

产物在 `outputs/member04_hog/final_v4/`，权重在 `weights/member04_hog/final_v4/`；原始图片不进入交付包。
