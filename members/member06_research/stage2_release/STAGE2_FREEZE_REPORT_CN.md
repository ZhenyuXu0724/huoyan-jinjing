# Baseline 2（Frozen ResNet18 + Linear）Stage 2 最终模型冻结报告

- **冻结状态**：`FROZEN_PRE_TEST`（正式 Test 评价前完成冻结，冻结后禁止因 Test 结果修改模型或参数）
- **冻结时间（UTC）**：`2026-11-24T12:30:03Z`
- **最终选定实验 ID**：`V4-R2_RAW_WEIGHTED_CE_SEED42`
- **模型版本标识**：`baseline2_final_v4_seed42_ep7`
- **源权重相对路径**：`members/member06_research/runs/final_training_v4_20261124_192110_PJSHyF/V4-R2_RAW_WEIGHTED_CE_SEED42/model/best_within_20.pt`
- **冻结交付权重路径**：`members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`
- **冻结权重 SHA-256**：`c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`（文件大小 `44809257` 字节）
- **Linear(512,2) 头参数哈希**：`7847cca3d9461393110661ffec441d5c201efc35779c3cc5b84e5de10a1878d8`
- **冻结 Backbone 参数哈希**：`b3b9896ad9efa14f33c6b3c3a084719503f1955b1d81e17df18e482e47e714d9`（与同目录 `best.pt` 逐张量核验一致：`True`）

## 1. 冻结模型结构、训练与推理协议

| 项目 | 冻结设定 |
|---|---|
| 模型结构 | `FrozenResNet18 (IMAGENET1K_V1) + Linear(512, 2)`（总参数 `11,177,538`：冻结 Backbone `11,176,512` + 线性头 `1,026`） |
| 官方预训练权重身份 | `resnet18-f37072fd.pth`（SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`） |
| 冻结与推理模式 | Backbone 参数与全部 BatchNorm 统计量完全冻结；推理强制 `model.eval()` + `torch.inference_mode()`；加载冻结权重无需联网下载预训练模型 |
| 输入尺寸与预处理 | `RGB`（不执行 EXIF 自动旋转；摄像头 BGR 显式转 RGB）→ `224×224 letterbox`（双线性插值，`pad_rgb=[0,0,0]` 居中填充）→ ImageNet 归一化（`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`） |
| 类别映射与阳性类 | `PASS = 0`，`FAIL = 1`（`FAIL` 为阳性类，输出未校准 `FAIL` 概率分数 `P(FAIL) = softmax(logits)[1]`） |
| 固定预期数量规则 | `expected_counts = [2, 2, 2, 2, 2]`（五类规定零件各 2 个）；非该计数请求显式返回 `UNSUPPORTED_COUNTS` |
| 分类决策与平局规则 | 默认 `argmax`（`P(FAIL) > P(PASS)` 判 `FAIL`；若 `P(FAIL) == P(PASS) == 0.5` 平局取类别索引 `0` 即 `PASS`），无事后阈值调优 |
| 能力边界声明 | 仅输出整图 `PASS/FAIL` 与 `FAIL` 概率分数；不伪造目标框 `objects`、逐类计数或 `non_catalogue_count` |
| 训练超参数与预算 | `Adam(lr=0.001, weight_decay=0)`，`batch_size=16`，最大预算 `20 Epoch`，训练损失 `Weighted CE [PASS=0.75, FAIL=1.50]`，验证选模为最低**无权** `Val CE`（最佳 `Epoch = 7`） |

## 2. 数据版本与清单哈希身份

- **小组统一数据集**：`members/member03_data/final_training_v4/`（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`）
- **训练清单 `manifests/resnet/train.csv`**（`249` 原图：`166 PASS / 83 FAIL`）：`8e77e41fc3da98f829d1cd65466daf58e3acdb8c5e6c4f1df0a8716be6115f3b`
- **验证清单 `manifests/resnet/val.csv`**（`34` 原图：`20 PASS / 14 FAIL`）：`4d5a04fdfe9a1b53a3c8aeb5bd9f8ea00ccafa15c29d3b46b8575914965e9c05`
- **测试清单 `manifests/resnet/test.csv`**（`30` 原图：`15 PASS / 15 FAIL`，冻结前仅登记文件哈希、未读取样本内容推理）：`b9b101f4d9948a8d75ca4a87797922860ca83e5f87d65c45838eb7e8382a455b`
- **总清单 `split_manifest.csv`**：`80b4e7500c0ff88236ba4a5252eba735cc9bad8ac001442507265726d7fccc42`

## 3. 冻结检查点的验证集与训练集参考表现（`Epoch = 7`）

| 评估集合 | 样本数 (`PASS/FAIL`) | 无权 CE Loss | Accuracy | Balanced Accuracy | FAIL Precision | FAIL Recall | FAIL F1 | FN / FP | 混淆矩阵 `[[TN,FP],[FN,TP]]` | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `Val`（选模依据） | 34 (`20/14`) | `0.353494` | `0.8824` (`88.24%`) | `0.9000` (`90.00%`) | `0.7778` | `1.0000` | `0.8750` | `0 / 4` | `[[16, 4], [0, 14]]` | `0.9679` |
| `Train`（固定权重重评） | 249 (`166/83`) | `0.301982` | `0.9076` (`90.76%`) | `0.8765` (`87.65%`) | `0.9286` | `0.7831` | `0.8497` | `18 / 5` | `[[161, 5], [18, 65]]` | `0.9630` |

## 4. 最终选型科学依据

1. **损失策略（`V4-R2` vs `V4-R1`）**：针对 `final_training_v4` 训练集 `2:1`（`166 PASS : 83 FAIL`）类别不平衡，采用逆频率 `Weighted CE [0.75, 1.50]`（验证保持无权 CE）在 3 个随机种子（`42, 43, 44`，20 轮预算）上将平均 `Val Balanced Accuracy` 从 `0.8643 ± 0.0357` 提升至 `0.8881 ± 0.0206`，平均漏放 `FN` 从 `1.00 ± 1.00` 降至 `0.33 ± 0.58`，且平均误拒 `FP`（`4.00`）未恶化。
2. **训练预算（`20 Epoch` 最大预算 + 最低无权 `Val Loss` 选模）**：`Seed=42` 与 `Seed=44` 在 10 轮与 20 轮预算内均分别锁定第 `7` 轮与第 `10` 轮最佳点；`Seed=43` 在第 `14` 轮进一步降低无权 `Val Loss`（`0.3923 → 0.3525`）并减少 1 例误拒。
3. **输入分辨率最终对照（`224` vs `384`）**：单变量受控实验 `V4-R4_RAW_WEIGHTED_CE_384` 未满足全部四项预设升级条件——三 seed 平均 `Val BA` 下降 `4.05` 个百分点（`0.8881 → 0.8476`），平均 `FN`（`0.33 → 1.00`）与 `FP`（`4.00 → 4.67`）双双恶化，代表种子 `Seed=42` 退步（`BA 0.9000 → 0.8393`, `FN 0 → 1`, `FP 4 → 5`），且 CPU 单图离线中位耗时增加至 `1.28x`（`161.76 ms → 207.25 ms`），故正式否决 `384`、保留 `224×224`。
4. **代表种子不事后择优**：正式代表种子在实验启动前即预先固定为 `seed=42`（`Epoch 7`, `Val Acc=88.24%`, `Val BA=90.00%`, `FN=0`, `FP=4`），不事后挑选 `seed=44` 的更低 `Val Loss`。
