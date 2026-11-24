# 成员 6：Baseline 2 · Frozen ResNet18 + Linear（Stage 2 交付版）

本目录为课程项目《火眼金睛——机械零件视觉检测》**成员 6（ResNet18 基线与 Stage 2 研究）**的完整代码、配置、冻结模型交付、正式测试评价与研究总结。

> **当前阶段状态**：已完成全部训练优化（`final_training_v4` 上 9 次正式训练）、最终模型冻结（`V4-R2_RAW_WEIGHTED_CE_SEED42`, `Epoch = 7`）与 30 张统一 `Test` 原图单次正式评价（`Accuracy = 70.00%`, `Balanced Accuracy = 70.00%`, `FN = 4, FP = 5`）。

---

## 1. 项目简介与成员 6 职责

- **核心任务**：实现并研究课程 **Baseline 2（整图监督二分类）**，对五类规定机械零件（`M4×25`、`M5×10`、`M4×10`、六角螺母、平垫圈）托盘照片进行整图合规判别（`PASS` / `FAIL`），输出未校准的异常概率分数 `P(FAIL)`。
- **能力边界**：Baseline 2 仅支持固定配方 `expected_counts = [2, 2, 2, 2, 2]`（五类各 2 个且无目录外零件），对其他数量请求显式返回 `UNSUPPORTED_COUNTS`，**不输出、也不伪造**目标检测框、逐类零件计数或 `non_catalogue_count`。

---

## 2. 模型架构与工作流程

1. **骨干网络（完全冻结）**：加载 ImageNet 预训练 `ResNet18`（`IMAGENET1K_V1`），移除全连接分类层输出 `512` 维全局平均池化（GAP）特征；训练全程设置 `requires_grad_(False)` 并强制 `backbone.eval()`，锁定全部 `BatchNorm` 运行统计量（`11,176,512` 冻结参数）。
2. **线性分类头（可训练）**：`Linear(512, 2)`（`1,026` 参数），`PASS = 0`、`FAIL = 1`（`FAIL` 为阳性类）。
3. **图像预处理**：读取存储 `RGB` 像素（不执行 EXIF 自动旋转；摄像头 `BGR` 帧显式反转通道为 `RGB`）→ **`224×224 letterbox`** 等比缩放并居中黑边填充（`pad_rgb=[0,0,0]`，保留边缘零件不被中心裁剪丢失）→ ImageNet 均值方差标准化。
4. **训练与选模协议**：`Adam(lr=0.001, weight_decay=0)`，`batch_size=16`，最大预算 `20 Epoch`；训练使用逆频率 `Weighted CE [PASS=0.75, FAIL=1.50]`，验证选模始终使用**无权 CE**，取最低无权 `Val Loss`（平局最早轮次优先）；推理决策采用默认 `argmax`（`P(FAIL) > 0.5` 判 `FAIL`，`0.5` 平局归 `PASS`）。

---

## 3. `final_training_v4` 数据说明

- **数据集来源**：小组统一最终版数据集 [members/member03_data/final_training_v4/](../member03_data/final_training_v4/README.md)（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`）。
- **划分构成**：
  - `Train`：`249` 张真实原图（`166 PASS / 83 FAIL`，`22` 个登记场景组；另有 `102` 张可选训练增强图，依据前期消融结论正式训练未并入增强图）。
  - `Val`：`34` 张真实原图（`20 PASS / 14 FAIL`，`11` 个登记场景组）。
  - `Test`：`30` 张真实原图（`15 PASS / 15 FAIL`，`11` 个登记场景组；冻结后单次盲推理评估，含训练/验证未见的目录外螺钉 `S` 与黄铜六角长柱 `H`）。

---

## 4. 安装环境

本模块已在 `Ubuntu 24.04.5 LTS`、`Python 3.11.17`（CPU-only）环境下完成全部训练、测试与验证（详细硬件与依赖说明见 [ENVIRONMENT.md](ENVIRONMENT.md)，精确版本锁见 [requirements-lock.txt](requirements-lock.txt)）：

```bash
python3.11 -m venv members/member06_research/.venv
source members/member06_research/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
python -m pip install -r members/member06_research/requirements.txt
python -m pip check
```

---

## 5. 快速运行（在仓库根目录执行）

```bash
# 1. 下载冻结权重至 members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt 并校验 SHA-256
sha256sum members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
# 期望: c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4

# 2. 单图推理（无需联网下载预训练权重）
members/member06_research/.venv/bin/python -m members.member06_research.stage2_adapter \
  --image path/to/image.jpg \
  --output /tmp/baseline2_single.json

# 3. 验证测试集预检（只读）与核心单元测试
bash members/member06_research/scripts/run_stage2_final_test.sh --check-only
members/member06_research/.venv/bin/python -m unittest members/member06_research/tests/test_stage2_release.py -v
```

---

## 6. 完整训练与对照复现方法

> **联网提示**：从头重训需首次缓存官方 `ResNet18_Weights.IMAGENET1K_V1`（SHA-256 `f37072fd...`）；仅加载冻结模型推理或复现 Test 评价**无需联网**。详细 15 步操作指南见 [stage2_release/USER_GUIDE_CN.md](stage2_release/USER_GUIDE_CN.md)。

```bash
# 1. 准备数据 provenance 清单（不修改原数据集）
members/member06_research/.venv/bin/python -m members.member06_research.v4_data prepare \
  --v4-root members/member03_data/final_training_v4 \
  --output-dir members/member06_research/data/final_training_v4_intake

# 2. 运行 224 核心 6 组实验（V4-R1 普通 CE vs V4-R2 Weighted CE × Seed 42/43/44 × 20 轮）
bash members/member06_research/scripts/run_final_training_v4.sh

# 3. 运行 384 分辨率 3 组对照实验（V4-R4 × Seed 42/43/44 × 20 轮，自动生成 224 vs 384 报告）
bash members/member06_research/scripts/run_final_training_v4_384.sh
```

---

## 7. 最终冻结模型及发布下载方式

- **最终模型**：`V4-R2_RAW_WEIGHTED_CE_SEED42`（`Epoch = 7`, `best_within_20.pt`）
- **文件大小 / SHA-256**：`44,809,257` 字节 / `c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`
- **本地存放相对路径**：`members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`
- **发布方式**：遵循仓库禁止大文件入普通 Git 的规范，冻结权重通过 **GitHub Release 资产**（配合 `stage2_release/SHA256SUMS.txt`）发布；详细发布操作与核验步骤见 [stage2_release/WEIGHT_RELEASE_NOTES_CN.md](stage2_release/WEIGHT_RELEASE_NOTES_CN.md)。

---

## 8. 单图及批量预测

- **Python 接口**：`Stage2ResNetAdapter`（支持路径、`PIL.Image`、原始字节流、`RGB`/`BGR` `ndarray`，可通过 `MEMBER06_CHECKPOINT` 环境变量配置权重路径）：
  ```python
  from members.member06_research.stage2_adapter import Stage2ResNetAdapter
  adapter = Stage2ResNetAdapter()
  res = adapter.predict("path/to/image.jpg")
  ```
- **共享协议批量导出（生成 `images.csv` + `run_config.json`）**：
  ```bash
  members/member06_research/.venv/bin/python -m members.member06_research.stage2_adapter \
    --input-dir path/to/images \
    --counts-file shared/expected_counts.example.json \
    --output /tmp/baseline2_batch_out
  ```

---

## 9. 实验比较及最终 Test 实测结果

### 9.1 `final_training_v4` 验证集（34 张原图）三种子对照汇总（20 轮预算）

| 策略 | 分辨率 | 训练 CE 权重 | Val Acc (`Mean±SD`) | Val Balanced Acc (`Mean±SD`) | FN /14 (`Mean±SD`) | FP /20 (`Mean±SD`) | 裁决结论 |
|---|---|---|---:|---:|---:|---:|---|
| `V4-R1_RAW_CE` | 224 | `[1.00, 1.00]` | `85.29% ± 2.94%` | `86.43% ± 3.57%` | `1.00 ± 1.00` | `4.00 ± 0.00` | 对照基线 |
| **`V4-R2_RAW_WEIGHTED_CE`** | **224** | **`[0.75, 1.50]`** | **`87.25% ± 1.70%`** | **`88.81% ± 2.06%`** | **`0.33 ± 0.58`** | **`4.00 ± 0.00`** | **入选（选定预设 Seed 42, Ep 7）** |
| `V4-R4_RAW_WEIGHTED_CE_384` | 384 | `[0.75, 1.50]` | `83.33% ± 1.70%` | `84.76% ± 1.44%` | `1.00 ± 0.00` | `4.67 ± 0.58` | 4 项预设门槛全未满足，否决 |

> 完整 9 次实验归档见 [stage2_release/experiments_v4/README.md](stage2_release/experiments_v4/README.md)。

### 9.2 最终冻结模型（`V4-R2_SEED42, Ep 7`）验证集 vs 正式 Test 实测结果

| 评估划分 | 样本数 (`PASS/FAIL`) | 覆盖率 | Accuracy | Balanced Acc | FAIL Prec | FAIL Recall | FAIL F1 | ROC-AUC | FN / FP | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Val`（选型参考） | 34 (`20/14`) | 100% | 88.24% | 90.00% | 77.78% | 100.00% | 87.50% | 0.9679 | `0 / 4` | `[[16, 4], [0, 14]]` |
| **`Test`（正式留出）** | **30 (`15/15`)** | **100%** | **70.00%** | **70.00%** | **68.75%** | **73.33%** | **70.97%** | **0.6667** | **`4 / 5`** | **`[[10, 5], [4, 11]]`** |

- **Test 按权威场景分组表现**：正常场景 `10/15`（`66.67%`, `FP=5`）、纯缺件 `0/1`（`0.00%`, `FN=1`）、已见干扰件 `3/3`（`100.00%`）、含未见干扰件混合场景 `8/11`（`72.73%`, `FN=3`）。详细逐图分析见 [STAGE2_FINAL_TEST_REPORT_CN.md](stage2_release/test_evaluation_v4_official/STAGE2_FINAL_TEST_REPORT_CN.md)。

---

## 10. 成员 7 实时系统接入

无需修改 `members/member07_app/server.py` 或 `index.html`，只需将 [member07_models_baseline2.example.json](stage2_release/member07_models_baseline2.example.json) 中的 `"baseline2"` 配置写入 `members/member07_app/models.json` 即可接入；34 张开发样本上的离线/单图/批量/字节流/RGB/BGR 与 `server.py` 接口一致性已实测通过（见 [dev_adapter_consistency.json](stage2_release/dev_adapter_consistency.json)）。具体集成说明见 [MEMBER07_INTEGRATION_GUIDE_CN.md](stage2_release/MEMBER07_INTEGRATION_GUIDE_CN.md)。*(注：尚未声称完成成员 7 实体摄像头联调或现场 20 帧实时延迟测试。)*

---

## 11. 已知局限与评估边界

1. **纯缺件不敏感**：ResNet18 末端全局平均池化（GAP）稀释局部计数变化，对无异物仅少 1 件样本（如 `Test` 中 `IMG_7209.png`，`P(FAIL)=0.1118`）高置信漏放。
2. **跨批次分布敏感**：`Test` 中 5 例误拒（`FP=5`）集中于训练样本占比较少的新批次 `11/23`（`3/5` 误拒）和 `11/20`（`2/2` PASS 误拒）。
3. **不支持动态预期数量与目标定位**：仅支持 `[2,2,2,2,2]`；`Test` 中含未见异物（`S`、`H`）的 11 张图像同时伴随缺件和已见垫圈 `W`，整图判 `FAIL` 不能等同于未见异物实例框级定位检出率。
4. **历史数据接触披露**：本模型在 `final_training_v4` 从头训练与选型过程中未读取 30 张 `Test` 图像；但早期数据集迭代（`v1/v2/v3`）曾包含部分相同原图（如 `11` 张曾处于 `v2` 训练集、`16` 张曾处于 `v3` 训练集）。

---

## 12. 仓库结构与核心文档索引

```text
members/member06_research/
├── README.md                              # 本主入口文档
├── progress.md                            # 成员 6 阶段进度记录
├── ENVIRONMENT.md                         # 真实软硬件环境说明
├── requirements.txt                       # 最小运行依赖
├── requirements-lock.txt                  # 实测 pip freeze 精确锁包
├── config.py / model.py / data.py / image_io.py / runtime.py / train.py / validate.py
├── v4_data.py / v4_train.py / v4_compare.py / v4_resolution_compare.py
├── stage2_freeze.py                       # 预测试冻结脚本
├── stage2_adapter.py                      # 成员 7 现场/批量/Python 统一适配器
├── stage2_test_eval.py                    # 两阶段 GT 隔离 Test 评价与一致性校验
├── configs/final_training_v4/             # 9 组 v4 实验配置 + 增强模板
├── scripts/                               # run_final_training_v4(_384).sh / run_stage2_final_test.sh
├── tests/                                 # 单元测试（含 test_stage2_release.py 等）
└── stage2_release/                        # Stage 2 最终交付包
    ├── README.md                          # 交付包索引
    ├── USER_GUIDE_CN.md                   # 15 步完整操作手册
    ├── STAGE2_FREEZE_REPORT_CN.md         # 冻结报告与 freeze_manifest.json
    ├── WEIGHT_RELEASE_NOTES_CN.md         # GitHub Release 权重发布指南与 SHA256SUMS.txt
    ├── MEMBER07_INTEGRATION_GUIDE_CN.md   # 成员 7 接入指南与示例配置
    ├── MEMBER06_COMPLETE_RESEARCH_SUMMARY_CN.md  # 12 节完整研究总结
    ├── TEAM_REPORT_10PAGE_MEMBER06_BRIEF_CN.md   # 组长 10 页报告与 decision_stage2.md 素材
    ├── GIT_SUBMISSION_CHECKLIST_CN.md     # Git 提交白名单与操作指南
    ├── experiments_v4/                    # 9 次 v4 正式实验归档与对比报告
    ├── test_evaluation_v4_official/       # 30 张正式 Test 原始产物与报告
    └── weights/                           # 本地冻结权重（不入普通 Git，由 Release 发布）
```
