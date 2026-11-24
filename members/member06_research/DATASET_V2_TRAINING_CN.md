# Dataset v2 第一阶段核心训练指南（R1 / R2 / R3）

本指南对应 `Dataset v2`（`438 / 92 / 79` 最终冻结版）的第一阶段成员 6 `Baseline 2`（`Frozen ResNet18 + Linear(512, 2)`）受控训练实验。

- **阶段标识**：`DEV_ONLY`（仅使用 `Dataset v2` 的 `Train` 与 `74` 张真实 `Val` 进行训练和选模；`18` 张 `Val` 离线增强图仅用于训练后鲁棒性诊断；`Test` 79 张完全隔离未接触）。
- **历史保护**：保留全部 `Dataset v1` 及历史 `DEV v1 / Weighted CE v2` 实验与归档不覆盖。

---

## 1. 进入仓库与激活环境

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/member06-v2-mpl
```

> 说明：`run_dataset_v2_phase1.sh` 内部已显式绑定 `members/member06_research/.venv/bin/python` 与 `TORCH_HOME`，不会误调系统 Python 或重新联网下载权重。

---

## 2. 训练前安全与配额预检（不会初始化 CNN、不会训练）

在正式启动前，先执行只读预检：

```bash
bash members/member06_research/scripts/run_dataset_v2_phase1.sh --check-only
```

预检会自动核验三组实验配置与 `Dataset v2` 全部 `Train (438)` + `Val (92)` 文件 SHA256，确认输出：
- `status`: `"DATASET_V2_READY_FOR_USER_TRAINING"`
- `R1_RAW_CE`: `294` 次曝光/轮（`149 PASS, 145 FAIL`），`19` 步/轮，10 轮共 `190` 步，`train_class_weight = "none"`
- `R2_PARENT_EQUAL_AUG_CE`: `294` 次曝光/轮（`149 PASS, 145 FAIL`），`19` 步/轮，10 轮共 `190` 步，`train_class_weight = "none"`（48 个含增强父图按 SHA-256 确定性偏移逐轮循环 4 个版本）
- `R3_FULL_AUG_BALANCED_CE`: `438` 次曝光/轮（`155 PASS, 283 FAIL`），`28` 步/轮，10 轮共 `280` 步，`train_class_weight = "balanced"`，类别权重约 `[1.41290323, 0.77385159]`
- `primary_val_images`: `74`（`37 PASS, 37 FAIL`，无权 CE）
- `post_train_val_aug_diagnostics_images`: `18`（仅训练后鲁棒性诊断）
- `test_content_read`: `false`

---

## 3. 由你本人启动核心训练

### 方式 A：一次顺序执行全部三组实验并自动生成比较报告（推荐）

```bash
bash members/member06_research/scripts/run_dataset_v2_phase1.sh
```

执行流程：
1. 自动新建唯一实验总目录：`members/member06_research/runs/dataset_v2_phase1_<时间戳>_<随机后缀>/`（绝不覆盖历史目录）。
2. 在 `R1` 首轮启动前，自动对已获批的 `438 Train + 92 Val = 530` 张唯一图像构建并校验身份绑定的冻结特征缓存（`members/member06_research/data/dataset_v2_phase1/feature_cache/`）；`R2` 与 `R3` 在严格核验图像哈希、预处理、Backbone 权重与软件版本一致后直接复用该缓存，显著降低 CPU 耗时且保持完整可独立推理的 `best.pt` / `last.pt`。
3. 依次运行 `R1_RAW_CE`、`R2_PARENT_EQUAL_AUG_CE`、`R3_FULL_AUG_BALANCED_CE` 各 10 轮。
4. 三组全部完成后，自动调用 `v2_compare` 生成 `<SUITE_DIR>/comparison/COMPARISON_PHASE1_CN.md` 及配套 CSV 与图表。

### 方式 B：分别单独运行某一组实验

可指定同一个 `--suite-dir` 分三次独立启动（当第 3 组跑完且检测到三组 `best.pt` 齐全时，同样会自动生成 `comparison/` 报告）：

```bash
SUITE="members/member06_research/runs/dataset_v2_phase1_manual_01"

bash members/member06_research/scripts/run_dataset_v2_phase1.sh --only R1 --suite-dir "$SUITE"
bash members/member06_research/scripts/run_dataset_v2_phase1.sh --only R2 --suite-dir "$SUITE"
bash members/member06_research/scripts/run_dataset_v2_phase1.sh --only R3 --suite-dir "$SUITE"
```

---

## 4. 实时观察训练日志

启动后终端会直接实时打印进度，并同步写入各自的 `console.log`。若在另一终端观察日志：

```bash
read -r -p '粘贴终端打印的 Suite directory 路径: ' SUITE
tail -f "$SUITE/R1_RAW_CE/console.log"
# 或观察 R2 / R3：
# tail -f "$SUITE/R2_PARENT_EQUAL_AUG_CE/console.log"
# tail -f "$SUITE/R3_FULL_AUG_BALANCED_CE/console.log"
```

---

## 5. 查看训练产物与自动比较报告

每组实验目录 `<SUITE>/<EXP_ID>/model/` 下均保存：
- `best.pt`（最小无权 74-Val CE 对应的完整 `FrozenResNet18` 权重，平局取最早 Epoch）与 `last.pt`（第 10 轮完整权重）
- `checkpoint_manifest.json`、`config.json`、`run_config.json`、`experiment_spec.json`
- `history.json` / `history.csv`（每轮 Train/Val Loss、Accuracy、Balanced Accuracy、FAIL Precision/Recall/F1、FN、FP、ROC-AUC、优化器步数与耗时）
- `eval_train_raw_294.json`（三组统一在相同 294 张真实训练原图上的固定权重重评估）
- `eval_train_supplementary.json`（438 全集及 R2 最佳轮实际曝光版本的固定权重补充参考）
- `eval_val_raw_74.json` 与 `best_val_predictions.json`（74 张真实 Val 固定权重重评估与逐图预测）
- `eval_val_aug_18_diagnostics.json`（18 张 Val 增强图相对 6 张父原图的分数变化与分类翻转诊断）
- `r2_exposure_log.json` / `r2_exposure_summary.csv`（仅 R2：逐轮逐父图实际选中版本与总曝光计数）

查看自动生成的比较报告：

```bash
cat "$SUITE/comparison/COMPARISON_PHASE1_CN.md"
ls -lh "$SUITE/comparison/"
```

如需手动指定新目录重新生成比较报告（只读，不重训、不覆盖已有报告目录）：

```bash
python -m members.member06_research.v2_compare \
  --r1 "$SUITE/R1_RAW_CE/model" \
  --r2 "$SUITE/R2_PARENT_EQUAL_AUG_CE/model" \
  --r3 "$SUITE/R3_FULL_AUG_BALANCED_CE/model" \
  --output "$SUITE/comparison_manual_01"
```

---

## 6. 第二阶段：训练预算（10轮 vs 20轮）与随机种子稳定性复核（R4: Seed 42 / 43 / 44）

第二阶段严格继承 `R1_RAW_CE` 配置（294 张真实训练原图、74 张真实验证原图、224 letterbox、无权 CE、Adam `lr=0.001`、默认 `argmax`），直接复用第一阶段已验收的 `224` 冻结特征缓存，一次运行 20 轮同时保存 `best_within_10.pt`、`best_within_20.pt`、`epoch10.pt`、`epoch20.pt`，并对 `seed=42` 前 10 轮与第一阶段 `R1_RAW_CE` 进行逐轮损失、数据顺序与模型张量复现核验。

### 6.1 第二阶段训练前只读预检（核实缓存复用与历史参考路径，不训练）

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_dataset_v2_phase2.sh --check-only
```

确认三组均显示 `"status": "DATASET_V2_READY_FOR_USER_TRAINING"` 且 `"feature_cache_preflight": {"cache_present": true, "reusable_verified": true, "cached_images": 530}`。

### 6.2 由你本人一键启动三组实验（Seed 42 / 43 / 44）并自动生成配对比较报告

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_dataset_v2_phase2.sh
```

如需分别单独运行单组实验（可选）：

```bash
SUITE_P2="members/member06_research/runs/dataset_v2_phase2_manual_01"
bash members/member06_research/scripts/run_dataset_v2_phase2.sh --only SEED42 --suite-dir "$SUITE_P2"
bash members/member06_research/scripts/run_dataset_v2_phase2.sh --only SEED43 --suite-dir "$SUITE_P2"
bash members/member06_research/scripts/run_dataset_v2_phase2.sh --only SEED44 --suite-dir "$SUITE_P2"
```

### 6.3 输出模型与比较报告路径

运行开始时终端打印 `Phase-2 Suite directory: <SUITE_P2>`（默认位于 `members/member06_research/runs/dataset_v2_phase2_<时间戳>_<6位后缀>/`）：
- 每组模型目录：`<SUITE_P2>/R4_RAW_CE_SEED{42,43,44}/model/`
  - 双预算与节点权重：`best_within_10.pt`、`best_within_20.pt`、`epoch10.pt`、`epoch20.pt`（另含兼容入口 `best.pt`、`last.pt`）
  - 双预算固定权重重评估：`budget_evaluations.json`、`best_within_10_val_predictions.json`、`best_within_20_val_predictions.json`
  - `seed=42` 前 10 轮历史复现核验：`reproduction_check_vs_r1.json`
- 自动配对比较与错误分析报告目录：`<SUITE_P2>/comparison/`
  - 主报告：`<SUITE_P2>/comparison/COMPARISON_PHASE2_CN.md`
  - 汇总与表格：`phase2_summary.json`、`phase2_per_seed_metrics.csv`、`phase2_val_74_prediction_changes.csv`、`r1_val_error_analysis.json`、`curves_phase2_budgets.png`

