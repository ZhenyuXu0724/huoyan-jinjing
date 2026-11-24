# Baseline 2：`final_training_v4` 最终重新训练运行指南

本指南对应小组统一采用的最终版数据集 `members/member03_data/final_training_v4/`（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`）上的成员 6 `Baseline 2`（`Frozen ResNet18 + Linear(512, 2)`）6 组核心对照实验。

---

## 1. 权威数据源与实验矩阵

- **权威清单**：
  - 训练：`members/member03_data/final_training_v4/manifests/resnet/train.csv`（`249` 张真实原图：`166 PASS`、`83 FAIL`，每轮 `249` 次曝光、`16` 次优化器更新，20 轮共 `320` 次更新）
  - 验证：`members/member03_data/final_training_v4/manifests/resnet/val.csv`（`34` 张真实原图：`20 PASS`、`14 FAIL`，始终使用无权 CE 选模）
  - 测试：`manifests/resnet/test.csv`（`30` 张原图：`15 PASS`、`15 FAIL`）**严格封存隔离，训练与选模阶段零访问**
- **核心实验（2 策略 × 3 随机种子 = 6 次运行，每次 20 轮，同步保存 10 轮与 20 轮双预算候选）**：
  1. `V4-R1_RAW_CE`（`seed = 42, 43, 44`）：普通无权 `CrossEntropyLoss`
  2. `V4-R2_RAW_WEIGHTED_CE`（`seed = 42, 43, 44`）：平衡类别权重 `CrossEntropyLoss`（`PASS = 0.75`，`FAIL = 1.50`；同一 `seed` 下与 `V4-R1` 严格共享相同初始线性头权重与相同 20 轮样本打乱顺序）

---

## 2. 训练前只读检查（核实上游三元组、6 组配置与权重，不训练）

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_final_training_v4.sh --check-only
```

确认输出：
- `Verified the v4 split and 283 physical image/LabelMe/YOLO triples.`
- 6 组实验全部显示 `"status": "FINAL_TRAINING_V4_READY_FOR_USER_START"`、`exposures_per_epoch: 249`、`optimizer_steps_per_epoch: 16`、`total_optimizer_steps: 320`、`V4-R2` 权重 `[0.75, 1.5]`、`test_content_read: false`。

---

## 3. 由你本人一键启动 6 组真实训练并自动生成统一比较报告

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_final_training_v4.sh
```

执行流程：
1. 自动创建独立运行总目录 `members/member06_research/runs/final_training_v4_<时间戳>_<6位后缀>/`（不覆盖任何历史实验）。
2. 首组运行时自动建立绑定 `final_training_v4` 数据与清单哈希的专属 `Train+Val (283 图)` 冻结特征缓存（`members/member06_research/data/final_training_v4_intake/feature_cache/`，在核验预处理/Backbone/文件 SHA-256 完全一致并现场抽检与实时前向误差 `≤1e-6` 后可加速复用合法单图冻结特征，绝不混入任何 `Test` 或历史训练分类头）；其余 5 组直接秒级加载该专属 `v4` 缓存。
3. 依次完成 `V4-R1_RAW_CE_SEED{42,43,44}` 与 `V4-R2_RAW_WEIGHTED_CE_SEED{42,43,44}` 共 6 组训练，每组均保存 `best_within_10.pt`、`best_within_20.pt`、`epoch10.pt`、`epoch20.pt`。
4. 6 组全部完成后，自动调用 `v4_compare` 生成统一比较报告。

### 可选：单组分别运行

```bash
SUITE_V4="members/member06_research/runs/final_training_v4_manual_01"
bash members/member06_research/scripts/run_final_training_v4.sh --only R1_S42 --suite-dir "$SUITE_V4"
bash members/member06_research/scripts/run_final_training_v4.sh --only R1_S43 --suite-dir "$SUITE_V4"
bash members/member06_research/scripts/run_final_training_v4.sh --only R1_S44 --suite-dir "$SUITE_V4"
bash members/member06_research/scripts/run_final_training_v4.sh --only R2_S42 --suite-dir "$SUITE_V4"
bash members/member06_research/scripts/run_final_training_v4.sh --only R2_S43 --suite-dir "$SUITE_V4"
bash members/member06_research/scripts/run_final_training_v4.sh --only R2_S44 --suite-dir "$SUITE_V4"
```

---

## 4. 实时日志与最终比较报告位置

- 实时观察某组日志：
  ```bash
  read -r -p '粘贴终端打印的 final_training_v4 Suite directory 路径: ' SUITE_V4
  tail -f "$SUITE_V4/V4-R1_RAW_CE_SEED42/console.log"
  ```
- 模型输出位置：
  - `<SUITE_V4>/V4-R1_RAW_CE_SEED{42,43,44}/model/{best_within_10.pt,best_within_20.pt,epoch10.pt,epoch20.pt}`
  - `<SUITE_V4>/V4-R2_RAW_WEIGHTED_CE_SEED{42,43,44}/model/{best_within_10.pt,best_within_20.pt,epoch10.pt,epoch20.pt}`
- 统一比较报告输出位置：
  - 主报告：`<SUITE_V4>/comparison/COMPARISON_V4_CN.md`
  - 结构化汇总与逐图/分组表：`<SUITE_V4>/comparison/v4_summary.json`、`v4_per_run_budget_metrics.csv`、`v4_val_34_per_image_comparison.csv`、`v4_val_34_grouped_slices.csv`、`curves_v4_6runs.png`

---

## 5. 最后一次训练优化：`V4-R4_RAW_WEIGHTED_CE_384`（224 vs 384 受控分辨率对照）

本阶段严格固定 `V4-R2_RAW_WEIGHTED_CE`（249 原图训练、34 原图无权 Val、`PASS=0.75 / FAIL=1.50`、20 轮预算、代表种子 `seed=42`），唯一改变输入分辨率 `image_size: 224 → 384`。`384` 使用完全独立的 `v4_resnet18_sz384_*` 专属特征缓存（三组种子共享同一 `384` 缓存，零混用 `224` 特征）。

### 5.1 启动前只读预检（不训练）

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_final_training_v4_384.sh --check-only
```

### 5.2 由你本人一键启动三组 384 实验（Seed 42 / 43 / 44）并自动生成最终分辨率对照报告

```bash
cd /home/grace/projects/AI/member06
bash members/member06_research/scripts/run_final_training_v4_384.sh
```

> 如需单组分别执行（可选）：
> ```bash
> SUITE_384="members/member06_research/runs/final_training_v4_384_manual_01"
> bash members/member06_research/scripts/run_final_training_v4_384.sh --only S42 --suite-dir "$SUITE_384"
> bash members/member06_research/scripts/run_final_training_v4_384.sh --only S43 --suite-dir "$SUITE_384"
> bash members/member06_research/scripts/run_final_training_v4_384.sh --only S44 --suite-dir "$SUITE_384"
> ```

### 5.3 模型与最终对照报告输出路径

终端打印 `final_training_v4 384 Suite directory: <SUITE_384>`（默认位于 `members/member06_research/runs/final_training_v4_384_<时间戳>_<6位后缀>`）：
- 三组 384 模型目录：`<SUITE_384>/V4-R4_RAW_WEIGHTED_CE_384_SEED{42,43,44}/model/`
- 最终 224/384 对照报告：`<SUITE_384>/comparison/COMPARISON_V4_FINAL_RESOLUTION_CN.md`
- 配套逐图变化、单图离线推理计时与曲线图：`<SUITE_384>/comparison/v4_224_vs_384_summary.json`、`v4_224_vs_384_per_seed_metrics.csv`、`v4_224_vs_384_val_34_changes.csv`、`curves_v4_224_vs_384.png`

