# 成员 6（Baseline 2 · Frozen ResNet18 + Linear）完整操作手册

本手册面向首次使用本项目的教师、组长及组员，覆盖**场景 A（直接使用最终冻结模型进行预测/集成/复现评价）**与**场景 B（从 `final_training_v4` 数据集从头重跑全部 9 次训练与对比实验）**。所有命令均已在仓库实际脚本与 CLI 中核验存在。

> 默认工作目录：请先 `cd` 进入仓库根目录（例如 `cd /path/to/huoyan-jinjing`），以下全部命令均从仓库根目录运行。

---

## 1. 获取代码与正确的数据版本

- **代码仓库**：成员 6 分支 `member06/resnet-baseline`（或已合入该目录的 `main` 分支）。
- **数据集版本（仅重跑训练或复现 Test 评价时需要）**：小组统一最终版数据集位于 `members/member03_data/final_training_v4/`，对应上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`。
  - 核验数据集完整性（默认核验 Train 249 + Val 34 = 283 组三元组；加 `--all` 核验全部 415 组含 Test 30 张与可选增强 102 张）：
  ```bash
  python members/member03_data/final_training_v4/verify_dataset.py --all
  ```
  - 若仅使用**最终冻结模型做单图/批量/摄像头推理**，无需下载整个数据集。

---

## 2. 创建及激活虚拟环境

```bash
python3.11 -m venv members/member06_research/.venv
source members/member06_research/.venv/bin/activate
```
*(以下若未激活虚拟环境，也可直接用 `members/member06_research/.venv/bin/python` 调用。)*

---

## 3. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
python -m pip install -r members/member06_research/requirements.txt
python -m pip check
```
> 更多环境细节与精确锁包清单见 [../ENVIRONMENT.md](../ENVIRONMENT.md) 与 [../requirements-lock.txt](../requirements-lock.txt)。

---

## 4. 获取（下载）最终冻结模型

由于仓库禁止将 `.pt` 权重提交进普通 Git，请从团队发布的 **GitHub Release 资产**（或团队授权存储）下载最终唯一冻结权重：
- **文件名**：`baseline2_v4_seed42_best_within_20.pt`（大小 `44,809,257` 字节）
- **标准存放路径**：
  `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`

---

## 5. 校验模型 SHA-256

放置权重后，务必校验哈希确保文件完整无损：

```bash
sha256sum members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
```
期望输出：
```text
c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4  members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
```

---

## 6. 配置模型路径与数据路径

- **默认免配置**：若权重放在第 4 步的标准相对路径，`stage2_adapter.py` 与测试脚本会自动定位并核验其 SHA-256，不依赖任何个人电脑绝对路径。
- **自定义权重路径**：
  - 方法 1：CLI 传 `--checkpoint /your/path/baseline2_v4_seed42_best_within_20.pt`；
  - 方法 2：设置环境变量 `export MEMBER06_CHECKPOINT=/your/path/baseline2_v4_seed42_best_within_20.pt`（可选 `export MEMBER06_DEVICE=cpu`）；
  - 方法 3：在成员 7 的 `members/member07_app/models.json` 中配置相对路径 `"weights"`。
- **重跑训练的数据路径准备（仅场景 B 需要）**：
  若在新克隆的仓库中重跑 `final_training_v4` 训练，先生成本机对应的 `provenance.json`（不修改原始数据集）：
  ```bash
  members/member06_research/.venv/bin/python -m members.member06_research.v4_data prepare \
    --v4-root members/member03_data/final_training_v4 \
    --output-dir members/member06_research/data/final_training_v4_intake
  ```

---

## 7. 单图 PASS/FAIL 预测（场景 A）

```bash
members/member06_research/.venv/bin/python -m members.member06_research.stage2_adapter \
  --image path/to/your_image.jpg \
  --output /tmp/baseline2_single_result.json
```
- 可选指定预期数量：`--expected-counts '[2,2,2,2,2]'`。若传入非 `[2,2,2,2,2]`（如 `'[2,2,2,3,2]'`），将显式返回 `status: "UNSUPPORTED_COUNTS"` 且 `predicted_label: null`。
- 输出 JSON 包含 `predicted_label`（`PASS` / `FAIL`）、`fail_probability`（未校准 `P(FAIL) ∈ [0,1]`）与 `pass_probability`。

---

## 8. 批量图片预测（场景 A）

```bash
members/member06_research/.venv/bin/python -m members.member06_research.stage2_adapter \
  --input-dir path/to/image_folder \
  --counts-file shared/expected_counts.example.json \
  --output /tmp/baseline2_batch_output
```
- 注意：需准备与 `input-dir` 中文件名一一对应的 `counts_file`（JSON 列表，每项仅含 `image_id` 与 `expected_counts`）。
- `--output` 目录必须**事先不存在**（防止覆盖历史证据），运行后生成符合 `shared/CONTRACT.md` 的 `images.csv` 与 `run_config.json`。

---

## 9. 重新运行 `final_training_v4` 六组核心训练实验（场景 B）

对应 `V4-R1_RAW_CE` 与 `V4-R2_RAW_WEIGHTED_CE` × `Seed 42/43/44` × `20 Epoch`（**首次训练需能访问 PyTorch 官方源下载 ResNet18 ImageNet 预训练权重或已本地缓存**）：

```bash
# 只读预检（不启动训练、不修改文件）
bash members/member06_research/scripts/run_final_training_v4.sh --check-only

# 启动 6 组完整训练（自动生成带时间戳的新运行目录，绝不覆盖历史结果）
bash members/member06_research/scripts/run_final_training_v4.sh
```

---

## 10. 重新运行 224 / 384 分辨率对照实验（场景 B）

对应 `V4-R4_RAW_WEIGHTED_CE_384` × `Seed 42/43/44` × `20 Epoch`：

```bash
# 只读预检
bash members/member06_research/scripts/run_final_training_v4_384.sh --check-only

# 启动 3 组 384 训练（可选 --suite-224 指定第 9 步新生成的 224 套件目录）
bash members/member06_research/scripts/run_final_training_v4_384.sh
```

---

## 11. 生成实验比较报告（场景 B）

上述两个一键脚本在全部种子跑完后**会自动**在套件目录下生成 `comparison/` 报告；如需手动对指定运行目录重新生成报告（输出到新目录）：

```bash
# 生成 6 组 224 核心实验比较报告
members/member06_research/.venv/bin/python -m members.member06_research.v4_compare \
  --suite-dir <224_SUITE_DIR> \
  --output <NEW_224_COMPARE_DIR>

# 生成 224 vs 384 最终分辨率对照报告
members/member06_research/.venv/bin/python -m members.member06_research.v4_resolution_compare \
  --suite-224 <224_SUITE_DIR> \
  --suite-384 <384_SUITE_DIR> \
  --output <NEW_384_COMPARE_DIR>
```

---

## 12. 复现最终冻结模型的 Test 评价（场景 A）

> **重要纪律**：已归档的官方正式 Test 证据位于 [test_evaluation_v4_official/](test_evaluation_v4_official/)，**严禁覆盖或重跑刷分**。如需在独立新目录复现核验，请显式指定新输出目录：

```bash
# 1. 只读预检（核验冻结档案哈希与 30 张 Test 原图三元组）
bash members/member06_research/scripts/run_stage2_final_test.sh --check-only

# 2. 复现推理与评价到新的临时目录（不覆盖正式目录）
members/member06_research/.venv/bin/python -u -m members.member06_research.stage2_test_eval test \
  --output-dir /tmp/reproduce_stage2_test_v4
```

---

## 13. 查看预测输出和结果报告

- **官方正式 Test 结果**：
  - 总结报告：[test_evaluation_v4_official/STAGE2_FINAL_TEST_REPORT_CN.md](test_evaluation_v4_official/STAGE2_FINAL_TEST_REPORT_CN.md)
  - 指标 JSON：[test_evaluation_v4_official/test_metrics.json](test_evaluation_v4_official/test_metrics.json)
  - 逐图预测与分组统计：[per_image_predictions.csv](test_evaluation_v4_official/per_image_predictions.csv)、[grouped_metrics.csv](test_evaluation_v4_official/grouped_metrics.csv)、[failure_cases.csv](test_evaluation_v4_official/failure_cases.csv)
- **9 次训练对比报告**：
  - 224 R1 vs R2：[experiments_v4/comparison_224_r1_vs_r2/COMPARISON_V4_CN.md](experiments_v4/comparison_224_r1_vs_r2/COMPARISON_V4_CN.md)
  - 224 vs 384：[experiments_v4/comparison_224_vs_384/COMPARISON_V4_FINAL_RESOLUTION_CN.md](experiments_v4/comparison_224_vs_384/COMPARISON_V4_FINAL_RESOLUTION_CN.md)

---

## 14. 接入成员 7 摄像头系统

1. 将 [member07_models_baseline2.example.json](member07_models_baseline2.example.json) 中的 `"baseline2"` 配置复制到 `members/member07_app/models.json`；
2. 保证冻结权重已放在第 4 步路径（或配置相对路径 `"weights"`）；
3. 启动成员 7 服务 `python members/member07_app/server.py`（无需修改 `server.py` 或 `index.html`），在网页下拉选择 `ResNet · baseline2` 即可拍照检测。详情见 [MEMBER07_INTEGRATION_GUIDE_CN.md](MEMBER07_INTEGRATION_GUIDE_CN.md)。

---

## 15. 常见错误及解决方法

| 错误现象 | 原因 | 解决方法 |
|---|---|---|
| `Missing virtual environment interpreter` | 未按第 2 步创建 `.venv` 或脚本路径不对 | 在仓库根目录创建 `members/member06_research/.venv` 并安装依赖 |
| `Frozen Baseline 2 checkpoint not found` | 尚未下载/放置 `.pt` 权重文件 | 按第 4–5 步下载权重并校验 SHA-256，或设置 `MEMBER06_CHECKPOINT` |
| `Checkpoint SHA256 mismatch` | 权重文件损坏或非正式冻结版 | 重新下载官方 `baseline2_v4_seed42_best_within_20.pt` 并核对 `c38c270a...` |
| `output directory already exists` | 保护机制拒绝覆盖已有实验/测试目录 | 换用不存在的新输出路径（如 `/tmp/...`） |
| `UNSUPPORTED_COUNTS` | 传入了非 `[2, 2, 2, 2, 2]` 的预期数量 | Baseline 2 仅支持五类各 2 件；动态数量核验需使用对象级模型（Baseline 3） |
| `PRETRAINED_WEIGHTS_UNAVAILABLE` | 重跑训练时无法联网下载 ImageNet 权重 | 仅推理/复现 Test 无需联网；重跑训练需提前联网或拷贝 `resnet18-f37072fd.pth` 到 `$TORCH_HOME/hub/checkpoints/` |
| `ModuleNotFoundError: No module named 'members'` | 未在仓库根目录执行 `python -m ...` | 先 `cd` 到仓库根目录再运行命令 |
