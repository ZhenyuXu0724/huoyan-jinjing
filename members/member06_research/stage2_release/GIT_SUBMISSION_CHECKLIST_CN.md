# GitHub 分支协作规范核查与 Stage 2 待提交文件清单

> 本清单依据主仓库 [CONTRIBUTING.md](../../../CONTRIBUTING.md)、[.gitignore](../../../.gitignore) 与自动范围检查脚本 [.github/check_scope.py](../../../.github/check_scope.py) 生成。**本轮未自动执行 `git commit`、`git push` 或创建 PR。**

---

## 1. 当前 Git 分支、远端与 PR 范围核查

- **当前工作分支**：`member06/resnet-baseline`（当前 `HEAD = e809b20`，已包含 `upstream/main` 提交 `bde8ea7`）
- **Fork 远程仓库（`origin`）**：`https://github.com/guoyu1106/member06.git`
- **团队主仓库（`upstream`）**：`https://github.com/ZhenyuXu0724/huoyan-jinjing.git`
- **重要分支提醒**：`member06/resnet-baseline` 在历史提交 `bc80be1` 中曾合入旧分支 `upstream/member03/data-20261114`，因此直接用 `member06/resnet-baseline` 向 `upstream/main` 发 PR 时，`.github/check_scope.py` 会检测到 `members/member03_data/` 路径差异而报错拦截。**建议提交至 `origin/member06/resnet-baseline` 保存个人完整分支后，再基于最新 `upstream/main` 新建干净的 PR 分支（如 `member06/stage2-release`），仅检出 `members/member06_research/` 的白名单文件**（具体命令见第 5 节）。
- **CI 范围红线（`.github/check_scope.py`）**：
  1. `member06/*` 分支向 `upstream/main` 的 PR **仅允许修改 `members/member06_research/` 目录下的文件**；
  2. 禁止提交任何 `.pt`、`.pth`、`.ckpt`、`.onnx`、`.npy`、`.npz`、`.jpg`、`.jpeg`、`.png`、`.zip`、`.pdf` 文件；
  3. 禁止提交路径中包含 `weights`、`checkpoints`、`datasets`、`sealed_test`、`private` 的文件，且单文件大小不得超过 `5 MiB`；
  4. 所有跟踪的 `.py` 文件必须通过 `ast.parse` 语法检查。

---

## 2. 精确待提交文件白名单（全部位于 `members/member06_research/`，不含权重/缓存/图片）

### A. 根入口、环境与文档
- `members/member06_research/README.md`
- `members/member06_research/progress.md`
- `members/member06_research/ENVIRONMENT.md`
- `members/member06_research/requirements.txt`
- `members/member06_research/requirements-lock.txt`
- `members/member06_research/TRAINING_GUIDE_CN.md`、`MEMBER6_LEARNING_GUIDE.md`、`WEIGHTED_CE_TRAINING_CN.md`、`DATASET_V2_TRAINING_CN.md`、`FINAL_TRAINING_V4_CN.md`

### B. 完整可运行源码、配置、脚本与测试
- 核心基线与训练/推理模块：`config.py`、`data.py`、`image_io.py`、`model.py`、`runtime.py`、`train.py`、`validate.py`、`adapter.py`、`inference.py`、`baseline2.py`、`batch_export.py`、`handoff.py`、`final_model.py`、`preflight.py`、`team_data.py`、`development.py`、`corpus_audit.py`、`group_metrics.py`、`experiments.py`、`evaluate_frozen.py`、`diagnose_first_run.py`、`weighted_dev.py`、`compare_dev_runs.py`、`candidate_test.py`、`plot_history.py`、`research_demo.py`、`research_tools.py`、`smoke_data.py`
- v2 / v4 实验引擎与报告生成器：`v2_data.py`、`v2_train.py`、`v2_compare.py`、`v2_phase2_compare.py`、`v2_error_analysis.py`、`v4_data.py`、`v4_train.py`、`v4_compare.py`、`v4_resolution_compare.py`
- Stage 2 冻结、成员 7 适配与正式 Test 评价：`stage2_freeze.py`、`stage2_adapter.py`、`stage2_test_eval.py`
- 脚本：`scripts/run_dataset_v2_phase1.sh`、`scripts/run_dataset_v2_phase2.sh`、`scripts/run_weighted_dev.sh`、`scripts/run_final_training_v4.sh`、`scripts/run_final_training_v4_384.sh`、`scripts/run_stage2_final_test.sh`
- 配置：`configs/` 下全部 `.json`（重点 `configs/final_training_v4/*.json`）
- 测试：`tests/test_*.py`（重点 `test_member06.py`、`test_final_training_v4.py`、`test_final_training_v4_384.py`、`test_stage2_release.py`、`test_weighted_ce.py`）

### C. Stage 2 交付包（`stage2_release/`，不含 `weights/` 与 `*.png`）
- `stage2_release/README.md`、`USER_GUIDE_CN.md`、`STAGE2_FREEZE_REPORT_CN.md`、`WEIGHT_RELEASE_NOTES_CN.md`、`MEMBER07_INTEGRATION_GUIDE_CN.md`、`MEMBER06_COMPLETE_RESEARCH_SUMMARY_CN.md`、`TEAM_REPORT_10PAGE_MEMBER06_BRIEF_CN.md`、`GIT_SUBMISSION_CHECKLIST_CN.md`
- `stage2_release/requirements.txt`、`SHA256SUMS.txt`、`freeze_manifest.json`、`model_config.json`、`inference_contract.json`、`dev_adapter_consistency.json`、`member07_models_baseline2.example.json`
- `stage2_release/experiments_v4/README.md`、`comparison_224_r1_vs_r2/*.md|*.csv|*.json`、`comparison_224_vs_384/*.md|*.csv|*.json` 及 `runs/<9次实验>/*.json|*.csv`（**排除 `*.png`**）
- `stage2_release/test_evaluation_v4_official/` 全部 9 个纯文本产物（`images.csv`、`run_config.json`、`counts_input.json`、`test_metrics.json`、`per_image_predictions.csv`、`grouped_metrics.csv`、`failure_cases.csv`、`failure_cases.json`、`STAGE2_FINAL_TEST_REPORT_CN.md`）

---

## 3. 严禁纳入 Git 提交的文件（不上传清单）

| 本地路径 / 类型 | 排除原因 |
|---|---|
| `members/member06_research/stage2_release/weights/`（`*.pt`） | 42.73 MiB 冻结权重，改由 GitHub Release 发布（见 `WEIGHT_RELEASE_NOTES_CN.md`） |
| `members/member06_research/stage2_release/experiments_v4/**/*.png` | `.png` 受 `.gitignore` 与 `check_scope.py` 限制，仅保留在本地归档 |
| `members/member06_research/runs/` | 全部历史中间 `.pt` 检查点与原始日志（已提取有效 v4 轻量化结果至 `experiments_v4/`） |
| `members/member06_research/data/` | 特征缓存 `features.pt`、本地摄入索引与归档 `.tar.gz` |
| `members/member06_research/.venv/` | 本地虚拟环境与 ImageNet 预训练缓存 |
| `members/member06_research/diagnostics/`、`evidence/`、`audits/`、`archives/`、`final_evaluation_v1/` | 旧数据集阶段历史中间工件继续保留在本地（2026-11-14 早期已跟踪的 `evidence/` 文本保留不动） |
| `members/member03_data/dataset_v2/`、`members/member03_data/analyses/` | 非成员 6 目录权限，含大批图片，禁止提交 |
| 仓库根目录未跟踪杂项文件 `"Net18 baseline and research tools\""` | 历史误建的根目录临时文件，切勿 `git add .` |

---

## 4. 下一步手动提交、推送与 PR 操作模板

```bash
cd /home/grace/projects/AI/member06

# 第 1 步：在当前分支精准暂存白名单文件并提交至个人 Fork (origin/member06/resnet-baseline)
git add \
  members/member06_research/README.md \
  members/member06_research/progress.md \
  members/member06_research/ENVIRONMENT.md \
  members/member06_research/requirements.txt \
  members/member06_research/requirements-lock.txt \
  members/member06_research/*.py \
  members/member06_research/*.md \
  members/member06_research/configs \
  members/member06_research/scripts \
  members/member06_research/tests \
  members/member06_research/stage2_release

# 确认无 weights/*.pt、无 *.png、无 member03_data/ 文件被暂存
git status --short

git commit -m "feat(member06): finalize Stage 2 Baseline 2 freeze, v4 archive, test eval, and docs"
git push origin member06/resnet-baseline

# 第 2 步（推荐向 upstream/main 发 PR 前执行）：从最新 upstream/main 切出干净 PR 分支，避免携带旧 member03/data-20261114 合并历史
git checkout -b member06/stage2-release upstream/main
git checkout member06/resnet-baseline -- members/member06_research
git commit -m "feat(member06): Stage 2 Baseline 2 release on final_training_v4"
git push -u origin member06/stage2-release
# 随后在 GitHub 上从 guoyu1106/member06:member06/stage2-release 向 ZhenyuXu0724/huoyan-jinjing:main 创建 PR
```
