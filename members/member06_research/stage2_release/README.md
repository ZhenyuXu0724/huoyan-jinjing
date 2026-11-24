# Member 6 · Baseline 2（Frozen ResNet18 + Linear）Stage 2 最终交付包

本目录为成员 6 在小组统一数据集 `final_training_v4`（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`）上完成的 **Stage 2 最终冻结模型、独立 Test 评估入口、成员 7 实时/批量系统接口交付与研究总结**。

---

## 1. 最终冻结模型身份摘要

- **实验 ID / 模型版本**：`V4-R2_RAW_WEIGHTED_CE_SEED42` / `baseline2_final_v4_seed42_ep7`
- **网络架构**：`FrozenResNet18 (IMAGENET1K_V1, Backbone+BN 完全冻结) + Linear(512, 2)`
- **输入与预处理**：`RGB`（无 EXIF 旋转，BGR 显式转 RGB）→ `224×224 letterbox`（`pad_rgb=[0,0,0]`）→ ImageNet 归一化
- **类别与决策规则**：`PASS=0, FAIL=1`；固定支持 `expected_counts=[2, 2, 2, 2, 2]`；默认 `argmax`（`P(FAIL) > 0.5` 判 `FAIL`，`0.5` 平局归 `PASS`）
- **训练策略与最佳轮次**：`249` 张真实原图，`Weighted CE [PASS=0.75, FAIL=1.50]`，无权 `Val CE` 选模，最大预算 `20` 轮，**最佳 `Epoch = 7`**（`Val Acc = 88.24%`, `Balanced Acc = 90.00%`, `FN = 0`, `FP = 4`, `ROC-AUC = 0.9679`）
- **冻结权重路径（本地 / Release 资产放置位置）**：
  `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`
- **权重 SHA-256**：
  `c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`

> **Git 与权重分发说明**：遵循仓库 `.gitignore` 与 `.github/check_scope.py`（禁止将 `.pt` 与 `weights/` 大文件提交入普通 Git），`weights/` 子目录保存在本地并通过 **GitHub Release 资产 / 团队授权网盘**分发；教师或组员下载后放入上述相对路径（或设置环境变量 `MEMBER06_CHECKPOINT`），无需联网下载 ImageNet 预训练模型即可直接复现。

---

## 2. 目录文件索引

| 文件 | 说明 |
|---|---|
| [freeze_manifest.json](freeze_manifest.json) | 正式 Test 前生成的不可变模型冻结档案（含模型/张量/数据清单 SHA-256、环境、选型依据） |
| [STAGE2_FREEZE_REPORT_CN.md](STAGE2_FREEZE_REPORT_CN.md) | Stage 2 最终模型冻结中文报告 |
| [WEIGHT_RELEASE_NOTES_CN.md](WEIGHT_RELEASE_NOTES_CN.md) | GitHub Release 权重发布指南（含文件大小、SHA-256 与组长发布说明） |
| [USER_GUIDE_CN.md](USER_GUIDE_CN.md) | 15 步完整操作手册（直接使用冻结模型 / 重跑 9 次 v4 训练与评价） |
| [../ENVIRONMENT.md](../ENVIRONMENT.md) | 真实硬件、OS、Python 与锁包环境说明 |
| [experiments_v4/README.md](experiments_v4/README.md) | `final_training_v4` 全部 9 次真实实验归档与 224/384 对比报告 |
| [test_evaluation_v4_official/STAGE2_FINAL_TEST_REPORT_CN.md](test_evaluation_v4_official/STAGE2_FINAL_TEST_REPORT_CN.md) | 30 张正式 Test 原图独立评价报告（`Accuracy = 70.00%`, `FN=4, FP=5`） |
| [model_config.json](model_config.json) | 冻结检查点完整 `Config` 参数 |
| [inference_contract.json](inference_contract.json) | 跨环境推理、预处理、分数定义与能力边界契约 |
| [SHA256SUMS.txt](SHA256SUMS.txt) | 冻结权重与核心配置文件 SHA-256 清单 |
| [requirements.txt](requirements.txt) | Python 3.11 锁定运行依赖 |
| [member07_models_baseline2.example.json](member07_models_baseline2.example.json) | 供成员 7 `members/member07_app/models.json` 直接接入的配置片段 |
| [dev_adapter_consistency.json](dev_adapter_consistency.json) | 34 张真实 `Val` 上离线推理、单图/批量、字节流、RGB/BGR 与 `server.py` 一致性核验记录 |
| [MEMBER07_INTEGRATION_GUIDE_CN.md](MEMBER07_INTEGRATION_GUIDE_CN.md) | 成员 7 现场摄像头系统与 CLI 批量导出接入指南 + 教师异地加载说明 |
| [MEMBER06_COMPLETE_RESEARCH_SUMMARY_CN.md](MEMBER06_COMPLETE_RESEARCH_SUMMARY_CN.md) | 成员 6 完整 12 节研究过程总结 |
| [TEAM_REPORT_10PAGE_MEMBER06_BRIEF_CN.md](TEAM_REPORT_10PAGE_MEMBER06_BRIEF_CN.md) | 供组长纳入 10 页小组报告与 `decision_stage2.md` 的精简素材 |
| [GIT_SUBMISSION_CHECKLIST_CN.md](GIT_SUBMISSION_CHECKLIST_CN.md) | Git 分支规范检查与待提交文件清单 |

---

## 3. 快速复现与正式 Test 一键命令（在仓库根目录执行）

```bash
# 1. 正式 Test 预检（只读校验冻结权重哈希与 30 张 Test 原图完整性，不执行推理）
bash members/member06_research/scripts/run_stage2_final_test.sh --check-only

# 2. 正式单次 Test 独立推理与分组评价（由成员 6 本人启动）
bash members/member06_research/scripts/run_stage2_final_test.sh
# 默认输出目录：members/member06_research/stage2_release/test_evaluation_v4_official/

# 3. 复核开发集（34 Val）离线推理与成员 7 适配器一致性
members/member06_research/.venv/bin/python -m members.member06_research.stage2_test_eval \
  verify-adapter-dev --output-json /tmp/check_consistency.json
```
