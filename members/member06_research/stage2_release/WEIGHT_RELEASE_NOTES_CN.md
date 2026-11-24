# Baseline 2 Stage 2 最终冻结权重发布说明（GitHub Release 准备）

> **当前状态**：最终权重已在本地冻结并校验无误，**尚未上传至 GitHub Release**（本说明不编造尚未创建的下载 URL）。

---

## 1. 唯一冻结权重与校验身份

| 项目 | 值 |
|---|---|
| 权重文件名 | `baseline2_v4_seed42_best_within_20.pt` |
| 本地相对路径 | `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt` |
| 文件精确大小 | `44,809,257` 字节（约 `42.73 MiB`） |
| **SHA-256** | **`c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`** |
| 对应冻结档案 | `members/member06_research/stage2_release/freeze_manifest.json`（SHA-256: `b72cf9bdaabc4962a102beb2abb346ab83ff55f8e760babd0b82ff638ea10416`） |
| 模型身份 | `V4-R2_RAW_WEIGHTED_CE_SEED42`（`Frozen ResNet18 + Linear(512, 2)`, `224×224 letterbox`, `Epoch = 7`） |
| 对应数据版本 | `members/member03_data/final_training_v4/`（上游主分支提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`） |

---

## 2. 发布权限核查与执行方式

1. **主仓库（`upstream`: `https://github.com/ZhenyuXu0724/huoyan-jinjing.git`）**：由组长（仓库所有者 `ZhenyuXu0724`）维护，成员 6 默认无主仓库直接发 Release 权限；本机亦未安装 `gh` CLI。
2. **Fork 仓库（`origin`: `https://github.com/guoyu1106/member06.git`）**：成员 6 本人拥有完整管理权限，可在 GitHub 网页端直接创建 Release。
3. **推荐发布流程（二选一，或双处同步）**：
   - **方式 A（成员 6 在个人 Fork 发布）**：在 `https://github.com/guoyu1106/member06/releases/new` 创建 Release（Tag 建议：`stage2-baseline2-v4`），上传以下两份文件：
     1. `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`
     2. `members/member06_research/stage2_release/SHA256SUMS.txt`
   - **方式 B（交由组长在主仓库统一发布）**：将上述两份文件（连同 `freeze_manifest.json`）通过课程/团队安全通道交给组长，由组长上传至 `ZhenyuXu0724/huoyan-jinjing` 的 `stage2-baseline2-v4` Release。
   - **严禁行为**：不得将 `.pt` 权重放入普通 Git 提交，也不得在公开文档写入包含个人访问 Token 的链接。

---

## 3. GitHub Release 正文模板（可直接复制粘贴）

```markdown
## Baseline 2（Frozen ResNet18 + Linear）Stage 2 最终冻结权重

- **模型版本**：`baseline2_final_v4_seed42_ep7`（`V4-R2_RAW_WEIGHTED_CE_SEED42`）
- **数据集**：`final_training_v4`（`upstream/main` 提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f`）
- **实测表现**：
  - `Val`（34 张原图，选型集，Ep 7）：`Accuracy = 88.24%`, `Balanced Accuracy = 90.00%`, `FN = 0, FP = 4`
  - `Test`（30 张原图，正式留出集）：`Accuracy = 70.00%`, `Balanced Accuracy = 70.00%`, `FAIL Precision = 68.75%`, `FAIL Recall = 73.33%`, `FAIL F1 = 70.97%`, `ROC-AUC = 0.6667`, `FN = 4, FP = 5`
- **SHA-256 校验**：`c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4` (`44,809,257` bytes)

### 下载后存放路径与校验命令
下载 `baseline2_v4_seed42_best_within_20.pt` 后放置到仓库相对路径：
`members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`

校验命令：
```bash
sha256sum members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
```
（加载该冻结权重进行推理或接入成员 7 系统**无需联网**下载 ImageNet 预训练模型。）
```
