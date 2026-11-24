## 2026-11-24 Stage 2 最终模型冻结、正式 Test 评价与 GitHub 提交准备完成

- **权威数据集切换与完成实验**：采用上游 `upstream/main` 提交 `bde8ea7116761bc8ee7fca1d47c336126daa995f` 之 `members/member03_data/final_training_v4/`（`Train=249, Val=34, Test=30`）；已完成 `V4-R1` vs `V4-R2 Weighted CE [0.75, 1.50]` × `10/20轮` × `Seed 42/43/44` 六组真实训练（`runs/final_training_v4_20261124_192110_PJSHyF`）及 `V4-R4 384×384` 三组最终分辨率对照（`runs/final_training_v4_384_20261124_195746_Rg1Mug`）。
- **最终科学裁决与模型冻结**：`384` 四项预设升级判据全部未满足，正式锁定并冻结 `V4-R2_RAW_WEIGHTED_CE_SEED42`（`224×224`, `Epoch=7`, `best_within_20.pt`, SHA-256 `c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4`, `Val Acc=88.24%`, `Val BA=90.00%`, `FN=0, FP=4`），冻结档案见 `stage2_release/freeze_manifest.json`，终止全部训练优化。
- **正式 Test（30 张原图）实测完成**：由成员 6 本人运行 `scripts/run_stage2_final_test.sh`，产物保存于 `stage2_release/test_evaluation_v4_official/`。实测 `Accuracy=70.00%`（`21/30`）、`Balanced Accuracy=70.00%`、`FAIL Precision=68.75%`、`FAIL Recall=73.33%`、`FAIL F1=70.97%`、`ROC-AUC=0.6667`、`FN=4, FP=5`、混淆矩阵 `[[10, 5], [4, 11]]`、有效覆盖率 `100.00%`。
- **工程打包与提交准备**：归档九次 v4 实验（`stage2_release/experiments_v4/`），补齐 `ENVIRONMENT.md`、`requirements-lock.txt`、15 步操作手册 `USER_GUIDE_CN.md`、GitHub Release 发布说明 `WEIGHT_RELEASE_NOTES_CN.md` 及成员 7 适配交付；完成干净隔离目录导入与推理核验。

## 2026-11-19 新数据接入（工程，不是真实训练实验）

- HEAD 5babd06，包含上游PR6/12c85c2；保留原工作区，仅改成员6，无commit/push/删除。
- 完整310图来源审计：原Train133PASS+52FAIL、Val24PASS+77FAIL、Test23PASS+1FAIL。截图分区缺逐图映射，不能认证82训练FAIL与181/129统计。
- DEV_ONLY入口/模型加载显式开关、分组指标及实时训练进度已实现；FORMAL_READY仍待真实审核/场景/规则确认。
- 本轮公开Test仅做来源审计，无训练、模型选择或测试预测；真实训练由用户执行。
- 详见 TRAINING_GUIDE_CN.md 与 evidence/2026-11-19/REVIEW.md；以下历史记录保留原时点含义。

# 个人时间线

成员 6｜ResNet18 基线与第三阶段研究

日期均为 2026 年；区间末日为当期交付截止。状态：未开始 / 进行中 / 待审核 / 完成。

| 时间 | 工作 | 交付证据 | 状态 / PR |
|---|---|---|---|
| 11.04–11.06 | 跑通冻结 ResNet18+Linear；协助初验几何特征可行性。 | 独立工程和合成数据验证见 evidence/ACCEPTANCE.md；真实尺寸观察待相机/标定 | 进行中；本地 member06/resnet-baseline，未提交 |
| 11.07–11.12 | 在 Pilot 接口联通；与成员 4 统一规则；查阅候选研究依据。 | 模型接口、文献笔记 | 未开始 |
| 11.13–11.18 | 训练线性头并验证选型；协助 baseline3 基础尺寸校验。 | 11-14 已完成真实数据工程审查与接入门控，见 TRAINING_GUIDE_CN.md；真实训练/选型未执行 | 进行中；等待可信双类标签、场景/规格与固定数量批准 |
| 11.19 | 冻结 baseline2 权重、预处理和支持数量。 | baseline2 冻结包 | 未开始 |
| 11.20–11.23 | 解读基线真实错误；区分错规格接受、漏框和光照退化。 | 研究候选排序与证据 ID | 未开始 |
| 11.24–11.26 | 写明唯一主线、假设、主指标、成本与消融计划。 | decision_stage2 内容、baseline2 章节 | 未开始 |
| 11.27–12.03 | 按决策实施改进；在训练/验证集试验并记录失败。 | final 候选、实验日志、论文到代码对应 | 未开始 |
| 12.04–12.05 | 完成验证选型，冻结 final 及所有正式消融设置。 | 最终模型/消融配置、选择依据 | 未开始 |
| 12.06–12.09 | 组织正式独立对照和消融，分析增益、误拒与代价。 | E3–E5 研究结论及限制 | 未开始 |
| 12.10–12.13 | 完成研究章节和最终决策记录，提交个人证据。 | 研究报告、decision_final 内容 | 未开始 |
| 12.14–12.16 | 准备研究问题、消融和负结果问答，验证 final 切换。 | 研究展示与现场模型验收 | 未开始 |

共同责任：首轮每人约 5 个 Pilot 场景；后续按任务卡参与采集、标注与交叉审核，不设人人拍满 35 场景的硬额度。测试集受控封存；不得用其调参。

## 2026-11-05 本轮工程交接

- **完成（工程）**：Frozen ResNet18 + Linear、BN 冻结、配置和清单、Train/Val、checkpoint、独立单图/批量推理、数量拒绝、像素 ROI/CNN/几何工具、自动测试、README 和中文学习指南。
- **SMOKE_ONLY**：真实 ImageNet 预训练权重 + 合成 Train=8 / Val=4、224×224、2 epoch；实际日志和权重哈希见 evidence。随机权重只用于离线结构测试。
- **实测**：成员模块 34 项中 33 通过、CUDA 1 项跳过；团队原有 6 项通过；pip check 和差异空白检查通过。
- **未完成 / 等待外部依赖**：真实 Train/Val、Pilot 数量确认、成员 4 评价器、成员 7 统一导出集成、成员 2 标定、成员 5 冻结 baseline3 与真实错误。没有正式准确率、真实尺寸结论或 Stage 3 方向/消融结论。
- **提交状态**：仅本目录本地改动，无 commit、push 或 PR。组长上游副本保持未修改。原有后续时间节点与科研决策未改写。

## 2026-11-14 真实数据训练工程接入

- 本轮起点 HEAD=90c77fa，原工程已由用户提交为 8d9ed44，并合入已获取的 upstream/main=f0126f5。本轮不再同步远程，不改 Git 历史、跟踪状态，不 commit/push。
- 读取组长 v3.0/公共 v0.1、成员 2 新采集说明、成员 3 完整数据包/划分/映射；未发现新的正式数量或场景批准。
- 实查 Train=112、Val=24，图片、哈希、对象框/计数一致；Test=24 仅查分区元数据，不读测试内容。Train/Val 无可信 scene_id、expected_counts 或整图标签，133 张带原始自动标注待复核标记，平垫片规格待核实。
- 数据结论为 D / NOT_READY。现有数量全部一致，不能只靠计数转换得到可信两类监督。没有生成正式训练清单，没有进行真实训练/Val 选型。
- 新增 team_data、preflight 和数据接入测试；复用原 ResNet18、BN 冻结、训练/推理结构。新增 --check-only，正式训练需获批来源，Val 仍为必需。
- 新增 TRAINING_GUIDE_CN.md，整理 README/旧学习指南中的真实数据入口；保留历史证据、环境和 Stage 3 工具，无文件删除。
- 实测成员模块 47 项：46 通过、CUDA 1 跳过；团队 6 项通过。现有 ImageNet 缓存初始化通过，CPU 环境与依赖保留。证据见 evidence/2026-11-14/。
- 待成员 3/组长：真实批准、逐图人工复核和场景分区、实物规格、可信 FAIL 样本及数据使用范围；待成员 7：统一导出集成。Stage 3 研究决策未推进。


## 2026-11-17 新数据和 HOG 衔接工程

- 已核验主仓库main 5dbb81a，当前本地520d17c。新增21训练/77验证接入，开发数据总Train133/Val101，完整性通过；未改变上游分区。
- 新增development v2多包来源/审核/准备及preflight兼容；Train缺FAIL、真实审核/scene/规格/固定数量仍待交接，正式训练未启动。
- 审查成员4 HOG PASS-only autoencoder；新增共同Val交接和共享CSV批量导出，实际用其compare消费合成输出。
- 新增用户手动实验plan/select/freeze与显式冻结后评测，完善Val逐图身份、误放/误拒指标；分步执行书已重写。
- 旧test已公开，成员4文档包含测试信息，本轮审查文档时已接触；未读取Test图片/标注或执行真实评测，不据此调参。
- 本轮仅工程合成测试与真实开发数据完整性审查；训练、Val选型/调参、正式测试留用户手动执行。验收见evidence/2026-11-17/REVIEW.md。
- 不修改其他成员、不提交推送、不删除环境/缓存/历史证据/Stage3工具；最终研究算法仍待baseline3真实错误。
