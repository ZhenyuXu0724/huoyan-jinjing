# 成员 4 · 进度与交接

## 当前状态：Baseline1_HOGMLP_v4_FinalDataset_Rebuild（2026-11-24）

在固定数据版本 `final_training_v4` 上完整重建 HOG+MLP 基线。旧版先归档到仓库同级 `Project/delivery/member04_history_before_v4_20261124/`，逐文件 SHA256 核对后清理活动目录；新流程 75 次正式训练、冻结、诊断、一次性 test 评价和本地交付均已完成。**本轮尚未取得 GitHub 上传许可，未 push、未建 PR、未发布 Release。**

| 阶段 | 当前状态 | 放行条件 |
|---|---|---|
| 规划/元数据审计 | 已完成 | 本文件＋dataset_audit.json＋START_HERE.md |
| 归档/清理 | 已完成 | 归档逐文件摘要全部通过、只清理成员4范围 |
| 新代码/开发数据审计 | 已完成 | 数据/接口/隔离测试通过、协议锁定 |
| 66次主实验 | 已完成 | 全部真实完成、重复组一致 |
| 9次有限改进 | 已完成 | R0末轮复现、固定规则选型 |
| 冻结/开发诊断 | 已完成 | 3代表＋新默认冻结、无test选型 |
| 最终评价/迁移验证 | 已完成 | test按门槛访问、指标可追溯、真实接口和包可用 |
| 本地交付/等待上传 | 进行中 | 提供完整成果清单，等待用户本次上传许可 |

- 数据：训练 249（166 PASS、83 FAIL）、验证 34（20/14）、test 30（15/15）；开发范围 862 文件、全量 1260 文件，`verify_dataset.py` 与自有 54+ 项审计均通过。
- 66 次主实验：AE 入选 A1（平均验证漏放 2.33、误拒 4.00）；监督入选 A2（0.33、2.67）。两对等价组 B2/C_without_geometry、A2/C_without_clahe 逐位一致。
- 9 次改进：R0 三个种子第 200 轮精确复现监督 A2 对应种子（权重、标准化器、预测、最大概率差 0）；按三种子最佳平均漏放/误拒选中 R0。
- 冻结：默认 `Baseline1_HOGMLP_v4_FinalDataset_Supervised_Capacity_L2_EpochSelection`（R0 seed42、第 4 轮、阈值 0.9850894510746002）；代表 SHA256 记录在 `freeze.json`。
- val：默认 R0 漏放 0、误拒 4；监督 A2 漏放 1、误拒 1；AE A1 漏放 2、误拒 4。
- 冻结后 test（只评价 3 个代表）：AE 漏放 4、误拒 7、FAIL F1 0.6667；监督 A2 漏放 10、误拒 3、F1 0.4348；默认 R0 漏放 6、误拒 4、F1 0.6429。11 张 S/H 未见件图中，三种方法分别漏放 2/7/4，仅图像级，非陌生件定位或分类。
- 稳定性：21 设置、283 图、3 模型共 17,829 条预测；原图翻转比例高，AE 在缩小居中设置下会把全部 train PASS 判 FAIL，不能把零翻转宣称为鲁棒。
- 成本：单图中位约 93–95ms，P95 约 216–229ms；监督模型文件约为 AE 的一半。

完整数字与限制见 [结果报告](docs/RESULTS.md)、[方法对照](docs/COMPARISON.md)、[分区结果](docs/SPLIT_RESULTS.md)、[错误分析](docs/ERROR_ANALYSIS.md)、[稳定性与成本](docs/STABILITY_COST.md)。

## 已验证产物

| 内容 | 位置 |
|---|---|
| 来源锁与 75 次任务单 | `configs/final_v4/{source_lock,protocol}.json` |
| 开发/全量审计、test 访问 | `outputs/member04_hog/final_v4/audit/` |
| 66 次主实验记录 | `outputs/member04_hog/final_v4/experiments/`、`weights/member04_hog/final_v4/{ae,supervised}/` |
| 9 次改进记录 | `outputs/member04_hog/final_v4/refinement/`、`weights/member04_hog/final_v4/refinement/` |
| 冻结与默认指针 | `outputs/member04_hog/final_v4/freeze.json`、`configs/active_model.json` |
| 诊断/评价/报告 | `outputs/member04_hog/final_v4/{diagnosis,evaluation,reports}/` |
| 验证证据 | `outputs/member04_hog/final_v4/verification/` |
| 历史归档与清单 | 仓库同级 `Project/delivery/member04_history_before_v4_20261124/` |

归档共 1489 个文件、2,926,577,008 字节，复制后逐文件复核通过；清理 1419 个旧文件，删除后归档复查通过。`v3_relocation_3cyz5ths` 因 ACL 拒绝读取未移动，作为未处理项保留在原位。

## 验证状态（2026-11-24）

- `final_v4.verify` 通过：84 份正式检查点（66 末轮 + 9×2）逐份摘要与重载校验通过；3 个代表在 val 34 图上单图判定与保存记录一致，最大得分差 3.58e-7（浮点路径差，不是标签差）。
- 模块测试 21 项、公共测试 6 项通过；依赖版本与 `requirements.txt` 一致；改动范围仅 `members/member04_hog/`。
- `.github/check_scope.py` 以 exit 1 结束，原因为 main 已跟踪成员3媒体；本地记录“成员范围通过、仓库框架未通过”，未修改他人数据或公共规则。
- 单图 OK/INPUT_ERROR/MODEL_UNAVAILABLE、批量 OK/INPUT_ERROR/UNSUPPORTED_COUNTS、缺权重启动失败且不建输出目录、旧 bundle 拒载均通过。
- 交付包 `Project/delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip`：928,957,751 字节、592 个条目，SHA256 `4449706dba29d50deb5ae745a84dc71224e63cdd6fd0479f81110955411223a3`；ZIP 内逐文件校验与解压迁移（模块测试、单图、批量、缺权重）全部通过，证据在同名 `_verification.json`、`_tests.txt`。ZIP 为打包时快照；打包后的 progress 变更不再重打包，最新交接状态以 Git 为准。
- 三个代表权重已作为拟上传附件放在 `Project/delivery/`（各带 `.sha256`）：AE `1ab77d31…87e4`、监督 `62436641…d916`、改进 `cc24c24b…037e`。本轮未上传。

## 失败与修复记录

- 归档前置检查发现 Git 对 428 个文本文件 LF→CRLF 检出，而 provider 的 image/json SHA256 锁定 CRLF 物理字节；数据审计同时核验 Git blob 规范化摘要与 provider 原始摘要，未改数据。
- 第一次改进汇总在 `refinement_key` 使用顶层 `fn/fp`，触发 `KeyError`；修正为 `best.fn/best.fp` 后重跑 9 次改进，R0 仍精确复现。原失败汇总移入 `outputs/member04_hog/final_v4/failed_attempts/refine_summary_keyerror_20261124/`。
- 第一次 test 评价的分组统计误用 HOG 清单（无 batch/condition 列），触发 `KeyError: filename`；改用 `split_manifest.csv` 后重跑，失败的首次评价目录移入 `failed_attempts/evaluate_grouped_keyerror_20261124/`，没有用 test 结果改模型。
- 全量审计最初把 102 张可选训练增强计入 train 标签/框，导致与原图总账不符；改为只统计 `usage=primary` 的原图后通过。该修复只影响审计统计，不改变数据、模型、阈值或选型；协议/冻结/检查点的代码摘要与 freeze 指针做了带记录的一致性刷新。
- test 的 S/H 子集最初把 63 张评价增强计入，修正为 11 张原始 test 图；只从已保存预测重算，未重新预测 test。

## test 访问记录

- `freeze.json` 创建于任何 test 解码之前，`test_accessed=false`。
- `prepare --scope full` 恢复 test/增强后写 `audit/test_access.json`（类型 restore）。
- `audit --scope full` 解码 30 张 test 并核验 415 组三联，`verify_dataset.py --all` 通过。
- `evaluate --frozen` 只评价三个冻结代表一次；`evaluation/test_access.json` 记录时间与冻结摘要。之后没有重训、换阈值、换种子或利用 test 选型。

## 下一条命令

```powershell
python -m members.member04_hog.final_v4.verify
python -m members.member04_hog.final_v4.package --out ../delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip
```

## 待交接

- 组长：本轮上传许可；公共 `baseline1` 注册映射不变，新默认由本模块 `configs/active_model.json` 指向。main 已跟踪成员3媒体，`.github/check_scope.py` 预期仍被公共媒体规则阻断；本地记录“成员范围通过/框架未通过”。
- 成员3：物理场景独立性与标签来源；本模块未修改划分或标注。
- 成员6/7：共同图像级评价和摄像头端到端耗时；本轮只做离线开发评价，不伪造其他成员预测。
- 旧 Release/PR：#5、#7、#8 与三个旧 Release 保持原地址、附件名和摘要；归档映射见 [ARCHIVE_MAPPING](docs/ARCHIVE_MAPPING.md)。
