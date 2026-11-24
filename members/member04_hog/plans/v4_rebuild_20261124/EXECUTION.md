# v4 重建执行记录

日期：2026-11-24。按 [PLAN.md](PLAN.md) 执行，结果已落入成员4活动目录与本地交付包；未做任何 GitHub 写入。

## 环境与 Git

- 工作分支：`member04/hog-v4-final-dataset`，从 `origin/main` = `bde8ea7116761bc8ee7fca1d47c336126daa995f` 建立；原 HEAD `9186a5b` 上的成员4内容与 main 无差异。
- 本机 Git HTTPS 使用 OpenSSL 后端与 Git 自带 CA（仅本地配置，未写入仓库）。
- 归档根：`Project/delivery/member04_history_before_v4_20261124/`；1489 文件 / 2,926,577,008 字节，复制后逐文件 SHA256 复核 PASS，清理 1419 个旧文件，删除后归档复查 PASS。
- `Project/delivery/v3_relocation_3cyz5ths` 被 ACL 拒绝读取，按计划列为未处理项保留。

## 数据

- 固定来源 commit/tree：`bde8ea7…` / `c6882fd…`；开发范围 862 文件，全量 1260 文件 / 1,789,463,157 tree 字节。
- 开发审计 54 项、全量审计在冻结后完成；provider `verify_dataset.py` 分别核验 283 / 415 组三联，均通过。
- 发现并记录 Git LF↔CRLF 口径：428 个文本文件 Git blob 为 LF、工作区为 CRLF；provider 的 image/json SHA256 锁定 CRLF 物理字节。恢复与审计同时核验两种口径，未修改数据。

## 75 次正式训练

- 66 次主实验：AE 入选 A1（平均 val 漏放 2.33、误拒 4.00）；监督入选 A2（0.33、2.67）。两对等价组逐位一致。
- 9 次改进：R0 三个种子第 200 轮精确复现监督 A2（权重、标准化器、预测、最大概率差 0）；R0 以平均最佳漏放 0、误拒 3.33 入选。
- 84 份正式检查点（66 last + 9 best/last）全部可重载。

## 冻结、诊断与 test

- 冻结默认：`Baseline1_HOGMLP_v4_FinalDataset_Supervised_Capacity_L2_EpochSelection`，R0 seed42 第 4 轮，阈值 0.9850894510746002；`test_accessed=false`。
- 诊断：283 图 ×21 设置 ×3 模型 = 17,829 条预测；AE 对缩小/居中补白极敏感，不能把零翻转解释为鲁棒。
- test：冻结后恢复、全量审计、一次性评价 3 个代表。AE 漏放 4/误拒 7、监督 A2 10/3、默认 R0 6/4；S/H 11 图三种方法漏放 2/7/4，仅图像级。

## 验证与交付

- `final_v4.verify` PASS：84 检查点、3 代表 val 逐图一致（最大浮点差 3.58e-7）、接口检查通过；模块 21 项、公共 6 项通过；依赖一致；改动仅成员4目录。
- `check_scope.py` exit 1：main 已跟踪成员3媒体；记录为“成员范围通过、框架未通过”，未绕过。
- 交付包：`Project/delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip`，928,957,751 字节、592 条目，SHA256 `4449706dba29d50deb5ae745a84dc71224e63cdd6fd0479f81110955411223a3`；打包后 ZIP 逐文件校验与解压迁移（模块测试、单图、批量、缺权重）全部通过，记录在同名 `_verification.json`、`_tests.txt`。ZIP 为打包时快照，后续文档/状态以 Git 为准。
- 三个代表权重作为拟上传附件放在 `Project/delivery/`，各带 `.sha256`；本轮未上传。

## 偏差与修复

1. 第一次改进汇总 `refinement_key` 误用顶层 fn/fp，KeyError；修正后重跑 9 次，R0 仍精确复现；旧失败目录在 `failed_attempts/refine_summary_keyerror_20261124/`。
2. 第一次 test 评价误用 HOG 清单做分组，KeyError；改用 split_manifest 后重跑；旧失败目录在 `failed_attempts/evaluate_grouped_keyerror_20261124/`，未用 test 改模型。
3. 全量审计曾把 102 张可选训练增强计入 train 原图总账；改为只统计 `usage=primary` 后通过。仅审计统计修复，未改数据/模型/阈值/选型；协议、冻结指针与检查点元数据做了带 `metadata_refresh` 记录的摘要刷新。
4. S/H 子集曾把评价增强计入 44 张；修正为 11 张 primary test 图，仅从已保存预测重算。
5. `evaluate.py` 的分组统计与 `delivery.py` 目录校验容错在冻结后做过报告/验证代码修复；没有重算 test、没有改模型或阈值。冻结摘要保留冻结时记录，差异在 progress.md 说明。

## 当前停止点

本地交付完成，**等待用户对本次 GitHub 上传的明确许可**。未 push、未创建/更新 PR、未创建标签、未发布/编辑 Release，也未上传附件。下一步（获许可后）才推送成员4分支、创建 PR 并按需上传交付包附件；不删除或改写历史 Release。
