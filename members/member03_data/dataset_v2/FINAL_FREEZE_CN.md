# Dataset v2 最终局部修订与冻结交付

日期：2026-11-24；版本：freeze_v1。工程状态 **READY**。数据集建设任务至此结束。

## 本轮事实来源与裁决

确认来源是用户本轮明确陈述，逐项记载于 `USER_CONFIRMATIONS.json`，不是脚本从时间间隔、相似度或预测结果推断：

1. 用户同意新版划分及将部分旧验证、历史评价数据重新用于新版开发训练。
2. 用户确认11/23正常照片每次拍摄前重新打乱摆放，属于不同物理摆放场景，允许从原Train六张中选择两张进入Test。
3. 用户确认171811与172012不是同一关联摆放；两张及其增强子图维持原分区。
4. 用户确认11/20、11/23新增数据未被参评模型用于训练、验证、选型、调参或模型错误分析。

上述四项限制已解除，不再要求重复确认。历史验收材料保留原文，以本报告说明新事实对旧待确认结论的更新。

## 实际目录调整

确定性规则：对修订前11/23 Train六张PASS原图的完整文件名字典序排序，取前两张。没有使用任何模型预测。六张候选与选择写入确认记录和划分策略。

两张为 `MVIMG_20261123_182036.jpg`、`MVIMG_20261123_182054.jpg`，及对应同名JSON，从 `train/PASS/original/` 移入 `test/PASS/original/`。它们没有增强子图。仅四个数据文件改变路径，其余1214个数据文件路径保持不变；全部数据内容保持不变。

|文件|SHA256（移动前后相同）|
|---|---|
|MVIMG_20261123_182036.jpg|`9c7449c41d372fbaa06fb26d4e4b295a7881e2ff9a52b95709e95e8cbd53b1d1`|
|MVIMG_20261123_182036.json|`aafbcf055f0bab92a85956ef513df5677294af3e84bb70e9b3b1920e04660393`|
|MVIMG_20261123_182054.jpg|`6c1f9d3f057a7a5d65868ac9ee5d086a085da01da115a37ec55e33d3603f3469`|
|MVIMG_20261123_182054.json|`589e308b80725005c316273a5b6c44b3a056a81eeaf8557cbfd3e3716f9e3d8b`|

## 最终分布

|分区|PASS原图|FAIL原图|PASS增强|FAIL增强|总数|
|---|---:|---:|---:|---:|---:|
|Train|149|145|6|138|438|
|Val|37|37|0|18|92|
|Test候选|15|16|0|48|79|
|总计|201|198|6|204|609|

原图399张、增强210张，共609张图片及609份JSON。Test的48张增强不增加独立场景支持数。

## 场景与正式评价资格

11/23全部19张正常原图按用户确认记录为19个独立摆放，分别具有 `U1123_<文件名stem>` 场景ID，其中15个在Test、4个在Train。171811/172012的指定关联疑点由用户裁决解除；不移动它们及任何增强子图。已记录的场景组及增强家族没有跨区。

Test当前包含31张原图（15 PASS、16 FAIL），满足照片数与类别照片数目标。**已确认的11/23 Test独立摆放为15个；11/20的16张Test原图仍沿用时间块场景记录，其互相之间及与其余开发场景之间的完整物理家族关系没有逐场景证据覆盖。** 本轮指定一对照片不相关的确认，不自动证明其余全部照片独立。因此总独立场景数尚不能认证为31，至少30个独立场景的目标尚未被证实。

正式独立测试状态维持 **NOT_READY / PROVISIONAL_HOLDOUT**。剩余具体信息仅为11/20 Test原图的物理摆放家族对应关系及其与开发场景的隔离；16个文件名列于 `evaluation_eligibility.json`。这不阻止工程交付，也不撤销四项用户裁决。不要求为此重新执行全量审查。

同背景纸、同拍摄会话仅作为评价局限，不视为同一摆放的证明。Test的PASS全部来自11/23、FAIL全部来自11/20，类别与拍摄批次混杂；即使后续独立性获确认，也不能仅凭这一测试集证明跨拍摄条件泛化。旧评价数据已获准转开发使用，旧实验成绩不能冒充新版独立测试成绩。文件完整性检查不是模型推理；没有进行训练或预测。

## 实际验收

- 609张图片、609份JSON完整；1218个文件与批准源文件、锁定Git源字节一致，SHA256校验全部通过。
- 图片像素、尺寸、文件名、JSON字段没有修改；没有删除或新增数据资源。
- 210张增强图父子跨区为0（70个原图父节点）；确认场景组跨区为0。
- 实际目录、清单、统计、增强映射、哈希表和建库计划一致；609条构建记录、56个管理场景组与实际管理文件一致。56不是已证明的独立物理场景数。
- 原始源锁文件中的源字节身份不变，仅同步目标分配；复现脚本包含本轮确定性选择，不会恢复旧配额。
- Dataset v2之外26913个文件内容/链接快照一致；84个历史验收文件哈希一致。证据：`freeze_v1/preservation_result.json`、`before.json`。
- 15项自动测试通过。完整命令、实际退出码和日志保存在 `freeze_v1/commands.json`。

|检查|实际退出码|结果|
|---|---:|---|
|regression|0|通过|
|plan|0|通过|
|existing_build_refusal|1|预期拒绝覆盖已有数据集|
|final_verifier|0|通过|
|independent_plan|0|通过|
|sha256sum|0|通过|

独立源校验：`freeze_v1/source_destination_checks.json`、`source_identity_verified.json`、`independent_summary.json`。构建一致性：`plan_recheck.json`。确认事项：`restriction_checks.json`。四次文件移动：`moves.json`。旧管理文件快照：`management_before/`。

## 版本身份和改动范围

- 工作树HEAD：`bc80be11ed5ed5eb1045dc5fe1904350f6a35bbc`；分支 `member06/resnet-baseline`。HEAD不代表本次未提交交付字节，后者另见 `freeze_v1/delivery_sha256.json`。
- 批准源提交：`755e038cc8f05b9701cceb28e3381d5fa3bc6a81`；已批准标注修订来源：`16d9239aae5360134aae4ae6eb0b9a19703364ba`。
- 同步更新清单、统计、场景、增强映射、SHA256SUMS、README、SPLIT_REPORT，以及建库和校验支持脚本。新增用户确认记录及本轮证据目录。
- `acceptance_v1/`、`final_revision_v1/`、`INDEPENDENT_ACCEPTANCE_CN.md`、`FINAL_REVISION_CN.md`保持历史原文。历史报告中的旧配额及已解除待确认事项由本报告取代，不回写历史证据。
- 没有修改Dataset v2外文件、成员3源数据或成员6训练代码；没有训练、Val/Test推理、Commit或Push。

## 交付与查看

数据集路径：`/home/grace/projects/AI/member06/members/member03_data/dataset_v2/`。
直接打开本报告、README和 `freeze_v1/commands.json` 查看最终状态。安全复核命令（仓库根目录）：

```bash
/usr/bin/python3 -B members/member03_data/dataset_v2/build_dataset_v2.py --plan
/usr/bin/python3 -B members/member03_data/dataset_v2/verify_dataset_v2.py
```

本轮未启动任何下一阶段训练工程。新版ResNet18科学方案由用户后续另行制定。
