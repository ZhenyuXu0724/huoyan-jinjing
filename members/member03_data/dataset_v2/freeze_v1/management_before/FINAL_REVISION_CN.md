# Dataset v2 提交前局部修订与最终验收

## 最终结论

**工程交付 READY；正式独立测试 NOT_READY。本轮完成局部资格复核、逐图管理限制修订与必要验收，没有移动任何图片或JSON。** 目标438/92/79没有实现，实际仍为440/92/77。不能把两张缺少历史用途与场景独立性证据的Train照片移入Test后填报达标。

本轮采用用户要求的兜底处理：无法确认具体关联范围时保留分区并标记完整候选范围待复核。没有擅自重设计整体划分、补拍、训练、Val/Test推理、Commit或Push。Dataset v2建设在本轮交付处结束。

## 1. 身份与保护范围

- 数据根：`/home/grace/projects/AI/member06/members/member03_data/dataset_v2/`。
- 当前本地分支 `member06/resnet-baseline`；起始HEAD `bc80be11ed5ed5eb1045dc5fe1904350f6a35bbc`（沿用现有工作树，无合并操作）。
- 最新批准源快照仍按 `source_snapshot_lock.json` 的 `755e038cc8f05b9701cceb28e3381d5fa3bc6a81` 验收；不自动引入未批准的新来源。本轮未获取远端更新。
- 修订前管理文件保存在 `final_revision_v1/management_before/`；原 `INDEPENDENT_ACCEPTANCE_CN.md` 及整个 `acceptance_v1` 均保持历史字节不变。

## 2. 最终实际分布（修改前后相同）

|分区|PASS原图|FAIL原图|PASS增强|FAIL增强|图片合计|JSON合计|
|---|---:|---:|---:|---:|---:|---:|
|train|151|145|6|138|440|440|
|val|37|37|0|18|92|92|
|test|13|16|0|48|77|77|

合计399原图＋210增强＝609图片，另609份对应JSON。Train共157 PASS/283 FAIL，Val37 PASS/55 FAIL，Test候选13 PASS/64 FAIL（64含48增强，真实FAIL仅16）。

Test原图29张不等于29个已证实独立场景，已核实的独立场景数未知。按照片数算仍少2张PASS、总原图低于30一张；即便后续补入2张合格PASS达到31原图，也必须验证物理场景与历史用途隔离，不能据31张自动宣称31场景。

## 3. 六张PASS的定点复核与未转入原因

本轮实际打开以下六张原始图片，并复用既有11/23全批联系表比较剩余Train与现有Test候选。零件布局/朝向存在变化，纸面笔迹及拍摄背景共享。没有仅用时间间隔、低IoU或横竖构图差异作为独立性依据。

|文件|实际可见特点|本轮归属|
|---|---|---|
|MVIMG_20261123_182036.jpg|两长螺丝左上/下中斜向，垫片下部左右；其余件横向分布。|保留Train|
|MVIMG_20261123_182054.jpg|两长螺丝左下近水平/右中斜向，垫片左上/右下。|保留Train|
|MVIMG_20261123_182108.jpg|两长螺丝上中近竖/左中斜向，螺母中下/下部。|保留Train|
|MVIMG_20261123_182122.jpg|两长螺丝中部斜向/右下斜向，垫片中下相邻。|保留Train|
|MVIMG_20261123_182138.jpg|竖幅：两长螺丝中上横向/右下竖向；构图变化不能单独证明重摆。|保留Train|
|MVIMG_20261123_182155.jpg|两长螺丝右上/左下，垫片中部/右部；共享笔迹与纸面。|保留Train|

- 本轮扫描成员4/5/6/7现有290份文本资料（排除虚拟环境、缓存及>20MB文件），六张的精确文件名或图片SHA匹配0处。完整检索范围及EXIF摘录见 `final_revision_v1/targeted_review.json`。这不能证明全组从未读取、训练、验证、选型或误差分析；未上传运行、外部release与线下记录不在证明范围。
- 六张均无可签收的物理场景家族记录和完整历史使用声明。因此不能识别“2张同时满足独立性及历史隔离条件的合格照片”。未任意选择两张，也没有把共享背景误写成已经证实同一摆放。
- 照片在本轮被用于数据质量审查，访问史明确记录；没有任何模型评分用于选择。六张均添加 `final_review_history_and_scene_unconfirmed_keep_train` 管理备注，未改Labelme JSON。

## 4. 171811/172012与前后照片的最终处理

复用 `acceptance_v1/nov20_B08_B09.jpg` 并重新实际打开两张原图。B08的171618、171632、171652、171704、171757、171811及B09的172012、172032、172048、172104、172132、172155均纳入相邻照片比较。

- 两根下部长螺丝及右侧两螺母的相对布局近似保留；左侧沉头螺丝邻近纸面笔迹；172012加入黄铜件并移除/替换圆形物体，上部物体也有转向/移动。支持局部摆放继承疑点，尚不能从两张图确认整个实际物理家族的精确边界。
- 前后其他照片有较明显位置变化，不能断言整个时间块都是同一个物理场景；也不能因块内布局变化就断言块间独立。
- **核心待复核家族**为以下8张，全部关联JSON同样保留；本轮没有遗漏三个派生版本：

- `MVIMG_20261120_171811.jpg`
- `MVIMG_20261120_171811__A01_lowlight.jpg`
- `MVIMG_20261120_171811__A02_warm.jpg`
- `MVIMG_20261120_171811__A03_softnoise.jpg`
- `MVIMG_20261120_172012.jpg`
- `MVIMG_20261120_172012__A01_lowlight.jpg`
- `MVIMG_20261120_172012__A02_warm.jpg`
- `MVIMG_20261120_172012__A03_softnoise.jpg`

- 保守扩展复核范围：B08/B09全部12原图＋36增强＝48张图片/48份JSON，逐图名单在 `final_review_policy.json` 的 `conservative_review_scope`，不声称这些都已证实为同一物理家族。
- 核心8张添加 `final_review_core_partial_layout_family_pending`，扩展48张添加 `final_review_B08_B09_family_scope_pending` 管理备注；所有增强继承对应限制。
- 依照“无法确认时标为待复核”的要求，保持原分区，所有Test继续PROVISIONAL_HOLDOUT，正式评价禁止放行。本轮没有足够证据把Train171811当作未开发新Test，也没有依据任意选择整个家族的迁移方向。
- 已知历史关联候选继续同区；本轮新增确认的物理家族为0。**场景关联隔离结论INCONCLUSIVE，不能填写PASS或声称已解决跨区疑点**。

## 5. 实际文件修订与复现一致性

|文件|本轮操作|
|---|---|
|图片及Labelme JSON|移动0、改名0、内容改动0；609对原位保留|
|dataset_manifest.csv|54张图片新增资格复核备注，包含6张PASS及48张家族扩展范围；计数/标签/路径/哈希未变|
|dataset_statistics.json、scene_groups.json、evaluation_eligibility.json|加入一致的final_revision状态，记录未达数量目标、场景未知、正式Test NOT_READY|
|final_review_policy.json|逐图冻结保留分区、核心8张与扩展48张及六张PASS限制|
|build_dataset_v2.py|按同一policy恢复管理备注；既有分配保持不变，不会复现成未经批准的438/92/79|
|verify_dataset_v2.py、final_revision_checks.py|新增资格限制、家族子图完整性与元数据一致性检查|
|independent_acceptance.py|新核查写final_revision_v1，避免覆盖历史acceptance_v1|
|augmentation_mapping.csv、SHA256SUMS.txt、source_snapshot_lock.json|无路径/数据变化，重新核验后保持字节原样；不制造无意义差异|
|README.md、SPLIT_REPORT_CN.md|明确指向最终报告，显示实际配额和未放行限制|

建库规划与实际manifest的609条记录逐项比对split、label、data_tier、parent_image_id、scene_group_id、augmentation_type、quality_notes，差异0。独立规划重算也通过。原有--build拒绝覆盖保护保留，不执行实体重建。

## 6. 实际验收结果

- 1218/1218数据文件与批准源文件SHA256、大小、文件名完全一致，且源文件Git blob匹配批准快照。无缺失、额外数据、重复分配、错误图/JSON配对、硬链接或符号链接。
- 1218个数据文件与本轮起始快照字节一致，路径也一致。未改任何原始来源。
- 210/210增强图父子类别、坐标/尺寸及分区一致，父子跨区0；核心两父图各3增强完整。
- 实际目录、各批次/类别统计、manifest、scene_groups、augmentation_mapping、SHA256SUMS一致。源内容核验复用已有独立校验器；未重跑全量布局相似度、PCA、阈值搜索或模型诊断。
- 15项定向/安全回归全部通过（既有12项＋禁止静默Train→Test、禁止READY虚报、增强子图限制不可遗漏3项）。
- Dataset v2外26913个文件（含ignored和.git）前后集合/记录哈希不变，新增0/删除0/变化0。历史验收43个文件哈希不变。证据见 `final_revision_v1/preservation_result.json`；只证明本轮执行期间，不倒推更早历史。

|检查|退出码|结果|
|---|---:|---|
|regression|0|通过|
|plan|0|通过|
|existing_build_refusal|1|预期拒绝覆盖，保护通过|
|final_verifier|0|通过|
|independent_plan|0|通过|
|sha256sum|0|通过|
|check_preservation.py|0|目录外、历史证据与数据字节不变|
|建库/manifest逐字段比对|0|609记录一致|

完整实际命令、cwd、退出码、耗时在 `final_revision_v1/commands.json`，日志同目录。

## 7. 上交路径及最小待确认事项

上交工程目录：`/home/grace/projects/AI/member06/members/member03_data/dataset_v2/`。最终报告：该目录 `FINAL_REVISION_CN.md`。可按工程合格、科学资格限制已披露的版本上交检查；不能包装成正式独立Test验收通过。

仅需补齐两类事实，不要求现在重拍或训练：

1. 拍摄人针对11/23六张及171811/172012、前后B08/B09照片提供：是否完全重新摆放、同一摆放继承的完整文件名单、可分离的场景家族及责任人/日期；同时确认候选Test场景台账。
2. 参评模型负责人给出六张PASS及候选Test的逐图历史用途声明（训练/验证/选型/误差分析/仅数据质检）。只有证据允许，才可另行批准整家族移区；本轮不预选两张、不预批移动。

安全复核命令（仓库根）：

```bash
cd /home/grace/projects/AI/member06
/usr/bin/python3 -B members/member03_data/dataset_v2/build_dataset_v2.py --plan
/usr/bin/python3 -B members/member03_data/dataset_v2/verify_dataset_v2.py
```

最终裁决：文件完整性PASS；字节零修改PASS；增强隔离PASS；实际划分/复现一致性PASS；真实场景隔离INCONCLUSIVE；工程READY；正式独立测试NOT_READY。本轮已完成全部具备证据的安全修订，数据建设任务到此结束。
