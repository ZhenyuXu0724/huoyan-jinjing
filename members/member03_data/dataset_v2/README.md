# Dataset v2：最终数据构建版本 freeze_v1

本版本已执行用户明确授权的局部修订。上交结论见 [FINAL_FREEZE_CN.md](FINAL_FREEZE_CN.md)，确认来源见 [USER_CONFIRMATIONS.json](USER_CONFIRMATIONS.json)。工程冻结仅指数据构建版本，不代表课程模型或正式测试协议已冻结。

| 分区 | PASS原图 | FAIL原图 | PASS增强 | FAIL增强 | 总图片 |
|---|---:|---:|---:|---:|---:|
| Train |149|145|6|138|438|
| Val |37|37|0|18|92|
| Test候选 |15|16|0|48|79|

609张图片＋609份对应Labelme JSON全部保留。仅将11/23原Train六张按文件名升序的前两张 `MVIMG_20261123_182036.jpg`、`MVIMG_20261123_182054.jpg` 及同名JSON移至 `test/PASS/original/`。无图片/JSON内容修改，没有新增强。

## 确认与评价边界

用户本轮已明确同意旧验证/历史评价数据重新用于开发训练，确认11/23每张照片前重新打乱摆放，确认171811/172012是不同关联摆放，确认11/20和11/23未用于参评模型训练、验证、选型、调参或模型错误分析。上述事项已解除待复核限制，无需重复确认。

Test有31张原图（15 PASS、16 FAIL），照片数量支持达标。15张11/23 PASS登记为用户确认的不同摆放。用户对171811/172012的否定关联不等于确认其余全部11/20候选的场景家族独立性；因此整体独立场景支持数尚不能确认为31，正式独立测试仍为NOT_READY，目录状态仍为PROVISIONAL_HOLDOUT。

共享纸面或会话仅作为评价局限，不直接判为同一物理摆放。Test PASS/FAIL来自不同日期，增强图不是独立场景。数据质量审查的图片接触记录不应伪称模型开发使用，也不应隐去。

## 使用路径

- 原图开发训练：`train/PASS/original/`、`train/FAIL/original/`，294张。
- 含既有增强的开发训练：加上Train对应`augmented/`，438张。
- 主验证原图：`val/{PASS,FAIL}/original/`，74张；18增强另报鲁棒性。
- Test候选原图31张，增强48张另报；不得合称79独立场景。
- **禁止递归扫描整个dataset_v2作为训练输入。** 本轮未改任何成员6训练工程，未启动训练或推理。

## 安全检查

仓库根目录使用已安装Pillow/NumPy/SciPy的系统Python：

```bash
/usr/bin/python3 -B members/member03_data/dataset_v2/build_dataset_v2.py --plan
/usr/bin/python3 -B members/member03_data/dataset_v2/verify_dataset_v2.py
/usr/bin/python3 -B -m unittest discover -s members/member03_data/dataset_v2 -p test_acceptance.py -v
```

建库默认只核验规划，`--build`遇已有目录或清单会拒绝覆盖。`source_snapshot_lock.json`保持755e038源字节锁，逐图分配已更新。建库与校验都执行用户裁决和最终分配规则。

新结果写`freeze_v1/`。`acceptance_v1/`、`final_revision_v1/`、`INDEPENDENT_ACCEPTANCE_CN.md`及`FINAL_REVISION_CN.md`保留历史原样；其旧结论由本轮用户裁决及最终报告更新，不再作为当前资格状态。
