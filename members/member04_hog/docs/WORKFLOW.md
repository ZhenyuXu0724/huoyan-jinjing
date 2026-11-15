# 从安装到交接的操作顺序

全部命令从仓库根目录执行。已有产物可先阅读报告，不必重复训练。Python 使用 3.11，环境安装见 [模块入口](../README.md)。不要把虚拟环境直接搬到新机器。

## 1. 开发数据准备

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.one_class.prepare --out data/member04_hog/dataset --phase development
```

需要联网，只下载固定版本的开发数据，核验后生成清单和来源记录。确认训练 133 PASS，验证 24 PASS + 77 FAIL。此时不接入 test 内容。

## 2. 训练基础组与消融

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -u -m members.member04_hog.train --train data/member04_hog/dataset/train.csv --valid data/member04_hog/dataset/valid.csv --out outputs/member04_hog/experiments --weights weights/member04_hog/experiments --epochs 200 --stage all
```

每组种子 42、43、44，先 A0，再 A1–A5、B1–B2，符合预定条件才运行 C 组。`--stage baseline` 仅跑 A0。短训练只验证程序，不能替代正式 200 轮结果。全部保存最后一轮，标准化器和权重只学习 PASS train。

同命令可核验并复用完整运行；不完整的中断目录会报错，应另建运行目录或明确移走该次未完成产物。数据、代码、轮数或配置不匹配时必须新建版本，不能拼接不同实验的记录。

## 3. 汇总与冻结

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.one_class.report --results outputs/member04_hog/experiments --weights weights/member04_hog/experiments --data data/member04_hog/dataset --destination weights/member04_hog/baseline1.joblib
```

查看 `summary.json` 的 `selected_group`、`REPORT.md` 和 `selection.json`。集成固定为该组 seed42；不选最佳种子。测试对照预先固定为 A0 seed42 与所选组 seed42。把以下 `SELECTED_GROUP` 替换成实际组名；若所选就是 A0，只传一个模型。

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.evaluate freeze --out outputs/member04_hog/assessment/freeze.json --models weights/member04_hog/experiments/A0/seed42/model.joblib weights/member04_hog/experiments/SELECTED_GROUP/seed42/model.joblib
```

冻结保存权重、阈值、配置、开发数据身份与推理评价代码校验值。之后修改受检内容会使校验失败，不能覆盖记录掩盖版本变化。

## 4. 冻结后接入提供的 test

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.one_class.prepare --out data/member04_hog/dataset --phase test --freeze outputs/member04_hog/assessment/freeze.json
.\.venv\Scripts\python.exe -m members.member04_hog.evaluate test --freeze outputs/member04_hog/assessment/freeze.json --test data/member04_hog/dataset/test.csv --out outputs/member04_hog/assessment/test
.\.venv\Scripts\python.exe -m members.member04_hog.one_class.audit --data data/member04_hog/dataset --freeze outputs/member04_hog/assessment/freeze.json --out outputs/member04_hog/assessment/data_audit.json
.\.venv\Scripts\python.exe -m members.member04_hog.one_class.report --results outputs/member04_hog/experiments --weights weights/member04_hog/experiments --data data/member04_hog/dataset --destination weights/member04_hog/baseline1.joblib --test-results outputs/member04_hog/assessment/test
```

测试输出目录须不存在，避免覆盖访问证据。结果仅汇报，不改阈值或方案。先看实际 PASS/FAIL 支持数，缺少某类时不能声称测得该类能力。公开 test 的限制见 [数据说明](DATA.md)。审计核对全部源文件、图片与框、对象计数及分区重复，不替代人工审核。

## 5. 成员 7 批量推理

复制 [数量请求示例](../examples/expected_counts.json)，把 `image_id` 改成真实文件名。请求只含文件名与要求数量，不能含真值。

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.inference --input_dir data/member04_hog/dataset/old/test --counts_file members/member04_hog/examples/expected_counts.json --output_dir outputs/member04_hog/camera_trial
```

递归搜索图片，重复文件名会拒绝。输出 `images.csv` 和 `run_config.json`；数量不支持为 `UNSUPPORTED_COUNTS`，图片缺失为 `INPUT_ERROR`。离线处理时间不包含摄像头或显示。

成员 6 使用 `compare.py --help` 查看共同评价入口；真值表、数量请求与模型导出格式见 [评价解释](INTERPRETATION.md)。

## 6. 检查与打包

第一阶段修订使用归档名 `../delivery/member04_hog_stage1_20261115.zip`，不覆盖原始归档。诊断命令见 [第一阶段报告](STAGE1.md)。真实权重接口验收可运行：

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.verify_stage1 --image "data/member04_hog/dataset/new/补强数据(训练）/IMG_20261114_160608.jpg" --out outputs/member04_hog/assessment/stage1_interface_repeat.json
```

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s members/member04_hog/tests -v
.\.venv\Scripts\python.exe .github/check_scope.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe predict.py --list-models
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m members.member04_hog.delivery --out ../delivery/member04_hog_20261115.zip
.\.venv\Scripts\python.exe -m members.member04_hog.delivery --verify ../delivery/member04_hog_20261115.zip
```

单元测试只验证程序逻辑，不是模型效果。ZIP 包含代码、默认与冻结对照权重、配置和运行证据，不包含原始图片、全部消融权重及虚拟环境。完整原始数据与全部实验权重仍保留在本地标准目录；完整迁移时请一并复制。ZIP 解压后按其中 `DELIVERY_README.md` 重建环境和校验。
