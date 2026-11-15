# 交付清单与接收检查

## 第一阶段修订版

版本 `member04-stage1-20261115`，归档名 `member04_hog_stage1_20261115.zip`。包含新增开发集诊断、阈值取舍表和真实接口验证；原始图片、诊断缩略图、虚拟环境均排除。默认模型仍为 A2 seed42，SHA256：`57714c697df9c102f27a21d26e71b6ee1907f8f4dd2613d46803f5d313b9fbab`。

计划发布到本仓库同名 GitHub Release；实际上传状态见 progress.md。压缩包 SHA256 随 `.sha256` 附件提供，解压后另有逐文件 `DELIVERY_SHA256.json`，避免将归档自身摘要写回归档形成循环。

模块 `.gitattributes` 保留历史文件字节，防止 Git 换行转换使冻结代码摘要失效。冻结检查基于实际字节，不应格式化已冻结的推理代码。

## 交付范围

当前版本使用成员 3 的真实采样数据：133 张 PASS 训练，101 张验证（24 PASS、77 FAIL），冻结后评测 24 张 test（23 PASS、1 FAIL）。所有图片、标注与正式运行记录保持原分区；数量规则为五类各 2，仍待小组最终确认。

| 内容 | 相对于仓库根目录的位置 | ZIP 是否包含 |
|---|---|---|
| 程序、配置、依赖、说明及测试 | `members/member04_hog/` | 是 |
| 默认集成模型 | `weights/member04_hog/baseline1.joblib` | 是 |
| 冻结对照 A0/A2 seed42 | `weights/member04_hog/experiments/` 的对应组 | 是 |
| 全部 33 次消融模型 | `weights/member04_hog/experiments/` | 仅冻结对照，其余留在本地 |
| 配置、训练历史、验证预测与汇总 | `outputs/member04_hog/experiments/` | 是 |
| 冻结、数据审计、测试与重载记录 | `outputs/member04_hog/assessment/` | 是 |
| 成员 7 批量推理示例与结果 | `outputs/member04_hog/handoff/` | 是 |
| 数据清单、来源和校验索引 | `data/member04_hog/dataset/` 顶层清单及模块 configs | 是 |
| 原始图片和对象标注 | `data/member04_hog/dataset/old/`、`new/` | 否，完整迁移时另行复制 |

ZIP 位于仓库外同级 `../delivery/member04_hog_20261115.zip`；旁边有 `.sha256`、`verification.json` 和 `verification-tests.txt`。ZIP 不携带虚拟环境或缓存。只接收 ZIP 时，可按固定来源索引重新取得图片；已有完整文件夹时直接保留标准数据目录。

## 接收顺序

1. 解压到新目录，先按 `DELIVERY_README.md` 检查文件校验值。
2. 使用 Python 3.11 重建环境并安装模块锁定依赖，运行成员 4 的单元检查。
3. 放入一张真实图片，用单图入口确认默认模型可以加载；批量调用格式见 [复现步骤](WORKFLOW.md)。
4. 将输出中的数量支持范围、得分类型、异常状态与成员 7 接口对齐。
5. 完整迁移时另行复制原始数据和其余消融权重；保留相对目录关系。

所有命令从解压根目录或仓库根目录执行，图片路径不依赖原机器盘符。ZIP 中的冻结校验值可以在解压后复核，测试访问记录与原始预测作为证据保留。

## 当前可确认与待确认事项

程序检查和重载一致性已通过。默认模型为 A2 seed42，验证平均 FAIL F1 为 0.9417；提供的 test 上漏放 1 张、误拒 10 张，FAIL F1 为 0。它是可复现实验基线，尚无可靠现场检测的证据。

成员 3 仍需核实真实场景独立性、规格及最终标注；组长确认数量规则并整合公共接口；成员 6 对齐共同图像级评价；成员 7 完成摄像头与界面联调、端到端耗时测量。详见 [进度与交接](../progress.md) 和 [实测报告](RESULTS.md)。
