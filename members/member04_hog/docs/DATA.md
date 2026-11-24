# 数据审计与来源锁定

本文件描述 `Baseline1_HOGMLP_v4_FinalDataset_Rebuild` 使用的唯一数据来源、恢复范围和审计结论。规划阶段的只读元数据审计保留在 `plans/v4_rebuild_20261124/dataset_audit.json`；执行时在 `outputs/member04_hog/final_v4/audit/` 重新计算了全部可解码内容。

## 固定来源

| 项目 | 值 |
|---|---|
| 仓库 | `ZhenyuXu0724/huoyan-jinjing` |
| 数据 commit | `bde8ea7116761bc8ee7fca1d47c336126daa995f` |
| 数据目录 | `members/member03_data/final_training_v4` |
| 目录 tree | `c6882fd823b1f458105b47fea2f52af0b357ef9c` |
| 引入提交 | `f5dad94b9a79243ce2f3f83041d62de118b8ed08` |
| 文件 / 字节 | 1260 / 1,789,463,157（tree 口径） |

供应方目录是只读来源；成员4自己的副本按固定 tree 恢复到 `data/member04_hog/final_training_v4/`，保留 `images/`、`labels/`、`manifests/` 和中文元数据文件名，使 HOG 清单中的 `../../images/...` 相对路径可直接解析。

## 恢复范围

- **development**：283 张 train/val 原图及其 LabelMe JSON、YOLO TXT，加 13 个元数据文件（README、classes.json、data.yaml、verify_dataset.py、split_manifest.csv、原图明细.csv、HOG 四张清单、ResNet 三张清单），共 862 个文件。
- **full**：在 development 基础上恢复 test 30 组三联、102 组可选训练增强三联、`测试集逐图核对.csv` 与 `train_resnet_v4.py`，共 1260 个文件；必须先写入 `freeze.json`，并记录 test 恢复事件。102 张离线增强不进入本轮 75 次训练。

开发阶段不复制、不解码 test 与增强媒体；test 像素只在冻结后的 `audit --scope full` 中解码一次。

## 分区与标签

| 分区 | PASS | FAIL | 原图 | 目标框 | 登记场景组 |
|---|---:|---:|---:|---:|---:|
| train | 166 | 83 | 249 | 2602 | 22 |
| val | 20 | 14 | 34 | 330 | 11 |
| test | 15 | 15 | 30 | 304 | 11 |
| 未采用 | 0 | 86 | 86 | — | 13 |

PASS 的条件为重算后的类别 1–5 各 2 个、类别 0（non_catalogue）为 0；否则 FAIL。S/H/N/W/B 是目录外干扰分析标签，不是检测类别，也不进入 HOG 输入或标准化。验证集中确认的空场景 FAIL 对应 0 字节 YOLO TXT，是合法样本。

## 交叉检查

- 文件数、字节、Git blob、大小与 provider `split_manifest.csv` 的 image/json SHA256 逐项一致。
- `verify_dataset.py` 在开发范围与全量范围分别通过，分别核验 283 / 415 组三联文件。
- 训练/验证/测试的 `scene_group`、`source_scene_group`、图片 SHA256 与解码像素 SHA256 两两交集为 0。
- LabelMe 尺寸、矩形边界、类别计数、YOLO 五字段与归一化框一致；增强父图字段与父图一致，父图为 11/20 训练 FAIL。
- 开发审计共 54 项检查全部通过；全量审计在冻结后完成并再次通过。

## 行尾与哈希口径

Git 对 428 个文本文件按 LF 存储，Windows 检出为 CRLF；provider `split_manifest.csv` 中 image/json 的 SHA256 锁定的是 CRLF 物理字节，415 个 JSON 在仓库中也以 CRLF 存储。执行时同时核验：

1. Git blob（文本按 CRLF→LF 规范化）与 tree 中记录的 blob SHA/大小一致；
2. provider 锁定的 image/json SHA256 与磁盘原始字节一致；
3. 13 个元数据文件的规范化 SHA256 与规划审计值一致。

这一区分记录在 `outputs/member04_hog/final_v4/audit/dataset_audit_development.json` 的 `eol` 字段中。

## 与成员4旧实验的关系

按 SHA256 匹配，旧 310 张原图在新总账中全部找到，PASS/FAIL 标签未改变；旧 train 185 张中 172 张进入新 train、13 张进入新 val；旧 valid/test 有部分进入新 train/val/test；86 张未采用 FAIL 包含旧的 6/7 干扰件图。新 test 含 4 张旧 valid、6 张旧 test 和 20 张此前不在成员4实验集的图，因此仍属历史接触，不能称为完全未见的独立测试。旧成绩只保留在归档，不进入新主报告。

## 已知限制

- 场景组字段互不重叠，但物理场景独立性没有独立证据。
- test 的 11 张未见件图同时含缺件和已见件；HOG 只输出整图判定，不能据此宣称陌生件定位或分类成功率。
- 数据划分、标签来源和拍摄条件由成员3维护；成员4不修改划分、标签或他人目录。
