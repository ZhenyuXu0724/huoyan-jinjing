# 机械零件视觉检测 · 最终版训练数据集 v4

三种基线共用本目录中的划分。下载本分支后，`images` 中就是原图和同名 LabelMe JSON，`labels` 下是同一张图对应的 YOLO TXT。文件按分区和 PASS / FAIL 归档。所有路径均相对于此目录。

| 分区 | PASS | FAIL | 原图合计 | 用途 |
|---|---:|---:|---:|---|
| images/train | 166 | 83 | 249 | 拟合模型 |
| images/val | 20 | 14 | 34 | 选模型、阈值 |
| images/test | 15 | 15 | 30 | 冻结模型后评估一次 |
| images/train_augmentations | 0 | 102 | — | 仅训练可选，均来自训练原图 |

共 313 张入选原图、102 张可选增强图，配有 415 份原始 JSON、415 份 YOLO TXT。例：`images/train/PASS/IMG_7174.png` 与同目录 `IMG_7174.json`、`labels/train/PASS/IMG_7174.txt` 对应。验证中已确认的空场景有一份零字节 TXT。第一轮统一只用 249 张训练原图；如将全部离线增强图等权加入，实际为 166 PASS、185 FAIL。

## 找标签和来源

- `原图明细.csv` 逐张列出 399 张原图的拍摄批次、去向、判定原因、场景组、正常件数量和目录外零件。86 张未用 FAIL 的文件路径为空。
- `测试集逐图核对.csv` 覆盖 30 张测试原图，列出已见与未见干扰件。此表供冻结后的评价使用。
- `split_manifest.csv` 记录全部 399 张原图和 210 张离线增强图。按 `usage=primary` 选正式原图，`optional_train_aug` 为可选训练增强。其余记录仅作追溯。清单含原始位置、v2/v3 去向、源文件大小、SHA-256、Git blob、增强父图、图像及两种标注路径。
- `classes.json` 定义六个检测类。0 为所有目录外零件的公共类别，1–5 依次为 M4×25、M5×10、M4×10、目标六角螺母、目标平垫圈。图像级 PASS 要求每种目标件各 2 个且无目录外零件。S/H/N/W/B 仅是干扰件分析用的细分字段。

## 分布与隔离

| 原图来源 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 11/11 | 143 | 8 | 9 |
| 11/14 | 21 | 5 | 1 |
| 11/18 | 44 | 8 | 0 |
| 11/20 | 34 | 6 | 15 |
| 11/23 | 7 | 7 | 5 |

训练和验证均覆盖五个批次，但背景与标签仍有关联。训练、验证、测试分别使用 22、11、11 个独立登记场景组；登记组、候选关联和增强父图无跨集。测试有 4 类干扰件，其中目录外螺钉 S（9 张）、黄铜色六角长柱 H（3 张）均未进入训练或验证，按种类计为 2/4。目录外螺母 N 和垫圈 W 为已见，轴承 B 仅出现在开发数据和未用数据中。11 张含未见件的测试图都摄于 11/20，也都含缺件和已见垫圈；评估陌生件定位时应另看其框。11 张新测试图曾在 v2 的训练集，16 张曾在 v3 的训练集，请用新的训练运行比较各基线。

## 训练输入

先在数据目录运行 `python verify_dataset.py`。默认核验 249+34 张开发原图的图片、原始标注 SHA-256、YOLO 框、类别计数及基线清单。完整交付验收可运行 `python verify_dataset.py --all`。训练脚本不会自动合入 102 张增强图。

### HOG

`manifests/hog/train_pass.csv` 供默认单类自编码器读取，`train.csv` 供监督分类读取。`val.csv` 和 `test.csv` 分别含 34、30 张。所有图片路径均相对清单所在目录。成员 4 的训练代码和依赖见 [HOG 实现](https://github.com/ZhenyuXu0724/huoyan-jinjing/tree/member04/hog-v3-stability-capacity/members/member04_hog)。从含该实现的代码仓库根目录运行，以下 `DATA` 表示此数据目录的实际路径。

```sh
python -m members.member04_hog.train --train "DATA/manifests/hog/train_pass.csv" --valid "DATA/manifests/hog/val.csv" --out outputs/hog_v4 --weights weights/hog_v4
```

历史监督版程序绑定了旧版划分、权重和审计记录；运行监督实验时使用本版 `train.csv` 与 `val.csv`，重新建立实验记录。

### Frozen ResNet18

`manifests/resnet/train.csv` 和 `val.csv` 采用成员 6 加载器的五列格式，标签 PASS=0、FAIL=1。另存的 `test.csv` 供最终评估。先安装成员 6 的依赖；`CODE` 指含 [成员 6 ResNet 实现](https://github.com/guoyu1106/member06/tree/520d17c94d385f0d031791a6bb7c9991880830e1/members/member06_research) 的代码仓库。随包的 v4 入口读取真实照片、复核完整性，并复用该成员的模型、预处理、训练轮次实现和验证损失选型。首次训练需要 torchvision 的 ImageNet 预训练权重。旧版 `team_data.py` 面向早期 PNG 专用交接格式；此处的 v4 入口直接读取冻结清单，未代签旧版人工审核记录。

```sh
python train_resnet_v4.py --code-root "CODE" --check-only
python train_resnet_v4.py --code-root "CODE" --output runs/resnet_v4
```

### YOLOv8n

`labels/{train,val,test}/{PASS,FAIL}` 中每一行是 `class_id x_center y_center width height`，坐标按原图宽高归一化。输入共 6 个类别。训练/验证/测试原图分别有 2,602 / 330 / 304 个框。可选训练增强另有 972 个框。`data.yaml` 提供相对目录和类别；成员 5 的通用入口要求绝对路径，运行以下第一步可生成与当前机器对应的 YAML。后两条命令从含 [成员 5 检测代码](https://github.com/ZhenyuXu0724/huoyan-jinjing/tree/member05/detector-foundation/members/member05_detector) 的仓库根目录执行，替换 YAML 和初始权重路径。

```sh
python verify_dataset.py --yolo-yaml data.local.yaml
python -m members.member05_detector.train --data "DATA/data.local.yaml" --check-only
python -m members.member05_detector.train --data "DATA/data.local.yaml" --weights "/path/to/yolov8n.pt" --name v4
```

源数据提交为 `755e038cc8f05b9701cceb28e3381d5fa3bc6a81`，队友 v2 对照提交为 `573000b1ab77571b1752fb075c58bc3b1092a636`。同名 LabelMe JSON 和照片原样保留，包括历史标注工具写入的 flags 与 description。YOLO 框由原始矩形逐框转换。训练环境和通用预训练权重由相应基线提供。
