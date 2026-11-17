# 成员 6 学习指南：从图像到可复现的 PASS/FAIL 基线

2026-11-14 更新：真实数据接入与当前可执行步骤请优先阅读 [TRAINING_GUIDE_CN.md](TRAINING_GUIDE_CN.md)。下文的合成数据例子用于学习；11-05 的验收是历史 SMOKE_ONLY 证据。当前真实数据缺可信双类标签、场景/规格与数量批准，尚未训练。

这份指南对应本目录的实际代码。现有实跑全部是 **SMOKE_ONLY**；工程能训练不等于已经学会检查真实机械零件。先读懂和复现工程，再接成员 3 的正式 Train/Val。

## 1. 先知道每个文件负责什么

| 文件与代码位置 | 读它时关注的问题 |
|---|---|
| `config.py:Config`、`configs/baseline2.json` | 类别编号、图像大小、数量规则、种子、数据和训练参数从哪里来？ |
| `image_io.py:read_image`、`data.py:Preprocess` | RGB 图像如何变成归一化的 `[3,H,W]` tensor？ |
| `data.py:ManifestDataset/check_disjoint` | 标签和 scene_id 从哪里读？如何拒绝跨集重叠？ |
| `model.py:FrozenResNet18`（第 7 行起） | backbone、512 维特征和 Linear 头如何连接？ |
| `runtime.py:seed_everything/choose_device/load_checkpoint` | 如何固定随机性、选 CPU/GPU、恢复同一个模型？ |
| `train.py:run_epoch`、`fit`（第 69 行起） | 梯度在哪更新？最佳模型依据是什么？ |
| `validate.py:main` | 如何用保存的配置重新计算 Val，而非重新训练？ |
| `adapter.py:ClassifierBackend/ClassificationAdapter` | UI 能调用什么？哪些输入必须拒绝？ |
| `inference.py:ResNetBackend/create_adapter/main` | 模型如何批量生成分数，并经适配器输出？ |
| `baseline2.py:predict/predict_batch` | 公共注册入口如何延迟加载环境变量指定的 checkpoint？ |
| `smoke_data.py:generate` | 如何生成确定性的示例图和清单，而不伪造真实数据？ |
| `plot_history.py:main` | 如何画 Train/Val 曲线和最佳验证混淆矩阵？ |
| `research_tools.py:crop_roi/segment_contours/geometry_features/LocalFeatureExtractor` | 如何测试 ROI、像素几何和局部 CNN？ |
| `research_demo.py:main` | 如何用 CLI 组合工具，且不输出研究模型判定？ |
| `tests/test_member06.py` | 如何用实际参数更新、重载与 CLI 证明工程工作？ |
| `evidence/` | 原始测试日志、环境快照、预训练验收与状态，不是正式研究结果 |
| `requirements.txt`、`README.md`、`progress.md` | 安装、操作、交接和团队时间线 |
| `final_model.py` | 团队原有占位入口，未实现；须等待 Stage 3 决策 |

数据流为：CSV → Dataset → DataLoader → 完整视野预处理 → 冻结 ResNet18 → 512 维向量 → Linear(512,2) → logits。训练时 logits 进入 CrossEntropyLoss，推理时 logits 经 softmax 得到分数。

## 2. ResNet18 和 512 维特征

打开 `model.py:8–24`。torchvision 的 ResNet18 先用 7×7 卷积和池化降低空间分辨率，再经过四组残差块。每组两个 BasicBlock，每块两个 3×3 卷积；通道依次为 64、128、256、512。残差连接把块的输入加回输出，必要时用投影匹配形状，帮助深层网络优化。

“18 层”按常用的有权重层计数：起始卷积 1 层 + 8 个块×2 层卷积 + 最后全连接 1 层；不是说模型只有 18 个 PyTorch 子模块。

对于 `[N,3,224,224]` 输入，最后卷积特征约为 `[N,512,7,7]`。全局自适应平均池化把每个通道的空间网格求平均，得到 `[N,512,1,1]`，展平为 `[N,512]`。512 来自最后一组卷积通道数，不是图片数量或零件类别数。

原 ImageNet 分类器是 Linear(512,1000)。本代码读取 `backbone.fc.in_features`，确认是 512，把原 fc 替换为 Identity，然后单独建立 `head=nn.Linear(512,2)`。`extract_features()` 返回 512 维；`forward()` 才调用整图二分类头。

64×64 的随机权重冒烟测试也能得到 512 维，因为自适应池化会处理变化后的空间尺寸。正式输入大小仍须在配置中冻结，不能据此认为所有分辨率效果相同。

## 3. 为什么冻结，Linear 头怎样训练

预训练网络已从 ImageNet 学到通用外观特征。课程数据较少时，只训练新头能减少待学习参数，降低训练成本，也为三基线比较提供清晰起点。这不保证数据域适配或错规格识别能力，效果只能用真实数据验证。

代码中有三层控制：

```python
# model.py:21–24
self.backbone.fc = nn.Identity()
self.backbone.requires_grad_(False)
self.head = nn.Linear(feature_dim, 2)
self.backbone.eval()
```

只设置 requires_grad=False 不够。BatchNorm 在 train 模式下会更新 running_mean/running_var，即使参数没有梯度，backbone 仍在变化。因此 `FrozenResNet18.train()`（第 26 行）在父类切换训练状态后，再强制 backbone.eval()。Linear 头仍处于 train 模式。`extract_features()` 用 no_grad，进一步避免保存 backbone 的反向图。

每张图产生向量 x，头计算 `logits = W @ x + b`。W 的形状是 `[2,512]`，b 为 `[2]`，共 1026 个可训练参数。标签默认 PASS=0、FAIL=1；它不是 shared 中的零件类别 ID。

`train.py:run_epoch` 对每批执行：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)                  # [N,2]
loss = nn.CrossEntropyLoss()(logits, labels)  # labels 是 int64 [N]
loss.backward()
optimizer.step()                       # optimizer 只包含 model.head.parameters()
```

不要在 CrossEntropyLoss 前自己做 softmax。该损失直接接收 logits，内部采用数值稳定的计算。推理才调用 `logits.softmax(1)`。本模型用 argmax 判定，没有用测试集选择阈值，也没有概率校准。

## 4. 先用示例跑完整流程

本机环境已安装。打开终端，从 Fork 根目录运行：

```bash
cd /home/grace/projects/AI/member06
M6_PY=members/member06_research/.venv/bin/python
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
# 新名字避免覆盖本轮 acceptance 工件
$M6_PY -m members.member06_research.smoke_data --output members/member06_research/data/learn01
$M6_PY -m members.member06_research.train --config members/member06_research/data/learn01/smoke_config.json --output members/member06_research/runs/learn_random01
$M6_PY -m members.member06_research.train --config members/member06_research/data/learn01/pretrained_smoke_config.json --output members/member06_research/runs/learn_pretrained01
```

第一条训练使用随机权重、64×64；第二条使用真预训练权重、224×224，都是两轮合成数据冒烟。终端每轮输出类似 `SMOKE_ONLY epoch=1 train_loss=... val_loss=...`。实际数值查看 `evidence/pretrained_train.log`；不要把示例的 loss 或准确率写成真实实验结果。

最佳权重在 `runs/learn_pretrained01/best.pt`。验证只做前向计算，不 backward：

```bash
$M6_PY -m members.member06_research.validate --checkpoint members/member06_research/runs/learn_pretrained01/best.pt --manifest members/member06_research/data/learn01/val.csv --allow-smoke --output members/member06_research/runs/learn_pretrained01/recheck.json
```

重新部署机器先按 README 安装 Python 3.11 和依赖；不要使用本机默认的 Python 3.14 直接启动。

## 5. 单图、批量推理和数量规则

```bash
$M6_PY -m members.member06_research.inference --checkpoint members/member06_research/runs/learn_pretrained01/best.pt --image members/member06_research/data/learn01/val/synthetic_0.png --expected-counts '[2,2,2,2,2]' --allow-smoke --output members/member06_research/runs/learn_pretrained01/one.json
$M6_PY -m members.member06_research.inference --checkpoint members/member06_research/runs/learn_pretrained01/best.pt --input-dir members/member06_research/data/learn01/val --allow-smoke --output members/member06_research/runs/learn_pretrained01/batch.json
```

输出 JSON 的 results 有 image_id、status、predicted_label、prediction_score、score_type、class_scores 等。示例 status 是 SMOKE_ONLY。prediction_score 是被选中类别的未校准 softmax，不是“模型有这么高概率正确”。查看 FAIL 的分数用 class_scores.FAIL，不能把 PASS 的 prediction_score 当作 FAIL 分数。

把 expected-counts 改为 `[1,2,2,2,2]`，应得到：

```json
{"status":"UNSUPPORTED_COUNTS","predicted_label":null,"prediction_score":null,"score_type":"uncalibrated_softmax","supported_counts":[2,2,2,2,2]}
```

这是可复现的拒绝规则，不需要模型猜实际数量。整图头不能提供框或五类计数；UI 不应依据它虚构这些信息。[2,2,2,2,2] 仍是 Pilot 前暂定规则。

Python 调用示例：

```python
from members.member06_research.inference import create_adapter
adapter = create_adapter("members/member06_research/runs/pretrained_final/best.pt", allow_smoke=True)
print(adapter.predict("members/member06_research/data/acceptance/val/synthetic_0.png", [2]*5))
```

适配器可接收 PIL.Image 或图像路径。正式公共 baseline2.predict 从 MEMBER06_CHECKPOINT 加载，默认拒绝示例 checkpoint。根 predict.py 的 CSV 导出仍待成员 7，当前使用成员 6 独立 CLI。

## 6. 怎样换成真实训练数据

先取得成员 3 冻结的场景级 Train/Val，确认数量规则和整图真值。不要自行把相邻视频帧随机分到两个集合；同一场景增强后的图也应属于同一分区。测试集由成员 3 独立封存，不能传进训练/选型脚本。

2026-11-14 已有真实包：`members/member03_data/full_dataset_upload/datasets/parts_v1/`，Train=112、Val=24，但目前只有对象标注，没有可信整图两类标签和 scene_id。数量仍待确认，标注复核与平垫片规格尚未完成。使用新增 `team_data.py` 审查；取得真实批准和逐图审核后才转换为本地 `data/team_parts_v1/` 下的清单和 provenance。

当前可以运行不更新权重的检查：

```bash
$M6_PY -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only
```

现阶段会以 WAITING_UPSTREAM 拒绝训练，这是待批准状态。不能通过把合成清单标成 REAL 或把默认数量标成已确认来绕过。完整的审批交接、实际路径和由你手动执行的训练操作见 TRAINING_GUIDE_CN.md。图片按 scene_id 隔离，原划分保持；无法自动判断的近重复和错误场景归属仍需成员 3 管理。

默认 letterbox：先保持宽高比缩小/放大，再放到方形画布中填边，桌面两侧不会被中心裁掉。stretch 也保留整图，但会改变长宽比。`data.py:Preprocess` 的 mean/std 与 padding 都可配置。几何长度必须在缩放前原图/ROI 测量，不能用 letterbox 后的像素当原图尺寸。

## 7. 看曲线、日志和过拟合

```bash
$M6_PY -m members.member06_research.plot_history --run-dir members/member06_research/runs/learn_pretrained01 --output members/member06_research/runs/learn_pretrained01/curves.png
```

图片有 Train/Val loss、accuracy 和最佳 Val confusion matrix。也可直接打开 CSV：

```bash
$M6_PY -c 'import csv; print(list(csv.DictReader(open("members/member06_research/runs/pretrained_final/history.csv"))))'
```

`history.json` 记录每一轮，`best_metrics.json` 只记录 Val loss 最小的一轮。不要把最后一轮当成最佳模型。矩阵行是真实类别、列是预测类别，顺序按配置编号：默认 `[PASS,FAIL]`，所以左下角是实际 FAIL 却预测 PASS，要特别检查。

如果训练 loss 持续下降、验证 loss 开始上升，且验证错误增多，是过拟合的信号。也要排查分区泄漏、类别不平衡、标注错误和分布差异，不能只看训练准确率。小 Val 上几张图片就能造成很大波动；保存逐图错误、按 scene_id 检查，再依据 Train/Val 比较学习率、正则、epoch 等。实验配置要事先记录，不得用封存测试集反复挑选方案。

`best_val_predictions.json` 保存图像路径、场景、目标和预测编号，可找到具体错误。`fail_precision/recall/f1` 分母为零时是 null，应解释未定义原因；不能为好看而改成 1。

本轮只有合成数据和两轮冒烟，不能用它判断正式模型是否过拟合或哪种研究改进有效。

## 8. Stage 3 工具怎样使用

### ROI：先明确坐标

```python
from members.member06_research.research_tools import crop_roi, geometry_features
crop = crop_roi("members/member06_research/data/acceptance/geometry_rectangle.png", [0,0,160,100])
print(crop["status"], crop["raster_xyxy"])   # OK (0, 0, 160, 100)
geo = geometry_features(crop["roi"], foreground="dark")
print(geo["features"]["length_px"], geo["features"]["width_px"])  # 100.0 30.0
```

xyxy 是原图左上角为原点、x 向右、y 向下，半开区间 `[x1,x2) × [y1,y2)`。整个 160×100 图的框为 `[0,0,160,100]`。负坐标、超界、x2≤x1、NaN、bool 都拒绝；不悄悄截断。小数框向外取整，结果附实际 raster_xyxy，方便追溯。

### 几何：阈值、轮廓和像素

`segment_contours` 先转灰度，再用 Otsu 或指定 threshold 二值化；dark 表示暗零件亮背景，light 相反。findContours 提取外轮廓，按 min_area_px2 去除过小区域。单个有效且不触边的轮廓才交给 minAreaRect 计算最小外接矩形。

返回长边 length_px、短边 width_px、面积 px²、周长 px、长宽比和 ROI 内矩形角点。100×30 是本合成矩形的像素轮廓跨度，不是毫米，不是一个真实螺丝规格。空白 NO_CONTOUR、多轮廓 AMBIGUOUS_CONTOURS、触边 BORDER_TOUCHING、异常 INPUT_ERROR 均返回 features=None。

```python
from PIL import Image
print(geometry_features(None)["status"])                          # INPUT_ERROR
print(geometry_features(Image.new("L", (50,50), 255))["status"])   # NO_CONTOUR
```

阈值结果会受光照、反光、阴影和背景影响；不能把每个轮廓都当作一个零件。无相机标定、透视与误差分析时，不能换算毫米。

### 局部 CNN：512 维描述，不做整图头的局部分类

```python
from members.member06_research.runtime import load_checkpoint
from members.member06_research.research_tools import LocalFeatureExtractor
model, cfg, device, _ = load_checkpoint("members/member06_research/runs/pretrained_final/best.pt", allow_smoke=True)
extractor = LocalFeatureExtractor(model, cfg, device)
features = extractor.extract("members/member06_research/data/acceptance/geometry_rectangle.png", [0,0,160,100])
print(features["status"], features["dimension"])  # SMOKE_ONLY 512
```

这个向量不输出 PASS/FAIL；整图头没经过局部类别训练，不能直接套在 ROI 上做对象分类。CNN 向量也不是毫米尺寸。

一条命令同时演示：

```bash
$M6_PY -m members.member06_research.research_demo --image members/member06_research/data/acceptance/geometry_rectangle.png --bbox 0 0 160 100 --checkpoint members/member06_research/runs/pretrained_final/best.pt --allow-smoke --output members/member06_research/runs/learn_pretrained01/roi.json
```

成员 5 的 CandidateProvider 负责原图候选框，成员 2 的 ScaleCalibration 负责有版本和误差的标定转换；现在只是 Protocol，不能当成已接入。真实 baseline3 冻结并分析错误后，团队再决定唯一改进、固定对照、消融因素和评价口径。

## 9. 常见错误与排查

| 现象 | 检查与处理 |
|---|---|
| No module named torch / cv2 | 使用本目录 .venv/bin/python；按 README 安装，运行 pip check |
| torchvision 算子不匹配 | torch 与 torchvision wheel 必须匹配；不要混装 CPU/CUDA 来源 |
| PRETRAINED_WEIGHTS_UNAVAILABLE | 看原始 DNS/网络/证书异常；配置正确 TORCH_HOME 或放入校验过的缓存；随机检查须显式 SMOKE_ONLY |
| 训练时 BN running_mean 变化 | 不要绕过 FrozenResNet18.train；运行冻结测试，确认 backbone.eval() |
| expected targets long / label invalid | CSV 标签是 PASS/FAIL；Dataset 映射为 int64，别使用对象类别 ID |
| train/val overlap | 找出冲突 scene_id/路径/像素，交数据负责人核对；别靠换名字绕过 |
| output directory must be empty | 新建实验名，保留旧配置/结果；工具不覆盖已有实验 |
| SMOKE_ONLY checkpoint requires explicit... | 示例加 --allow-smoke；正式接入须使用真实数据模型，不能改 checkpoint 标记 |
| UNSUPPORTED_COUNTS | 当前模型只支持配置中的固定向量；不是 UI 调一次数量就能适配 |
| MODEL_NOT_CONFIGURED | 公共入口需要 MEMBER06_CHECKPOINT；可先用独立 inference CLI 验收 |
| NO_CONTOUR / BORDER_TOUCHING | 检查图像、dark/light、ROI 是否截断；不要把空特征补零后称测量成功 |
| 根 predict.py 仍报 NOT_IMPLEMENTED | 成员 7 批量导出尚未实现，本轮使用成员 6 CLI |
| CUDA requested but unavailable | 当前是 CPU wheel；用 cpu/auto；GPU 安装与硬件另行验收 |

自动检查：

```bash
$M6_PY -m unittest discover -s members/member06_research/tests -v
$M6_PY -m unittest discover -s tests -v
```

测试用真实 ResNet 的前向、梯度、状态比较和 checkpoint，不靠固定输出骗过模型验收。适配器协议测试中的小型假后端只验证路由/标签映射，不能充当模型效果证据。具体实测结果与 CUDA 跳过项见 `evidence/ACCEPTANCE.md`。
