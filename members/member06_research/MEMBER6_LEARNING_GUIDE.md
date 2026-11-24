> 当前操作全部使用真实Train/Val和ImageNet预训练权重。DEV_ONLY是真实数据开发实验；尚未具备正式冻结资格。完整操作顺序见 [TRAINING_GUIDE_CN.md](TRAINING_GUIDE_CN.md)。

# 成员 6 学习指南：从图像到可复现的 PASS/FAIL 基线

这份指南对应本目录的实际代码。真实数据已接入，下面的训练命令会更新分类头；训练由你执行，当前尚未产生真实模型成绩。历史合成验收只能说明工程可运行。

## 1. 先知道每个文件负责什么

| 文件与代码位置 | 读它时关注的问题 |
|---|---|
| `config.py:Config`、`configs/baseline2.json` | 类别编号、图像大小、数量规则、种子、数据和训练参数从哪里来？ |
| `image_io.py:read_image`、`data.py:Preprocess` | RGB 图像如何变成归一化的 `[3,H,W]` tensor？ |
| `data.py:ManifestDataset/check_disjoint` | 标签和 scene_id 从哪里读？如何拒绝跨集重叠？ |
| `model.py:FrozenResNet18`（第 7 行起） | backbone、512 维特征和 Linear 头如何连接？ |
| `runtime.py:seed_everything/choose_device/load_checkpoint` | 如何固定随机性、选 CPU/GPU、恢复同一个模型？ |
| `train.py:run_epoch`、`fit` | 梯度在哪更新？最佳模型依据是什么？ |
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
| `corpus_audit.py:audit_corpus/audit_flat` | 核对来源、原分区与可选新分区，310张计数不等于截图分区已确认 |
| `development.py:prepare_dev/verify_dev` | 带限制的开发标签与来源绑定，和正式prepare隔离 |
| `group_metrics.py:grouped_validation` | 从标注元数据分组评价，数量不来自CNN预测 |
| `tests/test_dev_tier.py` | 开发开关、禁止升级、场景冲突和分组指标的合成测试 |
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

## 4. 使用真实数据训练

从仓库根目录执行。默认配置 `configs/baseline2.json` 已绑定185张真实Train和101张真实Val，`data_kind=REAL`、`weights=IMAGENET1K_V1`、`training_tier=DEV_ONLY`。数据不来自合成生成器。

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export M6=members/member06_research
export TORCH_HOME="$PWD/$M6/.venv/torch"
export MPLCONFIGDIR="$PWD/$M6/.venv/matplotlib"
export M6_CONFIG="$M6/configs/baseline2.json"
python -u -m members.member06_research.train --config "$M6_CONFIG" --check-only
```

check-only会读取真实数据检查哈希、标签、分区和输入，不创建CNN或更新权重。通过后，由你启动实际训练：

```bash
python -u -m members.member06_research.train --config "$M6_CONFIG" \
  --allow-dev-only --output "$M6/runs/dev_baseline_v1"
```

这条命令默认训练10轮，每轮实际计算Train梯度、更新Linear Head、评估Val并保存最佳模型。日志应显示`DEV_ONLY epoch=...`，不是SMOKE_ONLY。开发标记表示场景/规格/固定规则尚未正式批准，不代表模拟训练。输出目录不得覆盖已有实验；重跑请换版本名并同步后续路径。

## 5. 重新验证、单张和批量推理

以下命令依赖你已经训练生成的 `runs/dev_baseline_v1/best.pt`；没有这个文件就先完成训练，不能替换成旧合成checkpoint。

```bash
export M6_VAL=$(python -c 'import json,os; print(json.load(open(os.environ["M6_CONFIG"]))["val_manifest"])')
python -m members.member06_research.validate --checkpoint "$M6/runs/dev_baseline_v1/best.pt" \
  --manifest "$M6_VAL" --allow-dev-only --output "$M6/runs/dev_baseline_v1/validation.json"
export M6_IMAGE=$(python -c 'import csv,os; print(next(csv.DictReader(open(os.environ["M6_VAL"]))) ["image_path"])')
python -m members.member06_research.inference --checkpoint "$M6/runs/dev_baseline_v1/best.pt" \
  --image "$M6_IMAGE" --expected-counts '[2,2,2,2,2]' --allow-dev-only \
  --output "$M6/runs/dev_baseline_v1/one.json"
```

M6_IMAGE取真实Val清单第一张，仅用于演示调用，不作为新的独立测试。批量调用可用下面Python接口，传入你要检查的真实图像路径列表：

```python
import os
from members.member06_research.inference import create_adapter
adapter = create_adapter("members/member06_research/runs/dev_baseline_v1/best.pt", allow_dev_only=True)
results = adapter.predict_batch([os.environ["M6_IMAGE"]], [2,2,2,2,2])
print(results)
```

返回status=DEV_ONLY、predicted_label、prediction_score和class_scores。softmax未经校准，prediction_score属于预测类别，FAIL分数请读class_scores['FAIL']。不输出实际件数或框。数量规则不同返回UNSUPPORTED_COUNTS；坏图属于INPUT_ERROR，不算零件FAIL。公共baseline2入口默认拒绝开发模型，UI开发演示使用显式适配器。

## 6. 更换真实训练数据

当前Train为133PASS/52FAIL，Val为24PASS/77FAIL。原图来自成员3，不能挪Val/Test补Train。`configs/data_sources.json`记录来源，`configs/nov18_sources.json`记录新批次文件哈希，`development.prepare_dev/verify_dev`生成并验证清单和来源。

新数据由成员3发布后，更新来源描述、重新audit、prepare-dev到新版本目录；使用新生成config，不手工改CSV标签或哈希来绕过检查。正式数据需要真实规则/标签/规格审核和scene_id后执行prepare。默认baseline2.json是当前真实开发配置，不是审批模板。

`data.py:Preprocess`的letterbox保留全图，不做中心裁剪；stretch保留全图但改变比例。实际像素几何在原始ROI上测量，不能把224缩放图的像素当成原图尺寸。场景未知时UNVERIFIED标记不等于独立摆放；同场景跨分区必须处理，不能改文件名绕过。

## 7. 曲线、错误与过拟合

```bash
python -m members.member06_research.plot_history --run-dir "$M6/runs/dev_baseline_v1" \
  --output "$M6/runs/dev_baseline_v1/curves.png"
python - <<'PYCODE'
import json
from pathlib import Path
run=Path('members/member06_research/runs/dev_baseline_v1')
groups=json.loads((run/'best_val_groups.json').read_text())
print(groups['evidence_status'],groups['overall'])
print(groups['groups']['non_catalogue_count'])
for row in groups['errors']:
    print(row['image_path'],row['label'],'->',row['predicted_label'])
PYCODE
```

history.json/csv保存各轮结果，best.pt和best_metrics.json对应最小Val CE，不一定是最后一轮。曲线含loss、accuracy与混淆矩阵。False PASS是真FAIL判PASS；False FAIL是真PASS判FAIL。FAIL precision/recall/F1分母为零时保留null。

Train loss下降而Val loss持续上升提示过拟合，也要检查标签和分布差异。当前PASS/FAIL与采集批次明显混杂，高分不能证明学会数量或规格判断。分组指标中的干扰数量来自标注，不是模型数出来的。反复改设置也会对Val过拟合；不能根据Test选择参数。

## 8. Stage3工具：使用真实图片和已有标注ROI

这些工具不构成最终研究算法。以下示例从第5节选中的真实图片读取Labelme第一个矩形，使用已有标注坐标，不编造检测框：

```python
import os, json
from pathlib import Path
from members.member06_research.research_tools import crop_roi, geometry_features, LocalFeatureExtractor
from members.member06_research.runtime import load_checkpoint
image = Path(os.environ['M6_IMAGE'])
annotation = json.loads(image.with_suffix('.json').read_text())
shape = next(s for s in annotation['shapes'] if s['shape_type'] == 'rectangle')
(x1,y1),(x2,y2) = shape['points']
bbox = [min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]
crop = crop_roi(image,bbox)
print(crop['status'])
if crop['status'] == 'OK':
    print(geometry_features(crop['roi'],foreground='dark'))
model,cfg,device,_ = load_checkpoint('members/member06_research/runs/dev_baseline_v1/best.pt',allow_dev_only=True)
features = LocalFeatureExtractor(model,cfg,device).extract(image,bbox)
print(features['status'],features.get('dimension'))
```

ROI使用原图xyxy坐标。越界或无效框返回明确失败状态；合法裁剪后，局部CNN做配置一致的缩放归一化，输出512维特征。OpenCV阈值、轮廓和最小外接矩形只提供像素尺度；背景、反光和遮挡可能导致几何失效，需检查状态，不能强行填0或称为毫米。局部CNN特征不直接等于局部类别判定。成员2标定和成员5候选接口仍待正式交接，不在此自行决定融合方向。

## 9. 常见问题

| 问题 | 处理 |
|---|---|
| 看到SMOKE_ONLY或synthetic路径 | 停止该次启动，检查是否选了旧合成config/checkpoint；当前使用configs/baseline2.json |
| DEV_ONLY | 真实数据开发模式，需显式allow-dev-only，不是模拟训练 |
| PRETRAINED_WEIGHTS_UNAVAILABLE | 检查缓存/网络真实错误，不切随机权重冒充预训练模型 |
| best.pt不存在 | 尚未真实训练或未完成首轮验证，先看控制台日志，不取旧示例模型替代 |
| source/hash/manifest changed | 源文件变化，重新审查准备新数据版本 |
| output already exists | 使用新实验名，不删除历史来覆盖 |
| CUDA不可用 | auto使用CPU；当前CPU环境正常可运行 |
| WAITING_UPSTREAM | 正式审批不足，不伪造记录；开发实验使用独立DEV_ONLY入口 |

`tests/`和`smoke_data.py`仍用于隔离自动测试，不是你日常训练流程的依赖输入。测试中用合成夹具才能验证错误路径，而不污染真实数据；其历史成绩不会转换为真实实验成绩。你只需按第4节启动当前真实配置。
