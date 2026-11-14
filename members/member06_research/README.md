# 成员 6：Frozen ResNet18 + Linear 与 Stage 3 工具

工程已实现，**正式 baseline2 尚未训练/冻结**。本轮只使用程序生成的隔离示例数据；ImageNet 预训练权重真实可用，示例训练结果仍为 `SMOKE_ONLY`。详见 [验收记录](evidence/ACCEPTANCE.md) 和 [中文学习指南](MEMBER6_LEARNING_GUIDE.md)。

遵循组长 [v3.0](../member01_lead/PLAN.md)、[协作规范](../../CONTRIBUTING.md) 和 [公共接口 v0.1](../../shared/CONTRACT.md)。仅改本目录；`final_model.py` 仍未实现。没有接触正式测试集，没有提交、推送或修改远程仓库。

## 目录

```text
member06_research/
  baseline2.py               公共 predict / predict_batch，延迟加载
  config.py                 严格配置与数量向量校验
  image_io.py               与模型后端无关的 RGB 读图
  configs/baseline2.json     正式 Train/Val 配置模板（待填数据）
  data.py                   图像、letterbox、清单、跨集重叠检查
  model.py                  FrozenResNet18、二分类头、BN 冻结
  runtime.py                种子、设备、版本/源码哈希、checkpoint 加载
  train.py                  训练、验证选型、最佳 checkpoint、日志
  validate.py               使用 checkpoint 设置重新验证
  adapter.py                与 CNN 解耦的分类协议和数量规则
  inference.py              ResNet 后端、单图/批量 JSON CLI
  research_tools.py         ROI、局部 CNN、轮廓、像素几何、上游协议
  research_demo.py           工具演示，不执行融合决策
  smoke_data.py             生成合成示例图和 Train/Val 清单
  plot_history.py           训练曲线和验证混淆矩阵
  tests/test_member06.py     离线单元和端到端测试
  evidence/                 实跑记录、依赖快照、验收状态
  requirements.txt          直接依赖固定版本
  MEMBER6_LEARNING_GUIDE.md  中文教学文档
  progress.md               团队时间线和本轮交接
```

本地 `.venv/`、`data/`、`runs/` 被已有 `.gitignore` 排除；图像和权重不进入 Git。示例通过生成器交付，无需提交媒体文件。正式清单和逐图验证结果可能包含非公开信息，保持在被忽略的本地目录。

## 环境

所有命令从 Fork 仓库根目录执行：

```bash
cd /home/grace/projects/AI/member06
# 本轮已建好 .venv；重新部署时用 Python 3.11 创建：
python3.11 -m venv members/member06_research/.venv
M6_PY=members/member06_research/.venv/bin/python
$M6_PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
$M6_PY -m pip install -r members/member06_research/requirements.txt
$M6_PY -m pip check
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
```

本机 Python 3.11 路径为 `/home/grace/miniconda3/envs/jupyter-env/bin/python`，可代替 `python3.11`。默认 Python 3.14 不用于本交付。全量 CPU 版本见 [environment.freeze.txt](evidence/environment.freeze.txt)；严格复建可用：

```bash
$M6_PY -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r members/member06_research/evidence/environment.freeze.txt
```

`device=auto` 优先 CUDA，否则 CPU。当前安装 CPU wheel，CUDA 计算未实测；GPU 部署由组长按驱动选择匹配的官方 torch/torchvision CUDA wheel。请求不可用 CUDA 会明确失败。

首次使用 `IMAGENET1K_V1` 会下载官方权重（约 45 MB），导入模块和重载 checkpoint 不下载。网络失败返回 `PRETRAINED_WEIGHTS_UNAVAILABLE` 并保留原始异常，不静默降级。本轮沙箱 DNS 失败后通过获批网络访问成功下载；来源和 SHA256 见验收记录。离线可把同一经校验文件放到 `$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth`。

## 从零运行示例流水线

使用新目录 `smoke_demo`；已有目录拒绝覆盖，重复实验请换名字。示例 Train=8、Val=4，是图形和噪声组成的合成图片，没有真实零件标签意义。

```bash
$M6_PY -m members.member06_research.smoke_data --output members/member06_research/data/smoke_demo
# 随机 backbone，只能验证结构：
$M6_PY -m members.member06_research.train --config members/member06_research/data/smoke_demo/smoke_config.json --output members/member06_research/runs/random_demo
# 真预训练 backbone + 224x224，数据仍是合成数据：
$M6_PY -m members.member06_research.train --config members/member06_research/data/smoke_demo/pretrained_smoke_config.json --output members/member06_research/runs/pretrained_demo
$M6_PY -m members.member06_research.validate --checkpoint members/member06_research/runs/pretrained_demo/best.pt --manifest members/member06_research/data/smoke_demo/val.csv --allow-smoke --output members/member06_research/runs/pretrained_demo/reloaded_val.json
$M6_PY -m members.member06_research.inference --checkpoint members/member06_research/runs/pretrained_demo/best.pt --image members/member06_research/data/smoke_demo/val/synthetic_0.png --expected-counts '[2,2,2,2,2]' --allow-smoke --output members/member06_research/runs/pretrained_demo/single.json
$M6_PY -m members.member06_research.inference --checkpoint members/member06_research/runs/pretrained_demo/best.pt --input-dir members/member06_research/data/smoke_demo/val --allow-smoke --output members/member06_research/runs/pretrained_demo/batch.json
$M6_PY -m members.member06_research.plot_history --run-dir members/member06_research/runs/pretrained_demo --output members/member06_research/runs/pretrained_demo/curves.png
```

默认拒绝加载 SMOKE_ONLY checkpoint，示例须加 `--allow-smoke`。本轮实跑保留在 `data/acceptance/` 和 `runs/pretrained_final/`；可查看其中 `curves.png`、`best.pt`、`predictions.json`、`research.json`。这些是本地工件，复现者通过上述命令重新生成。

## 正式数据接入

由成员 3 提供按**场景**冻结的 Train/Val，本工具不重新分区。PASS/FAIL 真值须由团队按最终数量和目录规则确定，不能从文件名、候选预测或模型预测生成标签。

每份 UTF-8 CSV 恰含以下列，图像路径相对 CSV 所在目录，支持绝对路径：

```csv
image_path,label,scene_id,split,data_kind
train/scene001.jpg,PASS,scene001,train,REAL
train/scene002.jpg,FAIL,scene002,train,REAL
```

Val 文件的 split 全为 `val`；每份至少有 PASS/FAIL 两类。检查跨集路径、RGB 像素哈希、scene_id 重叠，拒绝错误标签、缺失图、重复路径和合成/正式混用。检查无法识别故意误标的测试数据、近重复帧或错误 scene_id；分区授权和关联场景管理仍由成员 3 负责。训练/选型入口不接受 test 角色。

复制 `configs/baseline2.json` 到本地配置，修改 `train_manifest` 和 `val_manifest`；配置内这两个路径相对**启动时仓库根目录**。显式 `class_to_idx={"PASS":0,"FAIL":1}` 可以交换编号，FAIL 始终是 P/R/F1 正类；这与 shared 中对象类别 ID 0–5 是不同概念。

保持 `weights=IMAGENET1K_V1`、`data_kind=REAL`。Pilot 后把确认的数量规则写入新实验配置；改配置不会让旧模型获得新的数量适应能力，需要相应数据/重训/验证。不要修改已有 checkpoint 的配置来绕过检查。

```bash
$M6_PY -m members.member06_research.train --config members/member06_research/configs/baseline2.json --output members/member06_research/runs/real_v1
$M6_PY -m members.member06_research.validate --checkpoint members/member06_research/runs/real_v1/best.pt --manifest data/member06/val.csv --output members/member06_research/runs/real_v1/reloaded_val.json
$M6_PY -m members.member06_research.inference --checkpoint members/member06_research/runs/real_v1/best.pt --input-dir data/member06/new_images --counts-file data/member06/counts.json --output members/member06_research/runs/real_v1/predictions.json
```

真实数据命令是接入模板，当前缺数据，未运行。counts.json 使用公共协议列表，每项恰有 image_id 和 expected_counts（字符串键 1–5 的非负整数），与输入文件一一对应，不含 GT。也可用 `--expected-counts` 为整批指定向量，两者互斥。目录推理不递归，只读取支持后缀图片，坏图保留 INPUT_ERROR。

## 训练与保存规则

预处理可选 letterbox 或 stretch，均不裁掉边缘；默认等比缩放填边、ImageNet mean/std，无随机增强。小零件仍可能因缩小损失细节，分辨率只能用 Train/Val 选择并在测试前冻结。不做 EXIF 旋转：数据负责人应统一方向，框对应存储栅格坐标。

训练使用 CrossEntropyLoss + Adam，仅优化 head，backbone 参数和 BN 状态均冻结。最小 Val loss 选择 best.pt，相同保留早期 epoch。决策为 softmax argmax，不调阈值；分数相同时选择编号 0。

best.pt 含完整模型状态、optimizer 状态、配置、epoch、验证指标和元数据。加载使用严格 state_dict 和 weights_only=True，仅加载可信工件。未提供断点续训，不能把 optimizer 字段当成已实现恢复训练。

| 工件 | 内容 |
|---|---|
| config.json | 标签、预处理、数量规则、设备、种子等全部设置 |
| run_config.json | Python/依赖/设备、源码与数据哈希、预训练来源、选型规则 |
| history.json / history.csv | 每 epoch Train/Val loss、accuracy；JSON 另含混淆矩阵和 FAIL P/R/F1 |
| best_metrics.json | 最佳 epoch 与混淆矩阵（行真值，列预测，class index 顺序） |
| best_val_predictions.json | Val 场景、路径、目标编号、预测编号、softmax 向量 |
| checkpoint_manifest.json | best.pt SHA256 |

P/R/F1 的分母为零时记录 null。种子控制 Python、NumPy、torch、DataLoader/worker，启用确定性算法；本轮同环境 CPU 重复训练一致。不同硬件、依赖或批大小可能产生浮点微差，不承诺跨平台逐位相同。

## 成员 7 适配接口

```python
from members.member06_research.inference import create_adapter
adapter = create_adapter("members/member06_research/runs/real_v1/best.pt")
one = adapter.predict("scene.jpg", expected_counts={str(i): 2 for i in range(1, 6)})
many = adapter.predict_batch(["a.jpg", "b.jpg"], expected_counts=[2, 2, 2, 2, 2])
```

ClassificationAdapter 依赖 ClassifierBackend.score_batch、类别编号及证据状态，后端可替换，数量规则不进入 ResNet18。批量共用一组数量；逐图不同要求用 predict 或 CLI 的 counts-file。不匹配数量在读图/模型推理前返回 UNSUPPORTED_COUNTS，非法数量为 INPUT_ERROR。未传数量使用 checkpoint 固定规则，并输出 counts_source=configured_fixed_rule，不表示从图像测得数量。

成功字段为 status、predicted_label、prediction_score、score_type、class_scores，另附证据状态和数量规则。prediction_score 是**预测类别**的 uncalibrated_softmax，不是校准后的正确概率。REAL checkpoint 状态为 OK，其 evidence_status=TRAIN_VAL_ONLY 不表示已正式验收；示例为 SMOKE_ONLY。失败时判定和分数为 null。

公共 baseline2.predict(image_path) 延迟加载环境变量 MEMBER06_CHECKPOINT；MEMBER06_DEVICE 可为 auto/cpu/cuda。未配置返回 MODEL_NOT_CONFIGURED 和 null，此扩展状态待成员 7/组长确认。公共入口不开放示例 checkpoint；导入模块不下载或训练。

根 predict.py 路由保留，批量导出原本就是 NOT_IMPLEMENTED，由成员 7 完成。本目录提供 JSON CLI，未擅改 UI/CSV 导出。图像分类器不生成 bbox、实际数量、objects.csv 或 counts.csv。

## Stage 3 基础工具

```bash
$M6_PY -m members.member06_research.research_demo --image members/member06_research/data/smoke_demo/geometry_rectangle.png --bbox 0 0 160 100 --checkpoint members/member06_research/runs/pretrained_demo/best.pt --allow-smoke --output members/member06_research/runs/pretrained_demo/research.json
```

crop_roi 接收原图半开 xyxy，拒绝反向、空、非数和越界坐标；小数框左上向下、右下向上取整，不静默截断。Preprocess 可用于 ROI 的 CNN 输入。LocalFeatureExtractor.extract 返回冻结 backbone 的 512 维向量，不调用整图头，不声称是局部零件类别。

segment_contours 提供 Otsu/固定阈值、dark/light 前景、最小面积。geometry_features 在**缩放前 ROI**上计算最小外接矩形、长宽、面积、周长、长宽比，坐标相对 ROI。示例矩形约 100×30 px；这是像素中心轮廓间距，未必等于占据像素数量。多轮廓、触边、空白、异常分别返回 AMBIGUOUS_CONTOURS、BORDER_TOUCHING、NO_CONTOUR、INPUT_ERROR，特征为空。阈值工具有光照/阴影局限，不能代替尺寸可行性实验。

CandidateProvider.propose(image) 是成员 5 预留协议，不接 expected_counts；ScaleCalibration.pixel_to_metric(...) 是成员 2 预留协议，均未接入真实实现。只有 px/px² 和 NOT_CALIBRATED，没有毫米或尺寸合规结论。不执行 CNN+几何融合、候选重分类、正式对照或消融决策。

## 测试

```bash
$M6_PY -m unittest discover -s members/member06_research/tests -v
$M6_PY -m unittest discover -s tests -v
$M6_PY predict.py --list-models
```

测试默认不联网：随机权重用于结构/梯度/流水线校验，预训练失败分支使用显式错误测试；官方预训练有单独实跑证据。最终测试数量和记录见 [ACCEPTANCE.md](evidence/ACCEPTANCE.md)。

## 待上游提供

| 负责人 | 缺失项 | 影响 |
|---|---|---|
| 组长/全组 | Pilot 后固定数量、预处理/评价协议、接口版本及状态扩展确认 | 模板仍为 PROVISIONAL，不正式冻结 |
| 成员 3 | 场景隔离的真实 Train/Val、整图真值定义、清单版本/许可、封存测试流程 | 正式训练/选型/指标未完成 |
| 成员 4 | 统一图像评价器、FAIL 正类、弃判/覆盖率口径 | 内部指标不能冒充共同评价结果 |
| 成员 2 | 相机设置、标定版本/误差、尺寸可辨观察 | 无毫米或真实规格判定 |
| 成员 5 | 候选格式确认、冻结 baseline3、真实错误清单 | Stage 3 方向/对照/消融待决策 |
| 成员 7 | UI、根 predict.py 批量 CSV 接入、错误状态验收 | 独立接口可测，统一应用待集成 |
| 组长/课程 | 仓库未附原始 Project-Man.pdf，需课程端核实最终提交要求 | 当前依可访问 PLAN v3.0 实施 |

## 贡献与资料来源

本轮在 AI 辅助下完成工程和本地验证，未生成正式科学结论。torchvision 提供 ResNet18 和 ImageNet 权重；公开可下载不表示获得第三方数据/权重再分发授权，权重不入 Git。

- [torchvision ResNet18](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)：模型和权重；本工程为保留桌面全视野采用 letterbox，没有沿用中心裁剪。
- [PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)：冻结特征提取器和替换分类头。
- [OpenCV 轮廓几何](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)：面积、周长、最小外接矩形。
- [ResNet 原论文](https://arxiv.org/abs/1512.03385)：基础阅读，不视为 Stage 3 改进依据。

正式文献到算法的对应、研究假设和消融方案，须在真实 baseline3 失败分析后由团队冻结。
