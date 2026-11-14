# 成员 5：第一次交付

本次为推理、训练入口和评价基础代码，不是完整 baseline3。尚无项目训练数据或已训练权重；尺寸拒绝、前景候选补查和真实识别效果未完成验收。禁止把测试用例结果当作课程实验指标。

## 文件与接口

- `model.py`：惰性加载本地权重；检查六类名称；输出原图 xyxy、类别和未校准分数。导入模块不下载、不训练；权重不存在明确报错。支持中文图像路径。
- `inference.json`：初始阈值，不是验证集选出的冻结参数。设置真实权重后可填 SHA256 校验。
- `train.py`：检查六类、图片与标签配对、归一化框范围、训练/验证路径，执行显式训练；不读取测试集。路径检查不能证明场景独立，仍需成员 3 的 scene_id 划分清单。
- `dataset.example.yaml`：项目 ID 0..5 与 YOLO 类索引保持一致。不要未经核验把公共数据类别硬映射成指定尺寸。
- `evaluation.py`：纯函数计算对象 P/R、0..5 计数 MAE、五类向量全对率与覆盖率；不被推理模块导入。
- `test_detector.py`：人工构造的程序测试。
- `check_environment.py`：真实 GPU 卷积前向和反向传播检查。

当前 `predict` 即使获得预测，也固定返回 `REVIEW_REQUIRED`。目标与目录外预测均是未经尺寸与候选覆盖验证的原型输出，不可用于自动 PASS 或正式成绩。失败状态的空列表也不是空场景证明。完成拒绝模块并通过验收后才能更改此策略；不提供配置开关绕过。组员 7 应显示原型提示并遵守失败/需复核状态。

## 环境与验证

在仓库根目录运行；独立环境使用 Python 3.11，不更改系统 Python。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r members/member05_detector/requirements.txt
python -m members.member05_detector.check_environment
python -m members.member05_detector.check_environment --architecture-smoke
python -m unittest members.member05_detector.test_detector -v
python -m unittest discover -s tests -v
```

若环境没有 pip，可用 `uv pip install --python .venv/Scripts/python.exe ...` 执行相同安装命令。CUDA 版由显式官方索引提供，不能把系统已有 CPU 版当成显卡训练环境。

本机验证记录见 `VALIDATION.txt`，完整版本快照见 `environment.lock.txt`。`--architecture-smoke` 只用包内网络结构和随机初始化权重测试 GPU 执行，绝不能当作训练结果。2026-11-05 已通过现有系统代理取得官方 yolov8n.pt，并验证预训练模型 GPU 推理及一轮合成数据训练；仍未训练真实零件模型。

```powershell
python -m members.member05_detector.download_weights
# 若直连失败，显式传入自己机器已有的 HTTP 代理；不修改系统设置：
# python -m members.member05_detector.download_weights --proxy http://127.0.0.1:7897
python -m members.member05_detector.check_environment --weights weights/yolov8n.pt
```

下载脚本使用固定官方 URL，校验记录的文件大小和 SHA256，使用 `.part` 临时文件成功后替换。GitHub 该发布项没有提供 digest；脚本 SHA256 为本次 HTTPS 下载的内容指纹，不能称为官方签名。已有文件不匹配时明确报错，不覆盖。

## 数据交接

向成员 3 获取训练/验证数据、类别目录、scene_id 划分清单和目录外型号说明。向成员 2 获取拍摄分辨率、尺度标定、姿态范围与经确认的尺寸容差。测试集封存，不用于调参。

私有数据目录示意：

```text
dataset-root/
  images/train/...
  images/val/...
  labels/train/...
  labels/val/...
```

每张图片有同相对路径、同名 `.txt` 标签；逐行 `class_id center_x center_y width height`，坐标归一化。经人工确认无零件的图像对应空标签文件；未标注图像不可冒充空场景。六类都应在合适数据分区有支持，但样本量与覆盖性由成员 3 报告。

复制 `dataset.example.yaml` 到本地私有目录，将 path 改为绝对路径。由正式来源单独下载 `yolov8n.pt` 放在被 Git 忽略的 weights 目录，记录来源及 SHA256。通用预训练权重用于初始化训练，不能直接传给项目推理接口冒充六类模型。

训练前可只执行数据检查，不加载模型：

```powershell
python -m members.member05_detector.train --data data/parts.yaml --check-only
```

该命令核查图片能否解码、标注配对与范围、六类映射和跨训练/验证集的完全相同像素，并打印每类实例数。它不能发现所有近重复帧，也不能证明场景独立或标注正确；这些仍须按 scene_id 与实物规格人工审核。

```powershell
python -m members.member05_detector.train --data data/parts.yaml --weights weights/yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --workers 0 --name pilot_v1
```

参数仅为起点。`amp=False` 避免 Ultralytics AMP 自检隐式下载额外模型；验证 AMP 兼容性后如更改需记录。训练输出路径由命令打印，勿假设覆盖旧结果。训练可能耗时，首次先用少量开发数据执行 1 轮检查流程。

真实六类权重可用后：

```powershell
$env:MEMBER05_WEIGHTS = (Resolve-Path 'runs/member05/pilot_v1/weights/best.pt').Path
python -m members.member05_detector.model --image data/example.jpg
```

权重环境变量优先于配置；相对权重路径相对于配置所在目录。`MEMBER05_CONFIG` 可以指定本地配置。原型命令打印真实预测后退出码为 2，提示尚不可作正式检验。根目录 predict.py 的批量导出仍由成员 7 实现，本次未修改。

## 评价规则（待成员 3 / 组长确认后冻结）

调用 `evaluate(cases)`，每项包含 image_id、status、objects 和 ground_truth。对象均有 class_id 与 bbox；预测另有 score，真实框可含 bool 型 ignore。只有 status=OK 且 evaluable=True 的图像进入指标；其他图像列入 excluded_images 并降低覆盖率。不能把 REVIEW_REQUIRED 改为 OK 以获取更好的成绩。

同类 IoU>=0.50，一对一匹配，按分数降序；分数并列用预测输入次序，IoU 并列用 GT 输入次序。先匹配有效 GT，未匹配预测若与任意忽略框达到阈值则中立，不计 FP；这是一项待全组确认的忽略策略。忽略图像或 counts_judgeable=False 不进入计数指标。空场景进入计数分母；无分母指标返回 null。五类向量准确不代表整图 PASS，因为 class 0 单独统计。

该函数只计算对象/计数指标，不实现图像级 F1、置信区间、条件分组和 CSV 加载器；图像级评价由成员 4 主责，数据加载/忽略策略需与成员 3 对齐。

可通过命令行对私有 cases JSON 运行，输出路径必须是新文件，避免覆盖既有实验记录：

```powershell
python -m members.member05_detector.evaluation --cases private/cases.json --output runs/eval01/metrics.json
```

cases 为列表，各项示意为 `{"image_id":"example.png","status":"OK","objects":[],"ground_truth":[]}`；此例只描述格式。预测有 class_id/bbox/score，真值有 class_id/bbox/ignore。标记与状态必须有效；失败/弃判另报覆盖率。不要把原型状态改成 OK 来获得正式指标。

## 实际权重来源

- 官方 URL：https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
- 本地位置：仓库 `weights/yolov8n.pt`，未加入 Git。
- 文件大小：6549796 bytes。
- 记录 SHA256：`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`。
- 用途：COCO 预训练初始化；不是五类零件与目录外拒绝的项目权重。
- 合成训练检查放在 `runs/member05/SYNTHETIC_SMOKE_NOT_PROJECT/`，不得用于现场或报告成绩。

## 下一步与来源

1. 获得 Pilot 数据后训练真实六类模型，记录漏框、错类、错规格。
2. 与成员 2/6 接入基础尺度拒绝；若目录外漏框，评估必要的前景补查。
3. 用独立核验样例复核评价器，再冻结配置和模型。

外部依赖：Ultralytics（https://github.com/ultralytics/ultralytics，AGPL-3.0）、PyTorch（https://pytorch.org）、OpenCV（https://opencv.org）。仓库未预设全组许可证，组长集成时需确认依赖许可。AI 辅助用于初版代码与测试设计；所有效果描述以实际运行记录为准，未训练项目模型。
