# Baseline 2 中文训练操作指南

更新日期：2026-11-14。操作目录：`/home/grace/projects/AI/member06`。

本轮没有执行真实数据训练或正式 Val 选型。本指南对应当前分支 `member06/resnet-baseline`、HEAD `90c77fa`；它已包含本地获取的 upstream/main `f0126f5`。不需要再次 pull/merge。

**先看当前结论：数据状态为 D，等待上游。** 真实 Train/Val 图片和对象框已在本地，但缺已批准的固定数量、可信整图标签和场景独立性记录，还有自动标注复核与规格问题。现在可以做环境检查、数据检查和工程测试；不能开始可信的二分类训练。下文提供带条件检查的完整训练入口，条件不满足时会停在训练之前。

## 1. 打开正确的终端和环境

在 Ubuntu VS Code 中打开终端，逐行执行：

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
python -c 'import sys; print(sys.executable); print(sys.version)'
```

`source` 让当前终端的 python 指向成员 6 虚拟环境，不会重新安装软件。成功时第一行应为 `/home/grace/projects/AI/member06/members/member06_research/.venv/bin/python`，版本为 Python 3.11.17。若不是，先检查当前目录和激活路径，不要在错误环境安装包。

设置本模块使用的可写缓存目录：

```bash
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
```

TORCH_HOME 指向已下载的 ImageNet 权重，避免尝试在默认缓存中重新下载。MPLCONFIGDIR 用于画图缓存，在当前受限环境中应设置。训练不需要设置 MEMBER06_CHECKPOINT；只有公共推理入口需要它。所有命令都从仓库根目录运行，不能 cd 到模块目录后继续照抄 `python -m members...`。

检查软件和设备：

```bash
python -m pip check
python -c 'import torch, torchvision; print("torch:",torch.__version__); print("torchvision:",torchvision.__version__); print("CUDA:",torch.cuda.is_available()); print("device:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
nvidia-smi
df -h .
```

pip check 应输出 No broken requirements found。本轮实测 torch=2.7.1+cpu、torchvision=0.22.1+cpu，CUDA=False。nvidia-smi 无法连接驱动，实际 GPU 名称与显存无法确认，所以当前使用 CPU/auto。磁盘剩余约 267 GiB，足以容纳现有约 907 MiB 数据和约 45 MB 的模型工件；每个实验仍应使用独立输出目录。

不要因为存在 nvidia-smi 命令就覆盖 PyTorch。GPU 环境需先由设备负责人确认驱动，再选择匹配 wheel；本轮保留原 .venv，没有安装或升级依赖。

权重检查命令：

```bash
sha256sum members/member06_research/.venv/torch/hub/checkpoints/resnet18-f37072fd.pth
```

预期 SHA256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。本轮已从该缓存成功初始化官方预训练 ResNet18，仅用零张量检查输出形状 `[1,2]`，没有拿真实图像做模型选型。

根目录 requirements.txt 仍是团队框架说明，不会安装完整训练依赖。需要重建环境时，用 Python 3.11 在**新环境目录**创建，不删除当前环境；依赖来源是 `members/member06_research/requirements.txt`，历史完整版本快照是 `evidence/environment.freeze.txt`。安装方法见模块 README，不需要在已通过 pip check 的环境重复安装。

## 2. 真实数据在哪里、现在有什么

数据根目录已经存在：

```text
members/member03_data/full_dataset_upload/datasets/parts_v1/
  split_manifest.csv
  README.txt
  train/拍摄条件/IMG_*.png + 同名 Labelme.json
  val/拍摄条件/IMG_*.png + 同名 Labelme.json
  test/ ...                         本模块不读取其图片或标注内容
  统一GT/
    class_mapping.json
    train/拍摄条件/同名.json
    val/拍摄条件/同名.json
    test/ ...                       本模块不读取
```

拍摄条件为标准光正视、暗光正视图、暗光侧视图、标准光侧视图。**条件目录不是零件类别。** 来源是成员 2 的 iPhone 7 白底采集，由成员 3 整理“完整数据集上传包”。开发使用范围与正式数据版本仍须由团队确认。

| 分区 | 图片 | 每种条件 | 当前状态 |
|---|---:|---:|---|
| Train | 112 | 28 | 全部已读图、校验哈希和对象标注；无可信整图标签 |
| Val | 24 | 6 | 全部已检查格式/完整性；未进行模型推理或选型 |
| Test | 24 | 6 | 数量来自 split_manifest 元数据；未读取标签、图片内容或结果 |

Train/Val 图片可以完整解码，图像 SHA256、原始标注来源哈希、框坐标、原始框与统一 GT 一致，未发现完整 RGB 内容重复。这个检查只能排除完全重复，不能证明相似帧或拍摄场景独立。

对象标注统计如下。这是**标注清单中的数量**，不是模型预测或经复核的物理规格保证：

| ID | 公共名称/规格 | Train 对象 | Val 对象 |
|---|---|---:|---:|
| 0 | non_catalogue | 0 | 0 |
| 1 | long_screw / M4×25 | 224 | 48 |
| 2 | thick_screw / M5×10 | 224 | 48 |
| 3 | thin_screw / M4×10 | 224 | 48 |
| 4 | hex_nut / M4 | 224 | 48 |
| 5 | flat_washer / M5，实物规格待核实 | 224 | 48 |

原始 Labelme 是 shapes/label/points；统一 GT 是 `member03.gt.v1`，包含 objects/class_id/bbox/actual_counts 和原始标注 SHA256。框是原图像素 xyxy 半开区间。它们都不是 ResNet 的整图分类清单。

本轮发现的阻塞：

1. 公共 catalogue 的数量规则仍为 PROPOSED_CONFIRM_AFTER_PILOT；不能自动把 `[2,2,2,2,2]` 作为正式标签规则。
2. 136 张开发图均缺 scene_id、expected_counts 和整图 PASS/FAIL。原分区按文件名分段，成员 3 明确说明尚不保证场景独立；本模块不重新分区，也不把文件名伪装成已审核 scene_id。
3. 109 张 Train、24 张 Val 的原始标注仍含 auto_label_review 或 human review required。统一 GT 没有保留这些草稿标记，不能仅看 GT 文件存在就当已人工验收。
4. 平垫片的实物规格未确认；没有尺度标定，不能从像素判断毫米规格。
5. 全部开发图的已注数量相同，ID 0 均为零。仅按固定数量规则无法形成 PASS/FAIL 两类监督；需要团队补充或审核真实 FAIL 场景。不能故意换 expected_counts、删对象、复制图片改标签来补负类。
6. 上游提交含测试数据/真值，与公共封存规则冲突。组长须处理该事实；本轮仍保持测试内容隔离。

因此 A（可信 Train+Val）、B（可信 Train）、C（规则已获批、只差转换）都尚不成立，当前是 D。也没有可报告的真实二分类类别数量；不能把图片数量表换写成 PASS/FAIL 数量。

## 3. 现在可以执行的数据检查

本轮已生成以下本地文件，位于被忽略的 data 目录，不提交 Git：

- `members/member06_research/data/team_audit_20261114.json`：完整 Train/Val 审查。
- `members/member06_research/data/team_approval_20261114.json`：PENDING 的批准记录模板。
- `members/member06_research/data/team_reviews_20261114.json`：逐图 PENDING 的复核/scene_id 模板。

查看现有结果：

```bash
python -c 'import json; p="members/member06_research/data/team_audit_20261114.json"; d=json.load(open(p)); print(d["status"], d["readiness"]); print(d["per_split"]); print("errors:", d["errors"]); print("test_content_read:",d["test_content_read"])'
```

这条命令只读已经生成的 JSON。当前应看到 NOT_READY、D、errors=[]、test_content_read=False。errors=[] 表示文件/格式检查通过，**不表示标签已经批准**。

需要重新检查新上游交付时：

```bash
M6_AUDIT_DIR=$(mktemp -d "$PWD/members/member06_research/data/recheck-XXXXXXXX")
python -m members.member06_research.team_data audit --output "$M6_AUDIT_DIR/audit.json" --approval-template "$M6_AUDIT_DIR/approval.json" --review-template "$M6_AUDIT_DIR/reviews.json"
```

运行文件为 `team_data.py`。输入是当前 split_manifest、Train/Val 图片、原始标注、统一 GT 和类别表；输出三个新 JSON 文件，不写 PASS/FAIL 标签、不加载 CNN、不读测试内容。图片全量解码可能需要几十秒到数分钟。成功时打印 Audit complete，具体错误列在 audit.json 的 errors；错误时退出码为 2。每次 mktemp 创建新目录，保留旧证据，不覆盖上轮审查。

用 `source_manifest_sha256` 区分源数据版本。本轮值为 `a3eac3429ab3f759c952c0fd7ece58019f0c65da4717918987c16dfdad8d1ed6`。文件或标注更新后必须重新审查并取得对应哈希的复核，不要把旧复核直接套到新数据。

## 4. 上游怎样交接可训练的标签

目前没有可访问的成员 3 Python 转换器。本轮新增的是成员 6 **本地适配格式**，不是擅自宣布新的公共协议。组长/成员 3 可以提供等价批准证据，再通过本地适配器接入。

`team_approval_20261114.json` 现在的 status=PENDING、fixed_expected_counts=null，确认字段为 false。真正完成审核后，交接需要包含：

| 字段 | 含义 |
|---|---|
| status / approved_by / decision_reference | 实际批准状态、负责人、可核对的团队决策出处 |
| dataset_version / source_manifest_sha256 / catalogue_sha256 | 批准针对的版本和文件，不是任意新图片 |
| fixed_expected_counts | 实际确认的五类固定数量；不能按默认值猜 |
| scene_split_verified | 成员 3 核实场景级分区，没有跨 Train/Val 场景 |
| catalogue_specs_verified | 五类实物规格已确认，包括平垫片 |
| dataset_usage_approved | 数据来源与使用范围已明确 |

逐图 reviews 需要真实 scene_id、APPROVED、judgeable、annotation_complete、physical_specifications_verified、reviewer 和 review_reference，以及匹配当前图片/原始标注/GT 的哈希。**不要仅为跑通程序把 PENDING 改成 APPROVED。** 哈希用于发现变动，不会替代人工真值审核或证明批准人的身份。

每张图必须有复核记录；无法判断、ignore/unjudgeable 或未知规格的图会被排除并记录原因，不猜二分类标签。原始自动草稿只有在针对同一哈希的明确人工复核记录成立后才能用于转换；已有 unknown/ignore 状态不会被静默当成 FAIL。

只有审批和逐图复核完整后，转换才按对象标注执行：五类实际数量全部等于已批准 expected_counts 且 ID 0 为零 → PASS；已可判断的缺失、多出或目录外 → FAIL。未知 ID 不硬映射。模型预测从不参与真值生成。

团队补齐后，由你手动执行下面的准备命令；**当前执行会明确提示 WAITING_UPSTREAM，不创建输出目录**：

```bash
python -m members.member06_research.team_data prepare --approval members/member06_research/data/team_approval_20261114.json --reviews members/member06_research/data/team_reviews_20261114.json --output members/member06_research/data/team_parts_v1
```

输入是已实际审核的批准记录/复核记录和团队源数据。成功才生成 train.csv、val.csv、approval.json、reviews.json、provenance.json；保留原始图像路径，不移动图片、不重新分区、不生成测试清单。provenance 记录来源、标签、场景数、排除项和哈希。

转换后的 CSV 为 `image_path,label,scene_id,split,data_kind`；label 为 PASS/FAIL，data_kind=REAL。每份 Train/Val 都必须包含两类。如果 Val 缺失或只剩一类，本工程保持等待，不用 Train Loss 选“最佳”模型。被排除图无监督标签，不能算作 FAIL。当前所有数量相同的数据，即使只补批准，也仍不足以通过两类检查。

源数据新增/纠正后，应先重新 audit，把新批准和复核记录放到上述明确路径；已有 team_parts_v1 时使用新版本目录，并同步修改下一节三个配置路径。本工具不会覆盖旧交接。若导出中断、缺少完整 provenance，训练前检查会拒绝使用它。

## 5. 配置检查：不会开始训练

当前真实配置就是 `members/member06_research/configs/baseline2.json`，已指向成员 6 接入后的本地清单，而不是不存在的根 data 示例目录：

```json
{
  "train_manifest": "members/member06_research/data/team_parts_v1/train.csv",
  "val_manifest": "members/member06_research/data/team_parts_v1/val.csv",
  "data_provenance": "members/member06_research/data/team_parts_v1/provenance.json"
}
```

上述是完整配置中的三个字段，其余字段保留，不要用这三行覆盖整个文件。当前 counts_rule_status=PROVISIONAL_PENDING_PILOT，是真实的待批准状态。

上游真实确认并成功 prepare 后，在同一配置中：把 supported_counts 改为实际获批向量，把 counts_rule_status 改为 CONFIRMED。保持 weights=IMAGENET1K_V1、data_kind=REAL、class_to_idx 的 PASS=0/FAIL=1、224、letterbox、device=auto。审批与配置数量不一致会被拒绝。

```bash
python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only
```

该命令运行 `train.py` 的检查分支，内部调用 `preflight.py`：核对审批、清单和源文件哈希、标签、Train/Val 隔离、输入形状，加载一个图像批次。没有 CNN 初始化、optimizer、反向传播或模型选型，不需要 --output。

**当前应退出 2 并显示“REAL training requires confirmed fixed counts”**；这是等待上游的准确结果。条件满足后才输出 READY_TRAIN_VAL_ONLY、Train/Val 两类数量、batch_shape 等，退出 0。READY 只表示数据和工程检查通过，不是课程冻结或科学效果合格。

## 6. 由你本人启动正式训练

本轮没有运行这一节。上游补齐、prepare 成功、配置已按依据确认后，在终端复制以下整块。它会先检查；等待状态下不会创建实验目录或训练：

```bash
if python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only; then
  mkdir -p members/member06_research/runs
  M6_RUN_DIR=$(mktemp -d "$PWD/members/member06_research/runs/real-XXXXXXXX")
  echo "本次实验目录：$M6_RUN_DIR"
  python -u -m members.member06_research.train --config members/member06_research/configs/baseline2.json --output "$M6_RUN_DIR" > "$M6_RUN_DIR/../$(basename "$M6_RUN_DIR").console.log" 2>&1
  M6_TRAIN_EXIT=$?
  cat "$M6_RUN_DIR/../$(basename "$M6_RUN_DIR").console.log"
  echo "训练退出码：$M6_TRAIN_EXIT"
else
  echo "等待上游：数据/批准/配置尚未通过，未启动训练。"
fi
```

运行文件为 train.py；输入是配置和经过复核的真实 Train/Val；输出目录由 mktemp 唯一生成，初始为空。console.log 放在实验目录旁边，避免日志先占用目录导致“output directory must be empty”。本命令以前台方式运行，直到训练结束；这段终端会暂时没有实时打印。

想看进度，在 VS Code 新开另一个终端，把上一终端打印出的目录复制进 VS Code 文件浏览器，打开旁边同名 `.console.log`；也可在原终端开始训练后，在新终端运行：

```bash
cd /home/grace/projects/AI/member06
M6_LOG=$(ls -t members/member06_research/runs/real-*.console.log 2>/dev/null | head -n 1)
if [ -n "$M6_LOG" ]; then tail -f "$M6_LOG"; else echo "尚未开始真实训练，没有真实日志。"; fi
```

这条命令查看最近创建的真实训练日志，不参与训练或选型。Ctrl+C 只停止这个查看日志的终端，不会停止另一个终端的训练。训练终端通常每个 epoch 输出 TRAIN_VAL_ONLY、epoch、train_loss、val_loss；退出码 0 且 checkpoint_manifest.json 存在表示正常完成。出错时看 console.log；不要根据“文件夹存在”判断训练完成。

实际优化行为仍是 Frozen ResNet18 + Linear：backbone.requires_grad_(False)，train() 后保持 backbone.eval()，仅 Adam(model.head.parameters()) 更新头。保留完整视野的 letterbox 预处理使用 ImageNet mean/std；没有新增增强、阈值调整或 Stage 3 算法。

程序按最小 Val CrossEntropyLoss 选择 best.pt，相同保留较早 epoch。Train Loss 不是替代 Val 的指标。当前实现没有 train-only 模式；Val 未完成时等待，不会把最后训练 epoch 伪装成验证最佳。

## 7. 输出文件与训练后的操作

同一个训练终端中的 M6_RUN_DIR 保留本次路径。若终端已关闭，先从 VS Code 文件浏览器找到该次真实实验路径，再在终端设置 `M6_RUN_DIR`；不要把别的实验或历史 SMOKE_ONLY 目录当成本轮结果。

| 文件 | 学习时如何使用 |
|---|---|
| config.json / run_config.json | 记录实际参数、种子、环境、源数据与源码哈希、准备记录哈希 |
| history.csv / history.json | 看每个 epoch 的 Train/Val loss、accuracy，JSON 另有 FAIL P/R/F1 和混淆矩阵 |
| best.pt | 完整 backbone+head、配置、optimizer 状态、最佳 epoch/Val 指标 |
| best_metrics.json | 看实际选中了哪个 epoch；行真值、列预测，按配置编号排列 |
| best_val_predictions.json | 找具体 Val 错误场景；不用于读取封存测试标签 |
| checkpoint_manifest.json | best.pt SHA256；用于交接，不是正式冻结签字 |

训练结束且数据未变化时，手动重新验证同一个已批准 Val：

```bash
if [ -n "${M6_RUN_DIR:-}" ] && [ -f "$M6_RUN_DIR/checkpoint_manifest.json" ]; then
  python -m members.member06_research.validate --checkpoint "$M6_RUN_DIR/best.pt" --manifest members/member06_research/data/team_parts_v1/val.csv --output "$M6_RUN_DIR/reloaded_val.json"
else
  echo "尚无已完成的真实实验，请先完成训练。"
fi
```

validate.py 读取 checkpoint 中的模型设置和获批 Val，不更新权重。它重新核对 provenance，只接受该 checkpoint 配置中的 Val 路径。源数据或清单变化时应建立新版本，而非覆盖旧验证输出。当前没有真实 checkpoint，所以本轮未执行此验证。

查看曲线：

```bash
if [ -n "${M6_RUN_DIR:-}" ] && [ -f "$M6_RUN_DIR/checkpoint_manifest.json" ]; then
  python -m members.member06_research.plot_history --run-dir "$M6_RUN_DIR" --output "$M6_RUN_DIR/curves.png"
else
  echo "尚无已完成的真实实验，当前没有真实训练曲线。"
fi
```

plot_history.py 从历史/最佳指标绘制 loss、accuracy 和混淆矩阵，输出独立 PNG，可在 VS Code 打开。已有 curves.png 时换新文件名，不覆盖。历史的 runs/pretrained_final 曲线是合成数据，不代表本轮训练。

加载已完成模型，给一张现有 Train 图片演示推理的命令如下；这只用于学接口，不作为泛化评价：

```bash
if [ -n "${M6_RUN_DIR:-}" ] && [ -f "$M6_RUN_DIR/checkpoint_manifest.json" ]; then
  python -m members.member06_research.inference --checkpoint "$M6_RUN_DIR/best.pt" --image members/member03_data/full_dataset_upload/datasets/parts_v1/train/标准光正视/IMG_7174.png --output "$M6_RUN_DIR/example_inference.json"
else
  echo "尚无真实 checkpoint，未执行真实推理。"
fi
```

inference.py 不读取该图片的标签，默认按 checkpoint 的固定数量要求推理；结果包含 PASS/FAIL 和未经校准的 softmax 分数，不提供实际计数或框。没有加 --allow-smoke，防止误加载示例模型。新图片可换成你有权使用的路径；单张格式是 --image，批量是 --input-dir；参数命名与根 predict.py 不同。根 predict.py 的 CSV 导出仍待成员 7。

## 8. 读懂实验结果

| 名称 | 在本工程中的含义 |
|---|---|
| Epoch | 完整遍历一次 Train；每轮结束再跑一次 Val |
| Batch Size | 每次梯度更新的图像数，当前模板 16；内存不足可在正式训练前减小并记录新配置 |
| Learning Rate | 头参数每次更新的步长，模板 0.001；只在 Train/Val 上决定，不看封存 Test |
| Train Loss | 训练样本交叉熵的平均值，越低说明拟合训练标签更好，并不保证泛化 |
| Validation Loss | 不更新参数时计算的 Val 交叉熵，本工程用它选择最佳 epoch |
| Accuracy | 全部判对的比例，类别单一时会误导，因此需要可信 PASS/FAIL 两类 |
| FAIL Precision | 预测 FAIL 中实际 FAIL 的比例 |
| FAIL Recall | 实际 FAIL 中检出 FAIL 的比例；错误 PASS 是本项目重点错误 |
| FAIL F1 | Precision/Recall 的调和平均；分母为零时记录 null |
| Checkpoint | 保存模型状态和配置，可重载推理；本工程不提供中断续训命令 |

默认标签 PASS=0、FAIL=1，因此混淆矩阵左下角是实际 FAIL 却预测 PASS。head 输出 logits，CrossEntropyLoss 直接接收 logits；推理才 softmax/argmax。prediction_score 是被预测类别的未校准分数，不是“模型判断一定正确的概率”。

Train loss 下降而 Val loss 上升、Val 错误增多，可能过拟合。先检查场景泄漏、近重复、标注质量、类别平衡和条件差异，再在 Train/Val 上比较学习率、epoch 等。保存每次独立配置和负结果，不能用测试集反复挑选。

SMOKE_ONLY 表示人工合成工程测试；TRAIN_VAL_ONLY 表示真实 Train/Val 工件仍处于开发阶段；READY 也只是前置检查状态。正式冻结还需要组长确认数量/接口/评价器、数据版本、模型与预处理哈希，以及正式冻结记录。本轮没有新增任何真实模型结果。

## 9. 常见问题

| 现象 | 操作 |
|---|---|
| 找不到路径/No module named members | 先 pwd，必须在 /home/grace/projects/AI/member06；配置中的清单路径相对启动目录，清单内图像路径相对清单或为绝对路径 |
| .venv 激活失败 | 检查 .venv/bin/python 的符号链接与 Python 3.11 是否还存在；不删旧环境，可按 README 在新路径重建 |
| No module named torch | 确认 sys.executable，使用成员环境；不要在根 requirements.txt 上期待获得 torch |
| CUDA 不可用 | 当前 CPU wheel 和驱动状态如此；使用 auto/cpu，不反复重装或强制 cuda |
| 权重下载失败 | 确认 TORCH_HOME 和 SHA256；缺失时按 README 从官方来源下载，失败保留真实异常，不随机降级 |
| 清单/来源哈希不匹配 | 源数据或审核记录变化，重新审查并取得对应批准；不要修改哈希绕过 |
| PASS/FAIL 不足/只有一类 | 需要真实复核的 FAIL/PASS 场景，不能生成假标签；当前数据正受此阻塞 |
| 输出目录已存在/非空 | 保留旧实验，换新目录；日志放在目录外；不用 git clean 或批量删除 |
| 内存/显存不足 | 停止该次实验，按 Train/Val 规则记录后调整 batch_size，另起新实验；不偷偷改变图像裁剪以省内存 |
| 训练中断 | Ctrl+C 退出 130。已完成的 best.pt 若已原子写入会保留，best.tmp 不能当正式 checkpoint；没有自动续训。检查日志后使用新目录重跑，不把半轮当完整训练 |
| UNSUPPORTED_COUNTS | 请求与 checkpoint 支持向量不同；整图模型不能适配任意数量，也不能通过改 checkpoint 获得新能力 |
| 训练完成却不能叫正式冻结 | best.pt 是 Val 选型产物；需要组长的冻结清单、评价协议和团队验收 |
| audit 无错误但训练被拒绝 | 文件完整性不等于数量规则、人工标注、场景分区和规格已批准 |
| reviews 模板仍是 PENDING | 等待成员 3 和组长的真实审核，不是让初学者自己猜 scene_id 和判定 |

## 10. 最简速查：按顺序执行

当前可以完整复制下面的环境/检查块；它只读现有审查和配置检查，最后明确等待，不启动训练：

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
python -c 'import sys,torch,torchvision; print(sys.executable); print(torch.__version__,torchvision.__version__,torch.cuda.is_available())'
python -m pip check
python -c 'import json; d=json.load(open("members/member06_research/data/team_audit_20261114.json")); print(d["status"],d["readiness"],d["errors"])'
if python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only; then
  echo "数据/配置检查通过，可按本指南第 6 节由你启动训练。"
else
  echo "等待上游：当前未开始真实训练。详见本指南第 2、4 节。"
fi
```

只有取得真实批准、逐图复核和两类 Train/Val 后，再依次执行：第 4 节 prepare → 第 5 节根据批准修改两个数量字段及实际版本路径 → 第 6 节带检查的训练块 → 第 7 节验证/曲线/推理。所有路径和参数已经给出；当前不会出现“表面训练成功、实际使用假标签”的结果。

审批完成后，先按第 5 节把配置中的 supported_counts 写成获批向量、counts_rule_status 写成 CONFIRMED。使用当前明确的 team_parts_v1 路径时，准备和训练也可以一次按顺序执行：

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
if [ -f members/member06_research/data/team_parts_v1/provenance.json ] || python -m members.member06_research.team_data prepare --approval members/member06_research/data/team_approval_20261114.json --reviews members/member06_research/data/team_reviews_20261114.json --output members/member06_research/data/team_parts_v1; then
  if python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only; then
    mkdir -p members/member06_research/runs
    M6_RUN_DIR=$(mktemp -d "$PWD/members/member06_research/runs/real-XXXXXXXX")
    echo "本次目录：$M6_RUN_DIR"
    python -u -m members.member06_research.train --config members/member06_research/configs/baseline2.json --output "$M6_RUN_DIR" > "$M6_RUN_DIR/../$(basename "$M6_RUN_DIR").console.log" 2>&1
    M6_TRAIN_EXIT=$?
    cat "$M6_RUN_DIR/../$(basename "$M6_RUN_DIR").console.log"
    echo "训练退出码：$M6_TRAIN_EXIT"
  else
    echo "数据或配置仍待上游，未开始训练。"
  fi
else
  echo "审批/数据准备尚未通过，未开始训练。"
fi
```

现阶段这块命令停在 PENDING 审批检查，不会更新模型。这里的存在判断只用于避免覆盖已准备目录，随后 preflight 仍完整核对哈希和批准，绝不会以“文件存在”替代就绪检查。
