# 成员 6 分步执行书（2026-11-19）

你负责启动真实训练、Val 选型和后续评测。本轮只执行来源审查、无权重更新的 preflight、模型结构检查及隔离合成测试。所有命令从仓库根目录运行；已有输出目录不覆盖，重跑请换版本名。

## 1. 当前能做什么

本地分支 `member06/resnet-baseline`，HEAD `5babd06ccc5e4e852a53aac02be806968e0ecef0`，包含上游 PR #6 `12c85c23bc3a57dea51f39915278f31d9133ce74` 及 11.18 新数据。只改成员 6，没有 commit/push。

按原始分区与暂定规则“五类合法件各 2，目录外件 0”推导：

| 原始分区 | PASS | FAIL | 总计 |
|---|---:|---:|---:|
| Train | 133 | 52 | 185 |
| Val | 24 | 77 | 101 |
| Test（仅来源审计） | 23 | 1 | 24 |
| 合计 | 180 | 130 | 310 |

**当前可以显式选择 DEV_ONLY 开发训练；尚未 FORMAL_READY。** 这些是真实图片、对象标注推导的开发标签；不是最终人工核准的实物规格结论。`FAIL` 指画面不满足约定，例如出现目录外件或合法件数量不符，不是图片文件损坏。损坏图片属于输入错误。

截图的 227/46/37 分区没有逐图清单，181/129 也与标注推导不一致。因此不采用它，不猜“82 张训练 FAIL”的成员，不挪 77 张历史 Val 到 Train。旧 Test 的 IMG_7209.png 标注 M5×10 少 1 个，仍为 FAIL。完整来源报告见 [本轮审计与验收](evidence/2026-11-19/REVIEW.md)。

Train 的 133 PASS 全来自旧批次，52 FAIL 全来自 11.18；Val 的 PASS/FAIL 也按批次分开。高分可能利用背景、相机或拍摄条件，不能证明计数/规格理解。新增 52 张全有 3 个目录外件；其中 9 张同时标注合法件少 1 个，需排查漏标。没有核实的物理场景台账；不同文件/hash 不等于独立摆放。

## 2. 进入目录、激活现有环境

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export M6=members/member06_research
export TORCH_HOME="$PWD/$M6/.venv/torch"
export MPLCONFIGDIR="$PWD/$M6/.venv/matplotlib"
python -m pip check
python -c 'import sys,torch,torchvision; print(sys.version); print(torch.__version__,torchvision.__version__,"CUDA:",torch.cuda.is_available())'
```

现有 Python 3.11、torch 2.7.1+cpu、torchvision 0.22.1+cpu 可运行。不要覆盖环境。默认 device=auto；CUDA 不可用时走 CPU。成员目录的 requirements.txt 才是本工程依赖。官方预训练缓存已可用；下载失败会报 PRETRAINED_WEIGHTS_UNAVAILABLE，绝不静默换随机权重。

## 3. 检查新数据、查看各异常数量

本轮已保存完整 310 图审计到 `data/corpus_audit_20261119/`，逐图来源在 images.csv，详细记录在 audit.json。该次审计**经本轮要求读取了公开 Test 图片和标注**，没有模型预测；不能再声称未接触旧 Test。

日常开发只审查 Train/Val：

```bash
python -m members.member06_research.corpus_audit --output "$M6/data/corpus_manual_v1"
python -m members.member06_research.development audit --output "$M6/data/review_manual_v1"
```

前者输出来源、标签、异常、SHA、重复候选，默认不读取 Test；后者还生成待填写的审核/批准模板。审查退出 0 表示格式完整，不等于正式批准。需要重做全 310 张的**来源审计**时，才显式加 `--include-public-test-audit`；不用于选择模型、参数或阈值。

```bash
python - <<'PY'
import json,collections
r=json.load(open('members/member06_research/data/corpus_audit_20261119/audit.json'))
print(r['original_split_label_counts'])
for split in ('train','val'):
    rows=[x for x in r['images'] if x['original_split']==split]
    print(split,'异常类型=',dict(collections.Counter(x['anomaly_type'] for x in rows)))
    print(split,'目录外件数=',dict(collections.Counter(x['non_catalogue_count'] for x in rows)))
print('分批次/条件支持数:',r['source_condition_support'])
PY
```

新 11.18 文件清单 `configs/nov18_sources.json` 是成员6根据已提交源文件生成的 SHA256 索引，不是成员3审核批准。原图/JSON 未修改。所有分组中的数量来自标注，不是分类模型预测数量。

## 4. 准备开发清单与 Preflight

本轮已经生成 `data/dev_20261119_v1/`；默认 `configs/baseline2.json` 已绑定这套真实数据，可直接使用，不需要重复 prepare：

```bash
export M6_CONFIG="$M6/configs/baseline2.json"
python -u -m members.member06_research.train --config "$M6_CONFIG" --check-only
```

预期输出 `status=DEV_ONLY`、Train185（PASS133/FAIL52）、Val101（PASS24/FAIL77）、`formal_readiness=NOT_READY`、`model_initialized=false`、`weights_updated=false`。扫描图片可能需要数分钟（本轮约137秒），此步骤不运行 CNN。

在另一个工作区或需新建版本时执行：

```bash
python -m members.member06_research.development prepare-dev --output "$M6/data/dev_manual_v1"
export M6_CONFIG="$M6/data/dev_manual_v1/config.json"
python -u -m members.member06_research.train --config "$M6_CONFIG" --check-only
```

输出 train.csv、val.csv、provenance.json、config.json；原图保持在成员3目录。清单示意：

```text
image_path,label,scene_id,split,data_kind
/实际来源/图片.jpg,FAIL,UNVERIFIED:实际图像SHA256,train,REAL
```

`UNVERIFIED:*` 是显式“场景未知”占位，不是场景隔离证明。可用 `--scene-ledger /实际台账.json` 提交覆盖全部开发图的真实场景：`{"schema_version":"member06.scene-ledger.v1","images":[{"image_path":"绝对路径","scene_id":"真实摆放ID"}]}`。已知同场景跨 Train/Val 将拒绝；不要用文件名伪造场景 ID。标注、来源或清单有冲突也会拒绝开发训练。

## 5. 由你开始一次真实开发训练

确认接受上面的开发限制后，由你执行以下命令。它会在真实 Train 上更新 Linear Head，并在绑定 Val 上选择最佳 Epoch。本轮交付没有执行这条真实训练命令。

```bash
mkdir -p "$M6/runs"
set -o pipefail
python -u -m members.member06_research.train \
  --config "$M6_CONFIG" --allow-dev-only \
  --output "$M6/runs/dev_baseline_v1" \
  2>&1 | tee "$M6/runs/dev_baseline_v1.console.log"
```

输出目录必须不存在。`--allow-dev-only` 是你的显式选择，缺少时会在模型创建前拒绝。日志先显示证据等级和限制，随后实时显示 epoch、train/val、batch进度和 running_loss，每轮汇总 train_loss/val_loss。CPU 可以较慢；不要把数据审查阶段误认为已开始训练。

模型保持预训练 ResNet18、冻结 backbone/BatchNorm、只训练 Linear(512,2)，CE+Adam；默认 PASS=0、FAIL=1，224 letterbox 保留整图边缘。默认 epochs/lr/batch_size 以生成 config 为准。可在开始前把 config 复制为同目录内的新文件（例如 config_run2.json）并更新 M6_CONFIG，再调整 epochs、learning_rate、weight_decay、batch_size；不要改来源、标签规则或 training_tier 来冒充正式数据。预处理变更属于新实验，记录原因。

每轮 Val loss 最小则保存 best.pt；平局保留较早 epoch。没有自动阈值调参、校准或自动续训。Ctrl+C 后可能已有最佳 checkpoint，但未完成实验不能进入自动候选 select；使用新目录重跑，保留旧记录。

## 6. 找到模型、曲线、混淆矩阵与错误

训练目录包含：

- `best.pt`：模型和完整配置，证据等级 DEV_ONLY。
- `config.json`、`run_config.json`：环境、源码/数据哈希、预训练来源与开发限制。
- `history.json/csv`：各轮 Train/Val loss 和指标。
- `best_metrics.json`、`best_val_predictions.json`：最佳 Val 指标与逐图预测。
- `best_val_groups.json`：异常类型、目录外件数、来源、拍摄条件的支持数/指标及错误图片。
- `checkpoint_manifest.json`：完整实验状态与 checkpoint 哈希。

训练后由你运行：

```bash
python -m members.member06_research.plot_history \
  --run-dir "$M6/runs/dev_baseline_v1" --output "$M6/runs/dev_baseline_v1/curves.png"
export M6_VAL=$(python -c 'import json,os; print(json.load(open(os.environ["M6_CONFIG"]))["val_manifest"])')
python -u -m members.member06_research.validate \
  --checkpoint "$M6/runs/dev_baseline_v1/best.pt" \
  --manifest "$M6_VAL" --allow-dev-only \
  --output "$M6/runs/dev_baseline_v1/validation.json"
```

PNG 同时包含 loss、accuracy、最佳 Val 混淆矩阵。validate 重新加载同一权重和预处理，只允许配置绑定的 Val，输出包含 `groups`。分组矩阵顺序明确为 `[PASS, FAIL]`；原始 epoch 矩阵按 class_to_idx，不能把类别编号当零件类别 ID。

```bash
python - <<'PY'
import json
r=json.load(open('members/member06_research/runs/dev_baseline_v1/best_val_groups.json'))
print(r['evidence_status'],r['overall'])
print('按干扰数分组:',r['groups']['non_catalogue_count'])
for row in r['errors']:
    print(row['image_path'],row['label'],'->',row['predicted_label'],row['anomaly_type'],row['fail_softmax'])
PY
```

FAIL 是阳性：False PASS=真实FAIL判PASS（漏报），False FAIL=真实PASS判FAIL（误报）。同时看 precision/recall/F1、支持数、覆盖率；无分母时为 null，不填成假分数。3 个干扰的训练图不是“模型最多识别3个”的数量上限，因为输出始终只有 PASS/FAIL；验证不同数量也不能单凭数量不同就称未知物体泛化。

Train loss 降而 Val loss 上升通常提示过拟合；两边高要检查标签和任务可分性。反复根据同一 Val 改设置也会对 Val 过拟合。批次混杂、只有某类异常、物理场景未确认等限制，不能被较高 accuracy 消除。

## 7. 有限调参、正式训练与冻结

开发阶段可先建立小网格，plan 只写配置：

```bash
python -m members.member06_research.experiments plan --config "$M6_CONFIG" \
  --output "$M6/runs/dev_plan_v1" --learning-rates 0.001 0.0001 --weight-decays 0 --epochs 10
```

生成 candidate_00.json、candidate_01.json 及 plan.json。你分别用 train 指定它们和对应 run_00/run_01，加 `--allow-dev-only`；全部完成后才执行：

```bash
python -m members.member06_research.experiments select --plan "$M6/runs/dev_plan_v1/plan.json" \
  --output "$M6/runs/dev_plan_v1/selection.json" --allow-dev-only
```

选择规则仍是最小 Val CE；源码变更后要新建计划。**DEV_ONLY 禁止正式 freeze、正式评测和导出 VERIFIED 真值。** 不能只改 JSON 等级来升级。

正式路径需要组长/成员3给出固定数量批准、逐图标签/实物规格审核、真实 scene_id 与场景隔离、数据用途与历史接触记录。在第3步生成的 approval.json/reviews.json 中填真实证据，不把模板直接改 APPROVED。然后：

```bash
python -m members.member06_research.development prepare \
  --approval "$M6/data/review_manual_v1/approval.json" \
  --reviews "$M6/data/review_manual_v1/reviews.json" --output "$M6/data/formal_v1"
python -m members.member06_research.train --config "$M6/data/formal_v1/config.json" --check-only
```

通过后使用该正式 config 新建实验、重新训练；正式入口不需要开发开关。plan/select 后 `experiments freeze --selection /实际selection.json --output /新freeze.json` 才可生成正式冻结记录。后续 `evaluate_frozen --freeze ... --manifest ... --approval ... --check-only --output ...` 检查交接，再由你显式去掉 check-only 评测；具体参数见 `python -m members.member06_research.evaluate_frozen --help` 和 `configs/evaluation_approval.template.json`。旧 Test 只能 PUBLIC_PARTITION_EVALUATION；独立测试需要新的封存场景证据，不从截图拼新测试集。

## 8. UI、单图与 HOG 衔接

开发模型调用也要显式选择开发模式：

```bash
python -m members.member06_research.inference --checkpoint "$M6/runs/dev_baseline_v1/best.pt" \
  --image /替换成实际图片.jpg --expected-counts '[2,2,2,2,2]' --allow-dev-only \
  --output "$M6/runs/dev_prediction_v1.json"
```

Python：`create_adapter(checkpoint, allow_dev_only=True)` 后调用 `predict(image, counts)` / `predict_batch(images, counts)`。返回 `status=DEV_ONLY`、predicted_label、class_scores、未经校准 softmax；数量不同返回 UNSUPPORTED_COUNTS。不会输出假框/实际件数。公共 baseline2.py 默认拒绝 DEV_ONLY，成员7若要开发演示应显式通过适配器选择，不自动修改公共协议。

`batch_export --checkpoint ... --input_dir ... --counts_file ... --output_dir ... --allow-dev-only` 输出共享 images.csv/run_config.json；counts 文件格式 `[{"image_id":"真实文件名.jpg","expected_counts":[2,2,2,2,2]}]`，不传标签给模型。

成员4实际 HOG+PASS-only MLP自编码器，使用重建误差；不能与 softmax 相加/平均。其比较器只把 status=OK 计为已判定，DEV_ONLY/SMOKE_ONLY 会保留为未判定，不能改成 OK 来刷覆盖率。开发分组指标由本成员输出并注明证据等级；正式共同评价要获批的同一批 Val、相同数量规则和完整状态。`handoff` 仅给获批数据生成 VERIFIED 真值。本轮只合成验证接口，未跑 HOG 真实模型。

成员5当前公开代码仍是占位；截图 YOLO/YOLO+OpenCV 结果缺原始日志，不能当作本成员成绩或据此决定 Stage3。

## 9. 回归测试和排错

```bash
python -m unittest discover -s members/member06_research/tests -v
python -m unittest discover -s tests -v
```

上述unittest仅是维护者的自动测试，不是日常训练命令。实际训练使用第5节REAL配置；不需要生成任何合成数据。测试夹具和历史SMOKE_ONLY工件保留隔离状态，不能改标记充当真实模型。

| 提示 | 处理 |
|---|---|
| DEV_ONLY requires allow-dev-only | 阅读限制后显式选择；正式freeze不接受该开关 |
| source/hash/manifest changed | 保留旧版，重新审查和prepare新版本，不手改哈希绕过 |
| known scene crosses / duplicate | 核查真实摆放与来源，交成员3修正，不重命名绕过 |
| WAITING_UPSTREAM | 正式确认不足；开发模式走独立prepare-dev，不能伪造批准 |
| PRETRAINED_WEIGHTS_UNAVAILABLE | 检查TORCH_HOME/缓存/下载真实错误，不能用随机模型冒充基线 |
| output exists / code changed | 新输出版本/新实验计划，保留历史 |
| CUDA unavailable / OOM | CPU或训练前减batch；不要换Test调参数 |
| INPUT_ERROR / UNSUPPORTED_COUNTS | 图片输入或规则问题，不算PASS |

Stage3仍是ROI/局部CNN/像素几何通用工具，见 MEMBER6_LEARNING_GUIDE.md。没有尺度标定不输出毫米。本轮没有删除环境、历史、真实数据或其他成员文件。
