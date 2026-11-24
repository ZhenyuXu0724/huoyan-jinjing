# Baseline 2 第二轮：仅 Train CE 类别权重对照

本实验为 **DEV_ONLY**，不是课程正式冻结或 Stage 3 Final。科学问题是：在固定数据、初始化和训练流程时，提高训练 FAIL 类的相对损失权重，是否改善既有 Val 的 FAIL 检出与排序？它不能单独分离背景、物体类型和拍摄批次的因果贡献。

## 1. 进入实际仓库与环境

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/member06-weighted-mpl
```

实际环境为成员6目录内的 `.venv`，不是仓库根 `.venv`。脚本直接使用该解释器，不依赖你是否激活。若提示虚拟环境不存在，检查该路径，不自动新建环境或换依赖。

## 2. 先校验 DEV v1 归档

```bash
python -m members.member06_research.weighted_dev --archive-only
```

输入：`members/member06_research/archives/dev_baseline_v1/`。输出应为 `archive_files_verified: 70`。该目录保存原始 run、数据清单/provenance、历史源码、诊断表与 SHA256.json。原模型仍在 `runs/dev_baseline_v1/`，归档 checkpoint 在 `archives/dev_baseline_v1/run/best.pt`。

SHA256 必须为 `57f7be37ed2e9543f3a3219636f527a78f9255acc8ccc4e193c38776fdcaaa65`。历史 best 为第5轮；不是第10轮模型。归档校验失败时停止，不覆盖、重新生成哈希或自动“修复”历史。

## 3. 数据、环境和单因素预检（不会训练）

```bash
bash members/member06_research/scripts/run_weighted_dev.sh --check-only
```

输入配置：`configs/baseline2_weighted_ce_dev_v2.json`，从第一次保存的配置复制，只增加 `train_class_weight: balanced`（旧模式语义为 `none`）。输入 Train/Val 为 `data/dev_20261119_v1/`。预检读取 Train/Val 及其 provenance，不访问 Test 图像/标签。哈希/图像检查可能需要数分钟；这是完整性检查，不是重复照片研究。

成功应显示 `DEV_ONLY_READY_FOR_USER_START`、Train PASS=133、FAIL=52，以及按类别编号计算的权重约 `[0.695489, 1.778846]`。任意其他配置差异、历史数据或已审查代码哈希变化、Python/torch等依赖或实际CPU/CUDA设备变化都会停止；不要为通过预检直接修改身份文件，需要先解释变化。

`scene_independence_verified: false` 和 `formal_readiness: NOT_READY` 是现有数据限制，不是假装已经审核通过。对象错标、9张少件图与场景关联仍待成员2/3确认。本人可以按授权在原标签/原分区上做这次受限开发对照；若上游随后修订数据，应另建版本，不把不同数据版本混为本次单因素实验。

## 4. 工程测试（只在合成数据上优化）

```bash
python -m unittest discover -s members/member06_research/tests -v
python -m unittest discover -s tests -v
bash -n members/member06_research/scripts/run_weighted_dev.sh
```

包含配置、权重索引、weighted/unweighted CE、无权Val、冻结Backbone/BN、历史checkpoint加载、与归档源码无权行为逐项一致、合成训练、错误退出码、不覆盖与比较报告测试。显示 `SMOKE_ONLY` 的测试仅为隔离临时目录中的工程测试；真实入口使用 ImageNet 权重和真实DEV数据，不能删去测试的真实性标记。CUDA不可用时只跳过CUDA专项测试。实际命令和结果见 `evidence/weighted_ce_v2/`。

## 5. 由你亲自启动第二轮真实训练

下面命令会真实训练，Codex 本轮未执行它：

```bash
bash members/member06_research/scripts/run_weighted_dev.sh
```

脚本先预检，再创建带时间戳与随机后缀的新目录：

```text
members/member06_research/runs/baseline2_weighted_ce_dev_v2_<时间>_<唯一后缀>/
  preflight.json / preflight.log
  console.log
  train_exit_code.txt
  model/
    config.json / run_config.json
    history.json / history.csv
    best.pt / checkpoint_manifest.json
    best_metrics.json / best_val_predictions.json / best_val_groups.json
  comparison.log
  comparison/
    COMPARISON_CN.md / comparison_metrics.json / comparison_by_group.csv
    experiment_identity.json / curves.png / score_distributions.png
```

`model/`在训练前不存在，不覆盖旧模型。脚本实时打印 batch 和 epoch；训练输出同时写 `console.log`。Python失败退出码被保留，失败不重试、不自动换参数。训练成功后自动运行只读比较；比较失败不代表训练模型丢失，查看 `train_exit_code.txt` 和 `comparison.log` 分清阶段。

## 6. 训练中看什么

复制终端打印的实验路径，例如将它完整粘贴到以下变量（此处不提供虚构的实际结果路径）：

```bash
read -r -p '粘贴终端打印的实验目录: ' RUN
 tail -f "$RUN/console.log"
```

在另一个终端看日志；`Ctrl+C` 只停止 `tail`，不要在训练终端随意中断。训练中观察是否推进到10轮、Val loss是否有限、是否出现错误。Train loss变小不是成功标准。保存最小**无权Val loss**的模型，可能早于第10轮。

每batch Train CE：`sum(w[target] * NLL) / sum(w[target])`，由PyTorch mean reduction实现。记录的epoch Train loss按各batch分子/分母累计。旧Train loss是普通平均NLL，两者不要直接比较大小。Val保持普通NLL均值，才是可直接比较的loss。

模型、224 letterbox、ImageNet初始化、冻结CNN/BN、Linear(512,2)、Adam、seed42、lr0.001、batch16、10轮、采样、数据和argmax均不变。二类softmax未经校准；恰好相等时argmax选择编号0，即PASS，不能把0.5边界自行改成另一种tie规则。

## 7. 训练完成后检查和比较

```bash
cat "$RUN/train_exit_code.txt"
ls "$RUN/model/best.pt" "$RUN/model/checkpoint_manifest.json" "$RUN/model/history.json"
cat "$RUN/comparison/COMPARISON_CN.md"
```

训练退出码应为0，history完整10轮，checkpoint哈希匹配。自动比较会验证以上条件，逐图重算指标并检查无权CE，不重新推理或训练。

如果自动比较未生成报告，或需要重新生成一个不同名字的报告目录：

```bash
python -m members.member06_research.compare_dev_runs \
  --new-run "$RUN/model" \
  --output "$RUN/comparison_manual_01"
```

该命令只读取两个已完成run及归档身份；输出目录已存在则拒绝。第二轮尚未完成时退出2并列出缺失文件，不生成假结果。不要传历史run充当第二轮。

若团队已有明确的Val PASS允许误拒数量，可追加 `--max-false-fail 整数`，用于报告成本提示，不参与训练、选epoch或修改阈值。未指定时明确“成本是否可接受待团队确认”，不代替团队制定预算。

在VS Code打开 `COMPARISON_CN.md`，可用预览阅读。打开 `curves.png` 看两次无权Val loss/accuracy，打开 `score_distributions.png` 看PASS/FAIL分数分布。`comparison_by_group.csv` 含标注目录外1、6、7及0组的支持数、FN/FP/召回和来源分组。`model/best_val_groups.json` 的errors包含具体误判文件名。

## 8. 如何决策与停止

- A候选：AUC、Balanced Accuracy、FAIL召回同时改善，且团队认为误拒代价可接受，才能保留候选；仍需独立场景确认，不自动替换v1。
- B：检出增加但误拒增加，提示可能主要是工作点移动；结合AUC和分组判断，不能说迁移已解决。
- C：AUC、Balanced Accuracy和异常分组都没有改善，停止继续搜索CE权重，保留负结果。
- D：身份、数据、配置、代码或指标不一致，先修复对照设计；工具失败退出，不输出貌似可信的比较数值。

报告不使用任意“显著改善”百分比冒充统计结论。一次种子、已反复使用的Val只能提供开发证据。若结果混合，不自动判成功，也不自动启动下一次训练。继续的依据应是新证据，而不是在同一Val上反复挑选。
