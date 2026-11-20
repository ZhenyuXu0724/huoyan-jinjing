# 成员 4 · HOG 基线交付指南

本模块交付 baseline1 整图 PASS/FAIL 判定、训练消融和共同图像级评价。默认旧版为 **HOG + PASS-only MLP 自编码器**，遵循旧版当时的“训练只使用 PASS”约束；其 FAIL 仅进入验证阈值选择和冻结后的评测。

2026-11-19 新增独立的 **baseline1_supervised_v2（HOG＋监督式MLP）**，使用133 PASS＋52 FAIL训练，完成全部11组×3种子×200轮及新旧比较。新训练没有合成FAIL，没有把验证集加入权重学习；旧模型、代码、33次实验与默认入口保留。

- [新旧完整对比报告](docs/COMPARISON_V2.md)：同组对照、误拒/漏放、分组错误、消融和同机计时。
- [分区识别结果与解释](docs/SPLIT_RESULTS_V2.md)：train/valid/test百分比、分子分母，以及学习、选型、检验的区别。
- [错误分析与改进建议](docs/IMPROVEMENT_ANALYSIS_V2.md)：阈值诊断、新旧组合候选及有限单因素实验建议，尚未部署或重训。
- [新版运行与交付说明](supervised_v2/README.md)：独立入口、数据合并、冻结和完整复现命令。
- [GitHub上传位置与目录关系](docs/UPLOAD_V2.md)：代码与Release附件的对应位置及解压层次。

新版按平均平衡准确率选择B2（CLAHE）seed42；验证漏放16、误拒2，旧A2 seed42为漏放4、误拒5。历史test新版漏放1、误拒3，旧版为漏放1、误拒10。全部11个同名组的新版平均F1和平衡准确率均低于旧版，不能宣称新版全面提升；默认仍使用旧版。

## 从哪里开始

本次接手诊断见 [第一阶段报告](docs/STAGE1.md)。保持 A2 seed42 权重与阈值不变；新增诊断和真实接口自检，不新增正式训练。上传与验收状态见 progress.md。

| 需要了解的内容 | 说明文件 |
|---|---|
| 有哪些数据、标签如何产生、哪些内容待审核 | [数据说明](docs/DATA.md) |
| 安装、训练、冻结、测试的完整顺序 | [复现步骤](docs/WORKFLOW.md) |
| 每个实验改了什么、为什么比较 | [实验分组](configs/groups/README.md) |
| 指标怎么读、如何理解模型限制 | [方法与结果解读](docs/INTERPRETATION.md) |
| 完整实测表、消融和冻结后失败记录 | [2026-11-15 结果报告](docs/RESULTS.md) |
| 本轮实际数值、选择依据、待办 | [进度与交接](progress.md) |
| 交哪些文件、如何确认接收成功 | [交付清单](docs/DELIVERY.md) |

## 文件与产物位置

```text
members/member04_hog/
  features.py          # 方向校正、补白、变换、HOG
  data.py              # 清单及跨分区重复检查
  train.py             # 训练入口
  one_class/           # 数据准备、训练、重建得分、报告
  model.py             # 加载模型与单图 predict
  inference.py         # 接收数量要求的批量推理
  evaluate.py          # 指标、冻结、冻结后测试
  compare.py           # 多模型共同图像级评价
  delivery.py          # 生成与校验便携交付包
  configs/             # 上游文件校验索引、实验分组
  examples/            # 清单及推理请求样例
  tests/               # 数据隔离、指标与接口回归检查
  docs/                # 数据、复现与解释
data/member04_hog/dataset/          # 原始数据、相对路径清单、来源记录
weights/member04_hog/experiments/   # 按组/seed 保存的模型
weights/member04_hog/baseline1.joblib
outputs/member04_hog/experiments/   # 按组/seed 保存配置、历史、验证预测
outputs/member04_hog/assessment/    # 冻结、访问记录、提供分区评测
```

后三类目录均为本地忽略目录，Git 克隆不会自动附带权重或图片。运行记录入口为 `outputs/member04_hog/experiments/REPORT.md`，原始数字在相邻的 `ablation_table.csv` 与各组 `metrics.json`。迁移时应连同这些本地产物复制，或按复现步骤重新生成。

本轮完成 11 组 × 3 种子 × 200 轮，选择 A2 seed42。验证平均 FAIL F1 为 0.9417；提供的 test 仅 1 张 FAIL，漏放 1 张、误拒 10 张，F1 为 0。当前是可复现的实验基线，尚不具备可靠现场检测的证据，不能只引用验证分数。

## 最小安装与预测

以下命令均从仓库根目录执行，使用 Python 3.11。迁移后重建虚拟环境，不要复制旧 `.venv/`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r members/member04_hog/requirements.txt
.\.venv\Scripts\python.exe -m members.member04_hog.evaluate predict --model weights/member04_hog/baseline1.joblib --image "data/member04_hog/dataset/new/补强数据(训练）/example.jpg"
```

将最后的示例图片路径替换成实际文件。程序也可调用 `from members.member04_hog.model import predict`，再执行 `predict(image_path)`。

返回 `status`、`predicted_label`、`prediction_score`、`score_type`。正常状态为 `OK`；图片失败为 `INPUT_ERROR`；默认权重缺失为 `MODEL_UNAVAILABLE`。得分是 HOG 重建均方误差，**不是 FAIL 概率**，越高越支持 FAIL。

批量入口缺失权重时抛出 FileNotFoundError 并非零退出，不生成输出；这与单图默认入口的 MODEL_UNAVAILABLE 返回方式不同。当前公共 predict.py 仅注册模型，批量联通仍由成员 7 完成。

## 数量与协作边界

当前模型只支持类别 1–5 各 2 个的候选规则；最终规格与数量仍需组长确认。单图 `predict` 默认使用模型内置数量，批量入口检查请求数量，不支持时返回 `UNSUPPORTED_COUNTS`。不输出零件框或实际数量，不能把空值当作“没有目录外零件”。

成员 3 负责场景、规格和标注审核；成员 4 负责基线与图像评价；成员 6 对齐共同指标；成员 7 接入摄像头并测端到端耗时。公共接口及其他成员目录由组长整合。提交前更新本说明与进度。

## 新研究与统一版本名称

参见[版本对应](docs/VERSIONS.md)及[v3研究入口](supervised_v3_study/README.md)。旧版和v2冻结产物不改名、不覆盖；新交付名称按Baseline1_HOGMLP_v版本号_方法命名。
