# Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection

本轮先诊断位置、尺度与补白敏感性，再比较R0原网络、R1小网络和R2强正则化。固定数据，不新增采集，不将验证或历史test加入训练。公共baseline1仍使用v1。

## 运行顺序

Python 3.11；依赖沿用上级requirements.txt。所有命令从仓库根目录执行，配置及模型内部路径相对仓库根定位。

```powershell
python -m members.member04_hog.supervised_v3_study.stability
python -m members.member04_hog.supervised_v3_study.train
python -m members.member04_hog.supervised_v3_study.assess
python -m members.member04_hog.supervised_v3_study.report
python -m members.member04_hog.supervised_v3_study.inference --image your_image.jpg
python -m unittest discover -s members/member04_hog/tests -v
```

已有诊断、训练目录禁止覆盖；完整重跑需独立工作区，恢复v2固定来源数据及全部参考资产。不能仅复制研究包就宣称具备全部训练输入。单图显式模型路径可使用`--model weights/member04_hog/supervised_v3_study/experiments/R0/seed42/best.joblib`；实际入选组以freeze.json为准。

批量入口使用`--input_dir`、`--counts_file`、`--output_dir`和可选`--model`，与v2请求格式一致：只接受image_id及expected_counts，不接受真值。五类各2为唯一支持数量；坏图/缺图返回INPUT_ERROR，数量不支持返回UNSUPPORTED_COUNTS。单图缺权重返回MODEL_UNAVAILABLE；批量缺权重沿用v2启动失败行为。未指定模型且缺少freeze.json时明确失败，不回退旧模型。

## 研究约定

- 训练185图，验证101图，原始分区不变。标准化器仅拟合训练原图；B2 CLAHE在训练与推理一致应用。
- 稳定性为每图21设置，原图加两个尺寸、两个补白值、五位置；保留全部画面。相同尺寸比较位置，相同位置尺寸比较填充，不把缩小称为纯位置变化。
- R0=(64,16), alpha=0.0001；R1=(32,8), alpha=0.0001；R2=(64,16), alpha=0.01。三个种子42/43/44各200轮，固定Adam/lr0.001/batch32，按原v2样本权重与顺序规则学习。
- 每轮在验证误拒不超过5的条件下选漏放最少、其次误拒少的阈值；同判定选较大有限代表切点。检查点先漏放少、误拒少，再取早轮次。组间按三种子平均漏放、平均误拒、网络大小、L2依次排序。
- 所有轮次仍训练完成；保存最佳和末轮两份权重、200轮损失及验证分数。R0末轮必须精确复现v2 B2三个种子，否则停止调查。
- 入选组固定seed42。冻结后才对历史test做一次回归。已使用的验证集结果属于开发证据，不保证泛化。

## 产物与版本对应

研究产物位于outputs/member04_hog/supervised_v3_study/；权重位于weights/member04_hog/supervised_v3_study/。这些属于被忽略的开发资产，不进入Git。代码、报告、小型汇总保存在成员4目录。统一版本对应见[版本命名](../docs/VERSIONS.md)。

完整结果见[研究报告](../docs/Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection.md)。新增版本只提供显式入口，不修改公共注册表。

入选R2 seed42第1轮；解释见[结果解读](../docs/Baseline1_HOGMLP_v3_Analysis.md)。此处只是独立研究入口，公共默认仍为v1。

prediction_score是未校准的FAIL分类输出，不与v1重建误差直接比较数值。模型只能做固定数量的整图判定，不输出逐类数量或目标框。

## 交付布局与下载

新发布入口为[Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection](https://github.com/ZhenyuXu0724/huoyan-jinjing/releases/tag/Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection)，发布是否完成以progress.md为准。

ZIP解压到新目录，保留members/、shared/、weights/、outputs/、data/层次。包含18份新检查点、新旧代表模型、训练与诊断记录，不含原始图片、虚拟环境、全部v1/v2参考权重。单独下载的同名joblib可用--model显式指定；若要放入原实验目录，应为weights/member04_hog/supervised_v3_study/experiments/R2/seed42/best.joblib，与冻结摘要一致。

迁移后重建Python3.11虚拟环境并安装上级requirements.txt。先运行`python -m members.member04_hog.delivery --verify .`校验；再用本页单图或批量命令运行。ZIP为已验证的实验快照，后续上传状态说明以Git仓库为准，不通过重打包修改已公布摘要。
