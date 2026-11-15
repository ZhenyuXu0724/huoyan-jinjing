# 第一阶段诊断（2026-11-15）

复现命令（仓库根目录；输出目录必须不存在）：

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.diagnose --out outputs/member04_hog/assessment/diagnosis_stage1_repeat
```

## 错误图抽查与解释

直接查看了 IMG_20261114_160244.jpg、IMG_20261114_160430.jpg、IMG_7289.png，并以训练正常图 IMG_20261114_160608.jpg 作为外观参考。前两张在 A0/A2 中均漏放，可见小型金属紧固件；第三张在两模型中均误拒，可见侧视、阴影和不同摆放。只记录可见外观，不凭缩略图确认规格或修改标签。原图与诊断缩略图留在本地，不进入代码或交付包。

两模型对 133 张训练 PASS 均判 PASS，在正常验证图上却发生误拒，提示训练与验证正常模式存在差异，不能据此确定原因。A2 seed42 比 A0 少漏放 5 张、多误拒 4 张，平衡准确率下降。验证零漏放需要误拒几乎全部正常图，说明仅调阈值不能同时解决现有漏放与误拒。

验证 PASS 全来自旧包，FAIL 全来自新增包，标签与来源混杂。新增训练 PASS 不能消除验证集本身的混杂；高 F1 不能证明模型只学习了零件异常。验证主要覆盖目录外干扰，不能推广到少件、多件和近规格混放。场景独立性及最终标注仍未确认，不要求新增采集来推进交付。

## 已知 test 与下一阶段

沿用既有冻结后评测：23 PASS、1 FAIL；A0 漏放 1、误拒 5，A2 漏放 1、误拒 10，两者 FAIL F1 均为 0。本次诊断未重新推理 test，也未将其用于重新选型。唯一 FAIL 的少件标签仍以成员 3 标注为准。

现有证据尚不能将特定 HOG 尺度、标准化方式或网络容量确定为主因。第二阶段默认保留现有基线，优先完成共同评价与联调；若提出新训练，先另写有针对性的实验任务单。成员 5/6 需提供共同图像预测，成员 3/组长确认已有真值及数量规则，成员 7 接入公共入口并测端到端延迟。第一阶段上传完成之前不开展第二阶段工作。

只读取既有权重；未新增正式训练，未替换默认模型或冻结阈值。33 次权重校验与指标记录一致，历史均为 200 轮；开发清单与选型记录一致，A0/A2 冻结校验通过。

## 固定 seed42 的结果

本表为集成种子的复算结果，不是三种子均值。训练只有 PASS，不解释其 FAIL F1 或平衡准确率。

| 模型 | 分区 | 样本 | 漏放 | 误拒 | F1 | 平衡准确率 | 覆盖率 | 中位得分 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0_seed42 | train | 133 | 0 | 0 | 不适用 | 不适用 | 100% | 0.3614 |
| A0_seed42 | valid | 101 | 9 | 1 | 0.9315 | 0.9207 | 100% | 1.3311 |
| A2_seed42 | train | 133 | 0 | 0 | 不适用 | 不适用 | 100% | 0.4547 |
| A2_seed42 | valid | 101 | 4 | 5 | 0.9419 | 0.8699 | 100% | 1.3041 |

## 分组统计

新包光照和视角未单独标注，标为 unrecorded，不凭外观自动补标签。各维度独立统计，同一图片会进入不同维度，不能累加各行。

| 模型 | 分组 | 样本 | 漏放 | 误拒 | 中位得分 | 最大得分 |
|---|---|---:|---:|---:|---:|---:|
| A0_seed42 | train/source/new | 21 | 0 | 0 | 0.3632 | 0.4996 |
| A0_seed42 | train/condition/补强数据(训练） | 21 | 0 | 0 | 0.3632 | 0.4996 |
| A0_seed42 | train/lighting/unrecorded | 21 | 0 | 0 | 0.3632 | 0.4996 |
| A0_seed42 | train/view/unrecorded | 21 | 0 | 0 | 0.3632 | 0.4996 |
| A0_seed42 | train/label/PASS | 133 | 0 | 0 | 0.3614 | 0.5640 |
| A0_seed42 | train/source/old | 112 | 0 | 0 | 0.3591 | 0.5640 |
| A0_seed42 | train/condition/暗光侧视图 | 28 | 0 | 0 | 0.2486 | 0.3957 |
| A0_seed42 | train/lighting/dark | 56 | 0 | 0 | 0.3454 | 0.5640 |
| A0_seed42 | train/view/side | 56 | 0 | 0 | 0.3035 | 0.4739 |
| A0_seed42 | train/condition/暗光正视图 | 28 | 0 | 0 | 0.5081 | 0.5640 |
| A0_seed42 | train/view/front | 56 | 0 | 0 | 0.4064 | 0.5640 |
| A0_seed42 | train/condition/标准光侧视图 | 28 | 0 | 0 | 0.3895 | 0.4739 |
| A0_seed42 | train/lighting/standard | 56 | 0 | 0 | 0.3638 | 0.5556 |
| A0_seed42 | train/condition/标准光正视 | 28 | 0 | 0 | 0.3253 | 0.5556 |
| A0_seed42 | valid/source/new | 77 | 9 | 0 | 1.4453 | 99.5040 |
| A0_seed42 | valid/condition/验证集_不要训练 | 77 | 9 | 0 | 1.4453 | 99.5040 |
| A0_seed42 | valid/lighting/unrecorded | 77 | 9 | 0 | 1.4453 | 99.5040 |
| A0_seed42 | valid/view/unrecorded | 77 | 9 | 0 | 1.4453 | 99.5040 |
| A0_seed42 | valid/label/FAIL | 77 | 9 | 0 | 1.4453 | 99.5040 |
| A0_seed42 | valid/source/old | 24 | 0 | 1 | 0.7404 | 1.0295 |
| A0_seed42 | valid/condition/暗光侧视图 | 6 | 0 | 1 | 0.8982 | 1.0295 |
| A0_seed42 | valid/lighting/dark | 12 | 0 | 1 | 0.7058 | 1.0295 |
| A0_seed42 | valid/view/side | 12 | 0 | 1 | 0.7753 | 1.0295 |
| A0_seed42 | valid/label/PASS | 24 | 0 | 1 | 0.7404 | 1.0295 |
| A0_seed42 | valid/condition/暗光正视图 | 6 | 0 | 0 | 0.6929 | 0.7377 |
| A0_seed42 | valid/view/front | 12 | 0 | 0 | 0.7261 | 0.9470 |
| A0_seed42 | valid/condition/标准光侧视图 | 6 | 0 | 0 | 0.7072 | 0.9122 |
| A0_seed42 | valid/lighting/standard | 12 | 0 | 0 | 0.7985 | 0.9470 |
| A0_seed42 | valid/condition/标准光正视 | 6 | 0 | 0 | 0.8243 | 0.9470 |
| A2_seed42 | train/source/new | 21 | 0 | 0 | 0.4383 | 0.5737 |
| A2_seed42 | train/condition/补强数据(训练） | 21 | 0 | 0 | 0.4383 | 0.5737 |
| A2_seed42 | train/lighting/unrecorded | 21 | 0 | 0 | 0.4383 | 0.5737 |
| A2_seed42 | train/view/unrecorded | 21 | 0 | 0 | 0.4383 | 0.5737 |
| A2_seed42 | train/label/PASS | 133 | 0 | 0 | 0.4547 | 0.7526 |
| A2_seed42 | train/source/old | 112 | 0 | 0 | 0.4615 | 0.7526 |
| A2_seed42 | train/condition/暗光侧视图 | 28 | 0 | 0 | 0.4081 | 0.4791 |
| A2_seed42 | train/lighting/dark | 56 | 0 | 0 | 0.4582 | 0.6180 |
| A2_seed42 | train/view/side | 56 | 0 | 0 | 0.4115 | 0.5558 |
| A2_seed42 | train/condition/暗光正视图 | 28 | 0 | 0 | 0.5369 | 0.6180 |
| A2_seed42 | train/view/front | 56 | 0 | 0 | 0.5635 | 0.7526 |
| A2_seed42 | train/condition/标准光侧视图 | 28 | 0 | 0 | 0.4115 | 0.5558 |
| A2_seed42 | train/lighting/standard | 56 | 0 | 0 | 0.4755 | 0.7526 |
| A2_seed42 | train/condition/标准光正视 | 28 | 0 | 0 | 0.6209 | 0.7526 |
| A2_seed42 | valid/source/new | 77 | 4 | 0 | 1.4094 | 93.2157 |
| A2_seed42 | valid/condition/验证集_不要训练 | 77 | 4 | 0 | 1.4094 | 93.2157 |
| A2_seed42 | valid/lighting/unrecorded | 77 | 4 | 0 | 1.4094 | 93.2157 |
| A2_seed42 | valid/view/unrecorded | 77 | 4 | 0 | 1.4094 | 93.2157 |
| A2_seed42 | valid/label/FAIL | 77 | 4 | 0 | 1.4094 | 93.2157 |
| A2_seed42 | valid/source/old | 24 | 0 | 5 | 0.6824 | 1.0167 |
| A2_seed42 | valid/condition/暗光侧视图 | 6 | 0 | 3 | 0.8333 | 1.0167 |
| A2_seed42 | valid/lighting/dark | 12 | 0 | 3 | 0.6714 | 1.0167 |
| A2_seed42 | valid/view/side | 12 | 0 | 3 | 0.7305 | 1.0167 |
| A2_seed42 | valid/label/PASS | 24 | 0 | 5 | 0.6824 | 1.0167 |
| A2_seed42 | valid/condition/暗光正视图 | 6 | 0 | 0 | 0.6598 | 0.6724 |
| A2_seed42 | valid/view/front | 12 | 0 | 2 | 0.6719 | 0.8273 |
| A2_seed42 | valid/condition/标准光侧视图 | 6 | 0 | 0 | 0.6915 | 0.7463 |
| A2_seed42 | valid/lighting/standard | 12 | 0 | 2 | 0.7305 | 0.8273 |
| A2_seed42 | valid/condition/标准光正视 | 6 | 0 | 2 | 0.7530 | 0.8273 |

## 阈值取舍：仅诊断

所有判定切点见交付包的 diagnosis_stage1/*_thresholds.csv。下表各约束取 F1 最大切点，并列取更大阈值；不替换冻结值，也不是现场推荐值。

| 模型 | 约束 | 阈值 | 漏放 | 误拒 | F1 | 平衡准确率 |
|---|---|---:|---:|---:|---:|---:|
| A0_seed42 | 验证零漏放 | 0.548495 | 0 | 24 | 0.8652 | 0.5000 |
| A0_seed42 | 验证零误拒 | 1.030964 | 12 | 0 | 0.9155 | 0.9221 |
| A2_seed42 | 验证零漏放 | 0.531893 | 0 | 23 | 0.8701 | 0.5208 |
| A2_seed42 | 验证零误拒 | 1.021200 | 11 | 0 | 0.9231 | 0.9286 |
