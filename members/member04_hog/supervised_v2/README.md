# baseline1_supervised_v2：完整增强对照

旧版默认 baseline1 保留为 PASS-only HOG＋MLP 自编码器。本目录新增监督式 HOG＋MLP，训练使用 133 PASS＋52 FAIL，不改变公共注册表或旧默认模型。所有命令从仓库根目录运行，Python 3.11；依赖使用成员 4 的锁定 requirements.txt。

## 数据与固定协议

来源提交 `12c85c23bc3a57dea51f39915278f31d9133ce74`；新增104份源文件的 Git blob 摘要见 sources.json。原开发数据仍放在 `data/member04_hog/dataset/`，新版清单放在 `data/member04_hog/supervised_v2/`。清单内图片路径相对于清单所在目录，指向旧数据或同目录 new18；这些相对目录必须一同保留。

训练185图（133 PASS、52 FAIL），验证101图（24 PASS、77 FAIL），历史test24图（23 PASS、1 FAIL）。原分区不变。审计时读取test仅为标注及身份去重，预测在新版冻结之后进行。场景独立性和近重复仍未核实，test已经公开且先前被接触，不是新的独立测试。

固定256×256等比例补白、灰度、9方向HOG、16×16单元、2×2块、L2-Hys。StandardScaler仅拟合训练原图特征。分类PASS=0/FAIL=1，MLP(64,16)、ReLU、Adam、学习率0.001、L2=0.0001、batch=32；每轮每图一个训练视图，200轮保存末轮。样本权重为n/(2×本类数量)。种子42/43/44，11组33次：A0–A5、B1 Gamma、B2 CLAHE、C0=A2+CLAHE、移除geometry与移除clahe。

阈值仅依据验证平衡准确率、少漏放、较高F1、更大阈值依次选择；方案按三种子平均平衡准确率、少漏放、启用项少、视图少、优先单项组排序。固定代表种子42。无测试选型，无新增参数搜索。

## 完整顺序

首次需存在旧数据及旧33次实验。若只有代码，先按旧版 [复现步骤](../docs/WORKFLOW.md) 恢复旧产物；不要重新覆盖已有冻结目录。旧训练权重不因新版而更新。

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.prepare
.\.venv\Scripts\python.exe -m unittest discover -s members/member04_hog/tests -v
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.train
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.assess
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.verify
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.report
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.package --out ../delivery/member04_hog_supervised_v2_20261119.zip
.\.venv\Scripts\python.exe -m members.member04_hog.delivery --verify ../delivery/member04_hog_supervised_v2_20261119.zip
```

prepare严格校验源文件及原数据版本，坏标注、跨分区重复会报错。train只复用协议及摘要一致的完整运行；不完整运行需先人工检查，不自动覆盖；失败保存在独立failure记录。assess要求已冻结模型，输出目录存在时拒绝重新覆盖历史测试访问。report只从记录生成报告。计时在无训练任务同时运行时执行，同机20张共同验证图、预热、单线程、交替先后顺序、重复3遍。

模型、结果、配置资源从模块位置推导的仓库根目录定位；CLI参数相对于当前仓库根目录。新权重位于 `weights/member04_hog/supervised_v2/`，全部运行证据位于 `outputs/member04_hog/supervised_v2/`。这些目录均不进入普通Git提交。

## 新版推理

```powershell
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.inference --image path/to/image.jpg
.\.venv\Scripts\python.exe -m members.member04_hog.supervised_v2.inference --input_dir data/images --counts_file members/member04_hog/examples/expected_counts.json --output_dir outputs/member04_hog/v2_demo
```

替换示例图片和数量请求为真实输入。程序接口为 `supervised_v2.inference.predict(image_path, model_path=DEFAULT)`；默认仅加载新版选择模型，不回退到旧版。保持status、predicted_label、prediction_score、score_type字段；分数为 `uncalibrated_fail_probability`，数值越大支持FAIL，但不是经校准的风险概率。

单图缺权重返回MODEL_UNAVAILABLE，坏图/缺图返回INPUT_ERROR；批量不支持的数量返回UNSUPPORTED_COUNTS并置空判定与分数。批量缺权重明确抛错，不生成成功文件；请求不得携带真值。批量仍用公共model_id=baseline1，run_config另记model_version=baseline1_supervised_v2。新旧输出须分目录保存，不能在只按model_id去重的公共比较器中当作两个不同模型一起输入；本版专用报告按版本比较。

## 报告与交付

新旧报告见 [COMPARISON_V2.md](../docs/COMPARISON_V2.md)。CSV和JSON汇总、逐图预测、训练历史、错误ID、冻结、计时与验证记录均随独立交付包提供。归档包含全部33次新版模型、旧默认模型、旧实验指标和运行代码，不包含原始图片、旧版全部33次权重或虚拟环境。归档逐文件SHA256及外部.zip.sha256用于验收。

本轮不替换旧默认入口，不重新发布或修改旧Release。新模型是否提升由结果报告决定，不以训练成功替代效果结论。
