# 成员 6 本轮验收记录

日期：2026-11-05（Asia/Shanghai）。这是独立工程验收，**不是正式模型效果验收**。

## 检查到的初始状态

- `/home/grace/projects/AI/member06` 是成员 Fork（origin=guoyu1106/member06，upstream=ZhenyuXu0724/huoyan-jinjing）；旁边的 `huoyan-jinjing` 是组长副本。初始代码一致，两者工作区均干净。
- 已读取 README、CONTRIBUTING、shared/CONTRACT v0.1、catalogue、组长 PLAN v3.0、decisions 和成员 6 占位实现；没有发现适用的额外 AGENTS.md。
- baseline2/final 原先仅抛 NotImplementedError；根 predict.py 的批量导出也未实现。不重建 Git、不接管成员 7，不修改公共协议或其他成员代码。
- 当前分支为本地 `member06/resnet-baseline`，未 commit / push / PR；上游副本保持干净。
- 默认 Python 3.14.6 没有 torch/torchvision/OpenCV/Pillow；使用已有 Python 3.11.17 创建本目录隔离 .venv。依赖完整快照见 [environment.freeze.txt](environment.freeze.txt)，运行硬件/源码哈希见 [run_environment.json](run_environment.json)。
- 真实 Train/Val、冻结 baseline3、候选/标定实现均未提供。PLAN v3.0 可访问，但仓库未附其引用的原始 Project-Man.pdf。

## 真实执行结果

| 验收项 | 状态 | 证据 |
|---|---|---|
| 成员模块自动测试 | PASS：34 项运行，33 通过，1 跳过，0 失败/错误 | [tests.log](tests.log)，20.969 秒 |
| CUDA 计算测试 | NOT_CERTIFIED：跳过 | CPU wheel，torch.cuda.is_available()=False；显式请求不可用 CUDA 会报错 |
| 团队原有合同/注册/目录规则测试 | PASS：6/6 | [team_tests.log](team_tests.log) |
| 环境依赖一致性 | PASS | [pip_check.log](pip_check.log) |
| 模型创建/输出、冻结、头更新、BN 状态 | PASS | 自动测试比较所有 backbone 参数和缓冲区，并验证头真实变化 |
| Train/Val、最佳保存、重载与重复训练 | PASS（SMOKE_ONLY） | 自动测试真实优化；同 CPU、种子、数据、配置重复训练历史和状态一致 |
| 预训练权重可用 | PASS | 官方 ImageNet 文件，非随机替代；来源和 SHA256 见下方 |
| 预训练完整流水线 | PASS（SMOKE_ONLY） | 224×224、2 epoch，8 合成 Train / 4 合成 Val；[pretrained_train.log](pretrained_train.log) |
| 训练后预训练 backbone 保持不变 | PASS | 与加载官方文件的参考 ResNet 全部参数/BN 缓冲区比较，见 [pretrained_verification.json](pretrained_verification.json) |
| 同权重/图片/预处理推理复现 | PASS（本环境 CPU） | 两次加载 logits 逐位相同，重载 Val loss 相同 |
| 单张/批量/逐图数量文件 CLI | PASS | 自动端到端测试，包含坏图、不支持数量、SMOKE_ONLY 默认拒绝；单图与批量浮点分数允许 1e-6 误差 |
| 适配器后端隔离 | PASS | 新进程使用小型测试后端运行适配器，不导入 torch/torchvision |
| ROI/坐标边界/异常输入 | PASS | 半开框、全图、向外取整、越界、空框、NaN/bool 等 |
| 轮廓和几何安全失败 | PASS | 空白、None、空数组、非 uint8、多轮廓、触边、错误参数均有明确状态 |
| ROI CNN / 原始像素几何 | PASS（工具） | 512 维局部向量；合成矩形 100×30 px，NOT_CALIBRATED |
| README 启动方式 | PASS | tests 内实际 subprocess 调用生成、训练、验证、单图、批量、数量文件、研究演示、绘图 CLI；另实跑预训练版本 |
| 正式预训练 baseline2 | NOT_TRAINED / 未正式验收 | 缺真实 Train/Val 与 Pilot 数量确认，不报告正式指标 |
| 统一应用 CSV 导出 | 待成员 7 集成 | 根入口原有 NOT_IMPLEMENTED 保留；独立 JSON 接口已交付 |
| Stage 3 最终研究模型/对照/消融 | NOT_DECIDED | 仅基础工具，等待冻结 baseline3 的真实错误与组长决策 |

开发中发现并修正了测试夹具属性 `run` 与 unittest.TestCase.run 的命名冲突；上表是修正后的最终实跑结果。没有修改模型行为或指标以强行通过测试。

## 预训练来源与失败处理

来源：`https://download.pytorch.org/models/resnet18-f37072fd.pth`

SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

首次在默认沙箱内失败：`URLError: <urlopen error [Errno -3] Temporary failure in name resolution>`；依赖安装也遇到 DNS 解析失败。工具审批允许网络下载后成功，未使用随机权重冒充预训练。缓存保存在本目录 `.venv/torch/hub/checkpoints/`，不入 Git。

历史官方文件没有显式保存 BN num_batches_tracked；PyTorch 的严格加载器初始化其为零。额外验收通过 torchvision 建立并严格加载官方参考模型后，比较完整参考状态（含这些缓冲区），而非忽略缺失项。

## 本地模型与结果

最终验收目录：`members/member06_research/runs/pretrained_final/`。

- `best.pt` SHA256：`a24c663f48554c89168b3573a4f8c0670a379572391e7976b33a73e987a802b9`。
- `config.json`、`run_config.json`、`checkpoint_manifest.json` 保存设置、来源、版本、源码和数据哈希。
- `reloaded_val.json`、`predictions.json`、`research.json` 和 `curves.png` 是已实际生成的本地工件。
- 供公开审阅的小型记录：[预训练历史](pretrained_history.json)、[验收数值](pretrained_verification.json)。后者含合成 Val 的准确率与混淆矩阵，仅用于确认流水线正常，不能外推真实零件效果。
- 最佳合成 Val loss 为 0.342935711145401，epoch=2；无正式精度/泛化结论。
- 图像、权重和完整运行目录按既有 .gitignore 留在本地；用 smoke_data 生成器复建示例，不上传媒体或大文件。

## 复验命令

从 `/home/grace/projects/AI/member06`：

```bash
M6_PY=members/member06_research/.venv/bin/python
$M6_PY -m unittest discover -s members/member06_research/tests -v
$M6_PY -m unittest discover -s tests -v
$M6_PY -m pip check
$M6_PY predict.py --list-models
```

生成、训练、验证、推理、绘图和研究工具的完整命令见 [README](../README.md)。自动测试默认不下载权重；单独的预训练实跑使用真实缓存。本轮未安装全组框架、未读取封存测试数据、未调用模型 API、未改团队分区/科研决策、未实现 final。

## 后续依赖

真实 Train/Val 与标签/场景版本（成员 3）、Pilot 后数量及接口冻结（组长）、统一评价器（成员 4）、相机标定与误差（成员 2）、YOLO 候选及冻结 baseline3 的错误清单（成员 5）、UI/统一 CSV（成员 7）。完整交接清单见 README。CUDA 分支仍需实际 GPU 环境验收。
