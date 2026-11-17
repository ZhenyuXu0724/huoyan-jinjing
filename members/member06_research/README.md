# 成员 6：Frozen ResNet18 + Linear Baseline 2

当前更新：2026-11-14。**真实数据已到本地，当前训练条件为 D / NOT_READY；本轮未执行真实训练或 Val 选型。**

从这里开始：[中文训练操作指南 TRAINING_GUIDE_CN.md](TRAINING_GUIDE_CN.md)。它解释当前数据、环境激活、逐步检查、上游交接、由你启动训练、结果分析和排错。

审查与测试：[2026-11-14 接入报告](evidence/2026-11-14/REVIEW.md)。网络远端未重新查询；本地 HEAD `90c77fa` 已包含已获取的 upstream/main `f0126f5`。原工程提交为 `8d9ed44`；本轮改动尚未提交/推送。

## 本轮数据结论

真实包位于 `members/member03_data/full_dataset_upload/datasets/parts_v1/`，由普通 Git 追踪，不是 LFS 指针。Train=112、Val=24、Test=24。只读了 Train/Val 的图片和标注，测试数量来自划分元数据，没有读取测试内容。

136 张开发图可解码，图像/标注哈希、GT/原始框、类别和数量一致。问题是：固定数量尚待 Pilot 确认；缺可信 scene_id、expected_counts、整图标签；133 张原标注仍含自动草稿复核标记；平垫片实物规格待确认。全部开发图的已注五类数量均为 2，ID 0 为零，仅凭当前数量不能提供两类监督。完整性通过不代表真值和场景隔离已批准。

因此暂不生成正式训练清单。成员 3/组长需要确认数据版本/使用范围、场景独立性、数量、规格和人工复核，并提供可信 PASS/FAIL 两类开发样本。原分区保留；不补造负类、不重新划分。根 predict.py 的统一 CSV 导出仍待成员 7。

## 环境与立即可运行的检查

当前 `.venv` 已可用，Python 3.11.17、torch 2.7.1+cpu、torchvision 0.22.1+cpu，依赖检查通过。CUDA=False，nvidia-smi 无法连接驱动；不宣称 GPU 已验收。环境和现有预训练缓存均保留，无依赖升级。

```bash
cd /home/grace/projects/AI/member06
source members/member06_research/.venv/bin/activate
export TORCH_HOME="$PWD/members/member06_research/.venv/torch"
export MPLCONFIGDIR="$PWD/members/member06_research/.venv/matplotlib"
python -m pip check
python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only
```

最后一条**当前预期退出 2 并报告 WAITING_UPSTREAM**，不开始训练。配置已指向未来经审核的 `members/member06_research/data/team_parts_v1/{train.csv,val.csv,provenance.json}`，这些文件当前没有生成；数量状态保持 PROVISIONAL_PENDING_PILOT。

本轮已生成的审查/待审批模板在本地忽略目录 `members/member06_research/data/`：team_audit_20261114.json、team_approval_20261114.json、team_reviews_20261114.json。重新检查使用新的输出目录：

```bash
M6_AUDIT_DIR=$(mktemp -d "$PWD/members/member06_research/data/recheck-XXXXXXXX")
python -m members.member06_research.team_data audit --output "$M6_AUDIT_DIR/audit.json" --approval-template "$M6_AUDIT_DIR/approval.json" --review-template "$M6_AUDIT_DIR/reviews.json"
```

审查成功退出 0 只代表完成格式/完整性检查；报告中的 NOT_READY 是科学就绪状态。`prepare` 需要真实审批和逐图审核，当前模板为 PENDING；完整接入和训练命令均在 TRAINING_GUIDE_CN.md，带前置检查，条件不足时不启动训练。

## 代码职责与复用

| 文件 | 职责 / 本轮状态 |
|---|---|
| model.py | 复用官方 ResNet18、512 维特征、Linear(512,2)；冻结 backbone 参数和 BN |
| data.py / image_io.py | 复用图像读取、letterbox/stretched 全视野预处理、清单和 RGB 去重 |
| config.py / configs/baseline2.json | 显式类别/参数；新增 data_provenance，更新真实接入目标路径 |
| team_data.py | 新增 member03.gt.v1 审查、待批模板、获批后转换、来源/场景/标签追溯 |
| preflight.py | 新增批准/哈希/双类 Train+Val/输入批次检查；不创建 CNN 或更新权重 |
| train.py | 复用 CrossEntropyLoss + Adam/head、Val loss 选型、保存；新增 --check-only 和真实数据前置检查/中断提示 |
| validate.py | 复用验证；REAL checkpoint 只能使用其配置中的获批 Val，并重验数据来源 |
| runtime.py | 复用种子、设备、版本/源码哈希、严格 checkpoint 加载 |
| baseline2.py / adapter.py / inference.py | 保留公共延迟加载和独立单图/批量接口，数量规则不进入 CNN |
| plot_history.py | 复用训练曲线和混淆矩阵 |
| smoke_data.py / tests/ | 保留有用的合成测试；新增数据接入测试，不接触真实模型训练 |
| research_tools.py / research_demo.py | Stage 3 ROI、局部 CNN、像素几何保留，当前不执行 Stage 3 实验 |
| final_model.py | 保留未实现状态，最终研究方向待真实 baseline3 错误和组长决策 |

标签配置默认 PASS=0/FAIL=1，与对象 ID 0–5 不同。图像分类输入不使用 YOLO 标签行直接监督。已审核的对象计数全部符合获批 expected_counts 且无目录外才生成 PASS；已可判断的缺失/多出/目录外生成 FAIL；ignore/unjudgeable/未知规格不猜标签。

新增审批/复核 JSON 是**成员 6 本地适配格式，待上游交接确认**，没有改公共 CONTRACT。哈希提供追溯，不是数字签名，也不替代科学批准。

## 训练、保存与推理约定

- 只训练 Linear Head；backbone.train() 后重新强制 eval()，BN 统计量不更新。
- 默认 224×224 letterbox + ImageNet mean/std，不中心裁剪；stretch 可配，无新增随机增强。无 EXIF 旋转，框对应存储图像栅格坐标。
- 最小 Val loss 保存 best.pt，平局保留早期 epoch；不调阈值，softmax argmax，平分选择类别编号 0。
- best.pt 保存完整模型/配置/optimizer/epoch/Val 指标；另存 config、run_config、history、最佳 Val 逐图结果、权重 SHA256。
- 继续要求可信 Train 和 Val 两类齐备。没有 train-only 选型或自动断点续训；中断保留已完成的原子 best.pt，重跑使用新目录。
- 只加载可信 checkpoint。SMOKE_ONLY 必须显式 --allow-smoke；REAL 开发工件仍为 TRAIN_VAL_ONLY，不等于正式冻结。
- 推理输出 score_type=uncalibrated_softmax，prediction_score 是预测类别的分数；不伪造框或实际数量。不支持数量返回 UNSUPPORTED_COUNTS，判定/分数为 null。
- 公共 baseline2.predict 使用 MEMBER06_CHECKPOINT 和可选 MEMBER06_DEVICE；未配置返回 MODEL_NOT_CONFIGURED。独立 JSON CLI 可批量，根统一 CSV 由成员 7 集成。

首次创建模型需要官方 ImageNet 权重；本机缓存已校验，SHA256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。缓存缺失/网络失败会明确报 PRETRAINED_WEIGHTS_UNAVAILABLE，不自动随机降级。

## 安装与安全工程测试

本轮无需重装环境。新机器使用 Python 3.11 在新虚拟环境创建，torch/torchvision 的 CPU 安装示例：

```bash
python3.11 -m venv members/member06_research/.venv-new
members/member06_research/.venv-new/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
members/member06_research/.venv-new/bin/python -m pip install -r members/member06_research/requirements.txt
members/member06_research/.venv-new/bin/python -m pip check
```

当前机器 Python 3.11 基解释器是 `/home/grace/miniconda3/envs/jupyter-env/bin/python`；如果 python3.11 不在 PATH，可用这个实际路径。不要删除现有 .venv；CUDA wheel 必须按目标驱动另行确认。根 requirements.txt 只有团队框架说明，直接依赖在本目录，历史全量快照在 evidence/environment.freeze.txt。

```bash
python -m unittest discover -s members/member06_research/tests -v
python -m unittest discover -s tests -v
python predict.py --list-models
```

本轮成员测试 47 项：46 通过、CUDA 1 项跳过；团队原有 6 项通过。原有测试会在临时目录训练随机头/合成图像，不用真实数据、不改已有权重。新增审批记录只在虚构单元测试夹具中成立，不能当作团队批准。真实数据仅做审查/格式加载；无真实训练、Val 选型、测试结果或 Stage 3 实验。

## 历史和交接

- [MEMBER6_LEARNING_GUIDE.md](MEMBER6_LEARNING_GUIDE.md)：PyTorch/ResNet、模型结构、训练概念和 Stage 3 工具学习。
- [2026-11-05 验收](evidence/ACCEPTANCE.md)：保留当时 SMOKE_ONLY 日志、工件哈希与结论，不改写历史。
- [2026-11-14 接入验收](evidence/2026-11-14/REVIEW.md)：本轮实际文件、数据、环境、测试、清理与外部依赖。
- [progress.md](progress.md)：个人时间线。

没有删除代码、数据、.venv、历史证据或 Stage 3 工具；仅整理过时文档入口和真实数据路径。原有未跟踪根文件保留，其他成员文件和公共规范均未修改。本轮不 commit/push，不自动更新远端或更改 Git 跟踪状态。

技术基础来源沿用：torchvision ResNet18/IMAGENET1K_V1、PyTorch 迁移学习教程、OpenCV 轮廓几何。工程由 Codex 辅助实现；预训练/数据可下载不表示获得再分发授权。正式研究文献、方向和消融待团队依据真实错误决定。
