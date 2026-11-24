# 成员 5 · 完整检测基线与对象级评价

## 最终版 v4 · 纯 YOLOv8n

小组选择纯 YOLO 作为展示方案。版本 `member05-final-v4-pure-yolo`，数据采用 [final_training_v4 固定划分](https://github.com/ZhenyuXu0724/huoyan-jinjing/tree/bde8ea7116761bc8ee7fca1d47c336126daa995f/members/member03_data/final_training_v4)。使用 249 张训练原图、34 张验证原图和 30 张测试原图，从官方 YOLOv8n 预训练权重训练 40 轮。置信阈值 0.50，NMS IoU 0.70，输入尺寸 640。

验证集整图判对 34/34，测试集整图判对 29/30。15 张 FAIL 中误放行 1 张，15 张 PASS 全部通过。测试集六类数量全对 17/30，干扰件正确检出 26/41。现场展示仍需验收。

[下载纯 YOLO 权重包（ZIP）](https://github.com/user-attachments/files/35330452/member05-final-v4-pure-yolo.zip)。压缩包包含 `best.pt`、推理脚本和配置，已完成解压推理验证，并核对了从 GitHub 下载的 ZIP 的 SHA-256。[附件发布记录](https://github.com/ZhenyuXu0724/huoyan-jinjing/pull/13#issuecomment-7131284012)。解压后在 Python 3.11 和 ultralytics 8.3.230 环境运行以下命令。

```sh
python predict.py --image example.jpg
```

权重 `best.pt` 的 SHA-256 为 `6669be04c689e210a1593ae825ae5190cfc8cfb986ce964031c2a249a5e87abe`。压缩包 SHA-256 为 `ad674c5e68a8a680863f5a97433d6d69d605adfa16c9847cdf1728c2c0868728`。[冻结推理配置](experiments/final-v4-pure-yolo/inference.json)

## 职责

YOLOv8n 训练、类别映射、基本尺寸拒绝/必要候选补查、计数、对象框匹配及计数指标。

**交付物：** 完整 baseline3；拒绝机制与阈值；对象匹配评价器；各类 P/R、Count MAE、全向量正确率；错误案例。

**协作：** 成员 2/6 支持尺度与基础校验；成员 3 提供可信框和评价例子；成员 7 负责统一调用与导出。

**验收：** 目标与目录外均可定位；任意合法预期数量可用；对象输出不随预期数量变化；匹配与计数经独立核验。

**边界：** 标准模型优先，不自行扩展多个新架构；候选补查若为必需能力须在 Stage 2 完成；评测由组长复核。

## 提交方式

任务分支 `member05/简短任务名`，仅修改 `members/member05_detector/`。公共接口/跨目录变更交组长整合。时间线见 [progress.md](progress.md)。

## 本模块运行说明（首次代码交付时补齐）

- 当前状态：已建立本地推理/训练入口与评价器原型；无项目训练数据、权重和正式效果。详见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
- 环境与新增依赖：填写本目录 requirements.txt。
- 最小运行命令、安装步骤、输入输出和限制见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
- 输入、输出和样例：遵守 [公共接口](../../shared/CONTRACT.md)。
- 模型/数据版本、访问方式与 SHA256：待填写；不提交大文件和封存 GT。
- 验证命令与真实结果：待填写。
- 已知限制/失败状态：待填写。

## 交接记录

| 日期 | PR/版本 | 交付内容 | 验证证据 | 接收人 | 状态 |
|---|---|---|---|---|---|
| 待填写 | | | | | 未开始 |

## 个人贡献与来源

记录自己负责的代码提交、公开可分享的实验 ID、技术决策、失败案例及外部来源/许可证/AI 辅助。个人报告、同伴评分、私人信息与封存测试真值在课程提交渠道单独管理，不放公开仓库。
