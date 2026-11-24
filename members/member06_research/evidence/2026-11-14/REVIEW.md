# 2026-11-14 Baseline 2 真实数据工程接入审查报告

结论：**工程接入完成；当前数据为 D / NOT_READY；未执行真实训练、正式 Val 选型或 Stage 3 实验。** 操作入口是 [TRAINING_GUIDE_CN.md](../../TRAINING_GUIDE_CN.md)。

## 1. Git 和本轮边界

- 实际根目录 `/home/grace/projects/AI/member06`，分支 `member06/resnet-baseline`，HEAD `90c77fa1091f780150f900a3a8db2cef14fda1bd`。
- HEAD 为用户已完成的上游合并，包含本地 upstream/main 和 origin/main `f0126f5b349c34caf2cbbc8519c95d0cb25d79c2`（11-11）。祖先检查通过，没有未解决冲突，初始追踪文件无未提交修改。
- 原工程已由用户提交为 `8d9ed44`，不是本轮新提交。本轮未 fetch/pull/merge/reset/clean/rebase/commit/push，也未改跟踪状态；不能确认此刻远端是否又有新提交，只对本地已获取版本负责。
- 初始未跟踪根文件 `Net18 baseline and research tools"`（1963 bytes）保留，不认定为无用，也不删除。
- 新增和修改均限成员 6 目录；其他成员、shared、根 requirements 和 Git 历史保持原状。
- 详见 [git_review.json](git_review.json)。

## 2. 实际读取的规范、团队更新与影响

| 文件 | 实际内容/影响 |
|---|---|
| 根 README、CONTRIBUTING、artifacts/README | Python 3.11、成员目录边界、模型注册、媒体/权重/测试隔离规则；仍有效 |
| shared/CONTRACT.md、catalogue.json、contracts.py、expected_counts.example.json | v0.1；五类和 ID 0、半开 xyxy、分类器不输出对象计数、FAIL/PASS/复核语义；数量仍 PROPOSED_CONFIRM_AFTER_PILOT |
| member01_lead/PLAN.md、README、progress、decisions | 可访问的正式计划仍为 v3.0，未发现替代版/数量批准；D01 仍待真实错误，冻结日期和决策不改 |
| member02_setup/README、progress、requirements | 新增 iPhone 7 白底四种条件各 40 张、每次打乱位置；无高度/角度实测、无标定/误差实验；无新依赖 |
| member03_data/README、progress、requirements | 新增完整数据包交付说明、场景/规格/使用范围尚待核验；文档仍写待审核，但 Git 已合并，需区分合并事实和真值验收 |
| member03_data/full_dataset_upload/README_上传说明.md | 普通 Git 完整包含测试图片/GT；声明与原封存、媒体/5 MiB 规则冲突；不据此放宽隔离 |
| parts_v1/README.txt、split_manifest.csv | 160 张按文件名分段，112/24/24；category 表示光照/视角，不表示零件类别；没有 scene_id |
| 统一GT/class_mapping.json、Train/Val 原始 JSON 和统一 GT | member03.gt.v1，显式 ID 与框；平垫片 physical_specification_verified=false；原始自动草稿标记在 GT 中未保留 |
| member06 原 README、progress、学习指南、历史 evidence 与所有核心模块 | 复用 Frozen ResNet18、冻结 BN、letterbox、训练/保存/推理；旧数据入口是成员自定义二分类 CSV，不能直接读取新对象包 |

从初始团队框架 `3958538` 到已获取上游 `f0126f5` 共 491 文件变化。除数据包外，更新集中在成员 2 的 README/progress 和成员 3 的 README/progress/上传说明；组长计划、公共协议和根依赖未更新。没有在本地成员 3 目录找到 Python 转换器；“已写自动标签脚本”的文字说明不能替代可访问实现。

## 3. 数据实际检查与训练状态

真实包：`members/member03_data/full_dataset_upload/datasets/parts_v1/`。完整 full_dataset_upload 为普通 Git 文件；本地没有 .gitattributes。读取的图片都是完整 PNG，不是 LFS 文本指针，不需要另行下载开发图片。

本轮仅打开 Train/Val 的图像/标注；Test 只读取 split_manifest 中的分区、文件名和哈希元数据。没有打开 test 图片、原始标注、统一 GT、测试 summary 或模型结果，也没有对测试真值做转换。

| 检查 | 实测 |
|---|---|
| Train | 112 张，四条件各 28；1120 个已注对象，每类 1–5 各 224，ID 0 为 0 |
| Val | 24 张，四条件各 6；240 个已注对象，每类 1–5 各 48，ID 0 为 0 |
| Test | 清单元数据列 24；内容完整性/标签未检查 |
| 开发文件完整性 | 136/136 解码、图片 SHA256、原始注释来源哈希、尺寸、原始/GT 框、类别和 actual_counts 一致；errors=[] |
| 完全重复 | 划分清单哈希元数据未发现重复，开发 RGB 内容未发现重复；不证明近重复或场景独立 |
| 场景/预期/整图真值 | 136 张均没有可信 scene_id、expected_counts 或整图 PASS/FAIL |
| 自动草稿 | Train 109 张、Val 24 张仍带 auto_label_review 或 human review required，共 133 张 |
| 规格/标定 | 平垫片实物规格未核实；相机未标定，无法声称毫米尺寸能力 |
| 两类监督 | 全部已注数量为 ID 0..5 的 [0,2,2,2,2,2]。固定规则下仅按数量不能生成两类，需要真实错误场景/复核 |
| 正式训练清单 | 未生成；team_parts_v1 目录不存在；无正式训练或 Val 模型选型 |

状态为 **D**：关键监督/规则/分区依据缺失。不是 A（可信 Train+Val）、不是 B（可信 Train）、也还不是 C（规则已批准只差转换）。本工程继续要求 Train/Val 两类齐备；即使将来只有 Train 先到，也不会用 Train loss 代替 Val 选最佳模型。

来源是成员 2 自采、成员 3 本地完整上传包，未自行引入公开规格不明的数据。数据使用范围和正式版本批准仍需交接。源 split_manifest SHA256 为 `a3eac3429ab3f759c952c0fd7ece58019f0c65da4717918987c16dfdad8d1ed6`。

可公开的小型汇总：[data_audit_summary.json](data_audit_summary.json)。逐图 Train/Val 记录和 PENDING 模板保留在本地忽略的 `members/member06_research/data/`，没有将它们伪装成正式监督标签。

## 4. 代码适配、复用和安全失败

- model.py、data.py、runtime.py、baseline2/adapter/inference 均复用；没有改为全量微调或换模型，类别编号和公共协议未改。
- 新增 team_data.py：审查实际 member03.gt.v1；单独生成 PENDING 的本地交接模板；只有真实数据批准和逐图人工复核/场景记录齐备才转换。
- 模板是成员 6 私有适配格式，待团队交接确认，不冒充既有公共规范或签名。所有批准字段初始为空/false/PENDING；不能仅靠修改字段替代实际批准。
- 转换依据获批 expected_counts 和可信对象注释：数量完全符合且无 ID 0 → PASS；已可判断的缺少/多出/目录外 → FAIL。ignore/unjudgeable/未知规格不猜标签，记录排除；未知 ID 不硬映射。
- 转换追踪每张图片、原始注释、GT 的哈希、scene_id、label、分区、两类支持数和排除项；不生成测试清单，不重新划分。每个开发图都必须有对应复核，过期审核拒绝。
- 新增 preflight.py 和 train --check-only：核对批准、固定数量、来源/清单/像素、标签与场景隔离，并加载图像批次。没有 CNN、optimizer 或 backward。REAL 训练不允许暂定数量或缺来源记录。
- train 仍用 CrossEntropyLoss + Adam/head、最小 Val loss 保存 best.pt，保留历史和逐图 Val 结果；新增准备记录哈希和 Ctrl+C 提示。checkpoint 原子 replace 机制保留，没有新增自动续训。
- validate 的 REAL 路径限定为 checkpoint 配置中获批 Val，并重验来源；不顺手执行正式验证。
- 未修改和执行 Stage 3 研究工具；保留将来复用的 ROI/局部 CNN/像素几何，不选择正式改进方向。

## 5. 本轮新增、修改和清理

修改：

- config.py：可选 data_provenance，旧 SMOKE_ONLY checkpoint 仍能加载。
- configs/baseline2.json：指向成员 6 准备后的真实数据版本；数量仍暂定，当前拒绝真实训练。
- train.py、validate.py：数据前置检查、--check-only、校验失败/中断提示与 Val 约束。
- README.md、MEMBER6_LEARNING_GUIDE.md、progress.md：更新实际路径/状态，清除误导的旧真实训练入口；保留历史事实。

新增：team_data.py、preflight.py、tests/test_team_data.py、TRAINING_GUIDE_CN.md，以及本目录本轮证据。实际文件与哈希见 delivery_manifest.json。

清理清单（**没有文件删除**）：

| 路径 | 原用途/是否被调用 | 处置与理由/影响 |
|---|---|---|
| README.md、学习指南第 6 节 | 旧真实数据示例路径与训练入口；供用户操作 | 改为现有团队包和带检查的接入路径，避免误以为 CSV 已存在；历史结论保留在原 evidence |
| smoke_data.py、tests/ | 原有测试仍调用；提供梯度/保存/重载验证 | 保留，不因本轮接真实数据而删有效测试 |
| evidence/ACCEPTANCE.md、原 evidence 日志/JSON、runs/pretrained_final | 原始工程验收/合成实验记录 | 保留，未改成真实实验结果 |
| research_tools.py、research_demo.py | 研究演示和测试调用，后续 Stage 3 可复用 | 保留，当前暂不使用 |
| .venv、权重缓存、依赖快照 | 当前可用环境与预训练源 | 保留，无升级/清空 |
| 根未跟踪文件 Net18 baseline and research tools" | 用途/所有者不明 | 保留，无证据证明可删 |

没有删除图片、标签、划分、其他成员文件或 Git 提交；没有使用 rm -rf、git clean 或重置。

## 6. 环境和本轮安全测试

现有 `.venv` 保留：Python 3.11.17、torch 2.7.1+cpu、torchvision 0.22.1+cpu、NumPy 2.2.6、Pillow 11.2.1、OpenCV 4.11.0.86、matplotlib 3.10.3；pip check 通过，无安装/依赖升级。nvidia-smi 无法连接驱动，CUDA=False，GPU 名称/显存不可核实，不能宣称 GPU 实测。

磁盘剩余约 267 GiB，现有开发数据和模型保存空间充足。现有权重缓存 46,830,571 bytes，SHA256=`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`；本轮官方预训练初始化和合成零张量 `[1,2]` 前向通过，不更新已有权重。

| 实际命令/操作 | 结果 |
|---|---|
| `members/member06_research/.venv/bin/python -m unittest discover -s members/member06_research/tests -v` | 47 项运行，46 通过、CUDA 1 跳过，零失败。耗时见 tests.log。包含旧的临时 SMOKE_ONLY 训练和新增来源/审核/路径/安全失败测试 |
| `members/member06_research/.venv/bin/python -m unittest discover -s tests -v` | 团队 6/6 通过，0.004 秒 |
| `members/member06_research/.venv/bin/python -m pip check` | PASS，No broken requirements found |
| `python -m members.member06_research.team_data audit --output ... --approval-template ... --review-template ...` | 全量 136 开发图检查通过，状态 D/NOT_READY，未生成正式标签 |
| `python -m members.member06_research.team_data prepare --approval members/member06_research/data/team_approval_20261114.json --reviews members/member06_research/data/team_reviews_20261114.json --output members/member06_research/data/team_parts_v1` | 预期退出 2，WAITING_UPSTREAM，未创建正式清单目录 |
| `python -m members.member06_research.train --config members/member06_research/configs/baseline2.json --check-only` | 预期退出 2，PRECHECK_FAILED/WAITING_UPSTREAM，无 CNN/优化器/训练 |
| 官方缓存初始化 + 合成零张量前向 | PASS，输出 [1,2]；不是模型效果测试 |
| Python AST/JSON/中文指南 Bash 语法、模块 --help、Git diff --check | 详见 static_checks.json，无真实训练操作 |

日志：[tests.log](tests.log)、[team_tests.log](team_tests.log)、[pip_check.log](pip_check.log)、[real_preflight.log](real_preflight.log)、[pending_prepare.log](pending_prepare.log)、[environment.json](environment.json)。测试中虚构批准记录只存在于隔离临时夹具，用于检查转换协议；没有拿它批准真实数据。

旧的 11-05 验收证据不覆盖、不升级；新的全部文件独立放在本日期目录。模型/配置保存机制和测试隔离维持原设计。

## 7. 下一步由谁提供什么

| 负责人 | 具体缺口 |
|---|---|
| 组长 | Pilot 后固定数量决策、实际数据使用与版本批准、公共接口/评价器/冻结规则；处理上游测试公开与原封存规定冲突 |
| 成员 3 | 原始自动草稿的人工复核、可信 scene_id 与跨分区独立性、完整可判断标注、可信整图标签/两类来源，必要时组织真实 FAIL 数据 |
| 成员 2/组长 | 五类实物规格核对，特别是平垫片；若需要尺寸能力，另提供标定版本和误差，不能用当前像素结果冒充毫米 |
| 成员 4 | 统一图像级评价器、FAIL 正类、弃判/覆盖率口径 |
| 成员 7 | 根 predict.py 的 UI/统一批量导出集成 |
| 成员 5/组长 | 冻结 baseline3 和真实错误，供之后 Stage 3 研究选择；本轮未推进 |

原始 Project-Man.pdf 在当前仓库仍未提供；没有发现比 PLAN v3.0 更高的可访问正式规定。上游文件已进入 Git，不代表上述科学/标签批准已完成。你可按 TRAINING_GUIDE_CN.md 先完成所有安全检查，待上游补齐后本人执行 prepare、preflight 和训练。
