# 复现、续跑与交接

所有命令从仓库根目录执行，使用 Python 3.11 和成员4锁定的依赖。`data/`、`weights/`、`outputs/` 为本地忽略目录，Git 克隆不会自动附带；按固定 commit 恢复数据后重新生成。

## 环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r members/member04_hog/requirements.txt
```

不使用旧虚拟环境或旧版本库；重新生成模型需要本机 Python 3.11.18 与 `requirements.txt` 中的版本。

## 完整顺序

```powershell
# 1. 恢复开发数据（283 原图三联 + 13 元数据，862 文件）
python -m members.member04_hog.final_v4.prepare --source-commit bde8ea7116761bc8ee7fca1d47c336126daa995f

# 2. 开发数据审计并锁定协议（66 + 9 任务单）
python -m members.member04_hog.final_v4.audit --scope development

# 3. 66 次主实验（11 组 × 3 种子 × 2 方法）
python -m members.member04_hog.final_v4.train --method ae --protocol members/member04_hog/configs/final_v4/protocol.json
python -m members.member04_hog.final_v4.train --method supervised --protocol members/member04_hog/configs/final_v4/protocol.json

# 4. 9 次有限改进
python -m members.member04_hog.final_v4.refine --protocol members/member04_hog/configs/final_v4/protocol.json

# 5. 冻结三个代表与默认模型（必须在任何 test 解码之前）
python -m members.member04_hog.final_v4.freeze

# 6. 开发稳定性诊断（21 设置）
python -m members.member04_hog.final_v4.diagnose --splits train val

# 7. 恢复全量数据并执行全量审计（必须已有 freeze）
python -m members.member04_hog.final_v4.prepare --scope full
python -m members.member04_hog.final_v4.audit --scope full

# 8. 冻结后一次性 test 评价
python -m members.member04_hog.final_v4.evaluate --frozen

# 9. 报告、验证与打包
python -m members.member04_hog.final_v4.report
python -m members.member04_hog.final_v4.verify
python -m members.member04_hog.final_v4.package --out ../delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip
```

ZIP 校验与解压迁移：

```powershell
python -m members.member04_hog.delivery --verify ../delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip
Expand-Archive ../delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip ../delivery/v4_relocation
python -m members.member04_hog.delivery --verify ../delivery/v4_relocation
python -m unittest discover -s members/member04_hog/tests -v
```

## 断点、失败与禁止事项

- 训练命令按 run 目录和权重摘要复用已完成 run；中断的 run 移入 `outputs/member04_hog/final_v4/failed_attempts/` 后按同配置同种子从头重跑，不保留一半优化器状态、不算额外候选。
- R0 第 200 轮必须精确复现主实验入选组；等价组必须逐位一致。不一致时停止调查，不换候选掩盖。
- 冻结后只允许修复报告文本，不重算 test、不重选模型。若发现影响模型或选择的程序错误，本轮标记无效并停止。
- 数据、来源、清单或训练代码摘要变化时协议失效，必须先重新审计；不能把旧训练目录直接当新结果。
- 不追加容量搜索、L2 网格、学习率搜索、特征截断、前景分割或离线增强训练；75 次之外不增加正式候选。
- 单图入口 `predict(image_path)` 返回 `status`、`predicted_label`、`prediction_score`、`score_type`；缺权重或版本不匹配返回 `MODEL_UNAVAILABLE`，坏图/缺图返回 `INPUT_ERROR`。批量入口只接受 `image_id` 与 `expected_counts`，五类各 2；缺权重启动失败且不创建输出目录。

## 证据位置

| 内容 | 路径 |
|---|---|
| 来源与审计 | `outputs/member04_hog/final_v4/audit/` |
| 协议与状态 | `configs/final_v4/protocol.json`、`outputs/member04_hog/final_v4/{protocol,state}.json` |
| 66 次主实验 | `outputs/member04_hog/final_v4/experiments/`、`weights/member04_hog/final_v4/{ae,supervised}/` |
| 9 次改进 | `outputs/member04_hog/final_v4/refinement/`、`weights/member04_hog/final_v4/refinement/` |
| 冻结与代表 | `outputs/member04_hog/final_v4/freeze.json`、`configs/active_model.json`、`weights/.../representatives/` |
| 诊断与评价 | `outputs/member04_hog/final_v4/{diagnosis,evaluation}/` |
| 报告与验证 | `outputs/member04_hog/final_v4/{reports,verification}/` |
| 历史归档 | 仓库同级 `Project/delivery/member04_history_before_v4_20261124/` |

`state.json` 记录当前阶段、已完成实验、失败原因、test 是否访问和下一条命令；接手时先读它和 `progress.md`。
