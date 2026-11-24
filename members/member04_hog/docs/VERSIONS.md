# Baseline1_HOGMLP 统一版本命名

统一采用 `Baseline1_HOGMLP_v版本号_修改或优化内容`。本轮起活动模块只保留 v4；v1/v2/v3 的实现、权重、运行记录与旧文档全部保存在仓库同级归档，Git 历史和已发布 Release 不删除、不改写。

| 统一名称 | 实际方法 | 当前状态 |
|---|---|---|
| Baseline1_HOGMLP_v4_FinalDataset_Rebuild | 固定 `final_training_v4` 上 HOG + MLP 的完整重建；含 `PASSOnly_Autoencoder`、`Supervised_MLP`、`Supervised_Capacity_L2_EpochSelection` 三个方法后缀 | 活动版本；默认模型由本轮冻结产生 |
| 旧版（已归档） | v1 PASS-only 自编码器、v2 监督 CLAHE、v3 稳定性/容量/L2/选轮次 | 仅保留在 `Project/delivery/member04_history_before_v4_20261124/`；详见 [归档映射](ARCHIVE_MAPPING.md) |

## 规则

- 组别与种子放在子目录（如 `A2/seed42`、`R1/seed43`），不虚构为独立版本。
- 注册表 `baseline1 -> members.member04_hog.model` 不变；旧默认权重已从活动目录移除，新默认由 `configs/active_model.json` 指向本轮冻结代表。
- Release 标题使用统一名称；旧 Release 的标签、附件文件名、下载地址与摘要保持不变。
- 旧版成绩不得与 v4 混表，也不能直接当作性能增量；旧版本只在归档映射中保留名称和位置，不抄入新主报告。
