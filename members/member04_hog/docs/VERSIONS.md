# Baseline1_HOGMLP 统一版本命名

统一采用 `Baseline1_HOGMLP_v版本号_修改或优化内容`。冻结版本通过名称映射保留原路径、内部标识和SHA256，不重写权重或撤销旧下载地址。

| 统一名称 | 实际方法 | 保留的实现与资产位置（仓库根相对） |
|---|---|---|
| Baseline1_HOGMLP_v1_PASSOnly_Autoencoder | PASS-only自编码器，A2 seed42 | members/member04_hog/one_class/；weights/member04_hog/baseline1.joblib |
| Baseline1_HOGMLP_v2_Supervised_CLAHE | 监督二分类，B2 seed42 | members/member04_hog/supervised_v2/；weights/member04_hog/supervised_v2/ |
| Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection | 稳定性诊断、小网络与强正则化、验证选轮次 | members/member04_hog/supervised_v3_study/；weights/member04_hog/supervised_v3_study/ |

v3的R0/R1/R2是同一版本内的实验配置，不虚构为三个正式版本。新交付ZIP和发布标题使用统一名称。旧版/v2已公开Release保持原地址和文件名，防止既有下载链接与摘要失效。公共baseline1默认始终为v1，研究版不自动替换。

2026-11-20已将三版Release展示标题统一为表中名称；旧标签、附件文件名、下载地址与校验值均保留。v3发布含5个已校验附件。
