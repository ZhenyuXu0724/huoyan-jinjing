# 旧版归档映射

v4 重建前，成员4活动模块、旧数据副本、旧权重、旧运行输出和旧交付产物已归档到仓库同级目录：

```text
Project/delivery/member04_history_before_v4_20261124/
  README_RESTORE.md
  legacy_inventory.json
  legacy_cleanup_report.json
  repository/members/member04_hog/...
  repository/data/member04_hog/...
  repository/weights/member04_hog/...
  repository/outputs/member04_hog/...
  delivery/...
```

`legacy_inventory.json` 记录 1489 个文件、2,926,577,008 字节的源路径、归档路径、大小与 SHA256；复制完成后逐文件复核，`verification.result=PASS`。只有全部摘要通过后才按白名单清理原位置；`legacy_cleanup_report.json` 记录删除的 1419 个文件、删除后归档复查结果和未处理项。

## 旧版本映射

| 旧统一名称/位置 | 方法与代表模型 | 归档位置 | 公开记录 |
|---|---|---|---|
| `Baseline1_HOGMLP_v1_PASSOnly_Autoencoder`；`members/member04_hog/one_class/`、`weights/member04_hog/baseline1.joblib`、`outputs/member04_hog/{experiments,assessment}` | HOG + PASS-only 自编码器；A2 seed42 | `repository/members/member04_hog/one_class/`、`repository/weights/member04_hog/baseline1.joblib`、`repository/outputs/member04_hog/` | PR #5；Release `member04-stage1-20261115` |
| `Baseline1_HOGMLP_v2_Supervised_CLAHE`；`members/member04_hog/supervised_v2/`、`weights/member04_hog/supervised_v2/`、`data/member04_hog/supervised_v2/` | 监督二分类；B2 seed42 | 对应 `repository/` 下同名路径 | PR #7；Release `member04-supervised-v2-20261119` |
| `Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection`；`members/member04_hog/supervised_v3_study/`、`weights/member04_hog/supervised_v3_study/` | R0/R1/R2 稳定性、容量与 L2；R2 seed42 | 对应 `repository/` 下同名路径 | PR #8；Release `Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection` |
| 旧交付包与迁移证据 | v1/v2/v3 ZIP、SHA256、代表权重、验证 JSON/TXT | `delivery/` | 旧 Release 附件保持原地址 |

旧 v1/v2/v3 的标签、附件文件名、下载地址和 SHA256 未修改；本归档只是把本地活动副本移出重建范围。成员4旧默认权重 `weights/member04_hog/baseline1.joblib` 不再存在于活动目录，也不被 v4 入口加载。

## 退役文件

以下旧文件已从活动模块删除，只保留在归档：

- `one_class/`、`supervised_v2/`、`supervised_v3_study/` 全部代码与子 README。
- 旧 `docs/`：`RESULTS.md`、`STAGE1.md`、`DELIVERY.md`、`INTERPRETATION.md`、`COMPARISON_V2.md`、`SPLIT_RESULTS_V2.md`、`UPLOAD_V2.md`、`IMPROVEMENT_ANALYSIS_V2.md`、`Baseline1_HOGMLP_v3_Analysis.md`、`Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection.md`、`Baseline1_HOGMLP_v3_summary.json`。
- 旧 `tests/`：`test_one_class.py`、`test_supervised_v2.py`、`test_supervised_v3_study.py`、`test_diagnose.py`。
- `configs/groups/`、`configs/member03_sources.json`、`diagnose.py`、`verify_stage1.py`。
- 旧 `examples/` 中仅适用于旧模型的说明。

旧成绩、旧混淆矩阵和旧上传状态只留在归档和旧 Release；新报告不以它们作为对照或性能增量。

## 未处理项

`Project/delivery/v3_relocation_3cyz5ths` 在归档时被 ACL 拒绝读取，未复制、未删除，保留在原位。该目录属于旧 v3 迁移验证产物，不参与 v4 训练或评测；如需处理，应由有权限的用户确认 ACL 后单独归档。

## 还原方法

把 `repository/` 下内容按相对路径复制回 `Project/huoyan-jinjing/`，把 `delivery/` 下内容复制回 `Project/delivery/`；按 `legacy_inventory.json` 重新计算 SHA256 核对。不要用归档中的旧模型参与 v4 训练、阈值选择、默认入口或 test 评价。
