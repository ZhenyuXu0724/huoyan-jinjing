# 新版 GitHub 交付位置与目录关系

## 代码仓库

仅在 `members/member04_hog/` 内新增和更新代码、说明、小型结果。公共默认baseline1保持旧版，不移动或覆盖旧模型实现。

```text
members/member04_hog/
  README.md                    # 新旧版本总入口
  progress.md                  # 实验及上传状态
  model.py                     # 原版默认入口，保留
  one_class/                   # 原版训练代码，保留
  supervised_v2/               # 新版独立训练、推理、评价与打包
    README.md                  # 新版复现命令
    sources.json               # 固定数据来源及摘要
  docs/
    COMPARISON_V2.md            # 11组新旧对比与分析
    SPLIT_RESULTS_V2.md         # train/valid/test百分比、分子分母与解释
    UPLOAD_V2.md                # 本说明
  tests/test_supervised_v2.py   # 新版回归检查
```

## Release 附件与解压位置

新版发布入口：[member04-supervised-v2-20261119](https://github.com/ZhenyuXu0724/huoyan-jinjing/releases/tag/member04-supervised-v2-20261119)。旧版 [Release](https://github.com/ZhenyuXu0724/huoyan-jinjing/releases/tag/member04-stage1-20261115) 保持原样。

| 附件 | 用途与位置 |
|---|---|
| member04_hog_supervised_v2_20261119.zip | 已验证实验快照，解压到新目录，目录即包的运行根目录 |
| 同名.zip.sha256 | 压缩包摘要 |
| baseline1_supervised_v2.joblib | 单独下载时放在仓库根下 weights/member04_hog/supervised_v2/ |
| member04_v2_relocation_verification.json | 解压迁移验证结果 |
| member04_v2_relocation_tests.txt | 迁移后的检查日志 |

ZIP内部维持仓库相对层次，不能将所有文件平铺到成员目录：

```text
解压根目录/
  members/member04_hog/                  # 运行代码及打包时文档
  shared/                               # 必需公共约定
  weights/member04_hog/
    baseline1.joblib                     # 旧默认模型
    supervised_v2/
      baseline1_supervised_v2.joblib     # 新代表模型
      experiments/组名/seed编号/         # 全部33次新模型
  outputs/member04_hog/
    experiments/                        # 旧实验指标
    supervised_v2/                      # 新配置、历史、预测及对比证据
  data/member04_hog/supervised_v2/        # 清单与来源记录，不含原始图片
  DELIVERY_SHA256.json                   # 包内逐文件校验
```

归档SHA256：`68dd8271e5dc2d3eae5c3d684e9f7910aaad0c0f54b1f80c47550e30a886b807`。
新代表模型SHA256：`77c8239cdb358925e272ffc5c939db5922d54b516a0c05403c89d0d6752d6456`。

ZIP保持已验证实验快照，不因上传说明更新而改写；新增的分区汇总说明及最新上传状态以Git仓库为准。原始图片、检查缩略图、虚拟环境和旧版全部消融权重不在ZIP中。重新训练需按固定来源恢复数据并保留相对目录关系；推理可提供自己的图片。
