# Dataset v2（机械零件视觉检测统一重划分数据包）

本目录在**不修改任何原有数据目录、不改动任何图像像素与 Labelme JSON 字节内容**的前提下，将 `members/member03_data/` 内现有的全部 **609 张图像（399 张真实拍摄原图 + 210 张派生离线增强图）** 与 **609 份同名 Labelme JSON** 按统一科学划分方案实体复制组织为 `Dataset v2`。

## 1. 目录结构

```text
members/member03_data/dataset_v2/
├── train/                        # DEVELOPMENT_TRAIN（共 440 张：原图 296 + 增强 144）
│   ├── PASS/
│   │   ├── original/             # 151 对（图像 + 同名 JSON）
│   │   └── augmented/            # 6 对
│   └── FAIL/
│       ├── original/             # 145 对
│       └── augmented/            # 138 对
├── val/                          # DEVELOPMENT_VAL（共 92 张：原图 74 + 增强 18）
│   ├── PASS/
│   │   ├── original/             # 37 对
│   │   └── augmented/            # 0 对（保留空目录以维持统一结构）
│   └── FAIL/
│       ├── original/             # 37 对
│       └── augmented/            # 18 对
├── test/                         # PROVISIONAL_HOLDOUT（共 77 张：原图 29 + 增强 48）
│   ├── PASS/
│   │   ├── original/             # 13 对
│   │   └── augmented/            # 0 对（保留空目录以维持统一结构）
│   └── FAIL/
│       ├── original/             # 16 对
│       └── augmented/            # 48 对
├── build_dataset_v2.py           # 可复现确定性建库脚本
├── verify_dataset_v2.py          # 独立完整性、字节不变性与防泄漏校验脚本
├── README.md                     # 本说明文档
├── SPLIT_REPORT_CN.md            # 完整划分设计、审查证据与局限说明报告
├── dataset_manifest.csv          # 609 张图像的逐图总账（来源、分区、计数、哈希、质量备注）
├── dataset_statistics.json       # 分区、来源矩阵、异常类型与实例计数统计
├── scene_groups.json             # 候选场景时间块与跨分区旋转不变框重叠核查记录
├── augmentation_mapping.csv      # 210 张离线增强图与 70 张父原图的逐一绑定映射
└── SHA256SUMS.txt                # 全部 1,218 个数据文件的标准 sha256sum 校验清单
```

## 2. 类别编号与整图判定规则

固定参考数量要求为 `[2, 2, 2, 2, 2]`（对应 ID `1..5` 各 2 个，且 `0: non_catalogue` 必须为 `0`）：

| Class ID | 公共目录名称 | Labelme 标签名 | 规格 | 参考预期数量 |
|---:|---|---|---|---:|
| `0` | `non_catalogue` | `non_catalogue` | 目录外类型或错规格（拒绝标签） | `0`（固定禁止） |
| `1` | `long_screw` | `M4x25` | M4 × 25 | `2` |
| `2` | `thick_screw` | `M5x10` | M5 × 10 | `2` |
| `3` | `thin_screw` | `M4x10` | M4 × 10 | `2` |
| `4` | `hex_nut` | `M4_hex_nut` | M4 | `2` |
| `5` | `flat_washer` | `flat_washer` | M5 | `2` |

- `PASS`：`count(0) == 0` 且 `count(1..5) == [2, 2, 2, 2, 2]`。
- `FAIL`：任一目标类数量不等于 `2`，或 `count(0) > 0`（含空检测台场景 `MVIMG_20261120_172557.jpg`）。

## 3. 使用规范与口径要求

1. **训练模式区分**：
   - **仅原图训练**：只读取 `train/{PASS,FAIL}/original/`（共 `296` 张：`151 PASS` + `145 FAIL`）。
   - **原图 + 离线增强训练**：读取 `train/{PASS,FAIL}/{original,augmented}/`（共 `440` 张：`157 PASS` + `283 FAIL`），用于受控研究离线增强的实际收益。
2. **验证与留出评价必须分三口径统计**：
   - **真实照片评价（主指标）**：仅统计 `original/`（Val `74` 张、Test 候选 `29` 张）。
   - **增强照片评价（光照/暖色/柔噪鲁棒性指标）**：仅统计 `augmented/`（Val `18` 张、Test 候选 `48` 张）。
   - **全部照片评价（补充统计）**：合并统计，**严禁将 29 张真实照片 + 48 张派生增强虚报为 77 个独立测试场景**。
3. **`test/` 分区状态为 `PROVISIONAL_HOLDOUT`**：
   - 当前包含 `29` 张真实原图（`13 PASS` + `16 FAIL`）及 `48` 张派生增强图，尚未达到课程 Stage 2 正式独立测试集“至少 30 个已核实独立物理场景且 ≥15 PASS / ≥15 FAIL”的目标；且 `11.14/验证集_不要训练` 用途调整与全组是否接触过 `11.20`/`11.23` 仍需组长与成员 3 签收确认。

## 4. 复现与校验命令

在仓库根目录执行：

```bash
# 重新构建（幂等重建 train/val/test 并重算清单）
python3 members/member03_data/dataset_v2/build_dataset_v2.py

# 独立验证目录配额、父子隔离、原件字节级不变性与非硬链接存储
python3 members/member03_data/dataset_v2/verify_dataset_v2.py

# 使用系统 sha256sum 校验全部 1,218 个复制文件
(cd members/member03_data/dataset_v2 && sha256sum -c SHA256SUMS.txt)
```
