# 完整数据集上传包

这是用户要求整理的完整本地上传候选包，尚未上传任何远端。

## 内容

```text
datasets/parts_v1/
  train/拍摄条件/图片.png + 同名Labelme.json
  val/拍摄条件/图片.png + 同名Labelme.json
  test/拍摄条件/图片.png + 同名Labelme.json
  统一GT/train、val、test/拍摄条件/同名.json
  统一GT/class_mapping.json
  统一GT/summary.json
  split_manifest.csv
```

完整复制了原“数据集划分”目录中的全部文件，原文件仍在原位置。
仅上传副本中的 split_manifest.csv 改为包内相对路径；其中 source 和 destination 均指向包内原图，sha256仍是对应PNG的哈希。
Labelme 标注和统一 GT 均保留；图片不缩放、不压缩，零件框和类别不改动。
train 112张、val 24张、test 24张，总160张图片、1599个对象。
Labelme 标签名称与公共 class_id 的映射见 统一GT/class_mapping.json。
其中 non_catalogue=0；本数据没有目录外样本，平垫片实物规格尚待确认。
当前划分仍待 scene_id 独立性核验。

## 上传位置和分批方式

若按用户本次要求上传完整数据，本次提交位置为 members/member03_data/full_dataset_upload/datasets/parts_v1/。
PNG共约906 MiB；完整目录约907 MiB，单张不超过7 MiB。
无需把全部图片压成一个大ZIP上传。网页上传可按 split/拍摄条件 分批，单次文件数不超过100。
SHA256SUMS.txt 可用于完整性校验。
程序代码、开发材料仍在旁边的 GitHub上传包/members/member03_data，不在本数据包重复放置。

## 与仓库现有约定的区别

本包明确包含测试图片和测试真值，不能称为已脱敏开发包。
仓库目前约定大批原图不提交普通Git、封存测试图片与GT不公开，并限制成员3改动自己的模块目录。
因此正式提交到该仓库前，需要由组长处理数据存储/公开范围及跨目录集成；如保持原约定，本包可作为受控外部存储交接包。
本次整理没有改变仓库规则，也没有实际发布数据。
