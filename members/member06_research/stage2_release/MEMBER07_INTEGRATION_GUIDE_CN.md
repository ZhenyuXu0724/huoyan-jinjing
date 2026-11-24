# 成员 6 → 成员 7 系统接口交付与跨环境复现指南（Baseline 2 · Frozen ResNet18）

本指南严格对齐主仓库现有的两套真实入口（`members/member07_app/server.py` 现场摄像头检查台与 `predict.py` / `members/member07_app/cli.py` 课程批量命令接口），**不另建重复摄像头系统，也不擅自修改 `members/member07_app/` 下的任何文件**。

---

## 1. 交付文件清单

| 用途 | 仓库相对路径 | 说明 |
|---|---|---|
| **统一推理与现场适配器** | `members/member06_research/stage2_adapter.py` | 同时实现 `server.py` 的 `load(config, root)` 接口、单图 `predict`、批量 `predict_batch` 及 `export_batch` |
| **冻结模型权重** | `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt` | `44,809,257` 字节（通过 GitHub Release / 团队网盘分发后放入该相对路径，或通过配置/环境变量指定） |
| **模型 SHA-256 校验值** | `c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4` | 加载时自动核验，防篡改或下载损坏 |
| **模型配置与推理契约** | `members/member06_research/stage2_release/model_config.json`、`inference_contract.json` | 完整预处理、类别映射、固定数量规则与能力边界 |
| **依赖清单** | `members/member06_research/stage2_release/requirements.txt` | Python 3.11；加载冻结权重**无需联网下载** ImageNet 预训练权重 |
| **`models.json` 接入配置样例** | `members/member06_research/stage2_release/member07_models_baseline2.example.json` | 供成员 7 直接拷入 `members/member07_app/models.json` 的 `"baseline2"` 字段 |
| **开发集接口一致性实测报告** | `members/member06_research/stage2_release/dev_adapter_consistency.json` | 34 张真实 `Val` 上验证离线、单图、批量、字节流、RGB/BGR 与 `server.py` 完全一致 |

---

## 2. 环境准备与教师异地加载说明

模型权重文件（`.pt`）已包含完整的 `ResNet18` 骨干参数与 `Linear(512, 2)` 分类头参数，`load_checkpoint(..., initialize=False)` 在任何离线或评测机器上加载时**均不会访问外网下载预训练模型**，且完全不依赖任何个人电脑绝对路径。

```bash
# 1. 安装依赖（Python 3.11，CPU 环境示例；如有 GPU 则安装对应 CUDA 版本的 torch/torchvision）
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
python -m pip install numpy==2.2.6 Pillow==11.2.1 opencv-python-headless==4.11.0.86

# 2. 将下载的冻结权重放到默认相对路径（或任意自定义路径并通过环境变量/配置指定）：
#    members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
sha256sum members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt
# 期望输出：c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4
```

**三种可配置权重路径方式（优先级从高到低）**：
1. 调用参数 / `models.json` 中的 `"weights"` 字段（支持相对路径，自动按 `members/member07_app/`、当前工作目录或仓库根目录解析）；
2. 环境变量 `MEMBER06_CHECKPOINT=/path/to/baseline2_v4_seed42_best_within_20.pt`（可选 `MEMBER06_DEVICE=cpu|cuda|auto`）；
3. 默认仓库相对路径 `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt`。

---

## 3. 成员 7 现场摄像头检查台（`members/member07_app/server.py`）最小修改接入方案

无需修改 `server.py` 或 `index.html` 任何代码！成员 7 只需更新 `members/member07_app/models.json` 中的 `"baseline2"` 项：

```json
  "baseline2": {
    "adapter": "../member06_research/stage2_adapter.py",
    "version": "baseline2_final_v4_seed42_ep7",
    "weights": "../member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt",
    "sha256": "c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4",
    "device": "auto"
  }
```

### 接口行为核对（100% 匹配 `server.py` 与 `index.html`）
- **单次初始化**：`server.py` 通过 `spec_from_file_location` 调用 `stage2_adapter.load(config, ROOT)` 时仅加载一次模型并缓存在 `CACHE[key]` 中，后续连续拍照/实时检测零重复加载。
- **输入与预处理一致性**：浏览器上传的 `image/jpeg` Base64 解码字节流经 `decode_rgb_image` 转为 `RGB`（无 EXIF 旋转），执行与离线训练/测试完全相同的 `224×224 letterbox`（`pad_rgb=[0,0,0]`、双线性插值、ImageNet 均值方差归一化），并在 `model.eval()` + `torch.inference_mode()` 下前向推理。
- **返回值样例**：
  ```json
  {
    "kind": "image",
    "decision": "FAIL",
    "fixed_expected": [2, 2, 2, 2, 2],
    "score": 0.914238
  }
  ```
  - `score` 含义：**未校准的 `FAIL` 类别概率分数 `P(FAIL) ∈ [0, 1]`**（`> 0.5` 判 `FAIL`，`<= 0.5` 判 `PASS`，`0.5` 平局判 `PASS`）。
  - **100 秒演示第 35–60 秒数量修改联动**：当界面把六角螺母预期数量从 `2` 改为 `3` 时，`index.html` 自动比对 `expected() !== result.fixed_expected` 并显示 **`清单超出该模型支持范围`**，真实反映整图分类基线仅支持固定 `[2, 2, 2, 2, 2]` 的方法学边界。
  - **损坏/空图像保护**：若字节流损坏或为空，`run(raw)` 显式抛出 `ValueError`，`server.py` 返回 HTTP 400 `{"error": ...}` 并清空旧画面判定，绝不返回假 `PASS/FAIL`。

---

## 4. Python 编程接口与课程批量 CLI 调用示例

### 4.1 Python 单图 / 批量 / OpenCV BGR 摄像头帧调用
```python
import cv2
from members.member06_research.stage2_adapter import Stage2ResNetAdapter

# 只需初始化一次（可传 checkpoint="..."，省略则自动读取默认冻结权重或环境变量 MEMBER06_CHECKPOINT）
adapter = Stage2ResNetAdapter(device="auto", batch_size=16)

# 1) 单张磁盘图片推理（默认 expected_counts=[2,2,2,2,2]）
res = adapter.predict("path/to/image.jpg")
# 返回:
# {
#   "status": "OK",
#   "predicted_label": "PASS" 或 "FAIL",
#   "prediction_score": 0.1234,          # P(FAIL) 概率分数
#   "fail_probability": 0.1234,          # P(FAIL)
#   "pass_probability": 0.8766,          # P(PASS)
#   "score_type": "uncalibrated_fail_softmax_prob",
#   "class_scores": {"PASS": 0.8766, "FAIL": 0.1234},
#   "expected_counts": [2, 2, 2, 2, 2],
#   "non_catalogue_count": None          # 绝不伪造框或逐类计数
# }

# 2) OpenCV 摄像头 BGR 帧直接推理（显式指定 color_order="bgr"，内部精确翻转为 RGB）
bgr_frame = cv2.imread("path/to/image.jpg", cv2.IMREAD_COLOR)
res_bgr = adapter.predict(bgr_frame, color_order="bgr")

# 3) 批量图片推理
batch_res = adapter.predict_batch(["img1.jpg", "img2.png"], expected_counts={"1": 2, "2": 2, "3": 2, "4": 2, "5": 2})

# 4) 不支持的预期数量 -> 显式返回 UNSUPPORTED_COUNTS（预测与分数均为 None）
unsup = adapter.predict("path/to/image.jpg", expected_counts={"1": 2, "2": 2, "3": 2, "4": 3, "5": 2})
# unsup["status"] == "UNSUPPORTED_COUNTS", unsup["predicted_label"] is None

# 5) 损坏或不存在的图片 -> 显式返回 INPUT_ERROR
err = adapter.predict("missing.jpg")
# err["status"] == "INPUT_ERROR", err["predicted_label"] is None
```

### 4.2 独立命令行单图与共享协议批量导出（生成 `images.csv` + `run_config.json`）
```bash
# 单图 JSON 推理
python -m members.member06_research.stage2_adapter \
  --image members/member03_data/final_training_v4/images/val/PASS/IMG_7288.png \
  --output /tmp/single_pred.json

# 共享契约批量推理导出（与成员 4 compare.py 及 shared/CONTRACT.md 完全兼容，仅交 images.csv 与 run_config.json）
python -m members.member06_research.stage2_adapter \
  --input-dir path/to/images \
  --counts-file shared/expected_counts.example.json \
  --output outputs/baseline2_batch_run
```

### 4.3 供成员 7 在统一 `members/member07_app/cli.py` 中对接 `baseline2` 的最小调用代码
当成员 7 实现根目录 `python predict.py --input_dir ... --model baseline2 --output_dir ... --counts_file ...` 路由时，针对 `model == "baseline2"` 可直接调用一行完成校验与导出：

```python
from members.member06_research.stage2_adapter import Stage2ResNetAdapter

adapter = Stage2ResNetAdapter()  # 支持环境变量 MEMBER06_CHECKPOINT 覆盖默认冻结权重路径
adapter.export_batch(args.input_dir, args.counts_file, args.output_dir, model_id="baseline2")
```
