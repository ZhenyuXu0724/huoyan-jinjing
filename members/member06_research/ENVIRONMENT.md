# 成员 6（Baseline 2 · Frozen ResNet18）真实运行环境说明

本文档中的所有系统、硬件、Python 与依赖版本均直接采自完成 `final_training_v4` 全部 9 次正式训练与 30 张 `Test` 正式评价的实际运行环境。

---

## 1. 实测硬件与软件环境（已验证）

| 项目 | 实测值 |
|---|---|
| 操作系统 | `Ubuntu 24.04.5 LTS`（`Linux-7.0.0-43-generic-x86_64-with-glibc2.39`） |
| CPU / 内存 | `13th Gen Intel(R) Core(TM) i7-13650HX`（20 逻辑核心），`23 GiB` RAM |
| GPU / CUDA | 无可用 CUDA（`torch.cuda.is_available() == False`，`torch.version.cuda == None`）；全部训练与正式 Test 均在 CPU 确定性模式运行 |
| Python 版本 | `3.11.17`（`pip 24.0`） |
| PyTorch / torchvision | `torch==2.7.1+cpu`、`torchvision==0.22.1+cpu` |
| 核心科学计算与图像库 | `numpy==2.2.6`、`pillow==11.2.1`、`opencv-python-headless==4.11.0.86`、`matplotlib==3.10.3` |
| 完整精确锁包清单 | 见 [requirements-lock.txt](requirements-lock.txt)（21 个包） |
| 最小运行依赖清单 | 见 [requirements.txt](requirements.txt) 与 [stage2_release/requirements.txt](stage2_release/requirements.txt) |

> **跨平台与 GPU 声明**：本模块仅在 **Linux x86_64 CPU** 环境下实际完成全量训练、Test 评价与单元测试。Windows 或 CUDA GPU 环境虽然代码层面通过 `device="auto"` 支持，但**尚未在本机实测验证**，不将其写成已验证环境。

---

## 2. 干净虚拟环境安装步骤

在仓库根目录（`huoyan-jinjing/`）执行：

```bash
# 1. 创建并激活 Python 3.11 虚拟环境（建议不要提交至 Git）
python3.11 -m venv members/member06_research/.venv
source members/member06_research/.venv/bin/activate

# 2. 先安装 CPU 版 PyTorch 与 torchvision（若使用 GPU，请按本机 CUDA 驱动改用对应官方 wheel）
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1

# 3. 安装其余运行依赖
python -m pip install -r members/member06_research/requirements.txt

# 4. 校验依赖完整性
python -m pip check
```

---

## 3. 预训练权重来源与离线/联网说明

1. **直接使用冻结模型进行推理或复现 Test 评价（无需联网）**：
   - 冻结权重 `baseline2_v4_seed42_best_within_20.pt` 已包含完整的 `ResNet18` 骨干与 `Linear(512, 2)` 分类头全部参数。
   - `runtime.load_checkpoint` 与 `stage2_adapter.Stage2ResNetAdapter` 调用 `FrozenResNet18(..., initialize=False)` 加载，**完全离线运行，不访问外网下载任何权重**。
2. **从头重跑 `final_training_v4` 训练实验（首次需缓存 ImageNet 预训练权重）**：
   - 官方来源：`ResNet18_Weights.IMAGENET1K_V1`（`https://download.pytorch.org/models/resnet18-f37072fd.pth`）
   - 官方文件 SHA-256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
   - 若本地未缓存该文件，首次运行训练时会通过 `torchvision` 自动下载至 `$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth`。若网络不可用，程序会明确报错 `PRETRAINED_WEIGHTS_UNAVAILABLE`，绝不静默降级为随机权重。

---

## 4. 常用环境变量说明

| 环境变量 | 作用 | 默认行为 |
|---|---|---|
| `MEMBER06_CHECKPOINT` | 覆盖冻结模型 `.pt` 文件路径（供 `baseline2.py`、`stage2_adapter.py` 或外部系统免改代码切换路径） | 默认读取仓库相对路径 `members/member06_research/stage2_release/weights/baseline2_v4_seed42_best_within_20.pt` |
| `MEMBER06_DEVICE` | 推理设备选择：`auto` / `cpu` / `cuda` | `auto`（无 CUDA 时自动使用 `cpu`） |
| `TORCH_HOME` | 指定 PyTorch 预训练模型缓存目录（重训时需要） | 使用用户默认 `~/.cache/torch`；配套脚本设为 `members/member06_research/.venv/torch` |
| `MPLCONFIGDIR` | Matplotlib 字体与配置缓存目录 | 配套脚本设为临时目录或 `.venv/matplotlib`，避免污染工作区 |
