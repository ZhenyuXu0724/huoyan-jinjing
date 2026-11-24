"""Stage 2 official pre-test model freeze for Member 6 Baseline 2 (Frozen ResNet18 + Linear).

Locates and verifies the exact selected checkpoint V4-R2_RAW_WEIGHTED_CE_SEED42 (best_within_20.pt, Epoch 7),
copies it byte-for-byte into stage2_release/weights/, and writes immutable pre-test freeze records.
Never retrains, never reinitializes, and never reads Test images or Test labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path

import torch

from .config import Config
from .runtime import environment, file_sha256, load_checkpoint, write_json

MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]

EXPECTED_SOURCE_CKPT_REL = (
    "members/member06_research/runs/final_training_v4_20261124_192110_PJSHyF/"
    "V4-R2_RAW_WEIGHTED_CE_SEED42/model/best_within_20.pt"
)
EXPECTED_SOURCE_SHA256 = "c38c270ae8eb5ab1dd02d4903c553a7d11fa47735b46bff55302413826b2e7a4"
EXPECTED_UPSTREAM_V4_COMMIT = "bde8ea7116761bc8ee7fca1d47c336126daa995f"
EXPECTED_TRAIN_MANIFEST_SHA256 = "8e77e41fc3da98f829d1cd65466daf58e3acdb8c5e6c4f1df0a8716be6115f3b"
EXPECTED_VAL_MANIFEST_SHA256 = "4d5a04fdfe9a1b53a3c8aeb5bd9f8ea00ccafa15c29d3b46b8575914965e9c05"
EXPECTED_SPLIT_MANIFEST_SHA256 = "80b4e7500c0ff88236ba4a5252eba735cc9bad8ac001442507265726d7fccc42"
EXPECTED_PRETRAINED_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"

FROZEN_WEIGHT_FILENAME = "baseline2_v4_seed42_best_within_20.pt"


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    overall = hashlib.sha256()
    head_hash = hashlib.sha256()
    backbone_hash = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        raw = tensor.numpy().tobytes()
        entry = f"{key}:{tuple(tensor.shape)}:{tensor.dtype}:".encode("utf-8")
        overall.update(entry)
        overall.update(raw)
        if key.startswith("head."):
            head_hash.update(entry)
            head_hash.update(raw)
        elif key.startswith("backbone."):
            backbone_hash.update(entry)
            backbone_hash.update(raw)
    return {
        "full_model_state_sha256": overall.hexdigest(),
        "backbone_state_sha256": backbone_hash.hexdigest(),
        "linear_head_state_sha256": head_hash.hexdigest(),
    }


def freeze_stage2_model(
    source_ckpt: Path | None = None,
    release_dir: Path | None = None,
    *,
    expected_sha256: str | None = EXPECTED_SOURCE_SHA256,
) -> dict:
    src = (source_ckpt or (REPO_ROOT / EXPECTED_SOURCE_CKPT_REL)).resolve()
    out_dir = (release_dir or (MEMBER_DIR / "stage2_release")).resolve()

    if not src.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {src}")
    freeze_manifest_path = out_dir / "freeze_manifest.json"
    if freeze_manifest_path.exists():
        raise FileExistsError(f"Freeze manifest already exists and is immutable: {freeze_manifest_path}")

    actual_sha256 = file_sha256(src)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")

    # Load and strictly validate checkpoint via existing runtime loader (no internet download)
    model, cfg, device, payload = load_checkpoint(src, device="cpu")
    if cfg.training_tier != "FORMAL_READY" or cfg.data_kind != "REAL":
        raise ValueError("Selected checkpoint must be REAL + FORMAL_READY")
    if cfg.seed != 42 or cfg.image_size != 224 or cfg.resize_mode != "letterbox":
        raise ValueError("Selected checkpoint does not match seed=42, 224x224 letterbox ruling")
    if cfg.epochs != 20 or cfg.batch_size != 16 or cfg.learning_rate != 0.001 or cfg.weight_decay != 0.0:
        raise ValueError("Selected checkpoint training hyperparameters do not match V4-R2 ruling")
    if cfg.train_class_weight != "balanced":
        raise ValueError("Selected checkpoint must use train_class_weight='balanced'")
    if cfg.class_to_idx != {"PASS": 0, "FAIL": 1} or cfg.fail_label != "FAIL":
        raise ValueError("Selected checkpoint class mapping must be PASS=0, FAIL=1")
    if cfg.supported_counts != [2, 2, 2, 2, 2]:
        raise ValueError("Selected checkpoint supported_counts must be [2, 2, 2, 2, 2]")
    if payload.get("epoch") != 7:
        raise ValueError(f"Expected selected epoch=7, got {payload.get('epoch')}")

    run_meta = payload.get("run_metadata", {})
    sel_proto = run_meta.get("selection_protocol", {})
    feat_cache = run_meta.get("feature_cache", {})
    if sel_proto.get("test_data_used") is not False or feat_cache.get("test_cached") is not False:
        raise ValueError("Selected checkpoint must have test_data_used=False and test_cached=False")
    if run_meta.get("experiment_id") != "V4-R2_RAW_WEIGHTED_CE_SEED42":
        raise ValueError(f"Unexpected experiment_id: {run_meta.get('experiment_id')}")
    if run_meta.get("upstream_v4_commit") != EXPECTED_UPSTREAM_V4_COMMIT:
        raise ValueError("Unexpected upstream_v4_commit in checkpoint metadata")

    loss_proto = run_meta.get("loss_protocol", {})
    if loss_proto.get("weights") != [0.75, 1.5] or loss_proto.get("validation") != "unweighted CE mean":
        raise ValueError(f"Unexpected loss_protocol in checkpoint: {loss_proto}")

    manifests_meta = run_meta.get("manifests", {})
    if manifests_meta.get("train_manifest_sha256") != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise ValueError("Train manifest SHA256 mismatch in checkpoint metadata")
    if manifests_meta.get("val_manifest_sha256") != EXPECTED_VAL_MANIFEST_SHA256:
        raise ValueError("Val manifest SHA256 mismatch in checkpoint metadata")
    if manifests_meta.get("split_manifest_sha256") != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError("split_manifest SHA256 mismatch in checkpoint metadata")

    pretrained_meta = run_meta.get("pretrained_source", {})
    if pretrained_meta.get("sha256") != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("Pretrained ResNet18 SHA256 mismatch in checkpoint metadata")

    val_metrics = payload.get("val_metrics", {})
    if val_metrics.get("n") != 34 or val_metrics.get("confusion_matrix") != [[16, 4], [0, 14]]:
        raise ValueError(f"Unexpected validation confusion matrix: {val_metrics.get('confusion_matrix')}")
    if val_metrics.get("false_pass") != 0 or val_metrics.get("false_fail") != 4:
        raise ValueError("Expected FN=0, FP=4 on 34-image Val")

    # Cross-check companion best.pt in same model directory has identical model_state tensors
    state_hashes = _state_dict_sha256(payload["model_state"])
    sibling_best = src.parent / "best.pt"
    sibling_best_verified = False
    if sibling_best.is_file():
        sibling_payload = torch.load(sibling_best, map_location="cpu", weights_only=True)
        sibling_hashes = _state_dict_sha256(sibling_payload["model_state"])
        if sibling_hashes != state_hashes:
            raise ValueError("best_within_20.pt and best.pt model_state tensors differ")
        sibling_best_verified = True

    budget_eval_path = src.parent / "budget_evaluations.json"
    train_249_metrics = None
    if budget_eval_path.is_file():
        be = json.loads(budget_eval_path.read_text(encoding="utf-8"))
        b20 = be.get("budgets", {}).get("20", {})
        if b20.get("sha256") != actual_sha256 or b20.get("selected_epoch") != 7:
            raise ValueError("budget_evaluations.json disagrees with best_within_20.pt")
        train_249_metrics = b20.get("unified_primary_train_metrics")

    total_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())

    # Create release directory structure and copy frozen checkpoint byte-for-byte
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    frozen_ckpt_path = weights_dir / FROZEN_WEIGHT_FILENAME
    shutil.copy2(src, frozen_ckpt_path)
    copied_sha256 = file_sha256(frozen_ckpt_path)
    if copied_sha256 != actual_sha256:
        raise RuntimeError("Copied frozen checkpoint SHA256 mismatch")

    frozen_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env_info = environment(device)

    portable_config = cfg.to_dict()
    write_json(out_dir / "model_config.json", portable_config)

    inference_contract = {
        "schema_version": "member06.stage2_inference_contract.v1",
        "model_id": "baseline2",
        "model_version": "baseline2_final_v4_seed42_ep7",
        "architecture": "FrozenResNet18 (IMAGENET1K_V1, backbone+BN frozen) + Linear(512, 2)",
        "weights_relpath": f"members/member06_research/stage2_release/weights/{FROZEN_WEIGHT_FILENAME}",
        "weights_sha256": copied_sha256,
        "weights_bytes": frozen_ckpt_path.stat().st_size,
        "class_to_idx": {"PASS": 0, "FAIL": 1},
        "idx_to_class": {"0": "PASS", "1": "FAIL"},
        "positive_class": "FAIL",
        "supported_expected_counts_list": [2, 2, 2, 2, 2],
        "supported_expected_counts_dict": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2},
        "unsupported_counts_status": "UNSUPPORTED_COUNTS",
        "invalid_image_status": "INPUT_ERROR",
        "preprocessing": {
            "color_space": "RGB (no EXIF auto-rotation; BGR camera arrays explicitly reversed to RGB before PIL conversion)",
            "image_size": 224,
            "resize_mode": "letterbox",
            "interpolation": "PIL.Image.Resampling.BILINEAR",
            "pad_rgb": [0, 0, 0],
            "pad_placement": "centered ((224 - width) // 2, (224 - height) // 2)",
            "tensor_scale": "[0.0, 1.0] float32 CHW",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "inference_mode": {
            "model_eval": True,
            "grad_enabled": False,
            "batchnorm_running_stats_frozen": True,
            "internet_pretrained_download_on_load": False,
        },
        "decision_rule": {
            "type": "argmax_uncalibrated_softmax",
            "probabilities": "softmax(logits, dim=1) -> [P(PASS), P(FAIL)]",
            "fail_score_definition": "P(FAIL) = softmax(logits)[1] (uncalibrated FAIL probability in [0, 1])",
            "predict_fail_condition": "P(FAIL) > P(PASS) (strictly > 0.5)",
            "tie_break_rule": "P(FAIL) == P(PASS) chooses class index 0 (PASS), matching max(range(2), key=probs.__getitem__)",
            "threshold_tuning_used": False,
        },
        "honest_capability_boundaries": {
            "outputs_bounding_boxes": False,
            "outputs_per_class_counts": False,
            "outputs_non_catalogue_count": False,
            "non_catalogue_count_value_in_csv": None,
        },
    }
    write_json(out_dir / "inference_contract.json", inference_contract)

    freeze_manifest = {
        "schema_version": "member06.stage2_freeze.v1",
        "status": "FROZEN_PRE_TEST",
        "frozen_at_utc": frozen_at_utc,
        "test_used_for_training_or_selection": False,
        "test_evaluated_prior_to_freeze": False,
        "post_test_modification_forbidden": True,
        "selected_model": {
            "model_id": "baseline2",
            "version_tag": "baseline2_final_v4_seed42_ep7",
            "experiment_id": "V4-R2_RAW_WEIGHTED_CE_SEED42",
            "strategy_id": "V4-R2_RAW_WEIGHTED_CE",
            "seed": 42,
            "max_budget_epochs": 20,
            "selected_epoch": 7,
            "selection_criterion": "minimum unweighted Val CrossEntropyLoss within 20-epoch budget; earliest epoch wins ties",
            "source_checkpoint_relpath": EXPECTED_SOURCE_CKPT_REL,
            "frozen_checkpoint_relpath": f"members/member06_research/stage2_release/weights/{FROZEN_WEIGHT_FILENAME}",
            "checkpoint_sha256": copied_sha256,
            "checkpoint_size_bytes": frozen_ckpt_path.stat().st_size,
            "sibling_best_pt_state_verified_identical": sibling_best_verified,
            "parameter_counts": {
                "total": total_params,
                "frozen_backbone": backbone_params,
                "trainable_linear_head": head_params,
            },
            "state_dict_sha256": state_hashes,
        },
        "training_and_loss_protocol": {
            "architecture": "FrozenResNet18 + Linear(512, 2)",
            "pretrained_backbone": {
                "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
                "official_url": pretrained_meta.get("url"),
                "checkpoint_sha256": pretrained_meta.get("sha256"),
                "backbone_and_bn_frozen": True,
            },
            "train_loss": "Weighted CrossEntropyLoss",
            "train_class_weights_pass0_fail1": [0.75, 1.5],
            "val_loss": "Unweighted CrossEntropyLoss mean",
            "optimizer": "Adam(lr=0.001, weight_decay=0.0)",
            "batch_size": 16,
            "drop_last": False,
            "online_or_offline_augmentation": "none (249 raw training originals only)",
            "exposures_per_epoch": 249,
            "optimizer_steps_per_epoch": 16,
            "total_budget_exposures": 4980,
            "total_budget_optimizer_steps": 320,
            "selected_epoch_cumulative_steps": 112,
        },
        "class_and_decision_protocol": {
            "class_to_idx": {"PASS": 0, "FAIL": 1},
            "positive_class": "FAIL",
            "supported_expected_counts": [2, 2, 2, 2, 2],
            "counts_rule_status": "CONFIRMED",
            "decision_rule": "argmax over uncalibrated 2-class softmax [P(PASS), P(FAIL)]; ties at 0.5 choose index 0 (PASS)",
        },
        "dataset_and_git_identity": {
            "dataset_id": "final_training_v4",
            "dataset_relpath": "members/member03_data/final_training_v4",
            "upstream_v4_commit": EXPECTED_UPSTREAM_V4_COMMIT,
            "member06_git_head_at_training": run_meta.get("environment", {}).get("git_head"),
            "member06_git_head_at_freeze": env_info.get("git_head"),
            "manifests_sha256": {
                "train_csv": EXPECTED_TRAIN_MANIFEST_SHA256,
                "val_csv": EXPECTED_VAL_MANIFEST_SHA256,
                "test_csv": file_sha256(REPO_ROOT / "members/member03_data/final_training_v4/manifests/resnet/test.csv"),
                "split_manifest_csv": EXPECTED_SPLIT_MANIFEST_SHA256,
            },
            "partition_counts": {
                "train_raw_originals": {"total": 249, "PASS": 166, "FAIL": 83},
                "val_raw_originals": {"total": 34, "PASS": 20, "FAIL": 14},
                "test_raw_originals": {"total": 30, "PASS": 15, "FAIL": 15},
            },
        },
        "validation_and_train_reference_metrics": {
            "val_34_metrics": val_metrics,
            "train_249_fixed_checkpoint_metrics": train_249_metrics,
        },
        "selection_rationale": {
            "ruling_1_strategy": (
                "Across 3 seeds (42/43/44) on final_training_v4 (20-epoch budget), V4-R2_RAW_WEIGHTED_CE ([0.75, 1.50]) "
                "improved mean Val Balanced Accuracy from 0.8643 +- 0.0357 (V4-R1 unweighted CE) to 0.8881 +- 0.0206, "
                "and reduced mean False PASS (FN) from 1.00 +- 1.00 to 0.33 +- 0.58 without worsening mean FP (4.00)."
            ),
            "ruling_2_budget": (
                "Retained 20-epoch max budget with minimum unweighted Val CE checkpoint selection: improved Seed 43 "
                "(Epoch 14 vs Epoch 8, BA 0.8393 -> 0.8643, FP 5 -> 4) while keeping Seed 42 (Epoch 7) and Seed 44 (Epoch 10) unchanged."
            ),
            "ruling_3_resolution": (
                "Controlled 384x384 experiment (V4-R4_RAW_WEIGHTED_CE_384, seeds 42/43/44) failed all 4 pre-registered "
                "upgrade criteria: mean Val BA dropped -0.0405 (0.8881 -> 0.8476), mean FN worsened (0.33 -> 1.00), "
                "mean FP worsened (4.00 -> 4.67), Seed 42 regressed (BA 0.9000 -> 0.8393, FN 0 -> 1, FP 4 -> 5), "
                "and offline single-image CPU median latency increased 1.28x (161.76 ms -> 207.25 ms)."
            ),
            "ruling_4_precommitted_seed": (
                "Representative delivery seed was pre-committed to seed=42 prior to comparing seeds (no post-hoc cherry-picking). "
                "V4-R2_RAW_WEIGHTED_CE_SEED42 achieved best Val CE=0.353494 at Epoch 7 (Val Acc=88.24%, Val BA=90.00%, FN=0, FP=4, ROC-AUC=0.9679)."
            ),
        },
        "software_environment_at_freeze": {
            "python": env_info["python"],
            "platform": env_info["platform"],
            "packages": env_info["packages"],
            "device": env_info["device"],
            "cuda_available": env_info["cuda_available"],
        },
        "training_comparison_references": {
            "v4_224_suite_report": "members/member06_research/runs/final_training_v4_20261124_192110_PJSHyF/comparison_repaired_v1/COMPARISON_V4_CN.md",
            "v4_224_vs_384_report": "members/member06_research/runs/final_training_v4_384_20261124_195746_Rg1Mug/comparison/COMPARISON_V4_FINAL_RESOLUTION_CN.md",
        },
    }
    write_json(freeze_manifest_path, freeze_manifest)

    sha_lines = [
        f"{copied_sha256}  weights/{FROZEN_WEIGHT_FILENAME}",
        f"{file_sha256(out_dir / 'model_config.json')}  model_config.json",
        f"{file_sha256(out_dir / 'inference_contract.json')}  inference_contract.json",
        f"{file_sha256(freeze_manifest_path)}  freeze_manifest.json",
    ]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    (weights_dir / "SHA256SUMS.txt").write_text(f"{copied_sha256}  {FROZEN_WEIGHT_FILENAME}\n", encoding="utf-8")

    md = [
        "# Baseline 2（Frozen ResNet18 + Linear）Stage 2 最终模型冻结报告",
        "",
        f"- **冻结状态**：`FROZEN_PRE_TEST`（正式 Test 评价前完成冻结，冻结后禁止因 Test 结果修改模型或参数）",
        f"- **冻结时间（UTC）**：`{frozen_at_utc}`",
        f"- **最终选定实验 ID**：`V4-R2_RAW_WEIGHTED_CE_SEED42`",
        f"- **模型版本标识**：`baseline2_final_v4_seed42_ep7`",
        f"- **源权重相对路径**：`{EXPECTED_SOURCE_CKPT_REL}`",
        f"- **冻结交付权重路径**：`members/member06_research/stage2_release/weights/{FROZEN_WEIGHT_FILENAME}`",
        f"- **冻结权重 SHA-256**：`{copied_sha256}`（文件大小 `{frozen_ckpt_path.stat().st_size}` 字节）",
        f"- **Linear(512,2) 头参数哈希**：`{state_hashes['linear_head_state_sha256']}`",
        f"- **冻结 Backbone 参数哈希**：`{state_hashes['backbone_state_sha256']}`（与同目录 `best.pt` 逐张量核验一致：`{sibling_best_verified}`）",
        "",
        "## 1. 冻结模型结构、训练与推理协议",
        "",
        "| 项目 | 冻结设定 |",
        "|---|---|",
        "| 模型结构 | `FrozenResNet18 (IMAGENET1K_V1) + Linear(512, 2)`（总参数 `11,177,538`：冻结 Backbone `11,176,512` + 线性头 `1,026`） |",
        "| 官方预训练权重身份 | `resnet18-f37072fd.pth`（SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`） |",
        "| 冻结与推理模式 | Backbone 参数与全部 BatchNorm 统计量完全冻结；推理强制 `model.eval()` + `torch.inference_mode()`；加载冻结权重无需联网下载预训练模型 |",
        "| 输入尺寸与预处理 | `RGB`（不执行 EXIF 自动旋转；摄像头 BGR 显式转 RGB）→ `224×224 letterbox`（双线性插值，`pad_rgb=[0,0,0]` 居中填充）→ ImageNet 归一化（`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`） |",
        "| 类别映射与阳性类 | `PASS = 0`，`FAIL = 1`（`FAIL` 为阳性类，输出未校准 `FAIL` 概率分数 `P(FAIL) = softmax(logits)[1]`） |",
        "| 固定预期数量规则 | `expected_counts = [2, 2, 2, 2, 2]`（五类规定零件各 2 个）；非该计数请求显式返回 `UNSUPPORTED_COUNTS` |",
        "| 分类决策与平局规则 | 默认 `argmax`（`P(FAIL) > P(PASS)` 判 `FAIL`；若 `P(FAIL) == P(PASS) == 0.5` 平局取类别索引 `0` 即 `PASS`），无事后阈值调优 |",
        "| 能力边界声明 | 仅输出整图 `PASS/FAIL` 与 `FAIL` 概率分数；不伪造目标框 `objects`、逐类计数或 `non_catalogue_count` |",
        "| 训练超参数与预算 | `Adam(lr=0.001, weight_decay=0)`，`batch_size=16`，最大预算 `20 Epoch`，训练损失 `Weighted CE [PASS=0.75, FAIL=1.50]`，验证选模为最低**无权** `Val CE`（最佳 `Epoch = 7`） |",
        "",
        "## 2. 数据版本与清单哈希身份",
        "",
        f"- **小组统一数据集**：`members/member03_data/final_training_v4/`（上游主分支提交 `{EXPECTED_UPSTREAM_V4_COMMIT}`）",
        f"- **训练清单 `manifests/resnet/train.csv`**（`249` 原图：`166 PASS / 83 FAIL`）：`{EXPECTED_TRAIN_MANIFEST_SHA256}`",
        f"- **验证清单 `manifests/resnet/val.csv`**（`34` 原图：`20 PASS / 14 FAIL`）：`{EXPECTED_VAL_MANIFEST_SHA256}`",
        f"- **测试清单 `manifests/resnet/test.csv`**（`30` 原图：`15 PASS / 15 FAIL`，冻结前仅登记文件哈希、未读取样本内容推理）：`{freeze_manifest['dataset_and_git_identity']['manifests_sha256']['test_csv']}`",
        f"- **总清单 `split_manifest.csv`**：`{EXPECTED_SPLIT_MANIFEST_SHA256}`",
        "",
        "## 3. 冻结检查点的验证集与训练集参考表现（`Epoch = 7`）",
        "",
        "| 评估集合 | 样本数 (`PASS/FAIL`) | 无权 CE Loss | Accuracy | Balanced Accuracy | FAIL Precision | FAIL Recall | FAIL F1 | FN / FP | 混淆矩阵 `[[TN,FP],[FN,TP]]` | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        f"| `Val`（选模依据） | 34 (`20/14`) | `{val_metrics['loss']:.6f}` | `{val_metrics['accuracy']:.4f}` (`88.24%`) | `{val_metrics['balanced_accuracy']:.4f}` (`90.00%`) | `{val_metrics['fail_precision']:.4f}` | `{val_metrics['fail_recall']:.4f}` | `{val_metrics['fail_f1']:.4f}` | `0 / 4` | `[[16, 4], [0, 14]]` | `{val_metrics['roc_auc']:.4f}` |",
    ]
    if train_249_metrics is not None:
        md.append(
            f"| `Train`（固定权重重评） | 249 (`166/83`) | `{train_249_metrics['loss']:.6f}` | `{train_249_metrics['accuracy']:.4f}` (`90.76%`) | `{train_249_metrics['balanced_accuracy']:.4f}` (`87.65%`) | `{train_249_metrics['fail_precision']:.4f}` | `{train_249_metrics['fail_recall']:.4f}` | `{train_249_metrics['fail_f1']:.4f}` | `18 / 5` | `[[161, 5], [18, 65]]` | `{train_249_metrics['roc_auc']:.4f}` |"
        )
    md += [
        "",
        "## 4. 最终选型科学依据",
        "",
        "1. **损失策略（`V4-R2` vs `V4-R1`）**：针对 `final_training_v4` 训练集 `2:1`（`166 PASS : 83 FAIL`）类别不平衡，采用逆频率 `Weighted CE [0.75, 1.50]`（验证保持无权 CE）在 3 个随机种子（`42, 43, 44`，20 轮预算）上将平均 `Val Balanced Accuracy` 从 `0.8643 ± 0.0357` 提升至 `0.8881 ± 0.0206`，平均漏放 `FN` 从 `1.00 ± 1.00` 降至 `0.33 ± 0.58`，且平均误拒 `FP`（`4.00`）未恶化。",
        "2. **训练预算（`20 Epoch` 最大预算 + 最低无权 `Val Loss` 选模）**：`Seed=42` 与 `Seed=44` 在 10 轮与 20 轮预算内均分别锁定第 `7` 轮与第 `10` 轮最佳点；`Seed=43` 在第 `14` 轮进一步降低无权 `Val Loss`（`0.3923 → 0.3525`）并减少 1 例误拒。",
        "3. **输入分辨率最终对照（`224` vs `384`）**：单变量受控实验 `V4-R4_RAW_WEIGHTED_CE_384` 未满足全部四项预设升级条件——三 seed 平均 `Val BA` 下降 `4.05` 个百分点（`0.8881 → 0.8476`），平均 `FN`（`0.33 → 1.00`）与 `FP`（`4.00 → 4.67`）双双恶化，代表种子 `Seed=42` 退步（`BA 0.9000 → 0.8393`, `FN 0 → 1`, `FP 4 → 5`），且 CPU 单图离线中位耗时增加至 `1.28x`（`161.76 ms → 207.25 ms`），故正式否决 `384`、保留 `224×224`。",
        "4. **代表种子不事后择优**：正式代表种子在实验启动前即预先固定为 `seed=42`（`Epoch 7`, `Val Acc=88.24%`, `Val BA=90.00%`, `FN=0`, `FP=4`），不事后挑选 `seed=44` 的更低 `Val Loss`。",
    ]
    (out_dir / "STAGE2_FREEZE_REPORT_CN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return freeze_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--release-dir", type=Path, default=None)
    args = parser.parse_args()
    manifest = freeze_stage2_model(args.source_checkpoint, args.release_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frozen_at_utc": manifest["frozen_at_utc"],
                "frozen_checkpoint": manifest["selected_model"]["frozen_checkpoint_relpath"],
                "checkpoint_sha256": manifest["selected_model"]["checkpoint_sha256"],
                "selected_epoch": manifest["selected_model"]["selected_epoch"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
