"""Dataset v2 Phase-1 & Phase-2 training engine with bound frozen feature caching and multi-budget checkpoints.

Guarantees:
1. Reuses FrozenResNet18, Preprocess, Config, class_weights, and load_checkpoint without altering legacy modules.
2. Enforces identical initial Linear(512,2) weights for a given seed and identical per-epoch parent order across R1/R2 and across 10ep/20ep budgets.
3. Enforces unweighted CE on the 74 real Val images for epoch selection (earliest epoch wins ties).
4. Supports optional budget_milestones (e.g. [10, 20]) saving best_within_10.pt, best_within_20.pt, epoch10.pt, epoch20.pt,
   plus exact parameter & trajectory reproduction verification against Phase-1 R1_RAW_CE for seed=42.
5. Restricts 18 Val augmented images strictly to post-training robustness diagnostics; never accesses Test.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config
from .data import Preprocess, image_digest
from .model import FrozenResNet18
from .runtime import choose_device, environment, file_sha256, load_checkpoint, seed_everything, write_json
from .train import class_weights
from .v2_data import r2_initial_offset, r2_select_variant, verify_prepared_dataset_v2

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_SCHEMA = "member06.v2_experiment.v1"
CACHE_SCHEMA = "member06.frozen_feature_cache.v1"


def _json_fingerprint(obj: dict | list) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_experiment_spec(path: str | Path) -> tuple[dict, Config]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError(f"unsupported experiment schema: {raw.get('schema_version')}")
    if raw.get("train_mode") not in {"raw_294", "parent_equal_cycle_294", "full_438"}:
        raise ValueError(f"unsupported train_mode: {raw.get('train_mode')}")
    if raw.get("online_augmentation") != "none":
        raise ValueError("protocol forbids online augmentation")
    if raw.get("lr_scheduler") != "none":
        raise ValueError("protocol forbids lr_scheduler")
    if raw.get("threshold_rule") != "default_argmax_pass0_fail1_no_threshold_search":
        raise ValueError("protocol requires default argmax without threshold search")
    if type(raw.get("use_feature_cache")) is not bool:
        raise ValueError("use_feature_cache must be boolean")
    cfg = Config(**raw["model_config"])
    if cfg.class_to_idx != {"PASS": 0, "FAIL": 1} or cfg.fail_label != "FAIL":
        raise ValueError("protocol requires PASS=0, FAIL=1")
    if raw["train_mode"] in {"raw_294", "parent_equal_cycle_294"} and cfg.train_class_weight != "none":
        raise ValueError(f"{raw['experiment_id']} requires unweighted train_class_weight='none'")
    if raw["train_mode"] == "full_438" and cfg.train_class_weight != "balanced":
        raise ValueError("full_438 (R3) requires train_class_weight='balanced'")

    milestones = raw.get("budget_milestones")
    if milestones is not None:
        if (
            not isinstance(milestones, list)
            or not milestones
            or any(type(m) is not int or m < 1 or m > cfg.epochs for m in milestones)
            or milestones != sorted(set(milestones))
            or milestones[-1] != cfg.epochs
        ):
            raise ValueError(f"invalid budget_milestones {milestones} for epochs={cfg.epochs}")

    ref_10 = raw.get("reference_10ep_run_dir")
    if ref_10 is not None:
        if not isinstance(ref_10, str) or not ref_10.strip():
            raise ValueError("reference_10ep_run_dir must be a nonempty string or null")
        if not milestones or 10 not in milestones:
            raise ValueError("reference_10ep_run_dir requires 10 in budget_milestones")

    return raw, cfg


def compute_rich_metrics(
    predictions: list[dict],
    loss_value: float,
    *,
    optimizer_steps: int = 0,
    cumulative_steps: int = 0,
    wall_seconds: float = 0.0,
) -> dict:
    if not predictions:
        raise ValueError("empty predictions")
    y = np.array([p["target_index"] for p in predictions], dtype=np.int64)
    d = np.array([p["predicted_index"] for p in predictions], dtype=np.int64)
    s = np.array([p["softmax"][1] for p in predictions], dtype=np.float64)

    tn = int(((y == 0) & (d == 0)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    fn = int(((y == 1) & (d == 0)).sum())
    tp = int(((y == 1) & (d == 1)).sum())
    n = len(y)
    n_pass = tn + fp
    n_fail = tp + fn

    tpr = tp / n_fail if n_fail else None
    tnr = tn / n_pass if n_pass else None
    bal_acc = 0.5 * (tpr + tnr) if (tpr is not None and tnr is not None) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tpr
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None

    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) and len(neg):
        auc = float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
    else:
        auc = None

    return {
        "loss": float(loss_value),
        "n": n,
        "support": {"PASS": n_pass, "FAIL": n_fail},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "matrix_axes": "rows=true [PASS=0, FAIL=1], cols=pred [PASS=0, FAIL=1]",
        "accuracy": float((tn + tp) / n),
        "balanced_accuracy": float(bal_acc) if bal_acc is not None else None,
        "fail_precision": float(prec) if prec is not None else None,
        "fail_recall": float(rec) if rec is not None else None,
        "fail_f1": float(f1) if f1 is not None else None,
        "false_pass": fn,
        "false_fail": fp,
        "false_pass_rate": float(fn / n_fail) if n_fail else None,
        "false_fail_rate": float(fp / n_pass) if n_pass else None,
        "roc_auc": auc,
        "score_median_PASS": float(np.median(neg)) if len(neg) else None,
        "score_median_FAIL": float(np.median(pos)) if len(pos) else None,
        "optimizer_steps": int(optimizer_steps),
        "cumulative_optimizer_steps": int(cumulative_steps),
        "wall_seconds": round(float(wall_seconds), 4),
    }


def _cache_identity_dict(config: Config, backbone_sha256: str, device: torch.device, pretrained_file_sha: str | None) -> dict:
    return {
        "schema_version": CACHE_SCHEMA,
        "image_size": config.image_size,
        "resize_mode": config.resize_mode,
        "mean": list(config.mean),
        "std": list(config.std),
        "pad_rgb": list(config.pad_rgb),
        "weights_name": config.weights,
        "pretrained_checkpoint_sha256": pretrained_file_sha,
        "backbone_state_sha256": backbone_sha256,
        "code_sha256": {
            "data.py": file_sha256(BASE_DIR / "data.py"),
            "model.py": file_sha256(BASE_DIR / "model.py"),
            "image_io.py": file_sha256(BASE_DIR / "image_io.py"),
        },
        "software_versions": {
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "Pillow": importlib.metadata.version("Pillow"),
            "numpy": np.__version__,
        },
        "device_type": device.type,
        "test_cached": False,
    }


def inspect_existing_cache_without_cnn(
    cache_root: str | Path,
    all_dev_items: list[dict],
    config: Config,
    device: torch.device,
    pretrained_file_sha: str | None,
) -> dict:
    """Read-only preflight check verifying whether an existing frozen feature cache matches current dataset/preprocess/software."""
    root = Path(cache_root).resolve()
    prefix = f"resnet18_sz{config.image_size}_"
    candidates = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir()) if root.is_dir() else []
    if not candidates:
        return {"cache_present": False, "reusable_verified": False, "reason": "no matching cache directory on disk yet"}

    unique_sha = {item["image_sha256"]: (item["v2_image_relpath"], item["split"]) for item in all_dev_items}
    expected_partial = _cache_identity_dict(config, "PLACEHOLDER", device, pretrained_file_sha)
    expected_partial.pop("backbone_state_sha256")

    for cdir in candidates:
        mfile = cdir / "cache_manifest.json"
        tfile = cdir / "features.pt"
        if not (mfile.is_file() and tfile.is_file()):
            continue
        saved = json.loads(mfile.read_text(encoding="utf-8"))
        ident = dict(saved.get("identity", {}))
        b_sha = ident.pop("backbone_state_sha256", None)
        if not b_sha or ident != expected_partial:
            continue
        if file_sha256(tfile) != saved.get("features_file_sha256"):
            raise ValueError(f"existing cache features.pt hash mismatch in {cdir}")
        saved_map = {e["image_sha256"]: (e["v2_image_relpath"], e["split"]) for e in saved.get("items", [])}
        if saved_map != unique_sha:
            raise ValueError(f"existing cache item inventory differs from current 530 Train/Val images in {cdir}")
        return {
            "cache_present": True,
            "reusable_verified": True,
            "cache_dir": str(cdir),
            "cached_images": len(saved_map),
            "features_file_sha256": saved["features_file_sha256"],
        }
    return {"cache_present": False, "reusable_verified": False, "reason": "no identity-matching cache found"}


def build_or_load_feature_cache(
    cache_root: str | Path,
    all_dev_items: list[dict],
    model: FrozenResNet18,
    config: Config,
    device: torch.device,
    pretrained_file_sha: str | None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build or strictly verify frozen 512-d backbone features for approved Train/Val ONLY."""
    for item in all_dev_items:
        if item["split"] not in {"train", "val"}:
            raise ValueError(f"REFUSED_CACHE_SCOPE: split={item['split']!r} is not train/val (Test caching forbidden)")

    ordered = sorted(all_dev_items, key=lambda r: (r["split"], r["v2_image_relpath"]))
    seen_sha = set()
    unique_items = []
    for item in ordered:
        if item["image_sha256"] not in seen_sha:
            seen_sha.add(item["image_sha256"])
            unique_items.append(item)

    backbone_sha = state_dict_sha256(model.backbone.state_dict())
    expected_identity = _cache_identity_dict(config, backbone_sha, device, pretrained_file_sha)
    id_hash = _json_fingerprint(expected_identity)[:16]
    cache_dir = Path(cache_root).resolve() / f"resnet18_sz{config.image_size}_{id_hash}"
    manifest_file = cache_dir / "cache_manifest.json"
    tensor_file = cache_dir / "features.pt"

    if manifest_file.is_file() and tensor_file.is_file():
        t0 = time.perf_counter()
        saved = json.loads(manifest_file.read_text(encoding="utf-8"))
        if saved.get("identity") != expected_identity:
            raise ValueError("FEATURE_CACHE_IDENTITY_MISMATCH: header differs from current model/preprocess/software")
        if saved.get("test_cached") is not False:
            raise ValueError("FEATURE_CACHE_IDENTITY_MISMATCH: invalid test_cached flag")
        if file_sha256(tensor_file) != saved.get("features_file_sha256"):
            raise ValueError("FEATURE_CACHE_IDENTITY_MISMATCH: features.pt hash mismatch")
        feats = torch.load(tensor_file, map_location="cpu", weights_only=True)
        if not isinstance(feats, torch.Tensor) or feats.dtype != torch.float32 or feats.shape != (len(saved["items"]), 512):
            raise ValueError("FEATURE_CACHE_IDENTITY_MISMATCH: invalid tensor shape or dtype")
        if not torch.isfinite(feats).all():
            raise ValueError("FEATURE_CACHE_IDENTITY_MISMATCH: non-finite cached features")
        saved_map = {entry["image_sha256"]: (idx, entry) for idx, entry in enumerate(saved["items"])}
        feature_lookup: dict[str, torch.Tensor] = {}
        for item in unique_items:
            sha = item["image_sha256"]
            if sha not in saved_map:
                raise ValueError(f"FEATURE_CACHE_IDENTITY_MISMATCH: missing image {item['v2_image_relpath']}")
            idx, entry = saved_map[sha]
            if entry["v2_image_relpath"] != item["v2_image_relpath"] or entry["split"] != item["split"]:
                raise ValueError(f"FEATURE_CACHE_IDENTITY_MISMATCH: path/split mismatch on {item['v2_image_relpath']}")
            if file_sha256(item["path"]) != sha:
                raise ValueError(f"FEATURE_CACHE_IDENTITY_MISMATCH: disk file modified for {item['v2_image_relpath']}")
            feature_lookup[sha] = feats[idx]
        elapsed = time.perf_counter() - t0
        return feature_lookup, {
            "cache_action": "LOADED_VERIFIED",
            "cache_dir": str(cache_dir),
            "cache_identity_hash": id_hash,
            "cached_images": len(unique_items),
            "features_file_sha256": saved["features_file_sha256"],
            "cache_wall_seconds": round(elapsed, 4),
            "test_cached": False,
        }

    # Build fresh cache
    t0 = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = Preprocess(config)
    model.eval()
    all_vecs: list[torch.Tensor] = []
    manifest_entries: list[dict] = []

    batch_size = max(1, config.batch_size)
    with torch.no_grad():
        for start in range(0, len(unique_items), batch_size):
            chunk = unique_items[start : start + batch_size]
            tensors = []
            for item in chunk:
                if file_sha256(item["path"]) != item["image_sha256"]:
                    raise ValueError(f"source image hash mismatch during cache build: {item['v2_image_relpath']}")
                tensors.append(transform(item["path"]))
                manifest_entries.append({
                    "v2_image_relpath": item["v2_image_relpath"],
                    "split": item["split"],
                    "image_sha256": item["image_sha256"],
                    "rgb_sha256": image_digest(item["path"]),
                })
            batch_x = torch.stack(tensors, dim=0).to(device)
            batch_feat = model.extract_features(batch_x).detach().cpu().to(torch.float32)
            if batch_feat.shape != (len(chunk), 512) or not torch.isfinite(batch_feat).all():
                raise ValueError("invalid extracted backbone features")
            all_vecs.append(batch_feat)

    full_tensor = torch.cat(all_vecs, dim=0).contiguous()
    tmp_tensor = cache_dir / "features.tmp"
    torch.save(full_tensor, tmp_tensor)
    tmp_tensor.replace(tensor_file)
    feat_sha = file_sha256(tensor_file)

    payload = {
        "schema_version": CACHE_SCHEMA,
        "identity": expected_identity,
        "test_cached": False,
        "num_items": len(manifest_entries),
        "feature_dim": 512,
        "features_file": "features.pt",
        "features_file_sha256": feat_sha,
        "items": manifest_entries,
    }
    write_json(manifest_file, payload)
    elapsed = time.perf_counter() - t0
    feature_lookup = {entry["image_sha256"]: full_tensor[idx] for idx, entry in enumerate(manifest_entries)}
    return feature_lookup, {
        "cache_action": "BUILT_AND_VERIFIED",
        "cache_dir": str(cache_dir),
        "cache_identity_hash": id_hash,
        "cached_images": len(manifest_entries),
        "features_file_sha256": feat_sha,
        "cache_wall_seconds": round(elapsed, 4),
        "test_cached": False,
    }


def _forward_items(
    model: FrozenResNet18,
    items: list[dict],
    config: Config,
    device: torch.device,
    feature_lookup: dict[str, torch.Tensor] | None,
    transform: Preprocess,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    cumulative_steps_start: int = 0,
    progress_prefix: str | None = None,
) -> tuple[dict, list[dict]]:
    training = optimizer is not None
    model.train(training)
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    if not training and criterion.weight is not None:
        raise ValueError("validation / fixed evaluation must use unweighted CrossEntropyLoss")
    if criterion.reduction != "mean":
        raise ValueError("only mean reduction supported")

    t0 = time.perf_counter()
    total_batches = math.ceil(len(items) / config.batch_size)
    loss_num = 0.0
    loss_den = 0.0
    steps = 0
    predictions: list[dict] = []

    with torch.set_grad_enabled(training):
        for b_idx, start in enumerate(range(0, len(items), config.batch_size), 1):
            chunk = items[start : start + config.batch_size]
            targets = torch.tensor([config.class_to_idx[r["label"]] for r in chunk], dtype=torch.int64, device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if feature_lookup is not None:
                feats = torch.stack([feature_lookup[r["image_sha256"]] for r in chunk], dim=0).to(device)
                logits = model.head(feats)
            else:
                imgs = torch.stack([transform(r["path"]) for r in chunk], dim=0).to(device)
                logits = model(imgs)
            if logits.shape != (len(chunk), 2) or not torch.isfinite(logits).all():
                raise ValueError("invalid logits")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
                steps += 1

            probs = logits.detach().softmax(dim=1).cpu()
            preds = probs.argmax(dim=1)
            for r, t_val, p_val, prob_vec in zip(chunk, targets.cpu().tolist(), preds.tolist(), probs.tolist()):
                predictions.append({
                    "sample_id": r["sample_id"],
                    "image_filename": r["image_filename"],
                    "v2_image_relpath": r["v2_image_relpath"],
                    "image_path": str(r["path"]),
                    "scene_id": r["scene_id"],
                    "label": r["label"],
                    "target_index": t_val,
                    "predicted_index": p_val,
                    "predicted_label": "FAIL" if p_val == 1 else "PASS",
                    "softmax": prob_vec,
                })
            batch_den = criterion.weight[targets].sum().item() if criterion.weight is not None else float(len(chunk))
            loss_den += batch_den
            loss_num += loss.item() * batch_den

            if progress_prefix and (b_idx == 1 or b_idx == total_batches or b_idx % max(1, total_batches // 4) == 0):
                print(f"{progress_prefix} batch={b_idx}/{total_batches} running_loss={loss_num / loss_den:.6f}", flush=True)

    elapsed = time.perf_counter() - t0
    metrics = compute_rich_metrics(
        predictions,
        loss_num / loss_den,
        optimizer_steps=steps,
        cumulative_steps=cumulative_steps_start + steps,
        wall_seconds=elapsed,
    )
    return metrics, predictions


def preflight_v2(spec_path: str | Path) -> dict:
    spec, config = load_experiment_spec(spec_path)
    prov, manifests, r2_doc = verify_prepared_dataset_v2(spec["prepared_dir"])
    device = choose_device(config.device)

    pretrained_sha = None
    if config.weights == "IMAGENET1K_V1":
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        if not cached.is_file():
            raise FileNotFoundError(f"missing cached ImageNet weights at {cached}")
        pretrained_sha = file_sha256(cached)

    if spec["train_mode"] in {"raw_294", "parent_equal_cycle_294"}:
        exp_rows = manifests["train_raw_294"]
        weights_list = None
    else:
        exp_rows = manifests["train_full_438"]
        weights_list = class_weights(exp_rows, config, "cpu").tolist()

    steps_per_epoch = math.ceil(len(exp_rows) / config.batch_size)
    r2_preview = None
    if spec["train_mode"] == "parent_equal_cycle_294":
        ep1 = [r2_select_variant(p, config.seed, 1) for p in r2_doc["parents"]]
        r2_preview = {
            "epoch1_variant_counts": dict(collections.Counter(x["variant_name"] for x in ep1)),
            "augmented_parents_offset_distribution": dict(
                collections.Counter(r2_initial_offset(config.seed, p["parent_id"], 4) for p in r2_doc["parents"] if p["has_augmented_variants"])
            ),
        }

    cache_status = None
    if spec["use_feature_cache"]:
        all_dev = manifests["train_full_438"] + manifests["val_raw_74"] + manifests["val_aug_18"]
        cache_status = inspect_existing_cache_without_cnn(
            spec["cache_dir"], all_dev, config, device, pretrained_sha
        )

    ref_status = None
    if spec.get("reference_10ep_run_dir"):
        ref_p = Path(spec["reference_10ep_run_dir"]).resolve()
        for req in ("history.json", "best.pt", "run_config.json", "checkpoint_manifest.json"):
            if not (ref_p / req).is_file():
                raise FileNotFoundError(f"missing reference_10ep_run_dir artifact: {ref_p / req}")
        ref_status = {
            "reference_dir": str(ref_p),
            "reference_best_sha256": file_sha256(ref_p / "best.pt"),
        }

    return {
        "status": "DATASET_V2_READY_FOR_USER_TRAINING",
        "experiment_id": spec["experiment_id"],
        "seed": config.seed,
        "epochs": config.epochs,
        "image_size": config.image_size,
        "budget_milestones": spec.get("budget_milestones"),
        "reference_10ep_check": ref_status,
        "train_mode": spec["train_mode"],
        "use_feature_cache": spec["use_feature_cache"],
        "feature_cache_preflight": cache_status,
        "device": str(device),
        "exposures_per_epoch": len(exp_rows),
        "total_exposures": len(exp_rows) * config.epochs,
        "train_exposure_class_counts": dict(collections.Counter(r["label"] for r in exp_rows)),
        "primary_val_images": len(manifests["val_raw_74"]),
        "primary_val_class_counts": dict(collections.Counter(r["label"] for r in manifests["val_raw_74"])),
        "post_train_val_aug_diagnostics_images": len(manifests["val_aug_18"]),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * config.epochs,
        "train_class_weight_mode": config.train_class_weight,
        "train_class_weights_pass0_fail1": weights_list,
        "r2_schedule_preview": r2_preview,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "verified_quotas_438_92_79": prov["verified_quotas_438_92_79"],
        "model_initialized": False,
        "weights_updated": False,
        "test_content_read": False,
    }


def _build_val_aug_diagnostics(
    val_raw_74: list[dict],
    val_raw_preds: list[dict],
    val_aug_18: list[dict],
    val_aug_preds: list[dict],
    val_aug_metrics: dict,
) -> dict:
    parent_pred_map = {p["image_filename"]: p for p in val_raw_preds}
    pairs = []
    by_aug_type: dict[str, list[dict]] = collections.defaultdict(list)

    for row, ap in zip(val_aug_18, val_aug_preds):
        parent_name = row["parent_image_id"]
        pp = parent_pred_map[parent_name]
        p_score = pp["softmax"][1]
        a_score = ap["softmax"][1]
        delta = a_score - p_score
        flipped = pp["predicted_label"] != ap["predicted_label"]
        rec = {
            "aug_sample_id": row["sample_id"],
            "aug_filename": row["image_filename"],
            "augmentation_type": row["augmentation_type"],
            "label": row["label"],
            "parent_filename": parent_name,
            "parent_predicted_label": pp["predicted_label"],
            "parent_fail_softmax": p_score,
            "aug_predicted_label": ap["predicted_label"],
            "aug_fail_softmax": a_score,
            "fail_softmax_delta": delta,
            "prediction_flipped": flipped,
            "flip_direction": f"{pp['predicted_label']}->{ap['predicted_label']}" if flipped else "unchanged",
        }
        pairs.append(rec)
        by_aug_type[row["augmentation_type"]].append(rec)

    aug_summary = {}
    for aug_type, items in sorted(by_aug_type.items()):
        deltas = [x["fail_softmax_delta"] for x in items]
        aug_scores = [x["aug_fail_softmax"] for x in items]
        detected = sum(x["aug_predicted_label"] == "FAIL" for x in items)
        flips = sum(x["prediction_flipped"] for x in items)
        aug_summary[aug_type] = {
            "n": len(items),
            "fail_detected": detected,
            "fail_recall": detected / len(items),
            "mean_aug_fail_softmax": float(np.mean(aug_scores)),
            "mean_score_delta_vs_parent": float(np.mean(deltas)),
            "median_score_delta_vs_parent": float(np.median(deltas)),
            "max_abs_score_delta": float(np.max(np.abs(deltas))),
            "prediction_flips": flips,
            "flip_directions": dict(collections.Counter(x["flip_direction"] for x in items)),
        }

    return {
        "role": "POST_TRAINING_ROBUSTNESS_DIAGNOSTIC_ONLY_NEVER_USED_FOR_SELECTION",
        "overall_18_augmented_FAIL": val_aug_metrics,
        "parent_6_original_FAIL_detected": sum(
            parent_pred_map[k]["predicted_label"] == "FAIL" for k in sorted({r["parent_image_id"] for r in val_aug_18})
        ),
        "total_flips_across_18": sum(x["prediction_flipped"] for x in pairs),
        "by_augmentation_type": aug_summary,
        "pairs": pairs,
    }


def _save_atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def verify_10ep_reproduction(
    new_out_dir: Path,
    ref_10ep_dir: str | Path,
    train_raw_294: list[dict],
    config: Config,
    epoch_sample_orders: dict[str, dict],
    history: list[dict],
    run_metadata: dict,
    tol: float = 1e-6,
) -> dict:
    """Verify epochs 1..10 trajectory, sample order, and best_within_10.pt parameters against Phase-1 R1."""
    ref_dir = Path(ref_10ep_dir).resolve()
    ref_run_cfg = json.loads((ref_dir / "run_config.json").read_text(encoding="utf-8"))
    ref_history = json.loads((ref_dir / "history.json").read_text(encoding="utf-8"))

    # 1. Initial backbone & head weights identity
    init_head_match = (
        run_metadata["determinism_notes"]["initial_head_sha256"]
        == ref_run_cfg["determinism_notes"]["initial_head_sha256"]
    )
    init_backbone_match = (
        run_metadata["determinism_notes"]["initial_backbone_sha256"]
        == ref_run_cfg["determinism_notes"]["initial_backbone_sha256"]
    )
    if not (init_head_match and init_backbone_match):
        raise RuntimeError("REPRODUCTION_CHECK_FAILED: initial head or backbone state_dict differs from reference R1")

    # 2. Reconstruct R1's exact 10-epoch sample orders from torch.Generator().manual_seed(seed) and verify 10/10 match
    ref_gen = torch.Generator().manual_seed(config.seed)
    order_matches = []
    for ep in range(1, 11):
        perm = torch.randperm(len(train_raw_294), generator=ref_gen).tolist()
        ref_ids = [train_raw_294[i]["sample_id"] for i in perm]
        expected_sha = _json_fingerprint(ref_ids)
        actual_sha = epoch_sample_orders[str(ep)]["sample_order_sha256"]
        order_matches.append(expected_sha == actual_sha)
    if not all(order_matches):
        raise RuntimeError(f"REPRODUCTION_CHECK_FAILED: sample order differed in epochs 1..10: {order_matches}")

    # 3. Compare per-epoch Train & Val losses and metrics across epochs 1..10
    if len(ref_history) != 10 or len(history) < 10:
        raise RuntimeError("REPRODUCTION_CHECK_FAILED: expected 10 epochs in reference history")
    epoch_diffs = []
    max_train_loss_diff = 0.0
    max_val_loss_diff = 0.0
    for ep_idx in range(10):
        h_ref = ref_history[ep_idx]
        h_new = history[ep_idx]
        tr_diff = abs(h_new["train"]["loss"] - h_ref["train"]["loss"])
        vl_diff = abs(h_new["val"]["loss"] - h_ref["val"]["loss"])
        max_train_loss_diff = max(max_train_loss_diff, tr_diff)
        max_val_loss_diff = max(max_val_loss_diff, vl_diff)
        epoch_diffs.append({
            "epoch": ep_idx + 1,
            "ref_train_loss": h_ref["train"]["loss"],
            "new_train_loss": h_new["train"]["loss"],
            "abs_train_loss_diff": tr_diff,
            "ref_val_loss": h_ref["val"]["loss"],
            "new_val_loss": h_new["val"]["loss"],
            "abs_val_loss_diff": vl_diff,
            "val_fn_match": h_new["val"]["false_pass"] == h_ref["val"]["false_pass"],
            "val_fp_match": h_new["val"]["false_fail"] == h_ref["val"]["false_fail"],
        })
        if tr_diff > tol or vl_diff > tol:
            raise RuntimeError(
                f"REPRODUCTION_CHECK_FAILED at epoch {ep_idx + 1}: train_diff={tr_diff}, val_diff={vl_diff} > tol={tol}"
            )

    # 4. Compare model parameters (not raw file SHA256, as checkpoint metadata/epochs=20 differ)
    allow_smoke = config.data_kind == "SMOKE_ONLY"
    allow_dev = config.training_tier == "DEV_ONLY"
    m_new10, _, _, p_new10 = load_checkpoint(
        new_out_dir / "best_within_10.pt", device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev
    )
    m_ref10, _, _, p_ref10 = load_checkpoint(
        ref_dir / "best.pt", device="cpu", allow_smoke=allow_smoke, allow_dev_only=allow_dev
    )
    if p_new10["epoch"] != p_ref10["epoch"]:
        raise RuntimeError(
            f"REPRODUCTION_CHECK_FAILED: best_within_10 epoch ({p_new10['epoch']}) != ref best epoch ({p_ref10['epoch']})"
        )

    sd_new = m_new10.state_dict()
    sd_ref = m_ref10.state_dict()
    if set(sd_new.keys()) != set(sd_ref.keys()):
        raise RuntimeError("REPRODUCTION_CHECK_FAILED: state_dict keys mismatch")

    head_w_diff = float((sd_new["head.weight"] - sd_ref["head.weight"]).abs().max().item())
    head_b_diff = float((sd_new["head.bias"] - sd_ref["head.bias"]).abs().max().item())
    backbone_max_diff = max(
        float((sd_new[k].float() - sd_ref[k].float()).abs().max().item())
        for k in sd_new
        if k.startswith("backbone.")
    )
    if max(head_w_diff, head_b_diff, backbone_max_diff) > tol:
        raise RuntimeError(
            f"REPRODUCTION_CHECK_FAILED: parameter diff exceeds tol={tol}: "
            f"head.weight={head_w_diff}, head.bias={head_b_diff}, backbone={backbone_max_diff}"
        )

    return {
        "status": "REPRODUCED_WITHIN_TOLERANCE",
        "tolerance": tol,
        "reference_run_dir": str(ref_dir),
        "best_epoch_within_10": p_new10["epoch"],
        "reference_best_epoch": p_ref10["epoch"],
        "initial_head_sha256_match": init_head_match,
        "initial_backbone_sha256_match": init_backbone_match,
        "all_10_epochs_sample_order_exact_match": True,
        "max_abs_train_loss_diff_epochs_1_10": max_train_loss_diff,
        "max_abs_val_loss_diff_epochs_1_10": max_val_loss_diff,
        "head_weight_max_abs_diff": head_w_diff,
        "head_bias_max_abs_diff": head_b_diff,
        "backbone_max_abs_diff": backbone_max_diff,
        "model_state_sha256_exact_match": state_dict_sha256(sd_new) == state_dict_sha256(sd_ref),
        "note": "Compared actual model tensors, losses, and shuffled data orders rather than serialized .pt file SHA256 (which embeds config.epochs=20 and run timestamps).",
        "per_epoch_comparison": epoch_diffs,
    }


def run_v2_experiment(spec_path: str | Path, output_dir: str | Path, *, allow_dev_only: bool = False) -> dict:
    spec, config = load_experiment_spec(spec_path)
    if config.training_tier == "DEV_ONLY" and not allow_dev_only:
        raise ValueError("DEV_ONLY training requires explicit --allow-dev-only")

    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory already exists and is non-empty (refusing overwrite): {out}")

    total_t0 = time.perf_counter()
    prov, manifests, r2_doc = verify_prepared_dataset_v2(spec["prepared_dir"])
    device = choose_device(config.device)

    pretrained_sha = None
    if config.weights == "IMAGENET1K_V1":
        cached_weights = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        if not cached_weights.is_file():
            raise FileNotFoundError(f"missing cached ImageNet weights: {cached_weights}")
        pretrained_sha = file_sha256(cached_weights)

    # 1. Deterministic model creation (fresh Linear(512,2) head from config.seed)
    seed_everything(config.seed, config.cpu_threads)
    model = FrozenResNet18(config.weights).to(device)
    initial_backbone_sha = state_dict_sha256(model.backbone.state_dict())
    initial_head_sha = state_dict_sha256(model.head.state_dict())

    # 2. Bound frozen feature cache over approved Train (438) + Val (92) ONLY
    feature_lookup = None
    cache_info = {"cache_action": "DISABLED", "test_cached": False}
    if spec["use_feature_cache"]:
        all_dev = manifests["train_full_438"] + manifests["val_raw_74"] + manifests["val_aug_18"]
        feature_lookup, cache_info = build_or_load_feature_cache(
            spec["cache_dir"], all_dev, model, config, device, pretrained_sha
        )
        if state_dict_sha256(model.backbone.state_dict()) != initial_backbone_sha:
            raise RuntimeError("backbone state mutated during feature cache build")

    # 3. Reset RNG immediately before training loop so 10-epoch & 20-epoch / cached & uncached share identical RNG
    seed_everything(config.seed, config.cpu_threads)
    epoch_rng = torch.Generator().manual_seed(config.seed)
    transform = Preprocess(config)

    if spec["train_mode"] == "full_438":
        weights_tensor = class_weights(manifests["train_full_438"], config, device)
    else:
        weights_tensor = None
    train_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    milestones: list[int] = spec.get("budget_milestones") or []
    milestone_best_loss = {m: float("inf") for m in milestones}
    milestone_best_epoch: dict[int, int | None] = {m: None for m in milestones}

    raw_by_filename = {r["image_filename"]: r for r in manifests["train_full_438"]}
    out.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "schema_version": "member06.v2_run_metadata.v1",
        "experiment_id": spec["experiment_id"],
        "train_mode": spec["train_mode"],
        "factor_role": spec["factor_role"],
        "budget_milestones": milestones if milestones else None,
        "evidence_status": config.evidence_status,
        "environment": environment(device),
        "determinism_notes": {
            "seed": config.seed,
            "cpu_threads": config.cpu_threads,
            "deterministic_algorithms": True,
            "initial_backbone_sha256": initial_backbone_sha,
            "initial_head_sha256": initial_head_sha,
            "parent_shuffle_generator": "torch.Generator().manual_seed(seed) drawing one randperm(N) per epoch",
            "r2_offset_rule": "SHA-256('R2_PARENT_EQUAL_AUG_CE|seed={seed}|parent_id={parent_id}')[:8] % 4",
            "non_guaranteed_factors": "Wall-clock latency and minor floating-point reduction differences across distinct CPU SIMD ISAs or GPU architectures.",
        },
        "feature_cache": cache_info,
        "selection_protocol": {
            "primary_validation_manifest": "val_raw_74.csv (74 real originals: 37 PASS, 37 FAIL)",
            "validation_loss": "unweighted CrossEntropyLoss mean",
            "selection_rule": f"minimum unweighted Val loss across epochs 1..{config.epochs}; earliest epoch wins ties",
            "decision_rule": "default argmax of uncalibrated 2-class softmax (PASS=0, FAIL=1; exact tie -> 0)",
            "val_aug_18_used_for_selection": False,
            "test_data_used": False,
        },
        "loss_protocol": {
            "train_class_weight": config.train_class_weight,
            "class_to_idx": config.class_to_idx,
            "weights": weights_tensor.detach().cpu().tolist() if weights_tensor is not None else None,
            "validation": "unweighted CE mean",
        },
        "data_provenance_sha256": file_sha256(Path(spec["prepared_dir"]) / "provenance.json"),
        "pretrained_source": (
            {"url": "https://download.pytorch.org/models/resnet18-f37072fd.pth", "sha256": pretrained_sha}
            if pretrained_sha else {"source": config.weights}
        ),
    }

    write_json(out / "experiment_spec.json", spec)
    write_json(out / "config.json", config.to_dict())
    write_json(out / "run_config.json", run_metadata)

    history: list[dict] = []
    epoch_sample_orders: dict[str, dict] = {}
    r2_epoch_exposures: dict[int, list[dict]] = {}
    r2_variant_counter: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    best_val_loss = float("inf")
    best_epoch = None
    cumulative_steps = 0

    train_wall_t0 = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if spec["train_mode"] == "raw_294":
            base_list = manifests["train_raw_294"]
            perm = torch.randperm(len(base_list), generator=epoch_rng).tolist()
            epoch_items = [base_list[i] for i in perm]
        elif spec["train_mode"] == "parent_equal_cycle_294":
            parents = r2_doc["parents"]
            perm = torch.randperm(len(parents), generator=epoch_rng).tolist()
            epoch_items = []
            ep_log = []
            for pos, p_idx in enumerate(perm):
                p_entry = parents[p_idx]
                chosen = r2_select_variant(p_entry, config.seed, epoch)
                row_item = raw_by_filename[chosen["image_filename"]]
                epoch_items.append(row_item)
                r2_variant_counter[p_entry["parent_id"]][chosen["variant_name"]] += 1
                ep_log.append({
                    "position": pos,
                    "parent_index": p_idx,
                    "parent_id": p_entry["parent_id"],
                    "label": p_entry["label"],
                    "has_augmented_variants": p_entry["has_augmented_variants"],
                    "initial_offset": chosen["initial_offset"],
                    "selected_slot": chosen["selected_slot"],
                    "variant_name": chosen["variant_name"],
                    "image_filename": chosen["image_filename"],
                })
            r2_epoch_exposures[epoch] = ep_log
        else:
            base_list = manifests["train_full_438"]
            perm = torch.randperm(len(base_list), generator=epoch_rng).tolist()
            epoch_items = [base_list[i] for i in perm]

        ordered_ids = [r["sample_id"] for r in epoch_items]
        epoch_sample_orders[str(epoch)] = {
            "sample_order_sha256": _json_fingerprint(ordered_ids),
            "count": len(ordered_ids),
            "first5": ordered_ids[:5],
            "last5": ordered_ids[-5:],
        }

        train_metrics, _ = _forward_items(
            model,
            epoch_items,
            config,
            device,
            feature_lookup,
            transform,
            optimizer=optimizer,
            criterion=train_criterion,
            cumulative_steps_start=cumulative_steps,
            progress_prefix=f"[{spec['experiment_id']}] epoch={epoch}/{config.epochs} train",
        )
        cumulative_steps = train_metrics["cumulative_optimizer_steps"]

        if state_dict_sha256(model.backbone.state_dict()) != initial_backbone_sha:
            raise RuntimeError(f"Backbone or BatchNorm running stats changed at epoch {epoch}!")

        val_metrics, _ = _forward_items(
            model,
            manifests["val_raw_74"],
            config,
            device,
            feature_lookup,
            transform,
            optimizer=None,
            criterion=val_criterion,
            cumulative_steps_start=cumulative_steps,
            progress_prefix=f"[{spec['experiment_id']}] epoch={epoch}/{config.epochs} val74",
        )

        history.append({
            "epoch": epoch,
            "experiment_id": spec["experiment_id"],
            "evidence_status": config.evidence_status,
            "sample_order_sha256": epoch_sample_orders[str(epoch)]["sample_order_sha256"],
            "train": train_metrics,
            "val": val_metrics,
        })
        write_json(out / "history.json", history)

        with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            fields = [
                "epoch", "optimizer_steps", "cumulative_optimizer_steps",
                "train_loss", "train_accuracy", "train_balanced_accuracy",
                "train_fail_precision", "train_fail_recall", "train_fail_f1",
                "train_false_pass", "train_false_fail", "train_roc_auc",
                "val_loss", "val_accuracy", "val_balanced_accuracy",
                "val_fail_precision", "val_fail_recall", "val_fail_f1",
                "val_false_pass", "val_false_fail", "val_roc_auc",
            ]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for h in history:
                tr, vl = h["train"], h["val"]
                writer.writerow({
                    "epoch": h["epoch"],
                    "optimizer_steps": tr["optimizer_steps"],
                    "cumulative_optimizer_steps": tr["cumulative_optimizer_steps"],
                    "train_loss": tr["loss"],
                    "train_accuracy": tr["accuracy"],
                    "train_balanced_accuracy": tr["balanced_accuracy"],
                    "train_fail_precision": tr["fail_precision"],
                    "train_fail_recall": tr["fail_recall"],
                    "train_fail_f1": tr["fail_f1"],
                    "train_false_pass": tr["false_pass"],
                    "train_false_fail": tr["false_fail"],
                    "train_roc_auc": tr["roc_auc"],
                    "val_loss": vl["loss"],
                    "val_accuracy": vl["accuracy"],
                    "val_balanced_accuracy": vl["balanced_accuracy"],
                    "val_fail_precision": vl["fail_precision"],
                    "val_fail_recall": vl["fail_recall"],
                    "val_fail_f1": vl["fail_f1"],
                    "val_false_pass": vl["false_pass"],
                    "val_false_fail": vl["false_fail"],
                    "val_roc_auc": vl["roc_auc"],
                })

        current_ckpt_payload = {
            "format_version": 1,
            "architecture": "frozen_resnet18_linear",
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "evidence_status": config.evidence_status,
            "val_metrics": val_metrics,
            "run_metadata": run_metadata,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            _save_atomic_checkpoint(out / "best.pt", current_ckpt_payload)
            write_json(out / "best_metrics.json", {
                "experiment_id": spec["experiment_id"],
                "epoch": epoch,
                "evidence_status": config.evidence_status,
                "class_to_idx": config.class_to_idx,
                **val_metrics,
            })

        for m in milestones:
            if epoch <= m and val_metrics["loss"] < milestone_best_loss[m]:
                milestone_best_loss[m] = val_metrics["loss"]
                milestone_best_epoch[m] = epoch
                _save_atomic_checkpoint(out / f"best_within_{m}.pt", current_ckpt_payload)
            if epoch == m:
                _save_atomic_checkpoint(out / f"epoch{m}.pt", current_ckpt_payload)

        print(
            f"[{spec['experiment_id']}] epoch={epoch}/{config.epochs} "
            f"steps={cumulative_steps} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_ba={val_metrics['balanced_accuracy']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f}",
            flush=True,
        )

    train_loop_wall = time.perf_counter() - train_wall_t0

    write_json(out / "epoch_sample_orders.json", epoch_sample_orders)

    last_payload = {
        "format_version": 1,
        "architecture": "frozen_resnet18_linear",
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": config.epochs,
        "evidence_status": config.evidence_status,
        "val_metrics": history[-1]["val"],
        "run_metadata": run_metadata,
    }
    _save_atomic_checkpoint(out / "last.pt", last_payload)

    if spec["train_mode"] == "parent_equal_cycle_294":
        write_json(out / "r2_exposure_log.json", {
            "seed": config.seed,
            "epochs": config.epochs,
            "exposures_per_epoch": 294,
            "total_exposures": 294 * config.epochs,
            "per_epoch_variant_counts": {
                str(ep): dict(collections.Counter(x["variant_name"] for x in rows))
                for ep, rows in r2_epoch_exposures.items()
            },
            "epochs_detail": {str(ep): rows for ep, rows in r2_epoch_exposures.items()},
        })
        with (out / "r2_exposure_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "parent_index", "parent_id", "label", "has_augmented_variants",
                "initial_offset", "original_count", "A01_lowlight_count",
                "A02_warm_count", "A03_softnoise_count", "total_exposures",
            ])
            w.writeheader()
            for p_entry in r2_doc["parents"]:
                pid = p_entry["parent_id"]
                cnt = r2_variant_counter[pid]
                w.writerow({
                    "parent_index": p_entry["parent_index"],
                    "parent_id": pid,
                    "label": p_entry["label"],
                    "has_augmented_variants": p_entry["has_augmented_variants"],
                    "initial_offset": r2_initial_offset(config.seed, pid, 4) if p_entry["has_augmented_variants"] else 0,
                    "original_count": cnt.get("original", 0),
                    "A01_lowlight_count": cnt.get("A01_lowlight", 0),
                    "A02_warm_count": cnt.get("A02_warm", 0),
                    "A03_softnoise_count": cnt.get("A03_softnoise", 0),
                    "total_exposures": sum(cnt.values()),
                })

    allow_smoke = config.data_kind == "SMOKE_ONLY"
    allow_dev = config.training_tier == "DEV_ONLY"

    def _evaluate_saved_checkpoint(ckpt_name: str) -> tuple[dict, list[dict]]:
        ckpt_model, _, _, ckpt_payload = load_checkpoint(
            out / ckpt_name, device=config.device, allow_smoke=allow_smoke, allow_dev_only=allow_dev
        )
        if state_dict_sha256(ckpt_model.backbone.state_dict()) != initial_backbone_sha:
            raise RuntimeError(f"reloaded {ckpt_name} backbone differs from initial frozen ImageNet backbone")
        tr294_m, _ = _forward_items(ckpt_model, manifests["train_raw_294"], config, device, feature_lookup, transform)
        vl74_m, vl74_p = _forward_items(ckpt_model, manifests["val_raw_74"], config, device, feature_lookup, transform)
        if abs(vl74_m["loss"] - ckpt_payload["val_metrics"]["loss"]) > 1e-6:
            raise RuntimeError(f"fixed-weight Val 74 loss disagrees with saved val_metrics in {ckpt_name}")
        vaug18_m, vaug18_p = _forward_items(ckpt_model, manifests["val_aug_18"], config, device, feature_lookup, transform)
        vaug_diag = _build_val_aug_diagnostics(manifests["val_raw_74"], vl74_p, manifests["val_aug_18"], vaug18_p, vaug18_m)
        return {
            "checkpoint": ckpt_name,
            "sha256": file_sha256(out / ckpt_name),
            "selected_epoch": ckpt_payload["epoch"],
            "primary_val_74_metrics": vl74_m,
            "unified_train_raw_294_metrics": tr294_m,
            "val_aug_18_diagnostics": vaug_diag,
        }, vl74_p

    # 4. Evaluate overall best.pt (maintains full compatibility with Phase-1 outputs)
    best_eval_bundle, eval_val_74_preds = _evaluate_saved_checkpoint("best.pt")
    write_json(out / "eval_train_raw_294.json", {
        "checkpoint": "best.pt",
        "best_epoch": best_epoch,
        "scope": "unified_294_real_train_originals_fixed_weights",
        "metrics": best_eval_bundle["unified_train_raw_294_metrics"],
    })

    best_model, _, _, _ = load_checkpoint(out / "best.pt", device=config.device, allow_smoke=allow_smoke, allow_dev_only=allow_dev)
    eval_train_438_metrics, _ = _forward_items(
        best_model, manifests["train_full_438"], config, device, feature_lookup, transform
    )
    supp_train = {
        "checkpoint": "best.pt",
        "best_epoch": best_epoch,
        "train_full_438_unweighted_eval": eval_train_438_metrics,
    }
    if spec["train_mode"] == "parent_equal_cycle_294":
        best_ep_items = [raw_by_filename[x["image_filename"]] for x in r2_epoch_exposures[best_epoch]]
        r2_ep_metrics, _ = _forward_items(best_model, best_ep_items, config, device, feature_lookup, transform)
        supp_train["r2_best_epoch_actual_294_exposures_eval"] = r2_ep_metrics
    write_json(out / "eval_train_supplementary.json", supp_train)

    write_json(out / "eval_val_raw_74.json", {
        "checkpoint": "best.pt",
        "best_epoch": best_epoch,
        "scope": "primary_74_real_val_originals_fixed_weights",
        "metrics": best_eval_bundle["primary_val_74_metrics"],
    })
    write_json(out / "best_val_predictions.json", eval_val_74_preds)
    write_json(out / "eval_val_aug_18_diagnostics.json", best_eval_bundle["val_aug_18_diagnostics"])

    # 5. Evaluate budget milestone checkpoints (e.g. best_within_10.pt and best_within_20.pt)
    milestone_manifests = {}
    if milestones:
        budget_evals = {}
        for m in milestones:
            b_name = f"best_within_{m}.pt"
            ep_name = f"epoch{m}.pt"
            b_eval, b_val_preds = _evaluate_saved_checkpoint(b_name)
            write_json(out / f"best_within_{m}_val_predictions.json", b_val_preds)
            budget_evals[str(m)] = {
                "budget_epochs": m,
                "exposures_within_budget": sum(h["train"]["n"] for h in history[:m]),
                "optimizer_steps_within_budget": history[m - 1]["train"]["cumulative_optimizer_steps"],
                **b_eval,
                "end_of_budget_checkpoint": {
                    "file": ep_name,
                    "epoch": m,
                    "sha256": file_sha256(out / ep_name),
                },
            }
            milestone_manifests[str(m)] = {
                "best_within_budget": {
                    "file": b_name,
                    "epoch": milestone_best_epoch[m],
                    "sha256": file_sha256(out / b_name),
                },
                "epoch_end": {
                    "file": ep_name,
                    "epoch": m,
                    "sha256": file_sha256(out / ep_name),
                },
            }
        write_json(out / "budget_evaluations.json", {
            "experiment_id": spec["experiment_id"],
            "seed": config.seed,
            "note": "All metrics below come from fixed-weight re-evaluations of saved best_within_K.pt checkpoints (never online epoch-K training metrics).",
            "budgets": budget_evals,
        })

    # 6. Verify 10-epoch reproduction against Phase-1 R1 if configured (e.g. seed=42)
    repro_summary = None
    if spec.get("reference_10ep_run_dir"):
        repro_summary = verify_10ep_reproduction(
            out,
            spec["reference_10ep_run_dir"],
            manifests["train_raw_294"],
            config,
            epoch_sample_orders,
            history,
            run_metadata,
        )
        write_json(out / "reproduction_check_vs_r1.json", repro_summary)

    total_wall = time.perf_counter() - total_t0
    ckpt_manifest = {
        "experiment_id": spec["experiment_id"],
        "seed": config.seed,
        "evidence_status": config.evidence_status,
        "best": {"file": "best.pt", "epoch": best_epoch, "sha256": file_sha256(out / "best.pt")},
        "last": {"file": "last.pt", "epoch": config.epochs, "sha256": file_sha256(out / "last.pt")},
        "milestones": milestone_manifests if milestone_manifests else None,
        "reproduction_check_status": repro_summary["status"] if repro_summary else None,
        "timing_seconds": {
            "feature_cache_seconds": cache_info.get("cache_wall_seconds", 0.0),
            "training_epochs_seconds": round(train_loop_wall, 4),
            "training_10_epochs_seconds": round(train_loop_wall, 4),
            "total_experiment_seconds": round(total_wall, 4),
        },
        "exposure_accounting": {
            "exposures_per_epoch": history[0]["train"]["n"],
            "total_exposures": sum(h["train"]["n"] for h in history),
            "total_exposures_10_epochs": sum(h["train"]["n"] for h in history[:10]) if len(history) >= 10 else None,
            "optimizer_steps_per_epoch": history[0]["train"]["optimizer_steps"],
            "total_optimizer_steps": cumulative_steps,
            "total_optimizer_steps_10_epochs": history[9]["train"]["cumulative_optimizer_steps"] if len(history) >= 10 else cumulative_steps,
        },
        "test_data_used": False,
    }
    write_json(out / "checkpoint_manifest.json", ckpt_manifest)
    return ckpt_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="path to V2 experiment JSON spec")
    parser.add_argument("--output", help="new empty output directory for model artifacts")
    parser.add_argument("--allow-dev-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-output", help="optional path to write precheck JSON")
    args = parser.parse_args()
    try:
        if args.check_only:
            res = preflight_v2(args.spec)
            if args.check_output:
                out_p = Path(args.check_output)
                if out_p.exists():
                    raise ValueError(f"check-output already exists: {out_p}")
                write_json(out_p, res)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            if not args.output:
                parser.error("--output is required when not running --check-only")
            res = run_v2_experiment(args.spec, args.output, allow_dev_only=args.allow_dev_only)
            print(json.dumps(res, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "INTERRUPTED: partial best.pt preserved if completed; use a fresh output directory to rerun.\n")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        prefix = "PRECHECK_FAILED" if args.check_only else "V2_TRAIN_FAILED"
        parser.exit(2, f"{prefix}: {exc}\n")


if __name__ == "__main__":
    main()
