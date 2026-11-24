"""Manifest-driven training engine for final_training_v4 (V4-R1 & V4-R2, multi-seed, 10/20-epoch budgets).

Reuses FrozenResNet18, Preprocess, Config, class_weights, load_checkpoint, and v2_train helpers
without hardcoding dataset sizes or legacy paths. Creates a dedicated final_training_v4-bound
Train/Val feature cache (optionally accelerating initial creation via verified single-image donor vectors).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import Config
from .data import Preprocess, image_digest
from .model import FrozenResNet18
from .runtime import choose_device, environment, file_sha256, load_checkpoint, seed_everything, write_json
from .train import class_weights
from .v2_train import (
    _cache_identity_dict,
    _forward_items,
    _json_fingerprint,
    _save_atomic_checkpoint,
    state_dict_sha256,
)
from .v4_data import load_verified_v4_manifests

V4_EXPERIMENT_SCHEMA = "member06.v4_experiment.v1"
V4_CACHE_SCHEMA = "member06.v4_frozen_feature_cache.v1"


def load_v4_experiment_spec(path: str | Path) -> tuple[dict, Config]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != V4_EXPERIMENT_SCHEMA:
        raise ValueError(f"unsupported v4 experiment schema: {raw.get('schema_version')}")
    if raw.get("online_augmentation") != "none":
        raise ValueError("v4 core protocol forbids online augmentation")
    if raw.get("lr_scheduler") != "none":
        raise ValueError("v4 core protocol forbids lr_scheduler")
    if raw.get("threshold_rule") != "default_argmax_pass0_fail1_no_threshold_search":
        raise ValueError("v4 core protocol requires default argmax without threshold search")
    if type(raw.get("use_feature_cache")) is not bool or type(raw.get("allow_optional_aug")) is not bool:
        raise ValueError("use_feature_cache and allow_optional_aug must be booleans")

    cfg = Config(**raw["model_config"])
    if cfg.class_to_idx != {"PASS": 0, "FAIL": 1} or cfg.fail_label != "FAIL":
        raise ValueError("v4 protocol requires PASS=0, FAIL=1")
    if not cfg.data_provenance:
        raise ValueError("v4 experiment requires data_provenance pointing to final_training_v4_intake/provenance.json")

    milestones = raw.get("budget_milestones")
    if (
        not isinstance(milestones, list)
        or not milestones
        or any(type(m) is not int or m < 1 or m > cfg.epochs for m in milestones)
        or milestones != sorted(set(milestones))
        or milestones[-1] != cfg.epochs
    ):
        raise ValueError(f"invalid budget_milestones {milestones} for epochs={cfg.epochs}")

    return raw, cfg


def _load_verified_donor_vectors(
    donor_root: str | Path | None,
    expected_base_identity: dict,
) -> dict[str, torch.Tensor]:
    """Load verified single-image frozen backbone vectors from a prior local cache ONLY if base identity matches 100%."""
    if not donor_root:
        return {}
    root = Path(donor_root).resolve()
    if not root.is_dir():
        return {}
    donor_map: dict[str, torch.Tensor] = {}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        mfile = cdir / "cache_manifest.json"
        tfile = cdir / "features.pt"
        if not (mfile.is_file() and tfile.is_file()):
            continue
        saved = json.loads(mfile.read_text(encoding="utf-8"))
        saved_id = dict(saved.get("identity", {}))
        cmp_expected = dict(expected_base_identity)
        saved_id.pop("schema_version", None)
        cmp_expected.pop("schema_version", None)
        if saved_id != cmp_expected or saved.get("test_cached") is not False:
            continue
        if file_sha256(tfile) != saved.get("features_file_sha256"):
            continue
        feats = torch.load(tfile, map_location="cpu", weights_only=True)
        if not isinstance(feats, torch.Tensor) or feats.dtype != torch.float32 or feats.shape != (len(saved["items"]), 512):
            continue
        if not torch.isfinite(feats).all():
            continue
        for idx, entry in enumerate(saved["items"]):
            if entry.get("split") in {"train", "val"}:
                donor_map[entry["image_sha256"]] = feats[idx]
    return donor_map


def build_or_load_v4_feature_cache(
    cache_root: str | Path,
    dev_items: list[dict],
    model: FrozenResNet18,
    config: Config,
    device: torch.device,
    pretrained_file_sha: str | None,
    provenance_doc: dict,
    *,
    donor_cache_root: str | Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Create or load a standalone cache strictly bound to final_training_v4 Train/Val manifests."""
    for item in dev_items:
        if item["split"] not in {"train", "val"}:
            raise ValueError(f"REFUSED_V4_CACHE_SCOPE: split={item['split']!r} (Test caching strictly forbidden)")
        if "/test/" in Path(item["path"]).as_posix():
            raise ValueError(f"REFUSED_V4_CACHE_SCOPE: test path {item['path']} forbidden")

    ordered = sorted(dev_items, key=lambda r: (r["split"], r["dest_image"]))
    seen_sha = set()
    unique_items = []
    for item in ordered:
        if item["image_sha256"] not in seen_sha:
            seen_sha.add(item["image_sha256"])
            unique_items.append(item)

    backbone_sha = state_dict_sha256(model.backbone.state_dict())
    base_identity = _cache_identity_dict(config, backbone_sha, device, pretrained_file_sha)
    v4_identity = {
        **base_identity,
        "schema_version": V4_CACHE_SCHEMA,
        "dataset_id": "final_training_v4",
        "upstream_commit": provenance_doc["upstream_commit"],
        "v4_provenance_sha256": file_sha256(config.data_provenance),
        "train_manifest_sha256": file_sha256(config.train_manifest),
        "val_manifest_sha256": file_sha256(config.val_manifest),
        "approved_dev_image_count": len(unique_items),
    }

    id_hash = _json_fingerprint(v4_identity)[:16]
    cache_dir = Path(cache_root).resolve() / f"v4_resnet18_sz{config.image_size}_{id_hash}"
    manifest_file = cache_dir / "cache_manifest.json"
    tensor_file = cache_dir / "features.pt"

    if manifest_file.is_file() and tensor_file.is_file():
        t0 = time.perf_counter()
        saved = json.loads(manifest_file.read_text(encoding="utf-8"))
        if saved.get("identity") != v4_identity or saved.get("test_cached") is not False:
            raise ValueError("V4_FEATURE_CACHE_IDENTITY_MISMATCH: header differs from current v4 manifests/model/software")
        if file_sha256(tensor_file) != saved.get("features_file_sha256"):
            raise ValueError("V4_FEATURE_CACHE_IDENTITY_MISMATCH: features.pt hash mismatch")
        feats = torch.load(tensor_file, map_location="cpu", weights_only=True)
        if not isinstance(feats, torch.Tensor) or feats.dtype != torch.float32 or feats.shape != (len(saved["items"]), 512):
            raise ValueError("V4_FEATURE_CACHE_IDENTITY_MISMATCH: invalid tensor shape or dtype")
        if not torch.isfinite(feats).all():
            raise ValueError("V4_FEATURE_CACHE_IDENTITY_MISMATCH: non-finite cached features")
        saved_map = {e["image_sha256"]: (idx, e) for idx, e in enumerate(saved["items"])}
        if len(saved_map) != len(unique_items):
            raise ValueError("V4_FEATURE_CACHE_IDENTITY_MISMATCH: item count mismatch")
        lookup: dict[str, torch.Tensor] = {}
        for item in unique_items:
            sha = item["image_sha256"]
            if sha not in saved_map:
                raise ValueError(f"V4_FEATURE_CACHE_IDENTITY_MISMATCH: missing {item['dest_image']}")
            idx, entry = saved_map[sha]
            if entry["dest_image"] != item["dest_image"] or entry["split"] != item["split"]:
                raise ValueError(f"V4_FEATURE_CACHE_IDENTITY_MISMATCH: role/path mismatch on {item['dest_image']}")
            if file_sha256(item["path"]) != sha:
                raise ValueError(f"V4_FEATURE_CACHE_IDENTITY_MISMATCH: disk file altered for {item['dest_image']}")
            lookup[sha] = feats[idx]
        elapsed = time.perf_counter() - t0
        return lookup, {
            "cache_action": "LOADED_V4_BOUND_CACHE",
            "cache_dir": str(cache_dir),
            "cache_identity_hash": id_hash,
            "cached_images": len(unique_items),
            "features_file_sha256": saved["features_file_sha256"],
            "cache_wall_seconds": round(elapsed, 4),
            "test_cached": False,
        }

    # Build dedicated final_training_v4 cache
    t0 = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = Preprocess(config)
    model.eval()

    donor_lookup = _load_verified_donor_vectors(donor_cache_root, base_identity)
    vecs: list[torch.Tensor | None] = [None] * len(unique_items)
    manifest_entries: list[dict] = []
    to_extract_indices: list[int] = []
    reused_indices: list[int] = []

    for idx, item in enumerate(unique_items):
        sha = item["image_sha256"]
        if file_sha256(item["path"]) != sha:
            raise ValueError(f"v4 image modified before caching: {item['dest_image']}")
        manifest_entries.append({
            "dest_image": item["dest_image"],
            "split": item["split"],
            "usage": item["usage"],
            "image_sha256": sha,
            "rgb_sha256": image_digest(item["path"]) if sha not in donor_lookup else "verified_by_file_sha256",
        })
        if sha in donor_lookup:
            vecs[idx] = donor_lookup[sha].clone()
            reused_indices.append(idx)
        else:
            to_extract_indices.append(idx)

    # Spot-check up to 3 donor-reused images against live CNN forward on final_training_v4 disk files
    spot_max_diff = 0.0
    if reused_indices:
        spot_idxs = [reused_indices[0], reused_indices[len(reused_indices) // 2], reused_indices[-1]]
        spot_idxs = sorted(set(spot_idxs))
        with torch.no_grad():
            spot_x = torch.stack([transform(unique_items[i]["path"]) for i in spot_idxs], dim=0).to(device)
            spot_live = model.extract_features(spot_x).detach().cpu().to(torch.float32)
        for local_i, item_idx in enumerate(spot_idxs):
            diff = float((vecs[item_idx] - spot_live[local_i]).abs().max().item())
            spot_max_diff = max(spot_max_diff, diff)
            if diff > 1e-6:
                raise RuntimeError(f"donor feature spot-check failed on {unique_items[item_idx]['dest_image']}: diff={diff}")

    # Extract fresh features for any remaining approved v4 Train/Val images
    batch_size = max(1, config.batch_size)
    with torch.no_grad():
        for start in range(0, len(to_extract_indices), batch_size):
            chunk_idxs = to_extract_indices[start : start + batch_size]
            batch_x = torch.stack([transform(unique_items[i]["path"]) for i in chunk_idxs], dim=0).to(device)
            batch_feat = model.extract_features(batch_x).detach().cpu().to(torch.float32)
            if batch_feat.shape != (len(chunk_idxs), 512) or not torch.isfinite(batch_feat).all():
                raise ValueError("invalid extracted backbone features")
            for local_i, item_idx in enumerate(chunk_idxs):
                vecs[item_idx] = batch_feat[local_i]

    full_tensor = torch.stack([v for v in vecs if v is not None], dim=0).contiguous()
    if full_tensor.shape != (len(unique_items), 512):
        raise RuntimeError("incomplete v4 feature cache assembly")

    tmp_tensor = cache_dir / "features.tmp"
    torch.save(full_tensor, tmp_tensor)
    tmp_tensor.replace(tensor_file)
    feat_sha = file_sha256(tensor_file)

    provenance_summary = {
        "reused_verified_single_image_vectors": len(reused_indices),
        "freshly_extracted_vectors": len(to_extract_indices),
        "donor_spot_check_count": len(set([reused_indices[0], reused_indices[len(reused_indices)//2], reused_indices[-1]])) if reused_indices else 0,
        "donor_spot_check_max_abs_diff": spot_max_diff,
    }
    write_json(manifest_file, {
        "schema_version": V4_CACHE_SCHEMA,
        "identity": v4_identity,
        "test_cached": False,
        "num_items": len(manifest_entries),
        "feature_dim": 512,
        "features_file": "features.pt",
        "features_file_sha256": feat_sha,
        "build_provenance": provenance_summary,
        "items": manifest_entries,
    })
    elapsed = time.perf_counter() - t0
    lookup = {entry["image_sha256"]: full_tensor[idx] for idx, entry in enumerate(manifest_entries)}
    return lookup, {
        "cache_action": "BUILT_V4_BOUND_CACHE",
        "cache_dir": str(cache_dir),
        "cache_identity_hash": id_hash,
        "cached_images": len(manifest_entries),
        "features_file_sha256": feat_sha,
        "build_provenance": provenance_summary,
        "cache_wall_seconds": round(elapsed, 4),
        "test_cached": False,
    }


def preflight_v4(spec_path: str | Path) -> dict:
    spec, config = load_v4_experiment_spec(spec_path)
    prov, train_rows, val_rows = load_verified_v4_manifests(
        config.train_manifest,
        config.val_manifest,
        config.data_provenance,
        allow_optional_aug=spec["allow_optional_aug"],
    )
    device = choose_device(config.device)

    pretrained_sha = None
    if config.weights == "IMAGENET1K_V1":
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        if not cached.is_file():
            raise FileNotFoundError(f"missing cached ImageNet weights at {cached}")
        pretrained_sha = file_sha256(cached)

    weights_list = (
        class_weights(train_rows, config, "cpu").tolist()
        if config.train_class_weight == "balanced"
        else None
    )
    steps_per_epoch = math.ceil(len(train_rows) / config.batch_size)

    return {
        "status": "FINAL_TRAINING_V4_READY_FOR_USER_START",
        "experiment_id": spec["experiment_id"],
        "strategy_id": spec["strategy_id"],
        "upstream_v4_commit": prov["upstream_commit"],
        "seed": config.seed,
        "epochs": config.epochs,
        "image_size": config.image_size,
        "budget_milestones": spec["budget_milestones"],
        "use_feature_cache": spec["use_feature_cache"],
        "device": str(device),
        "train_manifest": config.train_manifest,
        "train_manifest_sha256": file_sha256(config.train_manifest),
        "val_manifest": config.val_manifest,
        "val_manifest_sha256": file_sha256(config.val_manifest),
        "exposures_per_epoch": len(train_rows),
        "total_exposures": len(train_rows) * config.epochs,
        "train_class_counts": dict(collections.Counter(r["label"] for r in train_rows)),
        "val_images": len(val_rows),
        "val_class_counts": dict(collections.Counter(r["label"] for r in val_rows)),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": steps_per_epoch * config.epochs,
        "train_class_weight_mode": config.train_class_weight,
        "train_class_weights_pass0_fail1": weights_list,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "model_initialized": False,
        "weights_updated": False,
        "test_content_read": False,
    }


def run_v4_experiment(spec_path: str | Path, output_dir: str | Path) -> dict:
    spec, config = load_v4_experiment_spec(spec_path)
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory already exists and is non-empty (refusing overwrite): {out}")

    total_t0 = time.perf_counter()
    prov, train_rows, val_rows = load_verified_v4_manifests(
        config.train_manifest,
        config.val_manifest,
        config.data_provenance,
        allow_optional_aug=spec["allow_optional_aug"],
    )
    # Also load unified primary Train manifest if current run uses optional augmentations, so fixed-weight reference stays unified
    primary_train_csv = Path(prov["v4_root"]) / "manifests/resnet/train.csv"
    if Path(config.train_manifest).resolve() == primary_train_csv.resolve():
        primary_train_rows = train_rows
    else:
        _, primary_train_rows, _ = load_verified_v4_manifests(
            primary_train_csv, config.val_manifest, config.data_provenance, allow_optional_aug=False
        )

    device = choose_device(config.device)
    pretrained_sha = None
    if config.weights == "IMAGENET1K_V1":
        cached_weights = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        if not cached_weights.is_file():
            raise FileNotFoundError(f"missing cached ImageNet weights: {cached_weights}")
        pretrained_sha = file_sha256(cached_weights)

    # 1. Fresh initialization from ImageNet backbone + seed-specific Linear(512, 2) head
    seed_everything(config.seed, config.cpu_threads)
    model = FrozenResNet18(config.weights).to(device)
    initial_backbone_sha = state_dict_sha256(model.backbone.state_dict())
    initial_head_sha = state_dict_sha256(model.head.state_dict())

    # 2. Dedicated final_training_v4 Train/Val frozen feature cache
    feature_lookup = None
    cache_info = {"cache_action": "DISABLED", "test_cached": False}
    if spec["use_feature_cache"]:
        cache_items = list({r["image_sha256"]: r for r in (train_rows + primary_train_rows + val_rows)}.values())
        feature_lookup, cache_info = build_or_load_v4_feature_cache(
            spec["cache_dir"],
            cache_items,
            model,
            config,
            device,
            pretrained_sha,
            prov,
            donor_cache_root=spec.get("verified_donor_cache_root"),
        )
        if state_dict_sha256(model.backbone.state_dict()) != initial_backbone_sha:
            raise RuntimeError("backbone mutated during v4 feature cache construction")

    # 3. Reset RNG immediately before Epoch 1 so same-seed V4-R1 & V4-R2 share identical per-epoch sample order
    seed_everything(config.seed, config.cpu_threads)
    epoch_rng = torch.Generator().manual_seed(config.seed)
    transform = Preprocess(config)

    weights_tensor = (
        class_weights(train_rows, config, device)
        if config.train_class_weight == "balanced"
        else None
    )
    train_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    milestones: list[int] = spec["budget_milestones"]
    milestone_best_loss = {m: float("inf") for m in milestones}
    milestone_best_epoch: dict[int, int | None] = {m: None for m in milestones}

    out.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema_version": "member06.v4_run_metadata.v1",
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": prov["upstream_commit"],
        "experiment_id": spec["experiment_id"],
        "strategy_id": spec["strategy_id"],
        "budget_milestones": milestones,
        "evidence_status": config.evidence_status,
        "environment": environment(device),
        "determinism_notes": {
            "seed": config.seed,
            "cpu_threads": config.cpu_threads,
            "deterministic_algorithms": True,
            "initial_backbone_sha256": initial_backbone_sha,
            "initial_head_sha256": initial_head_sha,
            "shuffle_generator": "torch.Generator().manual_seed(seed) drawing one randperm(len(train_rows)) per epoch",
        },
        "feature_cache": cache_info,
        "manifests": {
            "train_manifest": config.train_manifest,
            "train_manifest_sha256": file_sha256(config.train_manifest),
            "val_manifest": config.val_manifest,
            "val_manifest_sha256": file_sha256(config.val_manifest),
            "v4_provenance_sha256": file_sha256(config.data_provenance),
            "split_manifest_sha256": prov["source_file_sha256"]["split_manifest.csv"],
        },
        "selection_protocol": {
            "validation_manifest": f"manifests/resnet/val.csv ({len(val_rows)} real originals)",
            "validation_loss": "unweighted CrossEntropyLoss mean",
            "selection_rule": f"minimum unweighted Val loss across budget epochs; earliest epoch wins ties",
            "decision_rule": "default argmax of uncalibrated 2-class softmax (PASS=0, FAIL=1; tie -> 0)",
            "test_data_used": False,
        },
        "loss_protocol": {
            "train_class_weight": config.train_class_weight,
            "class_to_idx": config.class_to_idx,
            "weights": weights_tensor.detach().cpu().tolist() if weights_tensor is not None else None,
            "validation": "unweighted CE mean",
        },
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
    best_val_loss = float("inf")
    best_epoch = None
    cumulative_steps = 0

    train_wall_t0 = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        perm = torch.randperm(len(train_rows), generator=epoch_rng).tolist()
        epoch_items = [train_rows[i] for i in perm]
        ordered_ids = [r["image_filename"] for r in epoch_items]
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
            raise RuntimeError(f"Backbone or BatchNorm running stats mutated at epoch {epoch}!")

        val_metrics, _ = _forward_items(
            model,
            val_rows,
            config,
            device,
            feature_lookup,
            transform,
            optimizer=None,
            criterion=val_criterion,
            cumulative_steps_start=cumulative_steps,
            progress_prefix=f"[{spec['experiment_id']}] epoch={epoch}/{config.epochs} val",
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

        ckpt_payload = {
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
            _save_atomic_checkpoint(out / "best.pt", ckpt_payload)
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
                _save_atomic_checkpoint(out / f"best_within_{m}.pt", ckpt_payload)
            if epoch == m:
                _save_atomic_checkpoint(out / f"epoch{m}.pt", ckpt_payload)

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

    allow_smoke = config.data_kind == "SMOKE_ONLY"
    allow_dev = config.training_tier == "DEV_ONLY"

    def _eval_ckpt(ckpt_name: str) -> tuple[dict, list[dict]]:
        ckpt_model, _, _, ckpt_payload = load_checkpoint(
            out / ckpt_name, device=config.device, allow_smoke=allow_smoke, allow_dev_only=allow_dev
        )
        if state_dict_sha256(ckpt_model.backbone.state_dict()) != initial_backbone_sha:
            raise RuntimeError(f"reloaded {ckpt_name} backbone differs from initial frozen ImageNet backbone")
        tr_m, _ = _forward_items(ckpt_model, primary_train_rows, config, device, feature_lookup, transform)
        vl_m, vl_p = _forward_items(ckpt_model, val_rows, config, device, feature_lookup, transform)
        if abs(vl_m["loss"] - ckpt_payload["val_metrics"]["loss"]) > 1e-6:
            raise RuntimeError(f"fixed-weight Val loss disagrees with saved val_metrics in {ckpt_name}")
        return {
            "checkpoint": ckpt_name,
            "sha256": file_sha256(out / ckpt_name),
            "selected_epoch": ckpt_payload["epoch"],
            "val_metrics": vl_m,
            "unified_primary_train_metrics": tr_m,
        }, vl_p

    best_bundle, best_val_preds = _eval_ckpt("best.pt")
    write_json(out / "best_val_predictions.json", best_val_preds)

    budget_evals = {}
    milestone_manifests = {}
    for m in milestones:
        b_name = f"best_within_{m}.pt"
        ep_name = f"epoch{m}.pt"
        b_bundle, b_preds = _eval_ckpt(b_name)
        write_json(out / f"best_within_{m}_val_predictions.json", b_preds)
        budget_evals[str(m)] = {
            "budget_epochs": m,
            "exposures_within_budget": sum(h["train"]["n"] for h in history[:m]),
            "optimizer_steps_within_budget": history[m - 1]["train"]["cumulative_optimizer_steps"],
            **b_bundle,
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
        "strategy_id": spec["strategy_id"],
        "seed": config.seed,
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": prov["upstream_commit"],
        "note": "All checkpoint metrics use unweighted CrossEntropyLoss in eval() mode on fixed saved checkpoints.",
        "budgets": budget_evals,
    })

    total_wall = time.perf_counter() - total_t0
    ckpt_manifest = {
        "experiment_id": spec["experiment_id"],
        "strategy_id": spec["strategy_id"],
        "seed": config.seed,
        "dataset_id": "final_training_v4",
        "upstream_v4_commit": prov["upstream_commit"],
        "evidence_status": config.evidence_status,
        "best": {"file": "best.pt", "epoch": best_epoch, "sha256": file_sha256(out / "best.pt")},
        "last": {"file": "last.pt", "epoch": config.epochs, "sha256": file_sha256(out / "last.pt")},
        "milestones": milestone_manifests,
        "timing_seconds": {
            "feature_cache_seconds": cache_info.get("cache_wall_seconds", 0.0),
            "training_epochs_seconds": round(train_loop_wall, 4),
            "total_experiment_seconds": round(total_wall, 4),
        },
        "exposure_accounting": {
            "exposures_per_epoch": history[0]["train"]["n"],
            "total_exposures": sum(h["train"]["n"] for h in history),
            "optimizer_steps_per_epoch": history[0]["train"]["optimizer_steps"],
            "total_optimizer_steps": cumulative_steps,
        },
        "test_data_used": False,
    }
    write_json(out / "checkpoint_manifest.json", ckpt_manifest)
    return ckpt_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-output")
    args = parser.parse_args()
    try:
        if args.check_only:
            res = preflight_v4(args.spec)
            if args.check_output:
                out_p = Path(args.check_output)
                if out_p.exists():
                    raise ValueError(f"check-output already exists: {out_p}")
                write_json(out_p, res)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            if not args.output:
                parser.error("--output is required when not running --check-only")
            res = run_v4_experiment(args.spec, args.output)
            print(json.dumps(res, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "INTERRUPTED: use a new empty output directory for rerun.\n")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        prefix = "V4_PRECHECK_FAILED" if args.check_only else "V4_TRAIN_FAILED"
        parser.exit(2, f"{prefix}: {exc}\n")


if __name__ == "__main__":
    main()
