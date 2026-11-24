"""Data/configuration checks only: no CNN initialization or optimizer updates."""
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ManifestDataset, check_disjoint
from .team_data import (PREPARED_SCHEMA, _gt_path, classify_reviewed_gt, development_path,
                        read_json, require_approval, sha256_file)


def _real_provenance(config):
    if config.counts_rule_status != "CONFIRMED":
        raise ValueError("WAITING_UPSTREAM: REAL training requires confirmed fixed counts; provisional [2,2,2,2,2] is not approval")
    if not config.data_provenance:
        raise ValueError("WAITING_UPSTREAM: REAL training needs approved data provenance from team_data prepare")
    path = Path(config.data_provenance).resolve()
    record = read_json(path)
    if record.get("schema_version") != PREPARED_SCHEMA or record.get("test_content_read") is not False:
        raise ValueError("unsupported prepared-data provenance")
    parent = path.parent
    for filename, key in (("approval.json", "approval_sha256"), ("reviews.json", "reviews_sha256")):
        if sha256_file(parent / filename) != record.get(key):
            raise ValueError(f"prepared {filename} changed; prepare a new version")
    approval = read_json(parent / "approval.json")
    expected = require_approval(approval)
    if expected != config.supported_counts or expected != record.get("expected_counts"):
        raise ValueError("configured fixed counts differ from approved label-generation rule")
    root = Path(record["dataset_root"]).resolve()
    for path, expected_hash in ((root / "split_manifest.csv", record.get("source_manifest_sha256")),
                                (root / "统一GT/class_mapping.json", record.get("class_mapping_sha256")),
                                (Path(record["catalogue_path"]), approval.get("catalogue_sha256"))):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source changed after approval: {path}")
    if approval.get("source_manifest_sha256") != record.get("source_manifest_sha256"):
        raise ValueError("approval and provenance identify different source manifests")
    for split, configured in (("train", config.train_manifest), ("val", config.val_manifest)):
        if Path(configured).resolve() != parent / f"{split}.csv":
            raise ValueError("config manifest path differs from prepared-data provenance")
        if sha256_file(configured) != record.get("manifest_sha256", {}).get(split):
            raise ValueError(f"prepared {split} manifest changed")
    return record, read_json(parent / "reviews.json"), root


def check_training_inputs(config):
    """Return datasets and a summary, preserving the requirement for BOTH Train/Val."""
    config.__post_init__()
    provenance = None
    dev_record = None
    source_record = None
    if config.data_kind == "REAL":
        if config.training_tier == "DEV_ONLY":
            from .development import verify_dev
            dev_record = verify_dev(config)
            source_record = dev_record
        elif config.data_provenance and Path(config.data_provenance).exists() and read_json(config.data_provenance).get("schema_version") == "member06.prepared-dev.v1":
            raise ValueError("DEV_ONLY data cannot be promoted by changing config tier")
        elif config.counts_rule_status == "CONFIRMED" and config.data_provenance and read_json(config.data_provenance).get("schema_version") == "member06.prepared-data.v2":
            from .development import verify
            source_record = verify(config)
        else:
            provenance = _real_provenance(config)
    train = ManifestDataset(config.train_manifest, config, "train")
    val = ManifestDataset(config.val_manifest, config, "val")
    check_disjoint(train, val)
    if provenance:
        record, reviews, root = provenance
        lineage_rows = record.get("lineage", [])
        lineage = {row["image_path"]: row for row in lineage_rows}
        review_rows = reviews.get("images", [])
        review_by_path = {row["image_path"]: row for row in review_rows}
        if len(lineage) != len(lineage_rows) or len(review_by_path) != len(review_rows):
            raise ValueError("duplicate provenance/review image identity")
        checked = set()
        for dataset in (train, val):
            for row in dataset.rows:
                relative = row["path"].relative_to(root).as_posix()
                source = lineage.get(relative)
                review = review_by_path.get(relative)
                if source is None or review is None or source.get("split") != row["split"]:
                    raise ValueError("missing lineage or mismatched partition")
                if source.get("label") != row["label"] or source.get("scene_id") != row["scene_id"] or review.get("scene_id") != row["scene_id"]:
                    raise ValueError("prepared label/scene differs from reviewed source")
                if source.get("rgb_sha256") != row["sha256_rgb"]:
                    raise ValueError("image pixels changed after preparation")
                image_path = development_path(root, relative, row["split"])
                annotation_path = image_path.with_suffix(".json").resolve()
                if not annotation_path.is_relative_to(root / row["split"]):
                    raise ValueError("annotation path escapes development partition")
                for path, key in ((image_path, "image_sha256"), (annotation_path, "annotation_sha256"),
                                  (_gt_path(root, relative, row["split"]), "gt_sha256")):
                    if sha256_file(path) != source.get(key) or source.get(key) != review.get(key):
                        raise ValueError(f"source/review content changed: {relative}, {key}")
                label = classify_reviewed_gt(read_json(_gt_path(root, relative, row["split"])), config.supported_counts, review)
                if label.get("label") != row["label"]:
                    raise ValueError("prepared label no longer matches approved object rule")
                checked.add(relative)
        if checked != set(lineage):
            raise ValueError("unaccounted images in provenance")
    if source_record:
        from .corpus_audit import describe
        source={r["image_path"]:r for r in source_record["audit"]["images"]}
        for dataset in (train,val):
            for row in dataset.rows:
                original=source[str(row["path"])]
                row.update(describe(original,config.supported_counts))
                row["source_package"]=original["source_package"]
                row["capture_condition"]=original.get("capture_condition","UNRECORDED")
    # Real image loading and input normalization without creating a model or updating weights.
    loader = DataLoader(train, batch_size=min(config.batch_size, len(train)), num_workers=0,
                        shuffle=False, generator=torch.Generator().manual_seed(config.seed))
    images, targets = next(iter(loader))
    if targets.dtype != torch.int64 or not torch.isfinite(images).all():
        raise ValueError("invalid input batch or target dtype")
    summary = {"status": "READY_TRAIN_VAL_ONLY" if config.data_kind == "REAL" else "SMOKE_ONLY",
               "data_kind": config.data_kind, "supported_counts": config.supported_counts,
               "counts_rule_status": config.counts_rule_status,
               "train_images": len(train), "val_images": len(val),
               "train_labels": dict(Counter(row["label"] for row in train.rows)),
               "val_labels": dict(Counter(row["label"] for row in val.rows)),
               "batch_shape": list(images.shape), "target_dtype": str(targets.dtype),
               "model_initialized": False, "weights_updated": False, "test_content_read": False}
    summary["training_tier"] = config.training_tier
    summary["formal_readiness"] = "NOT_READY" if dev_record else ("FORMAL_READY" if config.data_kind=="REAL" else "NOT_APPLICABLE")
    if dev_record:
        summary["status"] = "DEV_ONLY"
        summary["limitations"] = dev_record["limitations"]
        summary["scene_independence_verified"] = False
    return train, val, summary
