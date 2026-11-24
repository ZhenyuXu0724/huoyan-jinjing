"""One-time frozen evaluation on train/val/test plus grouped errors and cost."""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ..data import load_manifest, sha256
from .common import (
    AUDIT, EVALUATION, EXPERIMENTS, OUT, REFINEMENT, ROOT, WEIGHTS, ensure, now,
    prediction_rows, read, read_csv, record_stage, score_metrics, update_state, write,
    write_csv, write_predictions,
)


def bundle_paths(freeze):
    return {method: ROOT / record["model_path"]
            for method, record in freeze["representatives"].items()}


def load_everything():
    freeze_path = OUT / "freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("evaluate requires freeze.json")
    freeze = read(freeze_path)
    if freeze.get("test_accessed"):
        raise RuntimeError("freeze already records test access")
    audit_path = AUDIT / "dataset_audit_full.json"
    if not audit_path.is_file() or read(audit_path)["status"] != "PASS":
        raise RuntimeError("evaluate requires a passing full-scope audit")
    access_path = AUDIT / "test_access.json"
    if not access_path.is_file() or read(access_path).get("freeze_sha256") != sha256(freeze_path):
        raise RuntimeError("full-scope restore is missing or does not match this freeze")
    return freeze, freeze_path


def training_seconds(freeze, method):
    record = freeze["representatives"][method]
    if method == "refinement":
        path = REFINEMENT / record["group"] / "seed42/metrics.json"
        return float(read(path)["fit_seconds"])
    path = EXPERIMENTS / method / record["group"] / "seed42/metrics.json"
    return float(read(path)["fit_seconds"])


def grouped_rows(predictions, manifest_rows, model, split):
    metadata = {row["filename"]: row for row in manifest_rows}
    groups = defaultdict(list)
    for row in predictions:
        meta = metadata.get(row["image_id"])
        if meta is None:
            continue
        distractor = sum(int(meta[key]) for key in "NBSWH")
        keys = {
            "batch": meta["batch"], "condition": meta["condition"],
            "anomaly": meta["anomaly"], "scene_group": meta["scene_group"],
            "distractor_count": str(distractor), "label": row["label"],
        }
        for name, value in keys.items():
            groups[(name, value)].append(row)
    rows = []
    for (name, value), items in sorted(groups.items()):
        stats = score_metrics(items)
        rows.append({"split": split, "model": model, "group": name, "value": value,
                     "images": len(items), "tp": stats["tp"], "fp": stats["fp"],
                     "fn": stats["fn"], "tn": stats["tn"],
                     "f1": stats["f1"], "balanced_accuracy": stats["balanced_accuracy"],
                     "coverage": stats["coverage"]})
    return rows


def unseen_rows(predictions, manifest_rows):
    unseen = {row["filename"] for row in manifest_rows
              if int(row["S"]) > 0 or int(row["H"]) > 0}
    selected = [row for row in predictions if row["image_id"] in unseen]
    rest = [row for row in predictions if row["image_id"] not in unseen]
    return {"unseen_types_images": len(unseen),
            "unseen": score_metrics(selected) if selected else None,
            "rest": score_metrics(rest) if rest else None}


def cost_benchmark(freeze, bundles, val_samples):
    ordered = sorted(val_samples, key=lambda sample: sample.image_id)
    pass_samples = [sample for sample in ordered if sample.label == "PASS"][:10]
    fail_samples = [sample for sample in ordered if sample.label == "FAIL"][:10]
    if len(pass_samples) != 10 or len(fail_samples) != 10:
        raise RuntimeError("cost benchmark needs ten validation PASS and ten FAIL images")
    samples = pass_samples + fail_samples
    timings = defaultdict(list)
    rows = []
    with threadpool_limits(limits=1):
        for sample in samples[:2]:
            for method, bundle in bundles.items():
                from ..model import predict_with_model
                predict_with_model(sample.path, bundle)
        for round_number in (1, 2, 3):
            for sample in samples:
                for method, bundle in bundles.items():
                    from ..model import predict_with_model
                    started = time.perf_counter()
                    result = predict_with_model(sample.path, bundle)
                    elapsed = (time.perf_counter() - started) * 1000
                    if result["status"] != "OK":
                        raise RuntimeError(f"cost prediction failed for {sample.image_id}")
                    timings[method].append(elapsed)
                    rows.append({"round": round_number, "model": method,
                                 "image_id": sample.image_id, "label": sample.label,
                                 "milliseconds": elapsed})
    summary = {}
    for method, values in timings.items():
        array = np.asarray(values)
        summary[method] = {
            "n": int(array.size), "median_ms": float(np.median(array)),
            "p95_ms": float(np.percentile(array, 95)), "mean_ms": float(array.mean()),
            "model_bytes": (ROOT / freeze["representatives"][method]["model_path"]).stat().st_size,
            "training_seconds": training_seconds(freeze, method),
        }
    return summary, rows


def run():
    freeze, freeze_path = load_everything()
    destination = EVALUATION
    if destination.exists():
        raise RuntimeError("evaluation already exists; test is evaluated only once")
    ensure(destination)
    write(destination / "test_access.json", {
        "accessed_at": now(), "freeze_sha256": sha256(freeze_path),
        "scope": "one-time test evaluation of the three frozen representatives",
        "models": list(freeze["representatives"]),
    })
    bundles = {method: joblib.load(path)
               for method, path in bundle_paths(freeze).items()}
    split_names = ("train", "val", "test")
    samples = {split: load_manifest(ROOT / "data/member04_hog/final_training_v4/"
                                    f"manifests/hog/{split}.csv") for split in split_names}
    split_manifest = read_csv(ROOT / "data/member04_hog/final_training_v4/"
                              "split_manifest.csv")
    manifest_rows = {split: [row for row in split_manifest
                             if row["split"] == split and row["usage"] == "primary"]
                     for split in split_names}
    from ..model import predict_with_model

    predictions = {}
    split_summary = {}
    grouped = []
    errors = {}
    with threadpool_limits(limits=1):
        for split in split_names:
            for method, bundle in bundles.items():
                rows = []
                for sample in samples[split]:
                    result = predict_with_model(sample.path, bundle)
                    rows.append({"image_id": sample.image_id, "split": split,
                                 "label": sample.label, **result})
                predictions[(split, method)] = rows
                write_predictions(destination / f"{split}_{method}_predictions.csv", rows)
                metrics = score_metrics(rows)
                metrics.update({"split": split, "model": method,
                                "score_type": bundle["score_type"],
                                "threshold": float(bundle["config"]["threshold"])})
                split_summary[f"{split}/{method}"] = metrics
                errors[(split, method)] = set(metrics["errors"])
                grouped.extend(grouped_rows(rows, manifest_rows[split], method, split))
                print(f"Evaluated {split}/{method}: FN={metrics['fn']} FP={metrics['fp']}",
                      flush=True)
    grouped_written = []
    for row in grouped:
        grouped_written.append({key: value for key, value in row.items()})
    write_csv(destination / "group_metrics.csv", grouped_written)
    unseen = {}
    for method in bundles:
        unseen[method] = unseen_rows(predictions[("test", method)], manifest_rows["test"])
    error_rows = []
    for split in split_names:
        identifiers = sorted(set().union(*(errors[(split, method)] for method in bundles)))
        metadata = {row["filename"]: row for row in manifest_rows[split]}
        for image_id in identifiers:
            error_rows.append({
                "split": split, "image_id": image_id,
                "label": metadata[image_id]["label"],
                **{f"error_{method}": image_id in errors[(split, method)]
                   for method in bundles},
            })
    write_csv(destination / "error_ids.csv", error_rows) if error_rows else None
    constants = {}
    for split in split_names:
        support = Counter(sample.label for sample in samples[split])
        total = sum(support.values())
        constants[split] = {
            "images": total, "PASS": support["PASS"], "FAIL": support["FAIL"],
            "all_pass_accuracy": support["PASS"] / total if total else None,
            "all_fail_accuracy": support["FAIL"] / total if total else None,
            "all_fail_f1": (2 * support["FAIL"] / (total + support["FAIL"])
                            if total + support["FAIL"] else None),
        }
    cost, cost_rows = cost_benchmark(freeze, bundles, samples["val"])
    write_csv(destination / "cost_samples.csv", cost_rows)
    write(destination / "cost.json", cost)
    write(destination / "summary.json", {
        "schema_version": 1, "created_at": now(), "freeze_sha256": sha256(freeze_path),
        "reports": list(bundles), "split_metrics": split_summary,
        "unseen_types": unseen, "constant_baselines": constants,
        "error_ids": error_rows, "cost": cost,
        "notes": [
            "The AE learned from 166 PASS originals only; train metrics also score the 83 FAIL images it never learned from.",
            "The test split was decoded once after freeze and evaluated exactly once with the three frozen representatives.",
            "Test scores are development evidence, not an independent performance guarantee.",
        ],
    })
    record_stage("evaluate", "completed", "one-time test evaluation of three representatives")
    update_state(next_command="python -m members.member04_hog.final_v4.report")
    print("Evaluation complete", flush=True)
    return split_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", action="store_true", required=True)
    args = parser.parse_args()
    try:
        run()
    except Exception as error:
        record_stage("evaluate", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
