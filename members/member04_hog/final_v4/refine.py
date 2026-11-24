"""9 refinement runs (R0/R1/R2 x 3 seeds) on the selected supervised group.

Every epoch trains and then reads validation scores to choose the threshold and the
best checkpoint. Validation never changes weights. R0 epoch 200 must reproduce the
main supervised run for the same seed exactly.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import load_manifest
from ..features import read_image
from .common import (
    CELL, CLASSES, DATA, EPOCHS, EXPECTED_COUNTS, EXPERIMENTS, METHOD_SUFFIX,
    OUT, PROTOCOL, REFINEMENT, ROOT, SEEDS, SIZE, WEIGHTS, choose_threshold,
    dependencies, ensure, now, prediction_rows, read, record_stage, sample_weights,
    score_metrics, sha256, threshold_candidates, update_state, verify_protocol, write,
    write_csv, write_predictions,
)
from .groups import (
    GROUPS, base_features, parameter_count, validation_features, variant_features, view_count,
)
from .train import fit

CONFIGS = {
    "R0": {"layers": (64, 16), "alpha": 0.0001},
    "R1": {"layers": (32, 8), "alpha": 0.0001},
    "R2": {"layers": (64, 16), "alpha": 0.01},
}
ORDER = ("R0", "R1", "R2")


def refinement_key(name, rows):
    specification = CONFIGS[name]
    return (float(np.mean([row["best"]["fn"] for row in rows])),
            float(np.mean([row["best"]["fp"] for row in rows])),
            sum(specification["layers"]), specification["alpha"], ORDER.index(name))


def load_inputs(protocol_path=PROTOCOL):
    protocol = verify_protocol(protocol_path)
    supervised = read(EXPERIMENTS / "summary_supervised.json")
    group = supervised["selected_group"]
    if group not in GROUPS:
        raise ValueError("selected supervised group is missing")
    train = load_manifest(DATA / "manifests/hog/train.csv")
    valid = load_manifest(DATA / "manifests/hog/val.csv")
    images = [read_image(sample.path, SIZE) for sample in train]
    valid_images = [read_image(sample.path, SIZE) for sample in valid]
    originals = base_features(images, group)
    scaler = StandardScaler().fit(originals)
    variants = variant_features(images, group, scaler)
    valid_matrix = validation_features(valid_images, group, scaler)
    labels = np.asarray([int(sample.label == "FAIL") for sample in train])
    valid_labels = np.asarray([int(sample.label == "FAIL") for sample in valid])
    weights = sample_weights(labels)
    return {"protocol": protocol, "summary": supervised, "group": group, "train": train,
            "valid": valid, "labels": labels, "valid_labels": valid_labels,
            "weights": weights, "variants": variants, "valid_matrix": valid_matrix,
            "scaler": scaler, "protocol_path": protocol_path}


def bundle(context, config, seed, model, threshold, epoch):
    protocol = context["protocol"]
    return {
        "format_version": 2,
        "version": METHOD_SUFFIX["refinement"],
        "kind": "hog_supervised_classifier",
        "method": "refinement",
        "role": "local",
        "group": config,
        "seed": seed,
        "epoch": int(epoch),
        "model": model,
        "scaler": context["scaler"],
        "config": {"size": SIZE, "cell": CELL, "epochs": EPOCHS,
                   "threshold": float(threshold), "selected_epoch": int(epoch),
                   "expected_counts": EXPECTED_COUNTS,
                   "base_group": context["group"],
                   "geometry": GROUPS[context["group"]]["geometry"],
                   "gamma": GROUPS[context["group"]]["gamma"],
                   "clahe": GROUPS[context["group"]]["clahe"]},
        "score_type": "uncalibrated_fail_probability",
        "classes": CLASSES,
        "capabilities": {"image_level": True, "objects": False, "arbitrary_counts": False},
        "dependencies": dependencies(),
        "code_sha256": protocol["code_sha256"],
        "data": {"source_commit": protocol["source"]["commit"],
                 "dataset_tree": protocol["source"]["dataset_tree"],
                 "manifests_sha256": protocol["data"]["manifests_sha256"],
                 "counts": protocol["data"]["counts"]},
        "protocol_sha256": sha256(context["protocol_path"]),
    }


def checkpoint_metrics(context, config, seed, name, model, row, destination):
    scores = model.predict_proba(context["valid_matrix"])[:, 1]
    predicted = prediction_rows(context["valid"], scores, row["threshold"],
                                "uncalibrated_fail_probability")
    metrics = score_metrics(predicted)
    metrics.update({"threshold": float(row["threshold"]), "epoch": int(row["epoch"]),
                    "fn": metrics["false_pass"], "fp": metrics["false_fail"],
                    "group": config, "seed": seed, "method": "refinement",
                    "model_path": (WEIGHTS / "refinement" / config /
                                   f"seed{seed}/{name}.joblib").relative_to(ROOT).as_posix()})
    write_predictions(destination / f"{name}_valid_predictions.csv", predicted)
    return metrics


def reproduction_check(context, config, seed, last, destination):
    reference = joblib.load(WEIGHTS / "supervised" / context["group"] / f"seed{seed}/last.joblib")
    same_weights = all(
        np.array_equal(a, b) for a, b in
        zip(last.coefs_, reference["model"].coefs_, strict=True))
    same_weights = same_weights and all(
        np.array_equal(a, b) for a, b in
        zip(last.intercepts_, reference["model"].intercepts_, strict=True))
    same_scaler = (np.array_equal(context["scaler"].mean_, reference["scaler"].mean_)
                   and np.array_equal(context["scaler"].scale_, reference["scaler"].scale_))
    left = last.predict_proba(context["valid_matrix"])[:, 1]
    right = reference["model"].predict_proba(context["valid_matrix"])[:, 1]
    same_predictions = bool(np.array_equal(left, right))
    record = {
        "reference": reference["model_path"] if "model_path" in reference else
        (WEIGHTS / "supervised" / context["group"] / f"seed{seed}/last.joblib").relative_to(ROOT).as_posix(),
        "weights_exact": bool(same_weights), "scaler_exact": bool(same_scaler),
        "predictions_exact": same_predictions,
        "max_probability_difference": float(np.max(np.abs(left - right))),
    }
    write(destination / "reproduction.json", record)
    if not (same_weights and same_scaler and same_predictions):
        raise RuntimeError(f"R0 seed{seed} does not reproduce the main supervised run")


def run(protocol_path=PROTOCOL):
    started = time.perf_counter()
    context = load_inputs(protocol_path)
    decode_seconds = time.perf_counter() - started
    results = {}
    with threadpool_limits(limits=1):
        for config, specification in CONFIGS.items():
            rows = []
            for seed in SEEDS:
                destination = REFINEMENT / config / f"seed{seed}"
                best_artifact = WEIGHTS / "refinement" / config / f"seed{seed}/best.joblib"
                last_artifact = WEIGHTS / "refinement" / config / f"seed{seed}/last.joblib"
                if destination.exists() and best_artifact.exists() and last_artifact.exists():
                    metrics = read(destination / "metrics.json")
                    if (sha256(best_artifact) != metrics["best"]["model_sha256"]
                            or sha256(last_artifact) != metrics["last"]["model_sha256"]):
                        raise RuntimeError(f"existing refinement model changed: {destination}")
                    rows.append(metrics)
                    results[config] = rows
                    write(REFINEMENT / "progress.json", results)
                    print(f"  Reused refinement {config}/seed{seed}", flush=True)
                    continue
                if destination.exists() or best_artifact.exists() or last_artifact.exists():
                    failed = OUT / "failed_attempts" / f"{now().replace(':', '')}_refine_{config}_{seed}"
                    ensure(failed)
                    for label, path in (("output", destination),
                                        ("weights", best_artifact.parent)):
                        if path.exists():
                            shutil.move(str(path), str(failed / f"{label}_{path.name}"))
                    raise RuntimeError(f"incomplete refinement moved to {failed}")
                print(f"Train refinement {config}/seed{seed} started", flush=True)
                run_started = time.perf_counter()
                model = MLPClassifier(
                    hidden_layer_sizes=specification["layers"], activation="relu",
                    solver="adam", alpha=specification["alpha"], learning_rate_init=0.001,
                    batch_size=min(32, len(context["labels"])), shuffle=False, max_iter=1,
                    early_stopping=False, random_state=seed)
                last, best, best_row, history, epoch_scores = fit(
                    model, context["variants"], context["labels"], context["weights"],
                    seed, EPOCHS, collect=True, validation=context["valid_matrix"],
                    validation_labels=context["valid_labels"], score_type="supervised")
                if not np.array_equal(last.classes_, [0, 1]):
                    raise ValueError("unexpected probability column encoding")
                ensure(destination)
                ensure(best_artifact.parent)
                metrics = {"group": config, "seed": seed, "method": "refinement",
                           "base_group": context["group"],
                           "base_views": view_count(context["group"]),
                           "fit_seconds": time.perf_counter() - run_started,
                           "decode_seconds": decode_seconds,
                           "best": {}, "last": {}}
                if config == "R0":
                    reproduction_check(context, config, seed, last, destination)
                    metrics["reproduction"] = read(destination / "reproduction.json")
                for name, checkpoint, row in (("best", best, best_row),
                                              ("last", last, history[-1])):
                    target = WEIGHTS / "refinement" / config / f"seed{seed}/{name}.joblib"
                    payload = bundle(context, config, seed, checkpoint, row["threshold"],
                                     row["epoch"])
                    with target.open("xb") as stream:
                        joblib.dump(payload, stream)
                    metrics[name] = checkpoint_metrics(context, config, seed, name,
                                                       checkpoint, row, destination)
                    metrics[name]["model_sha256"] = sha256(target)
                    metrics[name]["model_bytes"] = target.stat().st_size
                write(destination / "config.json", {
                    "config": config, "seed": seed,
                    "layers": list(specification["layers"]),
                    "alpha": specification["alpha"], "epochs": EPOCHS,
                    "base_group": context["group"],
                    "base_views": view_count(context["group"]),
                    "learning_rate": 0.001, "batch_size": 32, "size": SIZE, "cell": CELL,
                })
                write(destination / "history.json", history)
                write(destination / "validation_scores_by_epoch.json",
                      {"image_ids": [sample.image_id for sample in context["valid"]],
                       "scores": epoch_scores})
                write(destination / "metrics.json", metrics)
                rows.append(metrics)
                results[config] = rows
                write(REFINEMENT / "progress.json", results)
                print(f"  Done {config}/seed{seed}: best epoch={best_row['epoch']} "
                      f"FN={best_row['fn']} FP={best_row['fp']}", flush=True)
    selected = min(results, key=lambda name: refinement_key(name, results[name]))
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "base_group": context["group"],
        "configs": {name: {"layers": list(spec["layers"]), "alpha": spec["alpha"],
                           "parameters": parameter_count("supervised", spec["layers"])}
                    for name, spec in CONFIGS.items()},
        "groups": results,
        "selected_config": selected,
        "selection_key": {name: list(refinement_key(name, rows))
                          for name, rows in results.items()},
        "representative_seed": 42,
        "protocol_sha256": sha256(context["protocol_path"]),
        "code_sha256": context["protocol"]["code_sha256"],
        "selection_rule": "mean FN, mean FP, fewer parameters, smaller alpha, table order",
    }
    write(REFINEMENT / "summary.json", summary)
    record_stage("refine", "completed", f"9 runs, selected {selected}")
    update_state(next_command="python -m members.member04_hog.final_v4.freeze")
    print(f"Refinement complete; selected {selected}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    try:
        run(args.protocol)
    except Exception as error:
        record_stage("refine", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
