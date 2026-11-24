"""66 formal main runs: 11 groups x 3 seeds for the AE and supervised methods.

Every epoch draws exactly one random view per original image; epoch 200 is used
without further checkpoint selection. Validation only chooses the threshold.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import load_manifest
from ..features import read_image
from .common import (
    CELL, CLASSES, DATA, EPOCHS, EXPECTED_COUNTS, EXPERIMENTS, FP_BUDGET,
    METHOD_SUFFIX, OUT, PROTOCOL, ROOT, SEEDS, SIZE, SOURCE_LOCK, WEIGHTS, choose_threshold,
    code_hashes, dependencies, ensure, now, prediction_rows, read, record_stage,
    reconstruction_scores, sample_weights, score_metrics, sha256, threshold_candidates,
    update_state, verify_protocol, write, write_csv, write_predictions,
)
from .groups import (
    EQUIVALENT_PAIRS, GROUPS, base_features, parameter_count, selection_key,
    validation_features, variant_features, view_count,
)

SCORE_TYPE = {"ae": "hog_reconstruction_mse",
              "supervised": "uncalibrated_fail_probability"}
KIND = {"ae": "hog_autoencoder", "supervised": "hog_supervised_classifier"}
LAYERS = {"ae": (64, 16, 64), "supervised": (64, 16)}
ALPHA = 0.0001


def load_samples():
    train = load_manifest(DATA / "manifests/hog/train.csv")
    valid = load_manifest(DATA / "manifests/hog/val.csv")
    if {sample.label for sample in valid} != {"PASS", "FAIL"}:
        raise ValueError("validation must contain PASS and FAIL")
    return train, valid


def fit(model, variants, labels, weights, seed, epochs, *, collect=False,
        validation=None, validation_labels=None, score_type="supervised"):
    order_rng = np.random.default_rng(seed)
    view_rng = np.random.default_rng(seed)
    history = []
    best = None
    best_model = None
    epoch_scores = []
    for epoch in range(1, epochs + 1):
        order = order_rng.permutation(len(variants[0]))
        chosen = view_rng.integers(len(variants), size=len(order))
        matrix = np.stack([variants[view][index]
                           for index, view in zip(order, chosen, strict=True)])
        if score_type == "ae":
            model.partial_fit(matrix, matrix)
        else:
            model.partial_fit(matrix, labels[order], classes=np.array([0, 1]),
                              sample_weight=weights[order])
        row = {"epoch": epoch, "train_loss": float(model.loss_)}
        history.append(row)
        if epoch % 50 == 0:
            print(f"    epoch {epoch}/{epochs} loss={model.loss_:.6f}", flush=True)
        if collect:
            scores = (reconstruction_scores(model, validation)
                      if score_type == "ae" else model.predict_proba(validation)[:, 1])
            choice = choose_threshold(validation_labels, scores)
            row.update({key: choice[key] for key in ("threshold", "fn", "fp")})
            epoch_scores.append([float(value) for value in scores])
            if best is None or (row["fn"], row["fp"], row["epoch"]) < (
                    best["fn"], best["fp"], best["epoch"]):
                best = row.copy()
                best_model = copy.deepcopy(model)
    return model, best_model, best, history, epoch_scores


def bundle(method, group, seed, model, scaler, threshold, epoch, protocol, protocol_path):
    settings = {"size": SIZE, "cell": CELL, "geometry": GROUPS[group]["geometry"],
                "gamma": GROUPS[group]["gamma"], "clahe": GROUPS[group]["clahe"],
                "epochs": EPOCHS, "threshold": float(threshold),
                "selected_epoch": int(epoch), "expected_counts": EXPECTED_COUNTS}
    return {
        "format_version": 2,
        "version": METHOD_SUFFIX[method],
        "kind": KIND[method],
        "method": method,
        "role": "local",
        "group": group,
        "seed": seed,
        "epoch": int(epoch),
        "model": model,
        "scaler": scaler,
        "config": settings,
        "score_type": SCORE_TYPE[method],
        "classes": CLASSES,
        "capabilities": {"image_level": True, "objects": False, "arbitrary_counts": False},
        "dependencies": dependencies(),
        "code_sha256": protocol["code_sha256"],
        "data": {
            "source_commit": protocol["source"]["commit"],
            "dataset_tree": protocol["source"]["dataset_tree"],
            "manifests_sha256": protocol["data"]["manifests_sha256"],
            "counts": protocol["data"]["counts"],
        },
        "protocol_sha256": sha256(protocol_path),
    }


def run_metrics(method, samples, scores, threshold):
    rows = prediction_rows(samples, scores, threshold, SCORE_TYPE[method])
    result = score_metrics(rows)
    result.update({"threshold": float(threshold), "epoch": EPOCHS,
                   "fn": result["false_pass"], "fp": result["false_fail"]})
    return rows, result


def reuse_or_fail(destination, artifact, group, seed, method, protocol):
    if destination.exists() and artifact.exists():
        metrics = read(destination / "metrics.json")
        if sha256(artifact) != metrics["model_sha256"]:
            raise RuntimeError(f"existing model changed: {artifact}")
        if metrics.get("group") != group or metrics.get("seed") != seed:
            raise RuntimeError(f"existing run identity mismatch: {artifact}")
        print(f"  Reused {method} {group}/{seed}", flush=True)
        return metrics
    if destination.exists() or artifact.exists():
        failed = OUT / "failed_attempts" / f"{now().replace(':', '')}_{method}_{group}_{seed}"
        ensure(failed)
        for label, path in (("output", destination), ("weights", artifact.parent)):
            if path.exists():
                shutil.move(str(path), str(failed / f"{label}_{path.name}"))
        raise RuntimeError(f"incomplete run moved to {failed}; rerun the same command")
    return None


def train_method(method, protocol_path):
    protocol = verify_protocol(protocol_path)
    started = time.perf_counter()
    train, valid = load_samples()
    if method == "ae":
        train = [sample for sample in train if sample.label == "PASS"]
        if len(train) != 166:
            raise ValueError("AE training accepts exactly the 166 PASS originals")
    images = [read_image(sample.path, SIZE) for sample in train]
    valid_images = [read_image(sample.path, SIZE) for sample in valid]
    decode_seconds = time.perf_counter() - started
    labels = np.asarray([int(sample.label == "FAIL") for sample in train])
    valid_labels = np.asarray([int(sample.label == "FAIL") for sample in valid])
    weights = sample_weights(labels) if method == "supervised" else None
    results = {}
    progress_path = EXPERIMENTS / f"progress_{method}.json"
    with threadpool_limits(limits=1):
        for group in GROUPS:
            feature_started = time.perf_counter()
            originals = base_features(images, group)
            scaler = StandardScaler().fit(originals)
            variants = variant_features(images, group, scaler)
            valid_matrix = validation_features(valid_images, group, scaler)
            feature_seconds = time.perf_counter() - feature_started
            rows = []
            for seed in SEEDS:
                destination = EXPERIMENTS / method / group / f"seed{seed}"
                artifact = WEIGHTS / method / group / f"seed{seed}/last.joblib"
                reused = reuse_or_fail(destination, artifact, group, seed, method, protocol)
                if reused is not None:
                    rows.append(reused)
                    results[group] = rows
                    write(progress_path, results)
                    continue
                print(f"Train {method} {group}/seed{seed} started", flush=True)
                run_started = time.perf_counter()
                if method == "ae":
                    model = MLPRegressor(
                        hidden_layer_sizes=LAYERS[method], activation="relu", solver="adam",
                        alpha=ALPHA, learning_rate_init=0.001,
                        batch_size=min(32, len(labels)), shuffle=False, max_iter=1,
                        early_stopping=False, random_state=seed)
                    last, best, choice, history, epoch_scores = fit(
                        model, variants, labels, None, seed, EPOCHS,
                        collect=False, validation=valid_matrix,
                        validation_labels=valid_labels, score_type="ae")
                    scores = reconstruction_scores(last, valid_matrix)
                else:
                    model = MLPClassifier(
                        hidden_layer_sizes=LAYERS[method], activation="relu", solver="adam",
                        alpha=ALPHA, learning_rate_init=0.001,
                        batch_size=min(32, len(labels)), shuffle=False, max_iter=1,
                        early_stopping=False, random_state=seed)
                    last, best, choice, history, epoch_scores = fit(
                        model, variants, labels, weights, seed, EPOCHS,
                        collect=False, validation=valid_matrix,
                        validation_labels=valid_labels, score_type="supervised")
                    if not np.array_equal(last.classes_, [0, 1]):
                        raise ValueError("unexpected probability column encoding")
                    scores = last.predict_proba(valid_matrix)[:, 1]
                chosen = choose_threshold(valid_labels, scores)
                candidates = threshold_candidates(valid_labels, scores)
                threshold = float(chosen["threshold"])
                predicted, metrics = run_metrics(method, valid, scores, threshold)
                if method == "ae":
                    metrics["train_reconstruction_mean"] = float(np.mean(
                        reconstruction_scores(last, scaler.transform(originals))))
                ensure(destination.parent)
                ensure(artifact.parent)
                payload = bundle(method, group, seed, last, scaler, threshold, EPOCHS,
                                 protocol, protocol_path)
                with artifact.open("xb") as stream:
                    joblib.dump(payload, stream)
                ensure(destination)
                write(destination / "config.json", {
                    "method": method, "group": group, "seed": seed,
                    "layers": list(LAYERS[method]), "alpha": ALPHA, "epochs": EPOCHS,
                    "learning_rate": 0.001, "batch_size": 32,
                    "size": SIZE, "cell": CELL,
                    "geometry": GROUPS[group]["geometry"],
                    "gamma": GROUPS[group]["gamma"], "clahe": GROUPS[group]["clahe"],
                    "views": view_count(group), "threshold": threshold,
                    "selected_epoch": EPOCHS,
                    "sample_weight": "n/(2*n_class)" if method == "supervised" else None,
                })
                write(destination / "history.json", history)
                write_csv(destination / "threshold_candidates.csv", candidates)
                write_predictions(destination / "valid_predictions.csv", predicted)
                metrics.update({
                    "group": group, "seed": seed, "method": method,
                    "model_sha256": sha256(artifact),
                    "model_path": artifact.relative_to(ROOT).as_posix(),
                    "model_bytes": artifact.stat().st_size,
                    "fit_seconds": time.perf_counter() - run_started,
                    "feature_seconds": feature_seconds,
                    "decode_seconds": decode_seconds,
                    "views": view_count(group),
                })
                write(destination / "metrics.json", metrics)
                rows.append(metrics)
                results[group] = rows
                write(progress_path, results)
                print(f"  Done {group}/seed{seed}: FN={metrics['fn']} FP={metrics['fp']} "
                      f"epoch={metrics['epoch']} {metrics['fit_seconds']:.1f}s", flush=True)
    selected = min(results, key=lambda name: selection_key(name, results[name]))
    equivalence = []
    for left, right in EQUIVALENT_PAIRS:
        pair_ok = True
        detail = []
        for seed in SEEDS:
            left_bundle = joblib.load(WEIGHTS / method / left / f"seed{seed}/last.joblib")
            right_bundle = joblib.load(WEIGHTS / method / right / f"seed{seed}/last.joblib")
            same_weights = all(
                np.array_equal(a, b) for a, b in
                zip(left_bundle["model"].coefs_, right_bundle["model"].coefs_, strict=True))
            same_weights = same_weights and all(
                np.array_equal(a, b) for a, b in
                zip(left_bundle["model"].intercepts_,
                    right_bundle["model"].intercepts_, strict=True))
            same_scaler = (np.array_equal(left_bundle["scaler"].mean_,
                                          right_bundle["scaler"].mean_)
                           and np.array_equal(left_bundle["scaler"].scale_,
                                              right_bundle["scaler"].scale_))
            left_features = validation_features(valid_images, left, left_bundle["scaler"])
            right_features = validation_features(valid_images, right, right_bundle["scaler"])
            if method == "ae":
                left_scores = reconstruction_scores(left_bundle["model"], left_features)
                right_scores = reconstruction_scores(right_bundle["model"], right_features)
            else:
                left_scores = left_bundle["model"].predict_proba(left_features)[:, 1]
                right_scores = right_bundle["model"].predict_proba(right_features)[:, 1]
            same_predictions = bool(np.array_equal(left_scores, right_scores))
            detail.append({"seed": seed, "weights": bool(same_weights),
                           "scaler": bool(same_scaler),
                           "predictions": same_predictions})
            pair_ok = pair_ok and same_weights and same_scaler and same_predictions
        equivalence.append({"pair": [left, right], "identical": bool(pair_ok),
                            "seeds": detail})
        if not pair_ok:
            raise RuntimeError(f"equivalent groups differ: {left} vs {right}")
    summary = {
        "schema_version": 1,
        "method": method,
        "created_at": now(),
        "groups": results,
        "selected_group": selected,
        "selection_key": {name: list(selection_key(name, rows))
                          for name, rows in results.items()},
        "representative_seed": 42,
        "equivalence": equivalence,
        "protocol_sha256": sha256(protocol_path),
        "code_sha256": protocol["code_sha256"],
        "selection_rule": "mean FN, mean FP, fewer augmentation types, fewer views, "
                          "table order; single-factor group wins exact duplicates",
    }
    write(EXPERIMENTS / f"summary_{method}.json", summary)
    record_stage(f"train_{method}", "completed",
                 f"{len(GROUPS) * len(SEEDS)} runs, selected {selected}")
    update_state(next_command=(
        "python -m members.member04_hog.final_v4.train --method supervised "
        "--protocol members/member04_hog/configs/final_v4/protocol.json"
        if method == "ae" else
        "python -m members.member04_hog.final_v4.refine "
        "--protocol members/member04_hog/configs/final_v4/protocol.json"))
    print(f"{method} complete; selected group {selected}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ae", "supervised"), required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    try:
        train_method(args.method, args.protocol)
    except Exception as error:
        record_stage(f"train_{args.method}", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
