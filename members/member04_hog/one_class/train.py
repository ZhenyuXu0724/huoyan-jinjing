"""Train HOG autoencoders only on PASS and select anomaly thresholds on validation."""
import argparse
import json
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import identity, load_manifest, reject_overlap, sha256
from ..evaluate import metrics, write_json, write_predictions
from ..features import extract, read_image
from .settings import GEOMETRY, PHOTOMETRIC, SEEDS
from .scoring import reconstruction_scores, threshold_from_validation


def load_development(train_path, valid_path):
    train, valid = load_manifest(train_path), load_manifest(valid_path)
    if any(sample.label != "PASS" for sample in train):
        raise ValueError("One-class autoencoder training accepts PASS samples only")
    if {sample.label for sample in valid} != {"PASS", "FAIL"}:
        raise ValueError("Validation must contain PASS and FAIL")
    reject_overlap(identity(train), identity(valid))
    return train, valid


def score_rows(samples, scores, threshold):
    return [{"image_id": sample.image_id, "scene_id": sample.scene, "label": sample.label,
             "status": "OK", "predicted_label": "FAIL" if score >= threshold else "PASS",
             "prediction_score": float(score), "score_type": "hog_reconstruction_mse"}
            for sample, score in zip(samples, scores, strict=True)]


def development_record(train, valid):
    return {"train": identity(train), "valid": identity(valid),
            "labels": {name: [(sample.image_id, sample.label) for sample in samples]
                       for name, samples in (("train", train), ("valid", valid))}}


def fit_group(train, valid, config, group, seed, output, weights, *, geometry="A0",
              gamma=False, clahe=False):
    if output.exists() or weights.exists():
        raise ValueError(f"Run already exists: {group}/seed{seed}")
    started = perf_counter()
    settings = {**config, "geometry": geometry, "gamma": gamma, "clahe": clahe}
    originals = [read_image(sample.path, config["size"]) for sample in train]
    original_features = np.stack([extract(image, settings) for image in originals])
    scaler = StandardScaler().fit(original_features)
    normal = scaler.transform(original_features)
    # No validation features, labels or errors enter optimizer updates.
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam", alpha=0.0001,
        learning_rate_init=0.001, batch_size=min(32, len(train)), shuffle=False,
        max_iter=1, early_stopping=False, random_state=seed)
    views, gammas = GEOMETRY[geometry], (0.8, 1.0, 1.2) if gamma else (1.0,)
    variants = {}
    for index, image in enumerate(originals):
        for turns, mirror in views:
            for exponent in gammas:
                key = (index, turns, mirror, exponent)
                variants[key] = scaler.transform(
                    extract(image, settings, turns, mirror, exponent).reshape(1, -1))[0]
    order_rng, view_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    history = []
    for epoch in range(1, config["epochs"] + 1):
        vectors = []
        for index in order_rng.permutation(len(train)):
            turns, mirror = views[view_rng.integers(len(views))]
            exponent = gammas[view_rng.integers(len(gammas))]
            vectors.append(variants[index, turns, mirror, exponent])
        vectors = np.stack(vectors)
        autoencoder.partial_fit(vectors, vectors)
        history.append({"epoch": epoch, "train_loss": float(autoencoder.loss_)})
        if epoch % 50 == 0:
            print(f"{group} seed{seed} epoch {epoch}/{config['epochs']} "
                  f"train_loss={autoencoder.loss_:.6f}", flush=True)
    valid_features = scaler.transform(np.stack(
        [extract(read_image(sample.path, settings["size"]), settings) for sample in valid]))
    scores = reconstruction_scores(autoencoder, valid_features)
    threshold = threshold_from_validation([sample.label for sample in valid], scores)
    settings["threshold"] = threshold
    training_id, valid_id = identity(train), identity(valid)
    bundle = {
        "format_version": 1,
        "kind": "hog_autoencoder", "role": "local", "group": group, "seed": seed,
        "model": autoencoder, "scaler": scaler, "config": settings,
        "development": {key: sorted(set(training_id[key]) | set(valid_id[key]))
                        for key in training_id},
        "local_data": development_record(train, valid),
        "training_code_sha256": sha256(__file__),
        "capabilities": {"image_level": True, "objects": False, "arbitrary_counts": False},
        "protocol": {"weight_fit": "PASS train only", "checkpoint": "last_fixed_epoch",
                     "threshold": "validation FAIL F1; balanced accuracy; fewer false FAIL; larger threshold",
                     "scene_independence": "UNVERIFIED", "count_rule": "PROPOSED"},
        "dependencies": {name: version(name) for name in
                         ("numpy", "Pillow", "scikit-image", "scikit-learn", "joblib")}}
    output.mkdir(parents=True)
    weights.parent.mkdir(parents=True, exist_ok=True)
    with weights.open("xb") as stream:
        joblib.dump(bundle, stream)
    rows = score_rows(valid, scores, threshold)
    valid_metrics = metrics(rows)
    specificity = (valid_metrics["tn"] / (valid_metrics["tn"] + valid_metrics["fp"]))
    valid_metrics.update(
        balanced_accuracy=(valid_metrics["recall"] + specificity) / 2,
        false_pass_rate=valid_metrics["fn"] / (valid_metrics["tp"] + valid_metrics["fn"]),
        false_fail_rate=1 - specificity,
        threshold=threshold, seconds=perf_counter() - started,
        model_bytes=weights.stat().st_size, model_sha256=sha256(weights),
        training_pass_images=len(train), validation_images=len(valid), seed=seed,
        train_reconstruction_mean=float(np.mean(reconstruction_scores(autoencoder, normal))),
        valid_pass_error_mean=float(np.mean(scores[[s.label == "PASS" for s in valid]])),
        valid_fail_error_mean=float(np.mean(scores[[s.label == "FAIL" for s in valid]])))
    write_json(output / "config.json", {**settings, "protocol": bundle["protocol"],
                                        "dependencies": bundle["dependencies"]})
    write_json(output / "history.json", history)
    write_predictions(output / "valid_predictions.csv", rows)
    write_json(output / "metrics.json", valid_metrics)
    print(f"Finished {group} seed{seed}: valid F1={valid_metrics['f1']:.4f}, "
          f"false PASS={valid_metrics['false_pass']}, false FAIL={valid_metrics['false_fail']}",
          flush=True)
    return valid_metrics


def run(train_path, valid_path, output, weights, epochs=200, stage="all"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train, valid = load_development(train_path, valid_path)
    config = {"expected_counts": {str(i): 2 for i in range(1, 6)}, "size": 256,
              "cell": 16, "epochs": epochs}
    options = {name: {"geometry": name} for name in GEOMETRY} | PHOTOMETRIC
    if stage == "baseline":
        options = {"A0": options["A0"]}
    results = {}
    def execute(group, settings):
        runs = []
        for seed in SEEDS:
            destination = output / group / f"seed{seed}"
            artifact = weights / group / f"seed{seed}/model.joblib"
            if destination.exists() and artifact.exists():
                meta = json.loads((destination / "config.json").read_text(encoding="utf-8"))
                saved = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                existing = joblib.load(artifact)
                if (saved["model_sha256"] != sha256(artifact) or meta["epochs"] != epochs
                        or existing["local_data"] != development_record(train, valid)
                        or existing.get("training_code_sha256") != sha256(__file__)
                        or any(meta.get(k) != v for k, v in settings.items())):
                    raise ValueError("Existing run is incompatible; use new directories")
                runs.append(saved)
                print(f"Reused completed {group} seed{seed}", flush=True)
                continue
            runs.append(fit_group(train, valid, config, group, seed, destination, artifact, **settings))
        results[group] = {
            "mean_f1": statistics.mean(row["f1"] for row in runs),
            "stdev_f1": statistics.stdev(row["f1"] for row in runs),
            "mean_false_pass": statistics.mean(row["false_pass"] for row in runs),
            "mean_false_fail": statistics.mean(row["false_fail"] for row in runs), "runs": runs}
        write_json(output / "progress.json", results)
    with threadpool_limits(limits=1):
        for group, settings in options.items():
            execute(group, settings)
        if stage == "all":
            best_geometry = max(GEOMETRY, key=lambda name:
                                (results[name]["mean_f1"], -len(GEOMETRY[name])))
            enabled = {}
            if results[best_geometry]["mean_f1"] > results["A0"]["mean_f1"]:
                enabled["geometry"] = best_geometry
            for group, setting in PHOTOMETRIC.items():
                if results[group]["mean_f1"] > results["A0"]["mean_f1"]:
                    enabled.update(setting)
            if len(enabled) > 1:
                combinations = {"C0": enabled}
                combinations.update({f"C_without_{key}": {k: v for k, v in enabled.items() if k != key}
                                     for key in enabled})
                options.update(combinations)
                for group, settings in combinations.items():
                    execute(group, settings)
    def rank(name):
        return (results[name]["mean_f1"], -len(options[name]),
                -len(GEOMETRY[options[name].get("geometry", "A0")]))
    selected = max(results, key=rank)
    write_json(output / "summary.json", {
        "method": "HOG + PASS-only MLP autoencoder", "selected_group": selected,
        "groups": results, "train_manifest_sha256": sha256(train_path),
        "valid_manifest_sha256": sha256(valid_path),
        "usage": "development_validation_not_independent_test"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "valid", "out", "weights"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stage", choices=("baseline", "all"), default="all")
    args = parser.parse_args()
    run(args.train, args.valid, args.out, args.weights, args.epochs, args.stage)


if __name__ == "__main__":
    main()
