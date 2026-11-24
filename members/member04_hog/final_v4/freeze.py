"""Freeze three method representatives and select the v4 default before test access."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np

from .common import (
    ACTIVE, AUDIT, DELIVERY_NAME, EXPERIMENTS, METHOD_SUFFIX, MODEL_CODE, OUT, PROTOCOL,
    REFINEMENT, ROOT, SOURCE_LOCK, WEIGHTS, code_hashes, dependencies, ensure, now, read,
    record_stage, sha256, update_state, validate_bundle, verify_protocol, write,
)
from .groups import parameter_count, view_count

PRIORITY = {"supervised": 0, "refinement": 1, "ae": 2}
KIND = {"ae": "hog_autoencoder", "supervised": "hog_supervised_classifier",
        "refinement": "hog_supervised_classifier"}
SCORE = {"ae": "hog_reconstruction_mse",
         "supervised": "uncalibrated_fail_probability",
         "refinement": "uncalibrated_fail_probability"}


def candidate_rows():
    supervised = read(EXPERIMENTS / "summary_supervised.json")
    ae = read(EXPERIMENTS / "summary_ae.json")
    refinement = read(REFINEMENT / "summary.json")
    refinement_group = refinement["selected_config"]
    refinement_layers = tuple(refinement["configs"][refinement_group]["layers"])
    rows = {
        "ae": {
            "group": ae["selected_group"],
            "rows": ae["groups"][ae["selected_group"]],
            "params": parameter_count("ae", (64, 16, 64)),
            "views": view_count(ae["selected_group"]),
            "epoch": 200,
            "model": WEIGHTS / "ae" / ae["selected_group"] / "seed42/last.joblib",
            "fn": lambda item: item["fn"],
            "fp": lambda item: item["fp"],
            "representative": lambda item: item,
        },
        "supervised": {
            "group": supervised["selected_group"],
            "rows": supervised["groups"][supervised["selected_group"]],
            "params": parameter_count("supervised", (64, 16)),
            "views": view_count(supervised["selected_group"]),
            "epoch": 200,
            "model": WEIGHTS / "supervised" / supervised["selected_group"] /
                     "seed42/last.joblib",
            "fn": lambda item: item["fn"],
            "fp": lambda item: item["fp"],
            "representative": lambda item: item,
        },
        "refinement": {
            "group": refinement_group,
            "rows": refinement["groups"][refinement_group],
            "params": parameter_count("supervised", refinement_layers),
            "views": view_count(refinement["base_group"]),
            "epoch": None,
            "model": WEIGHTS / "refinement" / refinement_group / "seed42/best.joblib",
            "fn": lambda item: item["best"]["fn"],
            "fp": lambda item: item["best"]["fp"],
            "representative": lambda item: item["best"],
        },
    }
    for method, row in rows.items():
        row["mean_fn"] = float(np.mean([row["fn"](item) for item in row["rows"]]))
        row["mean_fp"] = float(np.mean([row["fp"](item) for item in row["rows"]]))
        seed42 = next(item for item in row["rows"] if item["seed"] == 42)
        representative = row["representative"](seed42)
        row["threshold"] = float(representative["threshold"])
        row["epoch"] = int(representative["epoch"])
        row["model_sha256"] = representative["model_sha256"]
    return rows


def selection_key(method, row):
    return (row["mean_fn"], row["mean_fp"], row["params"], row["views"], PRIORITY[method])


def copy_model(source, destination):
    ensure(destination.parent)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"refusing to replace a different representative: {destination}")
        return sha256(destination)
    shutil.copyfile(source, destination)
    return sha256(destination)


def run():
    protocol = verify_protocol()
    state = read(OUT / "state.json") if (OUT / "state.json").exists() else {}
    if state.get("test_accessed"):
        raise RuntimeError("refusing to freeze after test access")
    rows = candidate_rows()
    candidates = []
    representatives = {}
    for method in ("ae", "supervised", "refinement"):
        row = rows[method]
        model_path = WEIGHTS / "representatives" / f"{method}.joblib"
        model_sha = copy_model(row["model"], model_path)
        bundle = joblib.load(model_path)
        ok, reason = validate_bundle(bundle, expected_version=METHOD_SUFFIX[method],
                                     expected_kind=KIND[method])
        if not ok:
            raise RuntimeError(f"{method} representative is invalid: {reason}")
        if bundle["seed"] != 42 or bundle["epoch"] != row["epoch"]:
            raise RuntimeError(f"{method} representative seed/epoch mismatch")
        if float(bundle["config"]["threshold"]) != row["threshold"]:
            raise RuntimeError(f"{method} representative threshold mismatch")
        representatives[method] = {
            "method": method, "group": row["group"], "seed": 42, "epoch": row["epoch"],
            "threshold": row["threshold"],
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": model_sha, "version": METHOD_SUFFIX[method],
            "kind": KIND[method], "score_type": SCORE[method],
        }
        candidates.append({
            "method": method, "group": row["group"], "seed": 42, "epoch": row["epoch"],
            "threshold": row["threshold"], "mean_fn": row["mean_fn"],
            "mean_fp": row["mean_fp"], "parameters": row["params"],
            "training_views": row["views"],
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": model_sha, "version": METHOD_SUFFIX[method],
            "kind": KIND[method], "score_type": SCORE[method],
        })
    selected = min(representatives, key=lambda method: selection_key(method, rows[method]))
    default_path = WEIGHTS / "representatives/default.joblib"
    default_sha = copy_model(WEIGHTS / "representatives" / f"{selected}.joblib", default_path)
    freeze = {
        "schema_version": 1,
        "created_at": now(),
        "name": DELIVERY_NAME,
        "source": {"repository": protocol["source"]["repository"],
                   "commit": protocol["source"]["commit"],
                   "dataset_tree": protocol["source"]["dataset_tree"]},
        "protocol_sha256": sha256(PROTOCOL),
        "source_lock_sha256": sha256(SOURCE_LOCK),
        "audit_sha256": sha256(AUDIT / "dataset_audit_development.json"),
        "candidates": candidates,
        "selected_default": selected,
        "selection_rule": "mean FN, mean FP, fewer trainable parameters, fewer training "
                          "views, then supervised main > supervised refinement > AE",
        "representatives": representatives,
        "default": {"method": selected,
                    "model_path": default_path.relative_to(ROOT).as_posix(),
                    "model_sha256": default_sha,
                    "version": METHOD_SUFFIX[selected], "kind": KIND[selected],
                    "score_type": SCORE[selected],
                    "threshold": representatives[selected]["threshold"],
                    "epoch": representatives[selected]["epoch"]},
        "code_sha256": code_hashes(MODEL_CODE),
        "dependencies": dependencies(),
        "test_accessed": False,
        "scope": "development validation selection; test untouched until prepare --scope full",
    }
    write(OUT / "freeze.json", freeze)
    freeze_sha = sha256(OUT / "freeze.json")
    active = {
        "schema_version": 1,
        "name": DELIVERY_NAME,
        "selected_method": selected,
        "version": METHOD_SUFFIX[selected],
        "model_path": default_path.relative_to(ROOT).as_posix(),
        "model_sha256": default_sha,
        "threshold": representatives[selected]["threshold"],
        "epoch": representatives[selected]["epoch"],
        "score_type": SCORE[selected],
        "expected_counts": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2},
        "representatives": {method: value["model_path"]
                            for method, value in representatives.items()},
        "frozen_at": freeze["created_at"],
        "freeze_sha256": freeze_sha,
    }
    write(ACTIVE, active)
    write(EXPERIMENTS / "summary.json", {
        "created_at": now(),
        "ae": read(EXPERIMENTS / "summary_ae.json"),
        "supervised": read(EXPERIMENTS / "summary_supervised.json"),
        "refinement": read(REFINEMENT / "summary.json"),
        "selected_default": selected,
        "freeze_sha256": freeze_sha,
    })
    state = read(OUT / "state.json") if (OUT / "state.json").exists() else {}
    state.update({"test_accessed": False, "upload_permission": False,
                  "stage": "freeze", "updated_at": now()})
    write(OUT / "state.json", state)
    record_stage("freeze", "completed",
                 f"selected default {selected}; 3 representatives frozen")
    update_state(next_command="python -m members.member04_hog.final_v4.diagnose --splits train val")
    print(f"Frozen; default method {selected}", flush=True)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except Exception as error:
        record_stage("freeze", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
