"""Verify checkpoints, representatives, interfaces, tests, dependencies and scope."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

from ..data import load_manifest
from .common import (
    AUDIT, DATA, EVALUATION, EXPERIMENTS, METHOD_SUFFIX, MODEL_CODE, OUT, PROTOCOL,
    REFINEMENT, ROOT, VERIFICATION, WEIGHTS, code_hashes, dependencies, ensure,
    prediction_rows, read, read_csv, record_stage, sha256, update_state, validate_bundle,
    verify_protocol, write,
)


def checkpoint_records():
    records = []
    for method in ("ae", "supervised"):
        for group_dir in sorted((WEIGHTS / method).iterdir()):
            if not group_dir.is_dir():
                continue
            for seed_dir in sorted(group_dir.iterdir()):
                artifact = seed_dir / "last.joblib"
                metrics = read(EXPERIMENTS / method / group_dir.name / seed_dir.name /
                               "metrics.json")
                records.append((method, group_dir.name, seed_dir.name, "last", artifact,
                                metrics["model_sha256"]))
    for config_dir in sorted((WEIGHTS / "refinement").iterdir()):
        if not config_dir.is_dir():
            continue
        for seed_dir in sorted(config_dir.iterdir()):
            metrics = read(REFINEMENT / config_dir.name / seed_dir.name / "metrics.json")
            for name in ("best", "last"):
                records.append(("refinement", config_dir.name, seed_dir.name, name,
                                seed_dir / f"{name}.joblib",
                                metrics[name]["model_sha256"]))
    return records


def verify_checkpoints():
    records = checkpoint_records()
    problems = []
    for method, group, seed, name, artifact, expected_sha in records:
        if not artifact.is_file():
            problems.append(f"missing {artifact}")
            continue
        if sha256(artifact) != expected_sha:
            problems.append(f"checksum {artifact}")
            continue
        bundle = joblib.load(artifact)
        ok, reason = validate_bundle(bundle, expected_version=METHOD_SUFFIX[method])
        if not ok:
            problems.append(f"{artifact}: {reason}")
        elif bundle["group"] != group or bundle["seed"] != int(seed[-2:]):
            problems.append(f"{artifact}: identity mismatch")
    return {"count": len(records), "problems": problems,
            "expected": 84, "pass": not problems and len(records) == 84}


def representative_validation():
    freeze = read(OUT / "freeze.json")
    valid = load_manifest(DATA / "manifests/hog/val.csv")
    from ..model import predict_with_model

    saved_paths = {
        "ae": EXPERIMENTS / "ae" / freeze["representatives"]["ae"]["group"] /
              "seed42/valid_predictions.csv",
        "supervised": EXPERIMENTS / "supervised" /
                      freeze["representatives"]["supervised"]["group"] /
                      "seed42/valid_predictions.csv",
        "refinement": REFINEMENT / freeze["representatives"]["refinement"]["group"] /
                      "seed42/best_valid_predictions.csv",
    }
    results, problems, maximum = [], [], 0.0
    for method, path in saved_paths.items():
        if not path.is_file():
            problems.append(f"missing saved predictions: {path}")
            continue
        saved = {row["image_id"]: row for row in read_csv(path)}
        bundle = joblib.load(ROOT / freeze["representatives"][method]["model_path"])
        label_mismatches = 0
        for sample in valid:
            result = predict_with_model(sample.path, bundle)
            previous = saved[sample.image_id]
            if result["predicted_label"] != previous["predicted_label"]:
                label_mismatches += 1
            maximum = max(maximum, abs(result["prediction_score"] -
                                       float(previous["prediction_score"])))
        results.append({"method": method, "images": len(valid),
                        "label_mismatches": label_mismatches})
        if label_mismatches:
            problems.append(f"{method}: {label_mismatches} label mismatches")
    return {"results": results, "max_score_difference": maximum,
            "problems": problems, "pass": not problems}


def interface_checks():
    from members.member04_hog import model as module
    from members.member04_hog import inference

    problems = []
    temporary = VERIFICATION / "interface_tmp"
    if temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        bad = temporary / "broken.png"
        bad.write_text("not an image", encoding="utf-8")
        assert module.predict_with_model(bad, joblib.load(
            ROOT / read(OUT / "freeze.json")["representatives"]["ae"]["model_path"]
        ))["status"] == "INPUT_ERROR"
        assert module.predict(temporary / "missing.png")["status"] == "INPUT_ERROR"
        original_config = module.ACTIVE_CONFIG
        try:
            record = read(original_config)
            missing = dict(record, model_path="weights/member04_hog/final_v4/"
                                               "representatives/missing.joblib")
            config_path = temporary / "missing.json"
            write(config_path, missing)
            module.ACTIVE_CONFIG = config_path
            if module.predict(temporary / "missing.png")["status"] != "MODEL_UNAVAILABLE":
                problems.append("missing weights did not return MODEL_UNAVAILABLE")
            bundle = joblib.load(ROOT / record["model_path"])
            legacy = dict(bundle, format_version=1,
                          version="Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection")
            legacy_path = temporary / "legacy.joblib"
            joblib.dump(legacy, legacy_path)
            legacy_record = dict(record, model_path=str(legacy_path),
                                 model_sha256=sha256(legacy_path))
            config_path = temporary / "legacy.json"
            write(config_path, legacy_record)
            module.ACTIVE_CONFIG = config_path
            if module.predict(temporary / "missing.png")["status"] != "MODEL_UNAVAILABLE":
                problems.append("legacy bundle was not rejected")
        finally:
            module.ACTIVE_CONFIG = original_config
        counts_path = temporary / "requests.json"
        candidates = sorted(path.name for path in (DATA / "images/val/PASS").iterdir()
                            if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not candidates:
            raise RuntimeError("no validation PASS image available for the batch check")
        write(counts_path, [
            {"image_id": candidates[0],
             "expected_counts": {str(i): 2 for i in range(1, 6)}},
            {"image_id": "missing.png", "expected_counts": {str(i): 2 for i in range(1, 6)}},
            {"image_id": "unsupported.png", "expected_counts": {str(i): 1 for i in range(1, 6)}},
        ])
        input_dir = DATA / "images/val/PASS"
        rows = inference.batch(input_dir, counts_path, temporary / "out")
        statuses = [row["status"] for row in rows]
        if statuses != ["OK", "INPUT_ERROR", "UNSUPPORTED_COUNTS"]:
            problems.append(f"unexpected batch statuses: {statuses}")
        if any(row["predicted_label"] for row in rows[1:]):
            problems.append("failed batch rows must have empty decisions")
        try:
            inference.batch(input_dir, counts_path, temporary / "no_model",
                            temporary / "missing.joblib")
            problems.append("missing weight did not raise")
        except FileNotFoundError:
            if (temporary / "no_model").exists():
                problems.append("missing weight created an output directory")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {"problems": problems, "pass": not problems}


def run_command(command, filename):
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=3600)
    text = ("COMMAND: " + " ".join(str(part) for part in command) + "\n"
            f"EXIT: {completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n")
    text = text.replace(str(ROOT), ".")
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    (VERIFICATION / filename).write_text(text, encoding="utf-8")
    return {"exit": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "file": filename}


def dependency_check():
    from importlib.metadata import version

    expected = {}
    for line in (ROOT / "members/member04_hog/requirements.txt").read_text(
            encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        expected[name.strip().lower()] = pinned.strip()
    installed = {}
    for name in expected:
        try:
            installed[name] = version(name)
        except Exception:
            installed[name] = None
    mismatches = {name: (pinned, installed.get(name))
                  for name, pinned in expected.items()
                  if installed.get(name) != pinned}
    return {"expected": expected, "installed": installed, "mismatches": mismatches,
            "pass": not mismatches}


def scope_check():
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    changed = []
    for line in status:
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append(path)
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "origin/main"],
                          capture_output=True, text=True).stdout.splitlines()
    paths = sorted(set(changed) | set(diff))
    outside = [path for path in paths if not path.startswith("members/member04_hog/")]
    return {"changed_paths": paths, "outside_member04": outside,
            "member_scope_ok": not outside}


def run():
    ensure(VERIFICATION)
    verify_protocol()
    freeze = read(OUT / "freeze.json")
    if freeze.get("test_accessed"):
        raise RuntimeError("freeze records test access before final verification")
    checkpoints = verify_checkpoints()
    representatives = representative_validation()
    interfaces = interface_checks()
    dependencies_result = dependency_check()
    module_tests = run_command(
        [sys.executable, "-m", "unittest", "discover", "-s",
         "members/member04_hog/tests", "-v"], "module_tests.txt")
    public_tests = run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        "public_tests.txt")
    list_models = run_command([sys.executable, "predict.py", "--list-models"],
                              "predict_list_models.txt")
    scope_run = run_command([sys.executable, ".github/check_scope.py"], "scope_check.txt")
    scope = scope_check()
    verification = {
        "schema_version": 1,
        "created_at": read(OUT / "state.json").get("updated_at"),
        "freeze_sha256": sha256(OUT / "freeze.json"),
        "checkpoints": checkpoints,
        "representatives": representatives,
        "interfaces": interfaces,
        "dependencies": dependencies_result,
        "tests": {"module": module_tests["exit"], "public": public_tests["exit"],
                  "module_file": module_tests["file"], "public_file": public_tests["file"]},
        "framework_check": {"exit": scope_run["exit"],
                            "file": scope_run["file"],
                            "note": "main already tracks member03 media; expected blocked "
                                    "by public media policy, recorded honestly"},
        "scope": scope,
        "test_accessed": False,
    }
    verification["pass"] = all((
        checkpoints["pass"], representatives["pass"], interfaces["pass"],
        dependencies_result["pass"], module_tests["exit"] == 0,
        public_tests["exit"] == 0, list_models["exit"] == 0,
        scope["member_scope_ok"]))
    write(VERIFICATION / "verification.json", verification)
    write(VERIFICATION / "scope_verification.json", scope)
    write(VERIFICATION / "dependencies.json", dependencies_result)
    if not verification["pass"]:
        raise RuntimeError("verification failed; see outputs/.../verification")
    record_stage("verify", "completed", "84 checkpoints and interfaces verified")
    update_state(next_command="python -m members.member04_hog.final_v4.package "
                              "--out ../delivery/Baseline1_HOGMLP_v4_FinalDataset_Rebuild.zip")
    print(f"Verification PASS: {checkpoints['count']} checkpoints", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except Exception as error:
        record_stage("verify", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
