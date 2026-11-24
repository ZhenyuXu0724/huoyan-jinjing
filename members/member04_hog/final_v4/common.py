"""Shared v4 paths, hashing, Git restore helpers and fixed selection rules."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "members/member04_hog"
DATA = ROOT / "data/member04_hog/final_training_v4"
SUPPLIER = ROOT / "members/member03_data/final_training_v4"
OUT = ROOT / "outputs/member04_hog/final_v4"
WEIGHTS = ROOT / "weights/member04_hog/final_v4"
CONFIG = MODULE / "configs/final_v4"
SOURCE_LOCK = CONFIG / "source_lock.json"
PROTOCOL = CONFIG / "protocol.json"
ACTIVE = MODULE / "configs/active_model.json"
AUDIT = OUT / "audit"
EXPERIMENTS = OUT / "experiments"
REFINEMENT = OUT / "refinement"
DIAGNOSIS = OUT / "diagnosis"
EVALUATION = OUT / "evaluation"
REPORTS = OUT / "reports"
VERIFICATION = OUT / "verification"
STATE = OUT / "state.json"

REPOSITORY = "ZhenyuXu0724/huoyan-jinjing"
SOURCE_COMMIT = "bde8ea7116761bc8ee7fca1d47c336126daa995f"
DATASET_PATH = "members/member03_data/final_training_v4"
DATASET_TREE = "c6882fd823b1f458105b47fea2f52af0b357ef9c"
INTRO_COMMIT = "f5dad94b9a79243ce2f3f83041d62de118b8ed08"
DATASET_BYTES = 1789463157

PREFIX = "Baseline1_HOGMLP_v4_FinalDataset"
DELIVERY_NAME = f"{PREFIX}_Rebuild"
METHOD_SUFFIX = {
    "ae": f"{PREFIX}_PASSOnly_Autoencoder",
    "supervised": f"{PREFIX}_Supervised_MLP",
    "refinement": f"{PREFIX}_Supervised_Capacity_L2_EpochSelection",
}
SEEDS = (42, 43, 44)
EPOCHS = 200
FP_BUDGET = 4
SIZE = 256
CELL = 16
EXPECTED_COUNTS = {str(i): 2 for i in range(1, 6)}
CLASSES = {"PASS": 0, "FAIL": 1}

# Metadata restored in both scopes (13 files). Everything else is media or test-only.
METADATA_FILES = (
    "README.md",
    "classes.json",
    "data.yaml",
    "verify_dataset.py",
    "split_manifest.csv",
    "原图明细.csv",
    "manifests/hog/train.csv",
    "manifests/hog/train_pass.csv",
    "manifests/hog/val.csv",
    "manifests/hog/test.csv",
    "manifests/resnet/train.csv",
    "manifests/resnet/val.csv",
    "manifests/resnet/test.csv",
)
FULL_ONLY_FILES = ("测试集逐图核对.csv", "train_resnet_v4.py")

DATA_CODE = (
    MODULE / "data.py",
    MODULE / "final_v4/common.py",
    MODULE / "final_v4/prepare.py",
    MODULE / "final_v4/audit.py",
)
TRAINING_CODE = (
    MODULE / "features.py",
    MODULE / "data.py",
    MODULE / "final_v4/common.py",
    MODULE / "final_v4/groups.py",
    MODULE / "final_v4/train.py",
    MODULE / "final_v4/refine.py",
)
INFERENCE_CODE = (
    MODULE / "model.py",
    MODULE / "inference.py",
    MODULE / "evaluate.py",
    MODULE / "final_v4/freeze.py",
)
MODEL_CODE = TRAINING_CODE + INFERENCE_CODE
PROTOCOL_CODE = DATA_CODE + TRAINING_CODE


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    ensure(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_bytes(path, payload):
    path = Path(path)
    ensure(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_predictions(path, rows):
    if not rows:
        raise ValueError("prediction rows must not be empty")
    write_csv(path, rows)


def write_csv(path, rows):
    if not rows:
        raise ValueError("table rows must not be empty")
    path = Path(path)
    ensure(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def find_git_ca():
    executable = shutil.which("git")
    if not executable:
        return None
    root = Path(executable).resolve().parents[1]
    for candidate in (
        root / "ucrt64/etc/ssl/certs/ca-bundle.crt",
        root / "usr/ssl/certs/ca-bundle.crt",
        root / "mingw64/ssl/certs/ca-bundle.crt",
        root / "etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    ):
        if candidate.is_file():
            return candidate
    return None


def git(*args, text=False):
    command = ["git", "-C", str(ROOT), "-c", "credential.helper="]
    ca = find_git_ca()
    if ca is not None:
        command += ["-c", "http.sslBackend=openssl", "-c", f"http.sslCAInfo={ca}"]
    result = subprocess.run(command + list(args), capture_output=True, timeout=7200)
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace')[:2000]}")
    return result.stdout.decode("utf-8", "replace") if text else result.stdout


def git_text(*args):
    return git(*args, text=True).strip()


def git_tree(commit, pathspec):
    """Return [{path,type,sha,size}] relative to `pathspec` for a recursive listing."""
    raw = git("ls-tree", "-r", "-l", "-z", commit, "--", pathspec)
    prefix = pathspec.rstrip("/") + "/"
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, kind, sha, size = meta.decode("ascii").split()
        name = path.decode("utf-8")
        if not name.startswith(prefix):
            raise RuntimeError(f"tree entry outside {pathspec}: {name}")
        entries.append({
            "path": name[len(prefix):],
            "type": kind,
            "mode": mode,
            "sha": sha,
            "size": None if size == "-" else int(size),
        })
    return entries


def git_eol_map(pathspec):
    """Map worktree-relative repo path -> eol flags from `git ls-files --eol`."""
    raw = git("ls-files", "--eol", "-z", "--", pathspec)
    mapping = {}
    for record in raw.split(b"\0"):
        if not record or b"\t" not in record:
            continue
        flags, path = record.split(b"\t", 1)
        mapping[path.decode("utf-8")] = flags.decode("utf-8").split()
    return mapping


def is_lf_to_crlf(flags):
    return "i/lf" in flags and "w/crlf" in flags


def eol_variant(payload, flags):
    """Bytes as stored in the Git blob when the worktree applied CRLF conversion."""
    if is_lf_to_crlf(flags):
        return payload.replace(b"\r\n", b"\n")
    return payload


def sample_weights(labels):
    labels = np.asarray(labels)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("training labels must contain PASS=0 and FAIL=1")
    return np.asarray([len(labels) / (2 * int(np.sum(labels == value))) for value in labels])


def threshold_candidates(labels, scores, budget=FP_BUDGET):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if set(labels.tolist()) != {0, 1} or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("threshold selection needs both classes and matching finite scores")
    unique = np.unique(scores)
    cuts = np.r_[unique[0], unique[:-1] + np.diff(unique) / 2,
                 np.nextafter(unique[-1], np.inf)]
    rows = []
    for threshold in cuts:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))
        rows.append({"threshold": float(threshold), "fn": fn, "fp": fp,
                     "tp": tp, "tn": tn, "feasible": fp <= budget})
    return rows


def choose_threshold(labels, scores, budget=FP_BUDGET):
    """FP<=budget; fewest FN, then fewest FP, then the larger representative threshold."""
    rows = [row for row in threshold_candidates(labels, scores, budget) if row["feasible"]]
    if not rows:
        raise ValueError("no feasible threshold")
    best = min(rows, key=lambda row: (row["fn"], row["fp"], -row["threshold"]))
    return best


def checkpoint_key(row):
    return (row["fn"], row["fp"], row["epoch"])


def score_metrics(rows):
    from ..evaluate import metrics

    result = metrics(rows)
    result["accuracy"] = ((result["tp"] + result["tn"]) / result["decided"]
                          if result["decided"] else None)
    result["specificity"] = (result["tn"] / (result["tn"] + result["fp"])
                             if result["tn"] + result["fp"] else None)
    result["balanced_accuracy"] = (
        (result["recall"] + result["specificity"]) / 2
        if all(result["support"].values()) and result["coverage"] == 1 else None)
    return result


def prediction_rows(samples, scores, threshold, score_type):
    return [{"image_id": sample.image_id, "label": sample.label, "status": "OK",
             "predicted_label": "FAIL" if float(score) >= threshold else "PASS",
             "prediction_score": float(score), "score_type": score_type}
            for sample, score in zip(samples, scores, strict=True)]


def reconstruction_scores(model, vectors):
    errors = vectors.astype(np.float64) - model.predict(vectors)
    return np.mean(errors ** 2, axis=1)


def code_hashes(paths):
    return {Path(path).relative_to(ROOT).as_posix(): sha256(path) for path in sorted(paths)}


def dependencies():
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "Pillow", "scikit-image", "scikit-learn", "joblib",
                 "scipy", "threadpoolctl"):
        result[name] = version(name)
    return result


def update_state(**fields):
    state = read(STATE) if STATE.exists() else {
        "schema_version": 1, "stage": "initialised", "test_accessed": False,
        "upload_permission": False, "stages": []}
    state.update(fields)
    state["updated_at"] = now()
    write(STATE, state)
    return state


def record_stage(name, status, detail, **fields):
    state = read(STATE) if STATE.exists() else {
        "schema_version": 1, "test_accessed": False, "upload_permission": False,
        "stages": []}
    stages = [stage for stage in state.get("stages", []) if stage.get("name") != name]
    stages.append({"name": name, "status": status, "detail": detail,
                   "recorded_at": now(), **fields})
    state["stages"] = stages
    state["stage"] = name
    state["updated_at"] = now()
    write(STATE, state)


def validate_bundle(bundle, expected_version=None, expected_kind=None):
    """Return (ok, reason). Rejects legacy bundles and inconsistent v4 metadata."""
    if not isinstance(bundle, dict):
        return False, "bundle is not a mapping"
    if bundle.get("format_version") != 2:
        return False, "unsupported bundle format_version"
    version = bundle.get("version")
    if not isinstance(version, str) or not version.startswith(PREFIX + "_"):
        return False, "bundle version is not a v4 final-dataset version"
    if expected_version is not None and version != expected_version:
        return False, "bundle version mismatch"
    kind = bundle.get("kind")
    if kind not in ("hog_autoencoder", "hog_supervised_classifier"):
        return False, "unsupported bundle kind"
    if expected_kind is not None and kind != expected_kind:
        return False, "bundle kind mismatch"
    expected_score = ("hog_reconstruction_mse" if kind == "hog_autoencoder"
                      else "uncalibrated_fail_probability")
    if bundle.get("score_type") != expected_score:
        return False, "score_type does not match bundle kind"
    config = bundle.get("config")
    if not isinstance(config, dict):
        return False, "missing config"
    for key in ("size", "cell", "threshold", "expected_counts"):
        if key not in config:
            return False, f"missing config.{key}"
    if config["size"] != SIZE or config["cell"] != CELL:
        return False, "unexpected preprocessing configuration"
    if config["expected_counts"] != EXPECTED_COUNTS:
        return False, "unexpected expected_counts"
    if not isinstance(config["threshold"], (int, float)):
        return False, "threshold must be numeric"
    if not callable(getattr(bundle.get("model"), "predict", None)):
        return False, "missing model"
    if not callable(getattr(bundle.get("scaler"), "transform", None)):
        return False, "missing scaler"
    return True, None


def safe_relative(path, root):
    path = Path(path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def verify_protocol(path=None):
    """Load and verify the locked protocol against data, source lock and model code."""
    path = Path(path) if path is not None else PROTOCOL
    protocol = read(path)
    if protocol.get("schema_version") != 1 or protocol.get("name") != DELIVERY_NAME:
        raise ValueError("unexpected protocol identity")
    lock = read(SOURCE_LOCK)
    if sha256(SOURCE_LOCK) != protocol["source"]["source_lock_sha256"]:
        raise ValueError("source lock changed after protocol creation")
    if lock["source_commit"] != protocol["source"]["commit"]:
        raise ValueError("source commit mismatch")
    audit_path = ROOT / protocol["audit"]["path"]
    if not audit_path.is_file() or sha256(audit_path) != protocol["audit"]["sha256"]:
        raise ValueError("development audit changed after protocol creation")
    for relative, expected in protocol["data"]["manifests_sha256"].items():
        if sha256(DATA / relative) != expected:
            raise ValueError(f"manifest changed after protocol creation: {relative}")
    if code_hashes(PROTOCOL_CODE) != protocol["code_sha256"]:
        raise ValueError("model/data code changed after protocol creation")
    if len(protocol.get("tasks", [])) != 66 or len(protocol.get("refinement", {}).get("tasks", [])) != 9:
        raise ValueError("protocol does not describe 66 + 9 runs")
    return protocol
