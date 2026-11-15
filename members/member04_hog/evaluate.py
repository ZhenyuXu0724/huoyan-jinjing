"""Image-level metrics, freezing and independent evaluation."""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .data import identity, load_manifest, reject_overlap, sha256
from .model import load_model, predict_with_model


def metrics(rows):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    support = dict(PASS=0, FAIL=0)
    errors, unjudged = [], []
    for row in rows:
        truth = row["label"]
        support[truth] += 1
        label = row["predicted_label"]
        if row["status"] != "OK" or label not in support:
            unjudged.append(row["image_id"])
            continue
        key = {("FAIL", "FAIL"): "tp", ("PASS", "FAIL"): "fp",
               ("FAIL", "PASS"): "fn", ("PASS", "PASS"): "tn"}[truth, label]
        counts[key] += 1
        if key in ("fp", "fn"):
            errors.append(row["image_id"])
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    decided = sum(counts.values())
    return {**counts, "false_pass": fn, "false_fail": fp, "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "total": len(rows), "decided": decided,
            "coverage": decided / len(rows) if rows else 0.0,
            "errors": errors, "unjudged": unjudged}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_predictions(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_code():
    module = Path(__file__).parent
    return {name: sha256(module / name)
            for name in ("data.py", "evaluate.py", "features.py", "model.py", "one_class/scoring.py")}


def freeze(output, models):
    if output.exists():
        raise ValueError("Freeze record already exists")
    entries = []
    reference = None
    seen = set()
    for path in models:
        bundle = load_model(path)
        if bundle["role"] != "local":
            raise ValueError("Freeze requires locally trained models")
        comparison = (bundle["config"]["expected_counts"], bundle["local_data"])
        if reference is not None and comparison != reference:
            raise ValueError("Models must use the same fixed counts and local split")
        reference = comparison
        name = f'{bundle["group"]}/seed{bundle["seed"]}'
        if name in seen:
            raise ValueError(f"Repeated run: {name}")
        seen.add(name)
        entries.append({"name": name, "path": os.path.relpath(path, output.parent),
                        "sha256": sha256(path), "config": bundle["config"],
                        "development": bundle["development"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"evaluation_rule": "fail-positive-v1", "code_sha256": evaluation_code(),
                   "assessment_scope": "provided public split; physical scene independence unverified",
                   "frozen_at": datetime.now(timezone.utc).isoformat(),
                   "models": entries},
                  stream, indent=2, ensure_ascii=False)


def validate_freeze(frozen):
    record = json.loads(frozen.read_text(encoding="utf-8"))
    if (record.get("evaluation_rule") != "fail-positive-v1"
            or record.get("code_sha256") != evaluation_code() or not record.get("models")):
        raise ValueError("Invalid freeze record")
    models = []
    for entry in record["models"]:
        path = frozen.parent / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen model changed: {path}")
        models.append((entry, load_model(path)))
    return record, models


def test(frozen, manifest, output):
    record, models = validate_freeze(frozen)
    output.mkdir(parents=True, exist_ok=False)
    access = {"accessed_at": datetime.now(timezone.utc).isoformat(),
              "freeze_sha256": sha256(frozen)}
    write_json(output / "test_access.json", access)
    access["test_manifest_sha256"] = sha256(manifest)
    write_json(output / "test_access.json", access)
    samples = load_manifest(manifest)
    test_data = identity(samples)
    for entry, bundle in models:
        reject_overlap(entry["development"], test_data)
        reject_overlap(bundle["development"], test_data)
    for entry, bundle in models:
        rows = [{"image_id": s.image_id, "scene_id": s.scene, "label": s.label,
                 **predict_with_model(s.path, bundle)} for s in samples]
        run = output / entry["name"]
        run.mkdir(parents=True)
        write_predictions(run / "predictions.csv", rows)
        write_json(run / "metrics.json", metrics(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--out", type=Path, required=True)
    frozen.add_argument("--models", type=Path, nargs="+", required=True)
    held_out = commands.add_parser("test")
    held_out.add_argument("--freeze", type=Path, required=True)
    held_out.add_argument("--test", type=Path, required=True)
    held_out.add_argument("--out", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out, args.models)
    elif args.command == "test":
        test(args.freeze, args.test, args.out)
    else:
        print(json.dumps(predict_with_model(args.image, load_model(args.model))))


if __name__ == "__main__":
    main()
