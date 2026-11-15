"""Evaluate exported image decisions under one shared fixed-count requirement."""
import argparse
import csv
import json
from pathlib import Path

from shared.contracts import decision_from_counts, validate_expected_counts

from .data import sha256
from .evaluate import metrics, write_json, write_predictions


def read_unique(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {required}")
        rows = {}
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in rows:
                raise ValueError(f"{path}: missing or duplicate image_id")
            rows[image_id] = row
    if not rows:
        raise ValueError(f"{path}: empty table")
    return rows


def count(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Ground-truth counts must be non-negative integers")
    return int(value)


def compare(truth_path, counts_path, predictions, output):
    """Consume only explicitly supplied evaluation exports, never infer images."""
    expected = json.loads(Path(counts_path).read_text(encoding="utf-8-sig"))
    cases = {}
    reference = None
    for case in expected:
        image_id = case["image_id"]
        requirement = validate_expected_counts(case["expected_counts"])
        if not image_id or image_id in cases:
            raise ValueError("Expected counts need unique image_id values")
        if reference is not None and requirement != reference:
            raise ValueError("E1 comparison requires one common fixed-count vector")
        reference = requirement
        cases[image_id] = requirement
    if not cases:
        raise ValueError("Empty expected counts")

    columns = [f"count_{i}" for i in range(1, 6)]
    truth = read_unique(truth_path, ["image_id", "annotation_status", *columns, "non_catalogue_count"])
    if set(truth) != set(cases):
        raise ValueError("Truth and expected counts must have identical image IDs")
    labels, excluded = {}, []
    for image_id, row in truth.items():
        if row["annotation_status"] == "UNJUDGEABLE":
            excluded.append(image_id)
            continue
        if row["annotation_status"] != "VERIFIED":
            raise ValueError(f"{image_id}: annotation_status must be VERIFIED or UNJUDGEABLE")
        actual = {str(i): count(row[f"count_{i}"]) for i in range(1, 6)}
        labels[image_id] = decision_from_counts(
            actual, cases[image_id], count(row["non_catalogue_count"]))["predicted_label"]
    if not labels:
        raise ValueError("No judgeable ground-truth images")

    reports = {}
    for path in predictions:
        exported = read_unique(path, ["image_id", "model_id", "status", "predicted_label"])
        if set(exported) != set(cases):
            raise ValueError(f"{path}: exported image IDs do not match the common input")
        model_ids = {row["model_id"] for row in exported.values()}
        if len(model_ids) != 1:
            raise ValueError(f"{path}: one model_id required per file")
        model_id = next(iter(model_ids))
        if model_id not in {"baseline1", "baseline2", "baseline3", "final"} or model_id in reports:
            raise ValueError(f"{path}: invalid or repeated model_id")
        rows = []
        for image_id, label in labels.items():
            row = exported[image_id]
            if row["status"] == "OK" and row["predicted_label"] not in {"PASS", "FAIL"}:
                raise ValueError(f"{path}: OK result requires PASS or FAIL")
            rows.append({"image_id": image_id, "model_id": model_id, "label": label,
                         "status": row["status"], "predicted_label": row["predicted_label"]})
        reports[model_id] = {"rows": rows, "metrics": metrics(rows), "prediction_sha256": sha256(path)}

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for model_id, report in reports.items():
        directory = output / model_id
        directory.mkdir()
        write_predictions(directory / "decisions.csv", report["rows"])
        write_json(directory / "metrics.json", report["metrics"])
        summaries[model_id] = {key: value for key, value in report.items() if key != "rows"}
    write_json(output / "comparison.json", {
        "truth_sha256": sha256(truth_path), "counts_sha256": sha256(counts_path),
        "expected_counts": reference, "excluded_unjudgeable": excluded,
        "models": summaries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.truth, args.counts, args.predictions, args.out)


if __name__ == "__main__":
    main()
