"""Export approved Val truth separately from GT-free shared inference requests."""
import argparse
import csv
from pathlib import Path

from .config import Config
from .preflight import check_training_inputs
from .team_data import read_json, write_new_json


def export(config_path, input_dir, output):
    config = Config.load(config_path)
    if config.training_tier == "DEV_ONLY":
        raise ValueError("DEV_ONLY cannot export VERIFIED truth; use development Val groups without claiming formal review")
    if config.data_kind != "REAL":
        raise ValueError("handoff requires reviewed REAL development provenance")
    _, val, _ = check_training_inputs(config)
    record = read_json(config.data_provenance)
    source = record["audit"]["images"] if record["schema_version"].endswith("v2") else record["lineage"]
    root = Path(record.get("dataset_root", "/"))
    by_path = {str((root / r["image_path"]).resolve()): r for r in source}
    input_dir = Path(input_dir).resolve()
    names = [r["path"].name for r in val.rows]
    if len(set(names)) != len(names):
        raise ValueError("shared protocol requires globally unique image filenames")
    if any(not r["path"].is_relative_to(input_dir) for r in val.rows):
        raise ValueError("input_dir does not contain all approved Val images")
    actual = {}
    for path in input_dir.rglob("*"):
        if path.is_file() and path.name in names:
            actual.setdefault(path.name, []).append(path.resolve())
    if any(actual.get(r["path"].name) != [r["path"]] for r in val.rows):
        raise ValueError("ambiguous input filename; choose an unambiguous input directory")
    truth, inputs = [], []
    for row in val.rows:
        original = by_path[str(row["path"])]
        truth.append({"image_id": row["path"].name, "annotation_status": "VERIFIED",
                      **{f"count_{i}": original["actual_counts"][str(i)] for i in range(1, 6)},
                      "non_catalogue_count": original["actual_counts"]["0"]})
        inputs.append({"image_id": row["path"].name,
                       "expected_counts": {str(i+1): n for i, n in enumerate(config.supported_counts)}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "counts.json", inputs)
    with (output / "val_truth.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(truth[0])); writer.writeheader(); writer.writerows(truth)
    write_new_json(output / "inputs_provenance.json", {"split": "val", "input_dir": str(input_dir),
        "images": [{"image_id": r["path"].name, "path": str(r["path"]),
                    "image_sha256": by_path[str(r["path"])]["image_sha256"]} for r in val.rows],
        "note": "Only counts.json goes to inference. val_truth.csv is evaluator input, never model input."})
    return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    try:
        print(f"Exported {len(export(a.config, a.input_dir, a.output))} approved Val cases")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f"HANDOFF_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
