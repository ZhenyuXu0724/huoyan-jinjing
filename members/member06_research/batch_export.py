"""Shared-contract export, compatible with member04 compare.py; no GT input."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from .config import normalize_counts
from .inference import create_adapter
from .runtime import file_sha256, write_json

COLUMNS = ["image_id", "model_id", "status", "predicted_label", "prediction_score", "score_type", "non_catalogue_count"]


def requests(path):
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not values:
        raise ValueError("counts file must be a nonempty list")
    result = {}
    for row in values:
        if not isinstance(row, dict) or set(row) != {"image_id", "expected_counts"}:
            raise ValueError("request must contain only image_id and expected_counts")
        name = row["image_id"]
        if not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in '/\\:') or name in result:
            raise ValueError("image_id must be a unique filename")
        result[name] = normalize_counts(row["expected_counts"])
    return result


def batch(checkpoint, input_dir, counts_file, output, device="auto", batch_size=16, allow_smoke=False, allow_dev_only=False):
    cases = requests(counts_file)
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists")
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError("input directory missing")
    paths = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name in cases:
            if path.name in paths or not path.resolve().is_relative_to(root):
                raise ValueError("duplicate filename or input symlink escape")
            paths[path.name] = path
    adapter = create_adapter(checkpoint, device, allow_smoke, batch_size, allow_dev_only)
    rows, times, hashes, results = [], {}, {}, {}
    names = list(cases)
    for start in range(0, len(names), batch_size):
        groups = {}
        for name in names[start:start + batch_size]:
            groups.setdefault(tuple(cases[name]), []).append(name)
        for requirement, group in groups.items():
            began = perf_counter()
            values = adapter.predict_batch([paths.get(name, root / name) for name in group], list(requirement))
            elapsed_per_image = (perf_counter() - began) * 1000 / len(group)
            for name, result in zip(group, values):
                results[name] = result
                times[name] = elapsed_per_image
                if name in paths:
                    hashes[name] = file_sha256(paths[name])
    for name in names:
        rows.append({"image_id": name, "model_id": "baseline2",
                     **{k: results[name].get(k) for k in ("status", "predicted_label", "prediction_score", "score_type")},
                     "non_catalogue_count": None})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "images.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "run_config.json", {
        "model_id": "baseline2", "model_sha256": file_sha256(checkpoint),
        "config": adapter.backend.config.to_dict(), "evidence_status": adapter.backend.evidence_status,
        "supported_expected_counts": adapter.supported_counts, "counts_rule_status": adapter.counts_rule_status,
        "input_expected_counts": {name: {str(i + 1): n for i, n in enumerate(value)} for name, value in cases.items()},
        "counts_file_sha256": file_sha256(counts_file), "image_sha256": hashes,
        "score_type": "uncalibrated_softmax", "score_semantics": "predicted-class softmax, not always FAIL score",
        "decision_rule": "argmax; ties choose class index 0", "capabilities": {"boxes": False, "actual_counts": False},
        "offline_processing_ms": times, "timing_scope": "amortized batch IO and inference; excludes hashing/camera/display/model loading"})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--counts_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--allow-dev-only", action="store_true")
    a = p.parse_args()
    try:
        rows = batch(a.checkpoint, a.input_dir, a.counts_file, a.output_dir, a.device, a.batch_size, a.allow_smoke, a.allow_dev_only)
        print(f"Exported {len(rows)} image decisions")
        return 0 if all(r["status"] in {"OK", "SMOKE_ONLY", "DEV_ONLY"} for r in rows) else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        p.exit(2, f"EXPORT_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
