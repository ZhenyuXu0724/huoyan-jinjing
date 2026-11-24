"""Image-level metrics and JSON/CSV helpers shared by member04 evaluation."""
import argparse
import csv
import json
from pathlib import Path


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
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    decided = sum(counts.values())
    return {**counts, "false_pass": fn, "false_fail": fp, "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "total": len(rows), "decided": decided,
            "coverage": decided / len(rows) if rows else 0.0,
            "errors": errors, "unjudged": unjudged}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        from .model import load_model, predict_with_model

        result = predict_with_model(args.image, load_model(args.model))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
