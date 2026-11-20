"""Pure object/count metrics. Not imported by inference; no GT leakage.

Input cases: image_id, status, objects (predictions), ground_truth (class_id,bbox,ignore).
Unjudgeable cases must have evaluable=False. Ignored objects are excluded from GT
counts; images with any ignored object are excluded from count metrics. Predictions
unmatched to valid GT and overlapping ANY ignored box at IoU>=threshold are neutral.
This proposed ignore policy must be agreed with member03/lead before freeze.
"""
import math
import argparse
import json
from pathlib import Path


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def validate_object(obj, prediction=False):
    if type(obj["class_id"]) is not int or obj["class_id"] not in range(6):
        raise ValueError("Class ID must be an integer 0..5")
    box = obj["bbox"]
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid original-pixel box")
    if prediction and (not math.isfinite(obj["score"]) or not 0 <= obj["score"] <= 1):
        raise ValueError("Invalid score")
    if not prediction and type(obj.get("ignore", False)) is not bool:
        raise ValueError("Normalize ignore flags to bool before evaluation")


def evaluate(cases, threshold=0.5):
    if not isinstance(cases, list):
        raise ValueError("Cases must be a JSON list")
    if not 0 < threshold <= 1:
        raise ValueError("Invalid matching threshold")
    metrics = {str(c): {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in range(6)}
    errors = {str(c): 0 for c in range(6)}
    seen = set()
    evaluated = counted = correct_vectors = ignored_predictions = 0
    excluded = []
    for case in cases:
        image_id = case["image_id"]
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a non-empty string")
        if case.get('status') not in ('OK','REVIEW_REQUIRED','UNSUPPORTED_COUNTS','INPUT_ERROR','NOT_IMPLEMENTED'):
            raise ValueError("Missing or unsupported prediction status")
        for flag in ('evaluable','counts_judgeable'):
            if flag in case and type(case[flag]) is not bool:
                raise ValueError(f"{flag} must be bool")
        if image_id in seen:
            raise ValueError("Duplicate image_id; evaluate different expected-count runs separately")
        seen.add(image_id)
        if case.get("status") != "OK" or not case.get("evaluable", True):
            excluded.append(image_id)
            continue
        predictions, truth = case["objects"], case["ground_truth"]
        for p in predictions:
            validate_object(p, True)
        for g in truth:
            validate_object(g)
        valid = [g for g in truth if not g.get("ignore", False)]
        ignored = [g for g in truth if g.get("ignore", False)]
        used = set()
        for g in valid:
            metrics[str(g["class_id"])]["support"] += 1
        # Stable order resolves tied prediction scores; lower GT row index resolves IoU ties.
        for p in sorted(predictions, key=lambda p: -p["score"]):
            candidates = [(iou(p["bbox"], g["bbox"]), j) for j, g in enumerate(valid)
                          if j not in used and g["class_id"] == p["class_id"]]
            best = max(candidates, key=lambda pair: (pair[0], -pair[1]), default=(0., -1))
            m = metrics[str(p["class_id"])]
            if best[0] >= threshold:
                used.add(best[1]); m["tp"] += 1
            elif any(iou(p["bbox"], g["bbox"]) >= threshold for g in ignored):
                ignored_predictions += 1
            else:
                m["fp"] += 1
        for j, g in enumerate(valid):
            if j not in used:
                metrics[str(g["class_id"])]["fn"] += 1
        evaluated += 1
        if not ignored and case.get("counts_judgeable", True):
            counted += 1
            differences = {}
            for c in range(6):
                differences[c] = abs(sum(p["class_id"] == c for p in predictions)-sum(g["class_id"] == c for g in valid))
                errors[str(c)] += differences[c]
            correct_vectors += all(differences[c] == 0 for c in range(1, 6))
    for c, m in metrics.items():
        m["precision"] = m["tp"]/(m["tp"]+m["fp"]) if m["tp"]+m["fp"] else None
        m["recall"] = m["tp"]/(m["tp"]+m["fn"]) if m["tp"]+m["fn"] else None
        m["count_mae"] = errors[c]/counted if counted else None
    return {"per_class": metrics, "evaluated_images": evaluated, "count_images": counted,
            "five_class_vector_accuracy": correct_vectors/counted if counted else None,
            "coverage": evaluated/len(seen) if seen else None,
            "excluded_images": excluded, "ignored_predictions": ignored_predictions,
            "iou_threshold": threshold}


def main():
    parser = argparse.ArgumentParser(description='Evaluate member05 object/count results; GT is read only here, never during inference')
    parser.add_argument('--cases', required=True, help='Private JSON list: image_id, status, objects, ground_truth, optional evaluable/counts_judgeable')
    parser.add_argument('--output', required=True, help='New metrics JSON; refuses to overwrite')
    parser.add_argument('--iou', type=float, default=.5)
    args = parser.parse_args()
    try:
        cases = json.loads(Path(args.cases).read_text(encoding='utf-8-sig'))
        report = evaluate(cases, args.iou)
        report['source_cases'] = Path(args.cases).name
        report['purpose'] = 'Object/count metrics only; interpretation depends on input provenance'
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, str(exc)+'\n')
    print(f'Wrote {destination}; evaluated {report["evaluated_images"]} images; excluded {len(report["excluded_images"])}')


if __name__ == '__main__':
    main()
