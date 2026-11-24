"""Batch baseline1 inference with the shared expected-count and output contract."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from shared.contracts import validate_expected_counts

from .data import sha256
from .evaluate import write_json
from .model import MODEL_PATH, load_active_model, load_model, predict_with_model


def _bundle(model_path):
    if model_path is None:
        return load_active_model(), None
    bundle = load_model(model_path)
    from .final_v4.common import validate_bundle

    ok, reason = validate_bundle(bundle)
    if not ok:
        raise ValueError(f"unsupported model bundle: {reason}")
    return bundle, model_path


def batch(input_dir, counts_file, output, model_path=None):
    cases = json.loads(Path(counts_file).read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("counts_file must be a nonempty list")
    requests = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"image_id", "expected_counts"}:
            raise ValueError("Each request contains only image_id and expected_counts")
        image_id = case["image_id"]
        if (not isinstance(image_id, str) or not image_id or image_id in requests
                or any(character in image_id for character in ("/", "\\", ":"))
                or image_id in (".", "..")):
            raise ValueError("Each image_id must be a unique filename")
        requests[image_id] = validate_expected_counts(case["expected_counts"])
    bundle, explicit = _bundle(model_path)
    paths = {}
    for path in Path(input_dir).rglob("*"):
        if path.is_file() and path.name in requests:
            if path.name in paths:
                raise ValueError(f"Duplicate input filename: {path.name}")
            paths[path.name] = path
    rows, timings = [], {}
    for image_id, requirement in requests.items():
        started = perf_counter()
        if requirement != bundle["config"]["expected_counts"]:
            result = {"status": "UNSUPPORTED_COUNTS", "predicted_label": None,
                      "prediction_score": None, "score_type": None}
        elif image_id not in paths:
            result = {"status": "INPUT_ERROR", "predicted_label": None,
                      "prediction_score": None, "score_type": None}
        else:
            result = predict_with_model(paths[image_id], bundle)
        timings[image_id] = (perf_counter() - started) * 1000
        rows.append({"image_id": image_id, "model_id": "baseline1", **result,
                     "non_catalogue_count": None})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "images.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    classes = json.loads((Path(__file__).resolve().parents[2] /
                          "shared/catalogue.json").read_text(encoding="utf-8"))["classes"]
    config = bundle["config"]
    write_json(output / "run_config.json", {
        "model_id": "baseline1",
        "model_version": bundle.get("version"),
        "method": bundle.get("method"),
        "model_sha256": sha256(explicit or MODEL_PATH),
        "class_catalogue": classes,
        "preprocessing": {key: config[key] for key in
                          ("size", "cell", "threshold", "expected_counts")},
        "capabilities": bundle["capabilities"],
        "supported_expected_counts": config["expected_counts"],
        "input_expected_counts": requests, "dependencies": bundle["dependencies"],
        "score_type": bundle["score_type"], "higher_score_supports": "FAIL",
        "offline_processing_ms": timings,
        "timing_scope": "offline processing, excludes camera and display"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--counts_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=None)
    args = parser.parse_args()
    batch(args.input_dir, args.counts_file, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()
