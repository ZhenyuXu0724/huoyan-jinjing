"""Post-freeze stability diagnosis on train/val originals; never used for selection."""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from threadpoolctl import threadpool_limits

from ..data import load_manifest
from ..features import extract, read_image
from .common import (
    DATA, DIAGNOSIS, OUT, ROOT, SIZE, WEIGHTS, ensure, now, read, record_stage,
    reconstruction_scores, score_metrics, sha256, update_state, write, write_predictions,
)


def border_gray(path):
    with Image.open(path) as source:
        array = np.asarray(ImageOps.exif_transpose(source).convert("L"))
    height, width = array.shape
    dy = max(1, int(np.ceil(height * 0.05)))
    dx = max(1, int(np.ceil(width * 0.05)))
    mask = np.zeros_like(array, dtype=bool)
    mask[:dy] = True
    mask[-dy:] = True
    mask[:, :dx] = True
    mask[:, -dx:] = True
    return int(np.median(array[mask]))


def variants(image, gray):
    yield "original", image
    raw = Image.fromarray(np.rint(image * 255).astype("uint8"))
    for size in (240, 224):
        small = np.asarray(raw.resize((size, size), Image.Resampling.LANCZOS),
                           dtype=np.float32) / 255
        gap = SIZE - size
        center = gap // 2
        positions = {"center": (center, center), "left": (0, center),
                     "right": (gap, center), "top": (center, 0), "bottom": (center, gap)}
        for fill, value in (("white", 255), ("border", gray)):
            for position, (x, y) in positions.items():
                canvas = np.full((SIZE, SIZE), np.float32(value / 255), dtype=np.float32)
                canvas[y:y + size, x:x + size] = small
                yield f"s{size}_{fill}_{position}", canvas


def score(bundle, matrix):
    features = bundle["scaler"].transform(extract(matrix, bundle["config"]).reshape(1, -1))
    if bundle["kind"] == "hog_autoencoder":
        return float(reconstruction_scores(bundle["model"], features)[0])
    return float(bundle["model"].predict_proba(features)[0, 1])


def save_pair(destination, model_name, row, image, gray):
    """Side-by-side original and one downscaled/border-filled variant."""
    raw = Image.fromarray(np.rint(image * 255).astype("uint8"))
    small = np.asarray(raw.resize((240, 240), Image.Resampling.LANCZOS),
                       dtype=np.float32) / 255
    canvas = np.full((SIZE, SIZE), np.float32(gray / 255), dtype=np.float32)
    canvas[8:248, 8:248] = small
    right = Image.fromarray(np.rint(canvas * 255).astype("uint8"))
    combined = Image.new("L", (SIZE * 2 + 10, SIZE), 255)
    combined.paste(Image.fromarray(np.rint(image * 255).astype("uint8")), (0, 0))
    combined.paste(right, (SIZE + 10, 0))
    draw = ImageDraw.Draw(combined)
    draw.text((4, 4), f"{model_name} {row['image_id']} original", fill=0)
    draw.text((SIZE + 14, 4), "240 border center", fill=0)
    ensure(destination)
    combined.save(destination / f"{model_name}_{Path(row['image_id']).stem}.png")


def run(splits=("train", "val")):
    freeze_path = OUT / "freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("diagnose requires freeze.json")
    freeze = read(freeze_path)
    bundles = {}
    for method, record in freeze["representatives"].items():
        bundles[method] = joblib.load(ROOT / record["model_path"])
    samples = {}
    for split in splits:
        samples[split] = load_manifest(DATA / f"manifests/hog/{split}.csv")
    destination = DIAGNOSIS
    if destination.exists():
        raise RuntimeError("diagnosis output already exists; do not overwrite")
    ensure(destination)
    rows = []
    pair_rows = []
    with threadpool_limits(limits=1):
        for split, items in samples.items():
            for number, sample in enumerate(items, 1):
                image = read_image(sample.path, SIZE)
                gray = border_gray(sample.path)
                views = list(variants(image, gray))
                hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
                          for name, array in views}
                aliases = {name: next(key for key, value in hashes.items() if value == digest)
                           for name, digest in hashes.items()}
                scores_by_model = {}
                for method, bundle in bundles.items():
                    scores = [score(bundle, array) for _, array in views]
                    scores_by_model[method] = scores
                    threshold = float(bundle["config"]["threshold"])
                    base = scores[0] >= threshold
                    for (name, _), value in zip(views, scores, strict=True):
                        rows.append({
                            "split": split, "model": method, "image_id": sample.image_id,
                            "label": sample.label, "variant": name, "status": "OK",
                            "predicted_label": "FAIL" if value >= threshold else "PASS",
                            "prediction_score": value,
                            "score_type": bundle["score_type"], "threshold": threshold,
                            "flipped": bool((value >= threshold) != base),
                            "original_prediction": "FAIL" if base else "PASS",
                            "duplicate_of": aliases[name], "border_gray": gray,
                        })
                    pair_rows.append({"split": split, "model": method,
                                      "image_id": sample.image_id,
                                      "any_flip": any((value >= threshold) != base
                                                      for value in scores)})
                if number % 20 == 0:
                    print(f"Diagnosis {split} {number}/{len(items)}", flush=True)
    write_predictions(destination / "predictions.csv", rows)
    summary = {}
    pairs_dir = destination / "pairs"
    saved_pairs = {}
    for split in splits:
        for method in bundles:
            selected = [row for row in rows if row["split"] == split and row["model"] == method]
            variant_names = list(dict.fromkeys(row["variant"] for row in selected))
            groups = {}
            for name in variant_names:
                group = [row for row in selected if row["variant"] == name]
                stats = score_metrics(group)
                stats.update({
                    "flips": sum(row["flipped"] for row in group),
                    "correct_to_wrong": sum(
                        row["original_prediction"] == row["label"]
                        and row["predicted_label"] != row["label"] for row in group),
                    "wrong_to_correct": sum(
                        row["original_prediction"] != row["label"]
                        and row["predicted_label"] == row["label"] for row in group),
                })
                groups[name] = stats
            any_flip = sum(any(row["flipped"] for row in selected
                               if row["image_id"] == sample.image_id) for sample in samples[split])
            lookup = {(row["image_id"], row["variant"]): row for row in selected}
            paired = {}
            for size in (240, 224):
                for fill in ("white", "border"):
                    base = f"s{size}_{fill}_center"
                    for position in ("left", "right", "top", "bottom"):
                        alternative = f"s{size}_{fill}_{position}"
                        paired[f"{alternative}_vs_{base}"] = sum(
                            lookup[(sample.image_id, alternative)]["predicted_label"]
                            != lookup[(sample.image_id, base)]["predicted_label"]
                            for sample in samples[split])
                for position in ("center", "left", "right", "top", "bottom"):
                    paired[f"s{size}_border_vs_white_{position}"] = sum(
                        lookup[(sample.image_id, f"s{size}_white_{position}")]["predicted_label"]
                        != lookup[(sample.image_id, f"s{size}_border_{position}")]["predicted_label"]
                        for sample in samples[split])
            summary[f"{split}_{method}"] = {
                "n": len(samples[split]), "any_flip": any_flip,
                "variants": groups, "paired_flips": paired}
            for pair in pair_rows:
                if pair["split"] == split and pair["model"] == method and pair["any_flip"]:
                    if saved_pairs.get((split, method), 0) >= 12:
                        continue
                    saved_pairs[(split, method)] = saved_pairs.get((split, method), 0) + 1
                    sample = next(item for item in samples[split]
                                  if item.image_id == pair["image_id"])
                    image = read_image(sample.path, SIZE)
                    gray = border_gray(sample.path)
                    save_pair(pairs_dir, method,
                              {"image_id": sample.image_id}, image, gray)
    write(destination / "summary.json", {
        "schema_version": 1, "created_at": now(), "freeze_sha256": sha256(freeze_path),
        "splits": list(splits), "bundles": {method: freeze["representatives"][method]["model_path"]
                                            for method in freeze["representatives"]},
        "variants": "original + 2 sizes x 2 fills x 5 positions = 21",
        "notes": ["Repeated variants are marked but counted per original image.",
                  "The AE never learned from train FAIL images.",
                  "Diagnosis is descriptive only; it never changes the default or triggers retraining."],
        "results": summary,
    })
    record_stage("diagnose", "completed", f"{len(rows)} predictions over {len(samples)} split(s)")
    update_state(next_command="python -m members.member04_hog.final_v4.prepare --scope full")
    print(f"Diagnosis complete: {len(rows)} predictions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=("train", "val"))
    args = parser.parse_args()
    try:
        run(tuple(args.splits))
    except Exception as error:
        record_stage("diagnose", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
