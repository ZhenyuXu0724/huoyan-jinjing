"""Generate isolated, reproducible SYNTHETIC fixtures, never official observations."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import Config
from .runtime import write_json


def generate(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("fixture directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(601)
    for split, count in (("train", 8), ("val", 4)):
        directory = output / split
        directory.mkdir()
        rows = []
        for i in range(count):
            array = rng.integers(200, 256, (80, 120, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            # Simple fixture variation exercises IO/training, not mechanical-part recognition.
            if i % 2:
                draw.ellipse((20 + i, 20, 60 + i, 60), fill=(30, 30, 30))
            else:
                draw.rectangle((5 + i, 25, 90, 45), fill=(40, 40, 40))
            path = directory / f"synthetic_{i}.png"
            image.save(path)
            rows.append({"image_path": f"{split}/{path.name}", "label": "FAIL" if i % 2 else "PASS",
                         "scene_id": f"synthetic_{split}_{i}", "split": split, "data_kind": "SMOKE_ONLY"})
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).rectangle((30, 35, 130, 65), fill="black")
    image.save(output / "geometry_rectangle.png")
    Image.new("RGB", (160, 100), "white").save(output / "geometry_empty.png")
    config = Config(weights="RANDOM_SMOKE_ONLY", data_kind="SMOKE_ONLY", image_size=64,
                    epochs=2, batch_size=4, train_manifest=str(output / "train.csv"), val_manifest=str(output / "val.csv"))
    write_json(output / "smoke_config.json", config.to_dict())
    pretrained = config.to_dict()
    pretrained["weights"] = "IMAGENET1K_V1"
    pretrained["image_size"] = 224
    write_json(output / "pretrained_smoke_config.json", pretrained)
    (output / "SMOKE_ONLY.txt").write_text("Synthetic fixtures only. Not real data or official metrics.\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(generate(args.output))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"FIXTURE_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
