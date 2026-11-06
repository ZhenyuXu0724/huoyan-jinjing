"""Explicit training; local inputs only. Never evaluates sealed test data."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_dataset(path):
    import yaml
    from PIL import Image
    from .model import validate_names
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_names(data["names"])
    root = Path(data["path"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Dataset path must be an existing absolute directory")
    root = root.resolve()
    split_paths = []
    hashes = {}
    audit = {}
    for split in ("train", "val"):
        directory = (root / data[split]).resolve()
        images = [p for p in directory.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
        if not images:
            raise ValueError(f"No images in {split}: {directory}")
        support = {str(i): 0 for i in range(6)}
        hashes[split] = set()
        for image in images:
            try:
                with Image.open(image) as opened:
                    decoded = opened.convert('RGB')
                    digest = hashlib.sha256(str(decoded.size).encode() + decoded.tobytes()).hexdigest()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot decode image: {image}") from exc
            hashes[split].add(digest)
            relative = image.relative_to(root)
            if relative.parts[0] != "images":
                raise ValueError("Use images/train and images/val with matching labels directories")
            label = (root / "labels" / Path(*relative.parts[1:])).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label (verified empty scenes need an empty file): {label}")
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    cid, xc, yc, w, h = map(float, line.split())
                    valid = cid.is_integer() and 0 <= cid <= 5 and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1
                    valid = valid and xc-w/2 >= -1e-6 and xc+w/2 <= 1+1e-6 and yc-h/2 >= -1e-6 and yc+h/2 <= 1+1e-6
                except (ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError(f"Invalid YOLO annotation: {label}")
                support[str(int(cid))] += 1
        audit[split] = {"images": len(images), "unique_decoded_images": len(hashes[split]),
                        "instances_per_class": support}
        split_paths.append(directory)
    a, b = split_paths
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Train/val directories must be distinct and non-nested")
    if hashes['train'] & hashes['val']:
        raise ValueError("Duplicate decoded image across train and val; split scenes before training")
    data['_audit'] = audit
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", help="Existing local yolov8n.pt; required for training")
    parser.add_argument("--check-only", action="store_true", help="Validate images/labels/splits without loading a model")
    parser.add_argument("--project", default="runs/member05")
    parser.add_argument("--name", default="pilot")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.check_only and (not args.weights or not Path(args.weights).is_file()):
        parser.error("Pretrained weights must exist locally; no automatic download")
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        parser.error("Invalid training settings")
    try:
        data = validate_dataset(args.data)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.check_only:
        print(json.dumps(data['_audit'], indent=2))
        return
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; install CUDA PyTorch or explicitly use --device cpu")
    from ultralytics import YOLO
    model = YOLO(str(Path(args.weights).resolve()), task="detect")
    model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers, seed=args.seed,
                project=args.project, name=args.name, exist_ok=False, deterministic=True,
                amp=False)
    # amp=False prevents implicit AMP-check checkpoint downloads; pilot default only.
    print(f"Training output: {model.trainer.save_dir}; validation only, not a held-out test")


if __name__ == "__main__":
    main()
