"""ResNet implementation of the adapter backend, and standalone JSON batch CLI."""
import argparse
import json
from pathlib import Path

import torch

from .adapter import ClassificationAdapter
from .data import Preprocess
from .runtime import file_sha256, load_checkpoint, write_json


class ResNetBackend:
    def __init__(self, checkpoint, device="auto", allow_smoke=False, batch_size=16):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.config, self.device, _ = load_checkpoint(checkpoint, device, allow_smoke=allow_smoke)
        self.class_to_idx = self.config.class_to_idx
        self.evidence_status = self.config.evidence_status
        self.transform = Preprocess(self.config)
        self.batch_size = batch_size

    @torch.inference_mode()
    def score_batch(self, images):
        self.model.eval()
        scores = []
        for start in range(0, len(images), self.batch_size):
            inputs = torch.stack([self.transform(image) for image in images[start:start + self.batch_size]]).to(self.device)
            scores.extend(self.model(inputs).softmax(1).cpu().tolist())
        return scores


def create_adapter(checkpoint, device="auto", allow_smoke=False, batch_size=16):
    backend = ResNetBackend(checkpoint, device, allow_smoke, batch_size)
    return ClassificationAdapter(backend, backend.config.supported_counts, backend.config.counts_rule_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--input-dir")
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument("--expected-counts", help='JSON vector or dictionary, e.g. [2,2,2,2,2]')
    counts.add_argument("--counts-file", help="shared protocol list with image_id/expected_counts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if Path(args.output).exists():
            raise ValueError("output already exists")
        if args.input_dir and not Path(args.input_dir).is_dir():
            raise ValueError("input directory does not exist")
        paths = [Path(args.image)] if args.image else sorted(p for p in Path(args.input_dir).iterdir()
                                                            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and p.is_file())
        if not paths:
            raise ValueError("no images found")
        expected = json.loads(args.expected_counts) if args.expected_counts else None
        adapter = create_adapter(args.checkpoint, args.device, args.allow_smoke, args.batch_size)
        if args.counts_file:
            records = json.loads(Path(args.counts_file).read_text(encoding="utf-8"))
            if not isinstance(records, list) or any(not isinstance(r, dict) or set(r) != {"image_id", "expected_counts"} for r in records):
                raise ValueError("invalid counts_file schema")
            mapping = {r["image_id"]: r["expected_counts"] for r in records}
            if len(mapping) != len(records) or set(mapping) != {p.name for p in paths}:
                raise ValueError("counts_file must match every input filename exactly once")
            results = [adapter.predict(p, mapping[p.name]) for p in paths]
        else:
            results = adapter.predict_batch(paths, expected)
        write_json(args.output, {"model_id": "baseline2", "checkpoint_sha256": file_sha256(args.checkpoint),
                                 "config": adapter.backend.config.to_dict(),
                                 "results": [{"image_id": p.name, **r} for p, r in zip(paths, results)]})
        print(f"wrote {len(results)} results to {args.output}")
        return 0 if all(r["status"] in {"OK", "SMOKE_ONLY"} for r in results) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"INFERENCE_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
