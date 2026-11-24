"""Pixel geometry and optional frozen local CNN extraction; no fusion prediction."""
import argparse
from pathlib import Path

from .research_tools import LocalFeatureExtractor, crop_roi, geometry_features
from .runtime import load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--foreground", choices=["dark", "light"], default="dark")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if Path(args.output).exists():
            raise ValueError("output already exists")
        crop = crop_roi(args.image, args.bbox)
        results = {k: v for k, v in crop.items() if k != "roi"}
        if crop["status"] == "OK":
            results["geometry"] = geometry_features(crop["roi"], foreground=args.foreground)
        if args.checkpoint:
            model, config, device, _ = load_checkpoint(args.checkpoint, allow_smoke=args.allow_smoke)
            results["cnn"] = LocalFeatureExtractor(model, config, device).extract(args.image, args.bbox)
        write_json(args.output, results)
        print(f"research tools -> {args.output}")
        return 0 if crop["status"] == "OK" else 2
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"RESEARCH_TOOLS_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
