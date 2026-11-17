"""Re-evaluate only a declared validation manifest using frozen checkpoint settings."""
import argparse
from pathlib import Path

from .data import ManifestDataset
from .preflight import check_training_inputs
from .runtime import load_checkpoint, write_json
from .train import make_loader, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    try:
        if Path(args.output).exists():
            raise ValueError("output already exists")
        model, config, device, _ = load_checkpoint(args.checkpoint, args.device, allow_smoke=args.allow_smoke)
        if config.data_kind == "REAL":
            if Path(args.manifest).resolve() != Path(config.val_manifest).resolve():
                raise ValueError("REAL validation must use the approved Val manifest stored in the checkpoint")
            _, dataset, _ = check_training_inputs(config)
        else:
            dataset = ManifestDataset(args.manifest, config, "val")
        metrics, predictions = run_epoch(model, make_loader(dataset, config), device, config.class_to_idx["FAIL"])
        write_json(args.output, {"evidence_status": config.evidence_status, "metrics": metrics, "predictions": predictions})
        print(f"{config.evidence_status}: validation results -> {args.output}")
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"VALIDATION_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
