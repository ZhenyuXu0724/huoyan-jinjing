"""Explicit post-freeze evaluation. Never called by training, planning or validation."""
import argparse
import csv
from pathlib import Path

from torch.utils.data import Dataset

from .config import normalize_counts
from .data import Preprocess, image_digest
from .experiments import code_hashes
from .runtime import file_sha256, load_checkpoint
from .team_data import read_json, write_new_json
from .train import make_loader, run_epoch


class EvaluationDataset(Dataset):
    def __init__(self, manifest, config, approval, development):
        self.rows, self.transform = [], Preprocess(config)
        root = Path(manifest).resolve().parent
        with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = {"image_path", "image_sha256", "label", "scene_id", "split", "data_kind"}
            if set(reader.fieldnames or []) != columns:
                raise ValueError(f"evaluation manifest requires {sorted(columns)}")
            for row in reader:
                if row["split"] != approval["split"] or row["data_kind"] != config.data_kind:
                    raise ValueError("evaluation role/data kind mismatch")
                if row["label"] not in config.class_to_idx or not row["scene_id"].strip() or row["scene_id"].upper().startswith("UNVERIFIED"):
                    raise ValueError("reviewed image labels and trusted scene IDs required")
                path = (root / row["image_path"]).resolve()
                if file_sha256(path) != row["image_sha256"]:
                    raise ValueError("evaluation image differs from approved manifest")
                self.rows.append({**row, "path": str(path), "sha256_rgb": image_digest(path),
                                  "target": config.class_to_idx[row["label"]]})
        if not self.rows:
            raise ValueError("empty evaluation set")
        for key in ("path", "scene_id", "sha256_rgb"):
            current = {row[key] for row in self.rows}
            if current & {str(row[key]) for row in development}:
                raise ValueError(f"evaluation/development overlap: {key}")
            if key != "scene_id" and len(current) != len(self.rows):
                raise ValueError(f"duplicate evaluation {key}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(row["path"]), row["target"]


def evaluate(freeze_path, manifest, approval_path, output, device="auto", allow_smoke=False, check_only=False):
    frozen, approval = read_json(freeze_path), read_json(approval_path)
    if frozen.get("schema_version") != "member06.freeze.v1" or frozen.get("test_used_for_selection") is not False:
        raise ValueError("valid pre-test freeze required")
    if frozen["source_sha256"] != code_hashes():
        raise ValueError("code changed after freeze")
    if file_sha256(frozen["selection"]) != frozen["selection_sha256"] or file_sha256(frozen["checkpoint"]) != frozen["checkpoint_sha256"]:
        raise ValueError("frozen selection/checkpoint changed")
    if approval.get("schema_version") != "member06.evaluation-approval.v1" or approval.get("status") != "APPROVED":
        raise ValueError("explicit evaluation handoff approval required")
    for key in ("approved_by", "decision_reference", "dataset_version"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError(f"missing evaluation {key}")
    if approval.get("labels_reviewed") is not True or approval.get("scene_split_verified") is not True:
        raise ValueError("evaluation labels and scene separation require review")
    if approval.get("split") not in {"test", "stage2_test", "stage3_test"}:
        raise ValueError("only declared post-freeze test roles are accepted")
    if approval.get("evaluation_scope") not in {"PUBLIC_PARTITION_EVALUATION", "INDEPENDENT_HELD_OUT"}:
        raise ValueError("evaluation scope must be explicitly declared")
    if approval["evaluation_scope"] == "INDEPENDENT_HELD_OUT" and approval.get("sealed_until_freeze") is not True:
        raise ValueError("independent evaluation needs sealed-until-freeze confirmation")
    if approval.get("freeze_sha256") != file_sha256(freeze_path) or approval.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("approval does not bind this freeze and evaluation manifest")
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; preserve prior evidence")
    # Decode the trusted checkpoint without constructing a model for check-only.
    import torch
    from .config import Config
    payload = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=True)
    config = Config(**payload["config"])
    if config.training_tier == "DEV_ONLY":
        raise ValueError("DEV_ONLY is not eligible for frozen formal evaluation")
    if config.data_kind == "SMOKE_ONLY" and not allow_smoke:
        raise ValueError("SMOKE_ONLY requires --allow-smoke")
    if normalize_counts(approval.get("expected_counts")) != config.supported_counts:
        raise ValueError("evaluation expected_counts differ from frozen rule")
    if payload["evidence_status"] != config.evidence_status or payload["run_metadata"]["test_data_used"] is not False:
        raise ValueError("invalid checkpoint evidence")
    development = [r for s in ("train", "val") for r in payload["run_metadata"]["datasets"][s]["images"]]
    dataset = EvaluationDataset(manifest, config, approval, development)
    result = {"evidence_status": config.evidence_status, "evaluation_scope": approval["evaluation_scope"],
              "split": approval["split"], "n": len(dataset), "class_to_idx": config.class_to_idx,
              "freeze_sha256": file_sha256(freeze_path), "manifest_sha256": file_sha256(manifest),
              "approval_sha256": file_sha256(approval_path), "test_used_for_selection": False,
              "check_only": check_only}
    if not check_only:
        model, _, selected_device, _ = load_checkpoint(frozen["checkpoint"], device, allow_smoke=allow_smoke)
        metrics, predictions = run_epoch(model, make_loader(dataset, config), selected_device, config.class_to_idx["FAIL"])
        result.update(metrics=metrics, predictions=[{**row, **pred} for row, pred in zip(dataset.rows, predictions)])
    write_new_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "manifest", "approval", "output"):
        p.add_argument(f"--{key}", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    try:
        result = evaluate(a.freeze, a.manifest, a.approval, a.output, a.device, a.allow_smoke, a.check_only)
        print(f"{result['evidence_status']}: {result['evaluation_scope']}, n={result['n']}, check_only={a.check_only}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        p.exit(2, f"EVALUATION_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
