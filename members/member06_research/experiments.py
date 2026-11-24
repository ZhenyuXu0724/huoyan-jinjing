"""Plan (no training), select completed Val runs, freeze selected model for evaluation."""
import argparse
import math
from pathlib import Path

import torch

from .config import Config
from .runtime import file_sha256
from .team_data import read_json, write_new_json


def code_hashes():
    return {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def plan(config_path, output, learning_rates, weight_decays, epochs):
    base = Config.load(config_path)
    output = Path(output).resolve()
    configurations = []
    for lr in learning_rates:
        for wd in weight_decays:
            values = {**base.to_dict(), "learning_rate": lr, "weight_decay": wd, "epochs": epochs}
            configurations.append(Config(**values))
    if not configurations or len({(c.learning_rate, c.weight_decay) for c in configurations}) != len(configurations):
        raise ValueError("empty or duplicate candidate grid")
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    for i, config in enumerate(configurations):
        path = output / f"candidate_{i:02d}.json"
        write_new_json(path, config.to_dict())
        candidates.append({"config": str(path), "config_sha256": file_sha256(path),
                           "run": str(output / f"run_{i:02d}")})
    record = {"schema_version": "member06.experiment-plan.v1", "selection": "minimum best Val cross entropy; plan order breaks ties",
              "training_executed": False, "test_used": False, "candidates": candidates,
              "source_sha256": code_hashes(), "base_config_sha256": file_sha256(config_path)}
    write_new_json(output / "plan.json", record)
    return record


def select(plan_path, output, allow_smoke=False, allow_dev_only=False):
    record = read_json(plan_path)
    if record.get("schema_version") != "member06.experiment-plan.v1" or record.get("test_used") is not False:
        raise ValueError("invalid experiment plan")
    if record["source_sha256"] != code_hashes():
        raise ValueError("code changed since plan; start a versioned plan")
    reports = []
    for candidate in record["candidates"]:
        if file_sha256(candidate["config"]) != candidate["config_sha256"]:
            raise ValueError("planned config changed")
        config = Config.load(candidate["config"])
        if config.data_kind == "SMOKE_ONLY" and not allow_smoke:
            raise ValueError("SMOKE_ONLY requires --allow-smoke")
        if config.training_tier == "DEV_ONLY" and not allow_dev_only:
            raise ValueError("DEV_ONLY selection requires --allow-dev-only")
        run = Path(candidate["run"])
        checkpoint = run / "best.pt"
        digest = file_sha256(checkpoint)
        if read_json(run / "checkpoint_manifest.json")["sha256"] != digest:
            raise ValueError("checkpoint changed or run incomplete")
        if read_json(run / "config.json") != config.to_dict():
            raise ValueError("run config differs from planned config")
        history = read_json(run / "history.json")
        if len(history) != config.epochs or [r["epoch"] for r in history] != list(range(1, config.epochs + 1)):
            raise ValueError("run not completed")
        if any(not math.isfinite(r["val"]["loss"]) for r in history):
            raise ValueError("nonfinite validation loss")
        best = min(history, key=lambda r: r["val"]["loss"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["config"] != config.to_dict() or payload["epoch"] != best["epoch"] or payload["val_metrics"] != best["val"]:
            raise ValueError("checkpoint/history selection mismatch")
        metadata = payload["run_metadata"]
        if metadata["test_data_used"] is not False or metadata["environment"]["source_sha256"] != record["source_sha256"]:
            raise ValueError("run code/evaluation scope differs from plan")
        data_identity = {s: metadata["datasets"][s]["manifest_sha256"] for s in ("train", "val")}
        if reports and data_identity != reports[0]["data_identity"]:
            raise ValueError("candidate runs used different data")
        reports.append({"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
                        "val_loss": best["val"]["loss"], "epoch": best["epoch"],
                        "data_identity": data_identity, "evidence_status": config.evidence_status})
    if not reports:
        raise ValueError("empty candidate plan")
    result = {"schema_version": "member06.selection.v1", "plan": str(Path(plan_path).resolve()),
              "plan_sha256": file_sha256(plan_path), "test_used": False,
              "candidates": reports, "selected": min(reports, key=lambda r: r["val_loss"])}
    write_new_json(output, result)
    return result


def freeze(selection_path, output, allow_smoke=False):
    selection = read_json(selection_path)
    if selection.get("schema_version") != "member06.selection.v1" or selection.get("test_used") is not False:
        raise ValueError("invalid selection")
    if selection.get("selected",{}).get("evidence_status") == "DEV_ONLY":
        raise ValueError("DEV_ONLY cannot be formally frozen; obtain reviewed data and run a new formal experiment")
    # Recompute selection from the fixed plan; don't trust a hand-edited winner.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        checked = select(selection["plan"], Path(tmp) / "selection.json", allow_smoke)
    if selection != checked:
        raise ValueError("selection changed; cannot freeze")
    winner = selection["selected"]
    result = {"schema_version": "member06.freeze.v1", **winner,
              "selection": str(Path(selection_path).resolve()), "selection_sha256": file_sha256(selection_path),
              "source_sha256": code_hashes(), "decision_rule": "argmax_uncalibrated_softmax",
              "test_used_for_selection": False}
    write_new_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--config", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.0001])
    q.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 0.0001])
    q.add_argument("--epochs", type=int, default=10)
    for name, input_name in (("select", "plan"), ("freeze", "selection")):
        q = sub.add_parser(name)
        q.add_argument(f"--{input_name}", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--allow-smoke", action="store_true")
        if name == "select": q.add_argument("--allow-dev-only", action="store_true")
    a = p.parse_args()
    try:
        if a.mode == "plan":
            plan(a.config, a.output, a.learning_rates, a.weight_decays, a.epochs)
        elif a.mode == "select":
            select(a.plan, a.output, a.allow_smoke, a.allow_dev_only)
        else:
            freeze(a.selection, a.output, a.allow_smoke)
        print(f"{a.mode}: {a.output}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        p.exit(2, f"EXPERIMENT_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
