"""Train and select on validation loss only. No test-manifest argument exists."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .model import FrozenResNet18
from .preflight import check_training_inputs
from .runtime import choose_device, environment, file_sha256, seed_everything, seed_worker, write_json


def summarize(loss_sum, n, confusion, fail_index):
    tp = confusion[fail_index][fail_index]
    fp = sum(confusion[i][fail_index] for i in range(2) if i != fail_index)
    fn = sum(confusion[fail_index][j] for j in range(2) if j != fail_index)
    # Undefined precision/recall remain null rather than invented numbers.
    return {"loss": loss_sum / n, "n": n, "accuracy": sum(confusion[i][i] for i in range(2)) / n,
            "confusion_matrix": confusion, "matrix_axes": "rows=true, columns=predicted, class index order",
            "support": {"FAIL": tp + fn, "PASS": n - tp - fn},
            "false_pass": fn, "false_fail": fp,
            "false_pass_rate": fn / (tp + fn) if tp + fn else None,
            "false_fail_rate": fp / (n - tp - fn) if n - tp - fn else None,
            "fail_precision": tp / (tp + fp) if tp + fp else None,
            "fail_recall": tp / (tp + fn) if tp + fn else None,
            "fail_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def class_weights(rows, config, device):
    """N/(2*n_c), in configured class-index order; Train manifest only."""
    counts = [0, 0]
    for row in rows:
        counts[config.class_to_idx[row["label"]]] += 1
    if not all(counts):
        raise ValueError("class weights require both Train classes")
    return torch.tensor([sum(counts) / (2 * n) for n in counts], dtype=torch.float32, device=device)


def run_epoch(model, loader, device, fail_index, optimizer=None, progress=None, criterion=None):
    training = optimizer is not None
    model.train(training)
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    if not training and criterion.weight is not None:
        raise ValueError("Validation must use unweighted CE")
    if criterion.reduction != "mean":
        raise ValueError("only CE mean reduction supported")
    denominator = 0.0
    loss_sum, n, confusion = 0.0, 0, [[0, 0], [0, 0]]
    predictions = []
    with torch.set_grad_enabled(training):
        for batch_number, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            if labels.dtype != torch.int64 or labels.ndim != 1 or not ((labels >= 0) & (labels < 2)).all():
                raise ValueError("targets must be int64 class indices in [0,1]")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all():
                raise ValueError("invalid model output")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            if training:
                loss.backward()
                optimizer.step()
            scores = logits.detach().softmax(1).cpu()
            predicted = scores.argmax(1)
            for target, pred, score in zip(labels.cpu().tolist(), predicted.tolist(), scores.tolist()):
                confusion[target][pred] += 1
                predictions.append({"target_index": target, "predicted_index": pred, "softmax": score})
            n += len(labels)
            batch_denominator = (criterion.weight[labels].sum().item()
                                 if criterion.weight is not None else len(labels))
            denominator += batch_denominator
            loss_sum += loss.item() * batch_denominator
            if progress and (batch_number == 1 or batch_number == len(loader) or batch_number % max(1,len(loader)//5) == 0):
                print(f"{progress} batch={batch_number}/{len(loader)} running_loss={loss_sum/denominator:.6f}",flush=True)
    if n == 0:
        raise ValueError("empty epoch")
    metrics = summarize(loss_sum, n, confusion, fail_index)
    metrics["loss"] = loss_sum / denominator
    return metrics, predictions


def make_loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config.seed),
                      pin_memory=False, drop_last=False)


def fit(config, output, *, allow_dev_only=False):
    if config.training_tier == "DEV_ONLY" and not allow_dev_only:
        raise ValueError("DEV_ONLY training requires explicit --allow-dev-only; no weights updated")
    print(f"{config.evidence_status}: checking data and provenance before model creation",flush=True)
    seed_everything(config.seed, config.cpu_threads)
    device = choose_device(config.device)
    train, val, preflight = check_training_inputs(config)
    for warning in preflight.get("limitations", []):
        print(f"LIMITATION: {warning}",flush=True)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty (no overwriting experiments)")
    model = FrozenResNet18(config.weights).to(device)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"evidence_status": config.evidence_status, "environment": environment(device), "data_preflight": preflight,
                "selection": "minimum validation cross-entropy; earliest epoch wins ties",
                "decision_rule": "argmax of uncalibrated softmax; ties choose class index 0",
                "test_data_used": False, "weights_source": config.weights,
                "datasets": {split: {"manifest": str(ds.manifest), "manifest_sha256": file_sha256(ds.manifest),
                                     "images": [{**row, "path": str(row["path"])} for row in ds.rows]}
                             for split, ds in (("train", train), ("val", val))}}
    if config.data_provenance:
        metadata["data_provenance_sha256"] = file_sha256(config.data_provenance)
    weights = class_weights(train.rows, config, device) if config.train_class_weight == "balanced" else None
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    metadata["loss_protocol"] = {
        "train_class_weight": config.train_class_weight,
        "class_to_idx": config.class_to_idx,
        "weights": weights.cpu().tolist() if weights is not None else None,
        "train_reduction": "mean: sum(weight[target]*NLL)/sum(weight[target]); epoch aggregates numerators/denominators",
        "validation": "unweighted CE mean", "selection": "minimum unweighted Val loss"}
    if config.weights == "IMAGENET1K_V1":
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        metadata["pretrained_source"] = {"url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                                          "sha256": file_sha256(cached)}
    write_json(output / "config.json", config.to_dict())
    write_json(output / "run_config.json", metadata)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader, val_loader = make_loader(train, config, True), make_loader(val, config)
    history, best_loss = [], float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, config.class_to_idx[config.fail_label], optimizer, progress=f"{config.evidence_status} epoch={epoch} train", criterion=train_criterion)
        val_metrics, predictions = run_epoch(model, val_loader, device, config.class_to_idx[config.fail_label], progress=f"{config.evidence_status} epoch={epoch} val")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics,
                        "evidence_status": config.evidence_status})
        write_json(output / "history.json", history)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
            writer.writeheader()
            for row in history:
                writer.writerow({"epoch": row["epoch"], **{f"{split}_{key}": row[split][key]
                                                          for split in ("train", "val") for key in ("loss", "accuracy")}})
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            payload = {"format_version": 1, "architecture": "frozen_resnet18_linear",
                       "config": config.to_dict(), "model_state": model.state_dict(),
                       "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                       "evidence_status": config.evidence_status, "val_metrics": val_metrics,
                       "run_metadata": metadata}
            temporary = output / "best.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "best.pt")
            write_json(output / "best_val_predictions.json", [
                {"image_path": str(row["path"]), "scene_id": row["scene_id"], **pred}
                for row, pred in zip(val.rows, predictions)])
            from .group_metrics import grouped_validation
            write_json(output / "best_val_groups.json", grouped_validation(val.rows, predictions, config))
            write_json(output / "best_metrics.json", {"epoch": epoch, "evidence_status": config.evidence_status,
                                                       "class_to_idx": config.class_to_idx, **val_metrics})
        print(f"{config.evidence_status} epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}", flush=True)
    write_json(output / "checkpoint_manifest.json", {"file": "best.pt", "sha256": file_sha256(output / "best.pt"),
                                                      "evidence_status": config.evidence_status})
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--allow-dev-only", action="store_true", help="explicitly run limited development training, never formal freeze")
    parser.add_argument("--check-only", action="store_true", help="check config/data/batch only; never initialize or train the CNN")
    args = parser.parse_args()
    try:
        config = Config.load(args.config)
        if args.check_only:
            _, _, summary = check_training_inputs(config)
            summary["device"] = str(choose_device(config.device))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            if not args.output:
                parser.error("--output is required for training")
            fit(config, args.output, allow_dev_only=args.allow_dev_only)
    except KeyboardInterrupt:
        parser.exit(130, "INTERRUPTED: completed best checkpoint is preserved if present; no automatic resume. Use a new output directory for a new run.\n")
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        prefix = "PRECHECK_FAILED" if args.check_only else "TRAIN_FAILED"
        parser.exit(2, f"{prefix}: {exc}\n")


if __name__ == "__main__":
    main()
