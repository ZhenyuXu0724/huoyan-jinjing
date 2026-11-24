"""Run member06's frozen ResNet18 and epoch implementation on the v4 manifests."""
import argparse
import json
from pathlib import Path
import sys

from verify_dataset import ROOT, file_digest, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True, help="checkout containing member06/resnet-baseline")
    parser.add_argument("--output", type=Path, help="new run directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("training requires --output")
    if not (args.code_root/"members/member06_research/train.py").is_file():
        parser.error("--code-root must contain the member06 training implementation")
    verify()
    sys.path.insert(0, str(args.code_root.resolve()))
    import torch
    from members.member06_research.config import Config
    from members.member06_research.data import ManifestDataset, check_disjoint
    from members.member06_research.model import FrozenResNet18
    from members.member06_research.runtime import choose_device, seed_everything
    from members.member06_research.train import make_loader, run_epoch

    config = Config(train_manifest=str(ROOT/"manifests/resnet/train.csv"),
                    val_manifest=str(ROOT/"manifests/resnet/val.csv"),
                    device=args.device, epochs=args.epochs)
    train = ManifestDataset(config.train_manifest, config, "train")
    val = ManifestDataset(config.val_manifest, config, "val")
    check_disjoint(train, val)
    features, targets = next(iter(make_loader(val, config)))
    if features.shape[1:] != (3,config.image_size,config.image_size) or targets.dtype != torch.int64:
        raise ValueError("Invalid ResNet input batch")
    if args.check_only:
        print(f"ResNet input verified: {len(train)} train, {len(val)} val.")
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed, config.cpu_threads)
    device = choose_device(config.device)
    model = FrozenResNet18(config.weights).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(train, config, True)
    val_loader = make_loader(val, config)
    source = {"dataset":"final_training_v4", "test_data_used":False,
              "selection":"minimum validation cross-entropy; earliest epoch wins ties",
              "manifests":{s:file_digest(ROOT/f"manifests/resnet/{s}.csv") for s in ("train","val")}}
    history = []
    best = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, config.class_to_idx["FAIL"], optimizer)
        val_metrics, _ = run_epoch(model, val_loader, device, config.class_to_idx["FAIL"])
        history.append({"epoch":epoch,"train":train_metrics,"val":val_metrics})
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            temporary = output/"best.tmp"
            torch.save({"format_version":1,"architecture":"frozen_resnet18_linear","epoch":epoch,
                        "config":config.to_dict(),"model_state":model.state_dict(),"val_metrics":val_metrics,
                        "evidence_status":config.evidence_status,"run_metadata":source}, temporary)
            temporary.replace(output/"best.pt")
        (output/"history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
        print(f"Epoch {epoch}: validation loss {val_metrics['loss']:.5f}")


if __name__ == "__main__":
    main()
