"""Export losses, accuracy and best validation confusion matrix to standalone PNG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    directory = Path(args.run_dir)
    history = json.loads((directory / "history.json").read_text())
    best = json.loads((directory / "best_metrics.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for split in ("train", "val"):
        for axis, key in zip(axes[:2], ("loss", "accuracy")):
            axis.plot([r["epoch"] for r in history], [r[split][key] for r in history], label=split, marker="o")
            axis.set(xlabel="Epoch", ylabel=key)
            axis.legend()
    matrix = best["confusion_matrix"]
    axes[2].imshow(matrix, cmap="Blues")
    labels = [label for label, _ in sorted(best["class_to_idx"].items(), key=lambda item: item[1])]
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
                xlabel="Predicted", ylabel="True", title=f"Best val epoch {best['epoch']}")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, matrix[i][j], ha="center", va="center")
    fig.suptitle(best["evidence_status"])
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
