"""Model evaluation script."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config
from .data_loader import discover_dataset
from .datasets import EEGDataset
from .models import EEGNet
from .train import get_device

log = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load model from checkpoint. Returns (model, config_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = EEGNet(
        num_classes=cfg["num_classes"],
        num_features=cfg["num_features"],
        input_time=cfg["input_time"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def evaluate(checkpoint_path: Path, output_path: Path | None = None):
    device = get_device()

    print("=" * 50)
    print("Project River - Evaluation")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model, cfg = load_model(checkpoint_path, device)

    # Load data
    pairs = discover_dataset()
    print(f"Found {len(pairs)} CSV files")

    dataset = EEGDataset(pairs, cfg["window_size"], cfg["stride"])
    loader = DataLoader(dataset, batch_size=64)
    print(f"Total samples: {len(dataset)}")

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    accuracy = (y_true == y_pred).mean()

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class results
    print("\nPer-class accuracy:")
    for label in range(config.num_classes):
        jamo = config.get_jamo(label)
        mask = y_true == label
        if mask.sum() > 0:
            acc = (y_pred[mask] == label).mean()
            print(f"  {jamo}: {acc:.4f} ({mask.sum()} samples)")

    # Confusion matrix
    if output_path:
        _plot_confusion_matrix(y_true, y_pred, output_path)
        print(f"\nConfusion matrix saved to {output_path}")

    print("=" * 50)
    return accuracy


def _plot_confusion_matrix(y_true, y_pred, output_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    labels = list(config.jamo_classes)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_eegnet.pth")
    parser.add_argument("--output", default="reports/confusion_matrix.png")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Run training first: python examples/train_eegnet.py")
        return

    evaluate(ckpt, Path(args.output))


if __name__ == "__main__":
    main()
