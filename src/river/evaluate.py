"""
Project River - Evaluation Script

Evaluate trained EEGNet model and generate confusion matrix.

Usage:
    python -m src.river.evaluate
    python -m src.river.evaluate --checkpoint checkpoints/best_eegnet.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load trained model from checkpoint.

    Returns:
        Tuple of (model, config_dict)
    """
    from .models import EEGNet

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = EEGNet(
        num_classes=config["num_classes"],
        num_channels=config["num_features"],
        input_time=config["input_time"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evaluate model and collect predictions.

    Returns:
        Tuple of (y_true, y_pred, accuracy)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    accuracy = (y_true == y_pred).mean()

    return y_true, y_pred, accuracy


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: Path,
) -> None:
    """
    Generate and save confusion matrix heatmap.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(args: argparse.Namespace = None) -> None:
    """Main evaluation function."""
    if args is None:
        args = parse_args()

    from .config import default_config
    from .data_loader import discover_dataset
    from .datasets import EEGJamoDataset
    from .train import get_device

    print("=" * 60)
    print("Project River - Model Evaluation")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"\nDevice: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Run training first: python examples/train_eegnet.py")
        return

    print(f"\n[Loading Model]")
    print(f"  Checkpoint: {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    print(f"  Input time: {config['input_time']}")
    print(f"  Window size: {config['window_size']}s")

    # Load dataset
    print(f"\n[Loading Data]")
    pairs = discover_dataset()
    print(f"  Found {len(pairs)} CSV files")

    try:
        dataset = EEGJamoDataset(
            pairs,
            window_size=config["window_size"],
            stride=config["stride"],
            normalize=True,
        )
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Total samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluate
    print(f"\n[Evaluation]")
    y_true, y_pred, accuracy = evaluate_model(model, loader, device)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class accuracy
    print(f"\n[Per-Class Results]")
    for label in range(default_config.num_classes):
        jamo = default_config.get_jamo_from_label(label)
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == label).mean()
            print(f"  {jamo} (label={label}): {class_acc:.4f} ({mask.sum()} samples)")
        else:
            print(f"  {jamo} (label={label}): N/A (0 samples)")

    # Confusion matrix
    print(f"\n[Confusion Matrix]")
    output_path = Path(args.output)
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=default_config.jamo_classes,
        output_path=output_path,
    )
    print(f"  Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained EEGNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_eegnet.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/confusion_matrix.png",
        help="Output path for confusion matrix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
