"""
Project River - Training Script

End-to-end training pipeline for EEGNet Jamo classifier.

Usage:
    python -m src.river.train --epochs 50 --window 5.0 --stride 2.0
    python examples/train_eegnet.py --epochs 50 --batch-size 32
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    """Get the best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def main(args: argparse.Namespace = None) -> None:
    """Main training function."""
    if args is None:
        args = parse_args()

    # Imports
    from .config import default_config, RANDOM_SEED
    from .data_loader import discover_dataset
    from .datasets import EEGJamoDataset, train_val_split
    from .models import build_default_eegnet

    # Set random seed
    torch.manual_seed(RANDOM_SEED)

    print("=" * 60)
    print("Project River - EEGNet Training")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"\nDevice: {device}")

    # Discover dataset
    print(f"\n[Data Loading]")
    pairs = discover_dataset()
    print(f"  Found {len(pairs)} CSV files")

    if len(pairs) < 2:
        print("  ERROR: Need at least 2 files for train/val split")
        return

    # Train/val split (file-level)
    train_pairs, val_pairs = train_val_split(pairs, val_ratio=args.val_ratio)
    print(f"  Train files: {len(train_pairs)}")
    print(f"  Val files: {len(val_pairs)}")

    # Create datasets
    print(f"\n[Dataset Creation]")
    print(f"  Window: {args.window}s, Stride: {args.stride}s")

    try:
        train_dataset = EEGJamoDataset(
            train_pairs,
            window_size=args.window,
            stride=args.stride,
            normalize=True,
        )
        val_dataset = EEGJamoDataset(
            val_pairs,
            window_size=args.window,
            stride=args.stride,
            normalize=True,
        )
    except ValueError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Input shape: {train_dataset.X.shape[1:]}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model
    print(f"\n[Model]")
    input_time = train_dataset.input_time
    model = build_default_eegnet(default_config, input_time=input_time)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: EEGNet")
    print(f"  Input: (1, {input_time}, {default_config.num_features})")
    print(f"  Output: {default_config.num_classes} classes")
    print(f"  Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n[Training]")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    best_val_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Log
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_eegnet.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": {
                        "num_classes": default_config.num_classes,
                        "num_features": default_config.num_features,
                        "input_time": input_time,
                        "window_size": args.window,
                        "stride": args.stride,
                    },
                },
                checkpoint_path,
            )
            print(f"    → Saved best model (acc={val_acc:.4f})")

    print(f"\n[Training Complete]")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint: checkpoints/best_eegnet.pth")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EEGNet for Jamo classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--window", type=float, default=5.0, help="Window size in seconds"
    )
    parser.add_argument(
        "--stride", type=float, default=2.0, help="Stride in seconds"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation set ratio"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
