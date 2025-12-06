"""Training script for EEGNet."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import config
from .data_loader import discover_dataset
from .datasets import EEGDataset, train_val_split
from .models import build_model

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainArgs:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    window: float = 5.0
    stride: float = 2.0
    val_ratio: float = 0.2


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / len(loader), correct / total


def train(args: TrainArgs | None = None) -> Tuple[nn.Module, float]:
    if args is None:
        args = TrainArgs()

    set_seed(config.random_seed)
    device = get_device()

    print("=" * 50)
    print("Project River - Training")
    print("=" * 50)
    print(f"Device: {device}")

    # Data
    pairs = discover_dataset()
    print(f"Found {len(pairs)} CSV files")

    train_pairs, val_pairs = train_val_split(pairs, args.val_ratio)
    print(f"Train: {len(train_pairs)} files, Val: {len(val_pairs)} files")

    train_ds = EEGDataset(train_pairs, args.window, args.stride)
    val_ds = EEGDataset(val_pairs, args.window, args.stride)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)

    # Model
    model = build_model(train_ds.input_time).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_acc = 0.0
    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_classes": config.num_classes,
                    "num_features": config.num_features,
                    "input_time": train_ds.input_time,
                    "window_size": args.window,
                    "stride": args.stride,
                },
            }, ckpt_dir / "best_eegnet.pth")
            print(f"  -> Saved best model")

    print(f"\nBest accuracy: {best_acc:.4f}")
    print("=" * 50)
    return model, best_acc


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=2.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    train(TrainArgs(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window=args.window,
        stride=args.stride,
        val_ratio=args.val_ratio,
    ))


if __name__ == "__main__":
    main()
