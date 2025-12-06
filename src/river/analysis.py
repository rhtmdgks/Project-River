"""Training analysis and visualization utilities."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


@dataclass
class TrainingLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float


@dataclass
class EvaluationResult:
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    num_samples: int
    num_samples_per_class: Dict[str, int]
    confusion_matrix: List[List[int]]
    best_class: str
    worst_class: str


def parse_training_log(log_text: str) -> List[TrainingLog]:
    """Parse training output into structured logs."""
    pattern = r"Epoch\s+(\d+)\s+\|\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)"
    logs = []
    for match in re.finditer(pattern, log_text):
        logs.append(TrainingLog(
            epoch=int(match.group(1)),
            train_loss=float(match.group(2)),
            val_loss=float(match.group(3)),
            val_acc=float(match.group(4)),
        ))
    return logs


def save_training_csv(logs: List[TrainingLog], path: Path) -> None:
    """Save training logs to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy\n")
        for log in logs:
            f.write(f"{log.epoch},{log.train_loss:.4f},{log.val_loss:.4f},{log.val_acc:.4f}\n")


def get_best_epoch(logs: List[TrainingLog]) -> TrainingLog:
    """Find epoch with best validation accuracy."""
    return max(logs, key=lambda x: x.val_acc)


def plot_loss_curve(logs: List[TrainingLog], output_path: Path) -> None:
    """Generate loss curve plot."""
    import matplotlib.pyplot as plt

    epochs = [l.epoch for l in logs]
    train_loss = [l.train_loss for l in logs]
    val_loss = [l.val_loss for l in logs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_curve(logs: List[TrainingLog], output_path: Path) -> None:
    """Generate accuracy curve plot."""
    import matplotlib.pyplot as plt

    epochs = [l.epoch for l in logs]
    val_acc = [l.val_acc for l in logs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_acc, label="Val Accuracy", marker="o", color="green")
    ax.axhline(y=1/9, color="red", linestyle="--", label="Random Baseline (11.1%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_class_distribution(output_path: Path) -> None:
    """Generate class distribution bar chart."""
    import matplotlib.pyplot as plt
    from .config import config
    from .data_loader import discover_dataset

    pairs = discover_dataset()
    counts = {j: 0 for j in config.jamo_classes}
    for _, label in pairs:
        jamo = config.get_jamo(label)
        counts[jamo] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.keys(), counts.values(), color="steelblue")
    ax.set_xlabel("Jamo Class")
    ax.set_ylabel("Number of Files")
    ax.set_title("Class Distribution (Files per Jamo)")

    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_evaluation(checkpoint_path: Path) -> EvaluationResult:
    """Run evaluation and return structured results."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix

    from .config import config
    from .data_loader import discover_dataset
    from .datasets import EEGDataset
    from .evaluate import load_model, get_device

    device = get_device()
    model, cfg = load_model(checkpoint_path, device)

    pairs = discover_dataset()
    dataset = EEGDataset(pairs, cfg["window_size"], cfg["stride"])
    loader = DataLoader(dataset, batch_size=64)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Per-class accuracy
    per_class_acc = {}
    per_class_samples = {}
    for label in range(config.num_classes):
        jamo = config.get_jamo(label)
        mask = y_true == label
        per_class_samples[jamo] = int(mask.sum())
        if mask.sum() > 0:
            per_class_acc[jamo] = float((y_pred[mask] == label).mean())
        else:
            per_class_acc[jamo] = 0.0

    # Best/worst class
    best_class = max(per_class_acc, key=per_class_acc.get)
    worst_class = min(per_class_acc, key=per_class_acc.get)

    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvaluationResult(
        overall_accuracy=float((y_true == y_pred).mean()),
        per_class_accuracy=per_class_acc,
        num_samples=len(y_true),
        num_samples_per_class=per_class_samples,
        confusion_matrix=cm,
        best_class=best_class,
        worst_class=worst_class,
    )


def save_evaluation_json(result: EvaluationResult, path: Path) -> None:
    """Save evaluation results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)


def generate_all_figures(logs: List[TrainingLog], output_dir: Path) -> None:
    """Generate all analysis figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(logs, figures_dir / "loss_curve.png")
    plot_accuracy_curve(logs, figures_dir / "accuracy_curve.png")
    plot_class_distribution(figures_dir / "class_distribution.png")

    print(f"Generated figures in {figures_dir}")
