"""Feature preprocessing utilities."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def normalize(
    X: np.ndarray,
    stats: Dict[str, np.ndarray] | None = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Z-score normalization per feature.

    Args:
        X: (N, F) or (N, T, F) array
        stats: Pre-computed {"mean": ..., "std": ...} for inference
        eps: Small value to avoid division by zero

    Returns:
        (X_normalized, stats)
    """
    if X.ndim not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

    if len(X) == 0:
        return X.astype(np.float32), {"mean": np.zeros(X.shape[-1]), "std": np.ones(X.shape[-1])}

    axes = 0 if X.ndim == 2 else (0, 1)

    if stats is None:
        mean = X.mean(axis=axes, keepdims=True)
        std = X.std(axis=axes, keepdims=True)
        std = np.where(std < eps, eps, std)
        stats = {"mean": mean.squeeze(), "std": std.squeeze()}
    else:
        mean = np.asarray(stats["mean"])
        std = np.asarray(stats["std"])
        shape = (1, -1) if X.ndim == 2 else (1, 1, -1)
        mean = mean.reshape(shape)
        std = std.reshape(shape)

    return ((X - mean) / std).astype(np.float32), stats


def save_stats(stats: Dict[str, np.ndarray], path: Path | str) -> None:
    """Save normalization stats to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)


def load_stats(path: Path | str) -> Dict[str, np.ndarray]:
    """Load normalization stats from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
