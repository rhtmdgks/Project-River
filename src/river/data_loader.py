"""Data loading utilities for EEG CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import config

log = logging.getLogger(__name__)

FileLabelPair = Tuple[Path, int]


def resolve_label(label: int | str) -> int:
    """Convert label (int or jamo string) to integer."""
    if isinstance(label, int):
        if not 0 <= label < config.num_classes:
            raise ValueError(f"Label {label} out of range [0, {config.num_classes})")
        return label

    if isinstance(label, str):
        if label in config.jamo_to_label:
            return config.jamo_to_label[label]
        # Try numeric string
        try:
            return resolve_label(int(label))
        except ValueError:
            pass
        raise ValueError(f"Invalid label: {label}")

    raise TypeError(f"Label must be int or str, got {type(label)}")


def load_session(
    path: Path | str,
    label: int | str,
    window_size: float | None = None,
    stride: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CSV session.

    Returns:
        (X, y) where X is (N, 20) for row-based or (N, T, 20) for windowed.
    """
    label = resolve_label(label)
    path = Path(path)

    df = pd.read_csv(path)

    # Validate columns
    missing = set(config.eeg_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    # Filter by HeadBandOn if present
    if "HeadBandOn" in df.columns:
        df = df[df["HeadBandOn"] == 1.0]

    data = df[list(config.eeg_columns)].dropna().values.astype(np.float32)
    n_samples = len(data)

    if n_samples == 0:
        return _empty_result(window_size)

    # Row-based mode
    if window_size is None:
        return data, np.full(n_samples, label, dtype=np.int64)

    # Windowed mode
    if stride is None:
        stride = window_size / 2

    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size and stride must be positive")

    win_len = int(window_size * config.sampling_rate)
    stride_len = max(1, int(stride * config.sampling_rate))

    if win_len > n_samples:
        return _empty_result(window_size)

    windows = []
    i = 0
    while i + win_len <= n_samples:
        windows.append(data[i : i + win_len])
        i += stride_len

    X = np.stack(windows)
    y = np.full(len(windows), label, dtype=np.int64)
    return X, y


def _empty_result(window_size: float | None) -> Tuple[np.ndarray, np.ndarray]:
    if window_size is not None:
        win_len = int(window_size * config.sampling_rate)
        return np.empty((0, win_len, config.num_features), np.float32), np.empty(0, np.int64)
    return np.empty((0, config.num_features), np.float32), np.empty(0, np.int64)


def load_sessions(
    pairs: List[Tuple[Path | str, int | str]],
    window_size: float | None = None,
    stride: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load multiple sessions and concatenate."""
    if not pairs:
        raise ValueError("No file-label pairs provided")

    X_all, y_all = [], []
    for path, label in pairs:
        try:
            X, y = load_session(path, label, window_size, stride)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")

    if not X_all:
        return _empty_result(window_size)

    return np.vstack(X_all), np.concatenate(y_all)


def discover_dataset(dataset_dir: Path | str | None = None) -> List[FileLabelPair]:
    """Find all CSV files organized by jamo subdirectories."""
    if dataset_dir is None:
        dataset_dir = config.dataset_dir
    dataset_dir = Path(dataset_dir)

    pairs = []
    for jamo, label in config.jamo_to_label.items():
        jamo_dir = dataset_dir / jamo
        if jamo_dir.exists():
            for csv in sorted(jamo_dir.glob("*.csv")):
                pairs.append((csv, label))

    return pairs
