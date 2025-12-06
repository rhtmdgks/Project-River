"""
Project River - EEG Jamo Dataset

PyTorch Dataset for loading windowed EEG band power features.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGJamoDataset(Dataset):
    """
    PyTorch Dataset for EEG Jamo classification.

    Loads CSV files, applies windowing, and returns tensors ready for EEGNet.

    Args:
        samples: List of (file_path, label) tuples from discover_dataset()
        window_size: Window size in seconds (default: 5.0)
        stride: Stride in seconds (default: 2.0)
        normalize: Whether to apply z-score normalization (default: True)
        sampling_rate: Sampling rate in Hz (default: None, uses config)

    Output shapes:
        X[i]: (1, T, C) - single channel, T time steps, C features
        y[i]: scalar int64 label
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], Union[int, str]]],
        window_size: float = 5.0,
        stride: float = 2.0,
        normalize: bool = True,
        sampling_rate: Optional[float] = None,
    ):
        from ..data_loader import load_multi_sessions
        from ..preprocessing import normalize_features

        if not samples:
            raise ValueError("samples list cannot be empty")

        # Load all data with windowing
        X, y = load_multi_sessions(
            samples,
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
        )

        if len(X) == 0:
            raise ValueError(
                "No windows generated. Check window_size and data availability."
            )

        # Normalize if requested
        self.stats = None
        if normalize:
            X, self.stats = normalize_features(X, method="zscore")

        # Store metadata
        self.window_size = window_size
        self.stride = stride
        self.input_time = X.shape[1]
        self.num_features = X.shape[2]

        # Convert to PyTorch tensors
        # X: (N, T, C) -> (N, 1, T, C) for Conv2d
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (X, y):
                - X: tensor of shape (1, T, C)
                - y: scalar tensor (label)
        """
        return self.X[idx], self.y[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.

        Returns:
            Tensor of shape (num_classes,) with inverse frequency weights
        """
        from ..config import default_config

        counts = torch.bincount(self.y, minlength=default_config.num_classes)
        # Avoid division by zero
        counts = counts.float().clamp(min=1)
        weights = 1.0 / counts
        # Normalize
        weights = weights / weights.sum() * len(counts)
        return weights


def train_val_split(
    samples: List[Tuple[Path, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split samples into train and validation sets at file level.

    Ensures that windows from the same CSV file don't appear in both
    train and validation sets.

    Args:
        samples: List of (file_path, label) tuples
        val_ratio: Fraction of files for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_samples, val_samples)

    Example:
        >>> pairs = discover_dataset()
        >>> train_pairs, val_pairs = train_val_split(pairs, val_ratio=0.2)
        >>> train_dataset = EEGJamoDataset(train_pairs)
        >>> val_dataset = EEGJamoDataset(val_pairs)
    """
    if not samples:
        raise ValueError("samples list cannot be empty")

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    # Group by label for stratified split
    from collections import defaultdict
    label_groups: dict = defaultdict(list)
    for path, label in samples:
        label_groups[label].append((path, label))

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    rng = random.Random(seed)

    for label, group in label_groups.items():
        # Shuffle within each class
        shuffled = group.copy()
        rng.shuffle(shuffled)

        # Split
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    # Shuffle final lists
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples
