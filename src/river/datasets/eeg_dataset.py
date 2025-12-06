"""PyTorch Dataset for EEG data."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import config
from ..data_loader import load_sessions
from ..preprocessing import normalize


class EEGDataset(Dataset):
    """
    Dataset for windowed EEG band power features.

    Args:
        samples: List of (path, label) tuples
        window_size: Window size in seconds
        stride: Stride in seconds
        do_normalize: Apply z-score normalization
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        window_size: float = 5.0,
        stride: float = 2.0,
        do_normalize: bool = True,
    ):
        X, y = load_sessions(samples, window_size, stride)

        if len(X) == 0:
            raise ValueError("No windows generated from provided samples")

        self.stats = None
        if do_normalize:
            X, self.stats = normalize(X)

        self.window_size = window_size
        self.stride = stride
        self.input_time = X.shape[1]

        # (N, T, C) -> (N, 1, T, C) for conv input
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_val_split(
    samples: List[Tuple[Path, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Stratified file-level split.

    Ensures windows from the same file stay in the same split.
    """
    if not samples:
        raise ValueError("Empty samples list")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    from collections import defaultdict

    by_label = defaultdict(list)
    for item in samples:
        by_label[item[1]].append(item)

    rng = random.Random(seed)
    train, val = [], []

    for label, group in by_label.items():
        shuffled = group.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
