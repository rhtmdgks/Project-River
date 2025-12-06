#!/usr/bin/env python3
"""Inspect EEG data samples."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.river.config import config
from src.river.data_loader import discover_dataset, load_session
from src.river.preprocessing import normalize


def resolve_jamo(s: str) -> tuple:
    """Resolve input to (jamo, label). Accepts 'ㄱ' or '0'."""
    if s in config.jamo_to_label:
        return s, config.jamo_to_label[s]
    try:
        label = int(s)
        if 0 <= label < config.num_classes:
            return config.get_jamo(label), label
    except ValueError:
        pass
    raise ValueError(f"Invalid input: {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jamo", default="ㄱ", help="Jamo character or label index")
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=2.0)
    args = parser.parse_args()

    jamo, label = resolve_jamo(args.jamo)

    print("=" * 50)
    print("Project River - Data Inspection")
    print("=" * 50)
    print(f"Dataset: {config.dataset_dir}")
    print(f"Sampling rate: {config.sampling_rate} Hz")

    pairs = discover_dataset()
    print(f"Total files: {len(pairs)}")

    # Find target files
    target_files = [p for p, l in pairs if l == label]
    if not target_files:
        print(f"No files found for {jamo}")
        return

    sample_file = target_files[0]
    print(f"\nSample: {sample_file.name}")
    print(f"Target: {jamo} (label={label})")

    # Row-based
    X_rows, _ = load_session(sample_file, label)
    print(f"\nRow-based: {X_rows.shape}")

    # Windowed
    X_win, _ = load_session(sample_file, label, args.window, args.stride)
    min_samples = int(args.window * config.sampling_rate)

    print(f"\nWindowed (window={args.window}s, stride={args.stride}s):")
    print(f"  Required samples: {min_samples}")
    print(f"  Available: {len(X_rows)}")

    if len(X_win) == 0:
        print(f"  No windows generated (need {min_samples} samples)")
        return

    print(f"  Windows: {X_win.shape}")

    # Stats
    X_norm, stats = normalize(X_win)
    print(f"\nNormalized: mean={X_norm.mean():.6f}, std={X_norm.std():.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
