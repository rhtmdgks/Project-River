#!/usr/bin/env python3
"""
Project River - Data Inspection Script

Quick inspection of EEG data quality and preprocessing pipeline.
Loads a sample CSV, applies windowing, and displays statistics.

Usage:
    python examples/inspect_sample.py
    python examples/inspect_sample.py --jamo ㄴ --window 5.0
    python examples/inspect_sample.py --jamo 0 --window 5.0 --stride 2.0
    python examples/inspect_sample.py --jamo ㄱ --visualize
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.river.config import default_config
from src.river.data_loader import (
    discover_dataset,
    get_min_samples_for_window,
    load_session_csv,
)
from src.river.preprocessing import normalize_features


def resolve_jamo_input(jamo_input: str) -> tuple:
    """
    Resolve jamo input to (jamo_char, label_int).

    Accepts:
        - Jamo character: 'ㄱ', 'ㄴ', etc.
        - Numeric string: '0', '1', etc.
        - Integer as string: '0' -> label 0 -> 'ㄱ'

    Returns:
        Tuple of (jamo_character, integer_label)
    """
    # Try as Jamo character
    if jamo_input in default_config.jamo_to_label:
        label = default_config.jamo_to_label[jamo_input]
        return jamo_input, label

    # Try as numeric string
    try:
        label = int(jamo_input)
        if 0 <= label < default_config.num_classes:
            jamo = default_config.get_jamo_from_label(label)
            return jamo, label
    except ValueError:
        pass

    raise ValueError(
        f"Invalid jamo input '{jamo_input}'. "
        f"Use Jamo ({default_config.jamo_classes}) or int (0-{default_config.num_classes - 1})"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect EEG data samples from Project River dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/inspect_sample.py --jamo ㄱ
    python examples/inspect_sample.py --jamo 0 --window 5.0
    python examples/inspect_sample.py --jamo ㄴ --window 5.0 --stride 2.0 --visualize
        """,
    )
    parser.add_argument(
        "--jamo",
        type=str,
        default="ㄱ",
        help="Jamo character ('ㄱ'-'ㅣ') or label index ('0'-'8') (default: ㄱ)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=2.0,
        help="Stride in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization of first window (requires matplotlib)",
    )
    return parser.parse_args()


def main() -> None:
    """Main inspection routine."""
    args = parse_args()

    print("=" * 60)
    print("Project River - Data Inspection")
    print("=" * 60)

    # Resolve jamo input
    try:
        jamo_char, target_label = resolve_jamo_input(args.jamo)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return

    # Config info
    print(f"\n[Config]")
    print(f"  Dataset dir: {default_config.dataset_dir}")
    print(f"  Sampling rate: {default_config.default_sampling_rate} Hz")
    print(f"  Jamo classes: {default_config.jamo_classes}")
    print(f"  Num features: {default_config.num_features}")

    # Discover dataset
    pairs = discover_dataset()
    print(f"\n[Dataset Discovery]")
    print(f"  Total CSV files: {len(pairs)}")

    if not pairs:
        print("  WARNING: No CSV files found!")
        print(f"  Check that data exists in: {default_config.dataset_dir}")
        return

    # Count per class
    class_counts: dict = {}
    for _, label in pairs:
        jamo = default_config.get_jamo_from_label(label)
        class_counts[jamo] = class_counts.get(jamo, 0) + 1
    print("  Files per class:")
    for jamo in default_config.jamo_classes:
        count = class_counts.get(jamo, 0)
        print(f"    {jamo}: {count} files")

    # Find sample file for requested Jamo
    target_files = [p for p, lbl in pairs if lbl == target_label]

    if not target_files:
        print(f"\n  ERROR: No files found for Jamo '{jamo_char}' (label={target_label})")
        return

    sample_file = target_files[0]
    print(f"\n[Sample File]")
    print(f"  Target: {jamo_char} (label={target_label})")
    print(f"  Path: {sample_file}")

    # Load without windowing (row-based)
    print(f"\n[Row-based Loading]")
    X_rows, y_rows = load_session_csv(sample_file, label=target_label)
    print(f"  X shape: {X_rows.shape}")
    print(f"  y shape: {y_rows.shape}")
    print(f"  X dtype: {X_rows.dtype}")
    print(f"  y dtype: {y_rows.dtype}")
    print(f"  Available samples: {len(X_rows)}")

    # Calculate minimum samples needed
    min_samples = get_min_samples_for_window(args.window)
    print(f"\n[Window Requirements]")
    print(f"  Window size: {args.window}s = {min_samples} samples (at {default_config.default_sampling_rate} Hz)")
    print(f"  Stride: {args.stride}s = {int(args.stride * default_config.default_sampling_rate)} samples")
    print(f"  Minimum samples needed: {min_samples}")
    print(f"  Available samples: {len(X_rows)}")

    # Load with windowing
    print(f"\n[Windowed Loading]")
    X_win, y_win = load_session_csv(
        sample_file,
        label=jamo_char,  # Test string label conversion
        window_size=args.window,
        stride=args.stride,
    )
    print(f"  X shape: {X_win.shape}")
    print(f"  y shape: {y_win.shape}")

    if len(X_win) == 0:
        print(f"\n  WARNING: No windows generated!")
        print(f"  File has {len(X_rows)} samples, but needs at least {min_samples} samples")
        print(f"  Suggestion: Use --window {max(1, len(X_rows) - 1)}.0 or collect more data")
        return

    window_len = X_win.shape[1]
    print(f"  Window length: {window_len} samples")
    print(f"  Num windows: {len(X_win)}")

    # Statistics before normalization
    print(f"\n[Raw Statistics (first window)]")
    first_window = X_win[0]
    print(f"  Mean per feature: {first_window.mean(axis=0)[:5]}... (first 5)")
    print(f"  Std per feature:  {first_window.std(axis=0)[:5]}... (first 5)")
    print(f"  Min: {first_window.min():.4f}, Max: {first_window.max():.4f}")

    # Normalize
    print(f"\n[Normalization (z-score)]")
    X_norm, stats = normalize_features(X_win, method="zscore")
    print(f"  Normalized shape: {X_norm.shape}")
    print(f"  Stats keys: {list(stats.keys())}")
    print(f"  Post-norm mean: {X_norm.mean():.6f}")
    print(f"  Post-norm std:  {X_norm.std():.6f}")

    # Visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            # Generate feature labels: Band_Channel format
            feature_labels = []
            for band in default_config.eeg_bands:
                for ch in default_config.eeg_channels:
                    feature_labels.append(f"{band[:2]}_{ch}")

            fig, axes = plt.subplots(2, 1, figsize=(14, 8))

            # Raw first window
            ax1 = axes[0]
            im1 = ax1.imshow(first_window.T, aspect="auto", cmap="viridis")
            ax1.set_title(f"Raw Window (Jamo: {jamo_char}, label={target_label})")
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Feature (Band_Channel)")

            # Set y-ticks for bands (every 4 features = 1 band)
            band_positions = [i * 4 + 1.5 for i in range(5)]
            ax1.set_yticks(band_positions)
            ax1.set_yticklabels(default_config.eeg_bands)
            plt.colorbar(im1, ax=ax1, label="Band Power")

            # Normalized first window
            ax2 = axes[1]
            im2 = ax2.imshow(X_norm[0].T, aspect="auto", cmap="viridis")
            ax2.set_title("Normalized Window (z-score)")
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Feature (Band_Channel)")
            ax2.set_yticks(band_positions)
            ax2.set_yticklabels(default_config.eeg_bands)
            plt.colorbar(im2, ax=ax2, label="Normalized Value")

            plt.tight_layout()

            output_path = Path("examples/sample_inspection.png")
            plt.savefig(output_path, dpi=150)
            print(f"\n[Visualization]")
            print(f"  Saved to: {output_path}")
            plt.show()

        except ImportError:
            print("\n[Visualization]")
            print("  Skipped (matplotlib not installed)")
            print("  Install with: pip install matplotlib")

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
