#!/usr/bin/env python3
"""
Project River - EEGNet Inference Example

Run inference on a single CSV file and predict the Jamo class.

Usage:
    python examples/infer_eegnet.py --file data/raw/ㄱ/mindMonitor_xxx.csv
    python examples/infer_eegnet.py --file data/raw/ㄴ/session.csv --checkpoint checkpoints/best_eegnet.pth
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run EEGNet inference on a single CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_eegnet.pth",
        help="Path to model checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    """Main inference function."""
    args = parse_args()

    from src.river.config import default_config
    from src.river.data_loader import load_session_csv
    from src.river.evaluate import load_model
    from src.river.preprocessing import normalize_features
    from src.river.train import get_device

    print("=" * 60)
    print("Project River - EEGNet Inference")
    print("=" * 60)

    # Check file exists
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"\nERROR: File not found: {input_file}")
        return

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Run training first: python examples/train_eegnet.py")
        return

    # Device
    device = get_device()
    print(f"\nDevice: {device}")

    # Load model
    print(f"\n[Loading Model]")
    print(f"  Checkpoint: {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)

    window_size = config["window_size"]
    stride = config["stride"]
    print(f"  Window: {window_size}s, Stride: {stride}s")

    # Load and preprocess input file
    print(f"\n[Loading Input]")
    print(f"  File: {input_file}")

    # Use dummy label (0) since we're just doing inference
    X, _ = load_session_csv(
        input_file,
        label=0,
        window_size=window_size,
        stride=stride,
    )

    if len(X) == 0:
        print(f"  ERROR: No windows generated from file")
        print(f"  File may be too short for window_size={window_size}s")
        return

    print(f"  Generated {len(X)} windows")
    print(f"  Shape: {X.shape}")

    # Normalize
    X_norm, _ = normalize_features(X, method="zscore")

    # Convert to tensor: (N, T, C) -> (N, 1, T, C)
    X_tensor = torch.from_numpy(X_norm).unsqueeze(1).float().to(device)

    # Inference
    print(f"\n[Inference]")
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Count predictions
    pred_counts = Counter(preds)
    most_common_label = pred_counts.most_common(1)[0][0]
    most_common_jamo = default_config.get_jamo_from_label(most_common_label)

    print(f"\n[Results]")
    print(f"  Predictions per window:")
    for label, count in sorted(pred_counts.items()):
        jamo = default_config.get_jamo_from_label(label)
        pct = count / len(preds) * 100
        print(f"    {jamo} (label={label}): {count} windows ({pct:.1f}%)")

    print(f"\n  ★ Final Prediction: {most_common_jamo} (label={most_common_label})")

    # Average confidence
    avg_probs = probs.mean(dim=0).cpu().numpy()
    top_idx = np.argmax(avg_probs)
    top_jamo = default_config.get_jamo_from_label(top_idx)
    print(f"  ★ By Avg Confidence: {top_jamo} ({avg_probs[top_idx]:.4f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
