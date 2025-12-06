#!/usr/bin/env python3
"""Run inference on a single CSV file."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.river.config import config
from src.river.data_loader import load_session
from src.river.evaluate import load_model
from src.river.preprocessing import normalize
from src.river.train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--checkpoint", default="checkpoints/best_eegnet.pth")
    args = parser.parse_args()

    input_file = Path(args.file)
    checkpoint = Path(args.checkpoint)

    if not input_file.exists():
        print(f"File not found: {input_file}")
        return
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return

    device = get_device()
    print(f"Device: {device}")

    model, cfg = load_model(checkpoint, device)
    print(f"Loaded model (window={cfg['window_size']}s)")

    # Load and preprocess
    X, _ = load_session(input_file, label=0, window_size=cfg["window_size"], stride=cfg["stride"])

    if len(X) == 0:
        print(f"No windows generated (file too short for {cfg['window_size']}s window)")
        return

    print(f"Generated {len(X)} windows")

    X, _ = normalize(X)
    X_tensor = torch.from_numpy(X).unsqueeze(1).float().to(device)

    # Inference
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).cpu().numpy()

    # Results
    counts = Counter(preds)
    print("\nPredictions:")
    for label, count in sorted(counts.items()):
        jamo = config.get_jamo(label)
        print(f"  {jamo}: {count} windows ({count/len(preds)*100:.1f}%)")

    # Final prediction
    top_label = counts.most_common(1)[0][0]
    top_jamo = config.get_jamo(top_label)
    avg_probs = probs.mean(0).cpu().numpy()
    conf_label = avg_probs.argmax()
    conf_jamo = config.get_jamo(conf_label)

    print(f"\nPrediction (majority): {top_jamo}")
    print(f"Prediction (confidence): {conf_jamo} ({avg_probs[conf_label]:.3f})")


if __name__ == "__main__":
    main()
