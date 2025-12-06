#!/usr/bin/env python3
"""
Project River - EEGNet Training Example

Train EEGNet model for Korean Jamo classification from EEG band power features.

Usage:
    python examples/train_eegnet.py
    python examples/train_eegnet.py --epochs 50 --window 5.0 --stride 2.0
    python examples/train_eegnet.py --epochs 100 --batch-size 32 --lr 0.001

Arguments:
    --epochs      Number of training epochs (default: 50)
    --batch-size  Batch size (default: 64)
    --lr          Learning rate (default: 0.001)
    --window      Window size in seconds (default: 5.0)
    --stride      Stride in seconds (default: 2.0)
    --val-ratio   Validation set ratio (default: 0.2)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.river.train import main, parse_args

if __name__ == "__main__":
    args = parse_args()
    main(args)
