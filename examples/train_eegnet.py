#!/usr/bin/env python3
"""Train EEGNet model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.river.train import main

if __name__ == "__main__":
    main()
