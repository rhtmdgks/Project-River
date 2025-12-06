"""
Project River: EEG-based Korean Jamo Classification

Classifies Korean Jamo (ㄱ, ㄴ, ㄷ, ...) from Muse 2 EEG band power features.
"""

__version__ = "0.2.0"

from .config import config
from .data_loader import discover_dataset, load_session, load_sessions

__all__ = ["config", "discover_dataset", "load_session", "load_sessions"]
