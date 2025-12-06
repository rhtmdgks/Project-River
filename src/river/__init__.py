"""
Project River: EEG-based Korean Jamo Classification

A research project for classifying Korean Jamo characters (ㄱ, ㄴ, ㄷ, ㄹ, ㅇ, ㅏ, ㅓ, ㅡ, ㅣ)
using band-power features from 4-channel EEG (Muse 2) and lightweight CNN (EEGNet).
"""

__version__ = "0.1.0"
__author__ = "Project River Team"

from .config import Config
from .data_loader import load_session_csv, load_multi_sessions

__all__ = [
    "Config",
    "load_session_csv",
    "load_multi_sessions",
]
