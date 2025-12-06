"""
Project River - Neural Network Models

EEGNet-based models for Korean Jamo classification from EEG band power features.
"""

from .eegnet import EEGNet, build_default_eegnet

__all__ = ["EEGNet", "build_default_eegnet"]
