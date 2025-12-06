"""
Project River - PyTorch Datasets

Dataset classes for EEG Jamo classification.
"""

from .eeg_dataset import EEGJamoDataset, train_val_split

__all__ = ["EEGJamoDataset", "train_val_split"]
