"""
Project River Configuration

Centralized configuration for paths, constants, and hyperparameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Config:
    """
    Project-wide configuration settings.

    Attributes:
        raw_dir: Directory for future raw EEG signal storage
        processed_dir: Directory for preprocessed data (numpy, parquet)
        dataset_dir: Root directory containing Jamo subdirectories (ㄱ/, ㄴ/, ...)
        eeg_bands: List of EEG frequency band names
        eeg_channels: List of Muse 2 electrode positions
        jamo_classes: List of target Jamo characters (single source of truth)
        random_seed: Random seed for reproducibility
        default_sampling_rate: Default EEG sampling rate in Hz
    """

    # Data paths
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    dataset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")

    # EEG configuration
    eeg_bands: List[str] = field(
        default_factory=lambda: ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    )
    eeg_channels: List[str] = field(
        default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"]
    )

    # Classification targets (single source of truth for label mapping)
    jamo_classes: List[str] = field(
        default_factory=lambda: ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅇ", "ㅏ", "ㅓ", "ㅡ", "ㅣ"]
    )

    # Training constants
    random_seed: int = 42
    # MindMonitor outputs band power at ~1 Hz (not raw EEG at 256 Hz)
    default_sampling_rate: float = 1.0

    # Windowing defaults (in seconds, but effectively in samples at 1 Hz)
    default_window_size: float = 5.0  # 5 samples per window
    default_stride: float = 2.0  # 2 sample stride

    @property
    def eeg_columns(self) -> List[str]:
        """Generate list of EEG column names (band_channel format)."""
        return [f"{band}_{ch}" for band in self.eeg_bands for ch in self.eeg_channels]

    @property
    def num_features(self) -> int:
        """Total number of EEG features (bands × channels)."""
        return len(self.eeg_bands) * len(self.eeg_channels)

    @property
    def num_classes(self) -> int:
        """Number of classification targets."""
        return len(self.jamo_classes)

    @property
    def label_to_jamo(self) -> Dict[int, str]:
        """Mapping from integer label to Jamo character."""
        return {i: jamo for i, jamo in enumerate(self.jamo_classes)}

    @property
    def jamo_to_label(self) -> Dict[str, int]:
        """Mapping from Jamo character to integer label."""
        return {jamo: i for i, jamo in enumerate(self.jamo_classes)}

    def get_jamo_label(self, jamo: str) -> int:
        """Convert Jamo character to integer label."""
        if jamo not in self.jamo_to_label:
            raise ValueError(
                f"Unknown Jamo '{jamo}'. Valid: {self.jamo_classes}"
            )
        return self.jamo_to_label[jamo]

    def get_jamo_from_label(self, label: int) -> str:
        """Convert integer label to Jamo character."""
        if label not in self.label_to_jamo:
            raise ValueError(
                f"Invalid label {label}. Valid range: 0-{self.num_classes - 1}"
            )
        return self.label_to_jamo[label]


# Default configuration instance
default_config = Config()

# Convenience exports
EEG_BANDS = default_config.eeg_bands
EEG_CHANNELS = default_config.eeg_channels
EEG_COLS = default_config.eeg_columns
JAMO_CLASSES = default_config.jamo_classes
RAW_DIR = default_config.raw_dir
PROCESSED_DIR = default_config.processed_dir
DATASET_DIR = default_config.dataset_dir
RANDOM_SEED = default_config.random_seed
DEFAULT_SAMPLING_RATE = default_config.default_sampling_rate
