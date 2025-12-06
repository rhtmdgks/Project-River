"""Project configuration."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Config:
    """Central configuration for Project River."""

    dataset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    eeg_bands: tuple = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
    eeg_channels: tuple = ("TP9", "AF7", "AF8", "TP10")
    jamo_classes: tuple = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅇ", "ㅏ", "ㅓ", "ㅡ", "ㅣ")

    random_seed: int = 42
    sampling_rate: float = 1.0  # MindMonitor outputs band power at ~1 Hz
    window_size: float = 5.0
    stride: float = 2.0

    @cached_property
    def eeg_columns(self) -> tuple:
        return tuple(f"{b}_{c}" for b in self.eeg_bands for c in self.eeg_channels)

    @property
    def num_features(self) -> int:
        return len(self.eeg_bands) * len(self.eeg_channels)

    @property
    def num_classes(self) -> int:
        return len(self.jamo_classes)

    @cached_property
    def jamo_to_label(self) -> dict:
        return {j: i for i, j in enumerate(self.jamo_classes)}

    @cached_property
    def label_to_jamo(self) -> dict:
        return dict(enumerate(self.jamo_classes))

    def get_label(self, jamo: str) -> int:
        if jamo not in self.jamo_to_label:
            raise ValueError(f"Unknown jamo: {jamo}")
        return self.jamo_to_label[jamo]

    def get_jamo(self, label: int) -> str:
        if label not in self.label_to_jamo:
            raise ValueError(f"Invalid label: {label}")
        return self.label_to_jamo[label]


config = Config()
