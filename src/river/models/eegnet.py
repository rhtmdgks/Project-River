"""EEGNet model for band power classification."""

import torch
import torch.nn as nn

from ..config import config


class EEGNet(nn.Module):
    """
    Lightweight CNN for EEG band power classification.

    Input: (batch, 1, T, C) where T=time steps, C=features (20)
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 9,
        num_features: int = 20,
        input_time: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_features = num_features

        self.features = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, T, C) -> (B, C, T)
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.features(x)
        return self.classifier(x)


def build_model(input_time: int = 5) -> EEGNet:
    """Create EEGNet with default config."""
    return EEGNet(
        num_classes=config.num_classes,
        num_features=config.num_features,
        input_time=input_time,
    )
