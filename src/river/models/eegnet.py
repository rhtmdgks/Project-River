"""
Project River - EEGNet Model

Simplified EEGNet architecture for EEG-based Jamo classification.
Adapted for band power features (not raw EEG).

Input shape: (batch, 1, T, C)
    - T: time steps (window length, e.g., 5 at 1 Hz)
    - C: features (bands × channels = 20)
Output: (batch, num_classes) logits
"""

import torch
import torch.nn as nn
from typing import Optional


class EEGNet(nn.Module):
    """
    Simplified EEGNet for EEG band power classification.

    This is a lightweight version adapted for:
    - Small temporal windows (T=5 at 1 Hz sampling)
    - Band power features (20 features = 5 bands × 4 channels)

    Architecture:
        1. Feature extraction with 1D convolutions over time
        2. Fully connected classification head

    Args:
        num_classes: Number of output classes (default: 9 for Jamo)
        num_channels: Number of input features (default: 20)
        input_time: Number of time steps in input window (default: 5)
        dropout_rate: Dropout probability (default: 0.5)
        hidden_dim: Hidden layer dimension (default: 64)
    """

    def __init__(
        self,
        num_classes: int = 9,
        num_channels: int = 20,
        input_time: int = 5,
        dropout_rate: float = 0.5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_time = input_time

        # Feature extraction
        # Input: (B, 1, T, C) -> treat as (B, C, T) after reshape
        self.features = nn.Sequential(
            # Temporal convolution across time
            nn.Conv1d(num_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            # Second conv layer
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Dropout(dropout_rate),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, T, C)
               - T: time steps
               - C: features (num_channels)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (B, 1, T, C), got {x.dim()}D"
        assert x.size(1) == 1, f"Expected 1 input channel, got {x.size(1)}"

        batch_size = x.size(0)
        T = x.size(2)
        C = x.size(3)

        # Reshape: (B, 1, T, C) -> (B, C, T)
        x = x.squeeze(1)  # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)

        # Feature extraction
        x = self.features(x)  # (B, hidden_dim, 1)

        # Classification
        x = self.classifier(x)  # (B, num_classes)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.

        Args:
            x: Input tensor of shape (batch, 1, T, C)

        Returns:
            Predicted class indices of shape (batch,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class EEGNetOriginal(nn.Module):
    """
    Original EEGNet architecture (for reference, requires larger input).

    Use this when input_time >= 64 (e.g., raw EEG at 256 Hz).
    For band power at 1 Hz with small windows, use EEGNet instead.
    """

    def __init__(
        self,
        num_classes: int = 9,
        num_channels: int = 20,
        input_time: int = 64,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_time = input_time

        # Block 1: Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise Convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        # Block 3: Separable Convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 3), padding=(0, 1), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate),
        )

        # Calculate flattened size
        self._flat_size = self._get_flat_size()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, num_classes),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_time, self.num_channels)
            x = self.temporal_conv(x)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


def build_default_eegnet(
    config: Optional[object] = None,
    input_time: int = 5,
    dropout_rate: float = 0.5,
    hidden_dim: int = 64,
) -> EEGNet:
    """
    Build EEGNet with default configuration from config module.

    Args:
        config: Config object with num_classes and num_features.
                If None, uses default_config.
        input_time: Number of time steps in input window
        dropout_rate: Dropout probability
        hidden_dim: Hidden layer dimension

    Returns:
        Configured EEGNet instance

    Example:
        >>> from src.river.config import default_config
        >>> model = build_default_eegnet(default_config, input_time=5)
        >>> print(model)
    """
    if config is None:
        from ..config import default_config
        config = default_config

    return EEGNet(
        num_classes=config.num_classes,
        num_channels=config.num_features,
        input_time=input_time,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
    )
