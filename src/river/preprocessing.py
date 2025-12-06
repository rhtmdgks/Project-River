"""
Project River Preprocessing Module

Signal processing utilities for EEG data preprocessing.

Implemented:
    - normalize_features(): Z-score and min-max normalization with stats save/load
    - compute_band_power(): Welch-based spectral power estimation

Deferred (Phase 3+):
    - bandpass_filter(): Butterworth band-pass filter
    - notch_filter(): IIR notch filter for power line removal
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


def normalize_features(
    X: np.ndarray,
    method: str = "zscore",
    stats: Optional[Dict[str, np.ndarray]] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize EEG features for model input.

    Supports 2D (N, F) and 3D (N, T, F) arrays. Normalization is computed
    per-feature (last axis) across all samples and time steps.

    Args:
        X: Feature array
           - 2D: shape (n_samples, n_features)
           - 3D: shape (n_samples, time_steps, n_features)
        method: Normalization method
                - "zscore": (x - mean) / std (default)
                - "minmax": scale to [0, 1] range
        stats: Pre-computed statistics for inference. If None, computed from X.
               For zscore: {"mean": ndarray, "std": ndarray}
               For minmax: {"min": ndarray, "max": ndarray}
        eps: Small value to prevent division by zero (default: 1e-6)

    Returns:
        Tuple of (X_normalized, stats):
            - X_normalized: Normalized array with same shape as input, dtype=float32
            - stats: Dictionary of normalization parameters for reuse

    Example:
        >>> # Training: compute stats
        >>> X_train = np.random.randn(100, 5, 20).astype(np.float32)
        >>> X_norm, stats = normalize_features(X_train, method="zscore")

        >>> # Save stats for later
        >>> save_normalization_stats(stats, "models/norm_stats.json")

        >>> # Inference: load and reuse stats
        >>> stats = load_normalization_stats("models/norm_stats.json")
        >>> X_test_norm, _ = normalize_features(X_test, stats=stats)
    """
    if X.ndim not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

    if len(X) == 0:
        # Return empty array with same shape
        empty_stats = {"mean": np.zeros(X.shape[-1]), "std": np.ones(X.shape[-1])}
        return X.astype(np.float32), empty_stats

    # Determine axes for computing statistics
    # For 2D (N, F): compute over axis 0 (samples)
    # For 3D (N, T, F): compute over axes (0, 1) (samples and time)
    if X.ndim == 2:
        reduce_axes: Union[int, Tuple[int, int]] = 0
    else:
        reduce_axes = (0, 1)

    if method == "zscore":
        if stats is None:
            mean = X.mean(axis=reduce_axes, keepdims=True)
            std = X.std(axis=reduce_axes, keepdims=True)
            std = np.where(std < eps, eps, std)
            stats = {"mean": mean.squeeze(), "std": std.squeeze()}
        else:
            mean = np.asarray(stats["mean"])
            std = np.asarray(stats["std"])
            # Reshape for broadcasting
            if X.ndim == 2:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            else:
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)

        X_norm = (X - mean) / std

    elif method == "minmax":
        if stats is None:
            x_min = X.min(axis=reduce_axes, keepdims=True)
            x_max = X.max(axis=reduce_axes, keepdims=True)
            x_range = x_max - x_min
            x_range = np.where(x_range < eps, eps, x_range)
            stats = {"min": x_min.squeeze(), "max": x_max.squeeze()}
        else:
            x_min = np.asarray(stats["min"])
            x_max = np.asarray(stats["max"])
            # Reshape for broadcasting
            if X.ndim == 2:
                x_min = x_min.reshape(1, -1)
                x_max = x_max.reshape(1, -1)
            else:
                x_min = x_min.reshape(1, 1, -1)
                x_max = x_max.reshape(1, 1, -1)
            x_range = x_max - x_min
            x_range = np.where(x_range < eps, eps, x_range)

        X_norm = (X - x_min) / x_range

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'minmax'.")

    return X_norm.astype(np.float32), stats


def save_normalization_stats(
    stats: Dict[str, np.ndarray],
    path: Union[str, Path],
) -> None:
    """
    Save normalization statistics to a JSON file.

    Args:
        stats: Dictionary containing normalization parameters
               (e.g., {"mean": ndarray, "std": ndarray})
        path: Output file path (should end with .json)

    Example:
        >>> X_norm, stats = normalize_features(X_train)
        >>> save_normalization_stats(stats, "models/norm_stats.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable = {k: v.tolist() for k, v in stats.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_normalization_stats(
    path: Union[str, Path],
) -> Dict[str, np.ndarray]:
    """
    Load normalization statistics from a JSON file.

    Args:
        path: Path to the JSON file containing stats

    Returns:
        Dictionary containing normalization parameters as numpy arrays

    Example:
        >>> stats = load_normalization_stats("models/norm_stats.json")
        >>> X_norm, _ = normalize_features(X_test, stats=stats)
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def compute_band_power(
    signal: np.ndarray,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    nperseg: Optional[int] = None,
) -> np.ndarray:
    """
    Compute spectral power in standard EEG frequency bands from raw signal.

    Uses Welch's method for power spectral density estimation.
    This function is for future use with raw EEG signals (not MindMonitor CSV).

    Args:
        signal: Raw EEG signal array
                - 1D: shape (n_samples,) for single channel
                - 2D: shape (n_samples, n_channels) for multi-channel
        fs: Sampling frequency in Hz (default: 256.0 for raw Muse EEG)
        bands: Dictionary mapping band names to (low, high) frequency tuples
               Default bands:
                 - Delta: 1-4 Hz
                 - Theta: 4-8 Hz
                 - Alpha: 8-13 Hz
                 - Beta: 13-30 Hz
                 - Gamma: 30-45 Hz
        nperseg: Length of each segment for Welch's method.
                 Default: min(256, signal_length)

    Returns:
        Band power array:
            - 1D input: shape (n_bands,)
            - 2D input: shape (n_bands, n_channels)

    Example:
        >>> # Single channel
        >>> raw = np.random.randn(512)
        >>> powers = compute_band_power(raw, fs=256.0)
        >>> print(powers.shape)  # (5,)

        >>> # Multi-channel (4 Muse channels)
        >>> raw = np.random.randn(512, 4)
        >>> powers = compute_band_power(raw, fs=256.0)
        >>> print(powers.shape)  # (5, 4)
    """
    from scipy.signal import welch

    # Default frequency bands
    if bands is None:
        bands = {
            "Delta": (1.0, 4.0),
            "Theta": (4.0, 8.0),
            "Alpha": (8.0, 13.0),
            "Beta": (13.0, 30.0),
            "Gamma": (30.0, 45.0),
        }

    # Handle 1D input
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    n_samples, n_channels = signal.shape

    # Set nperseg
    if nperseg is None:
        nperseg = min(256, n_samples)

    # Compute PSD using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, axis=0)

    # Compute band powers
    band_powers = []
    for band_name, (low, high) in bands.items():
        # Find frequency indices within band
        idx = np.logical_and(freqs >= low, freqs < high)
        # Integrate power in band (sum of PSD values * frequency resolution)
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        power = np.sum(psd[idx, :], axis=0) * freq_res
        band_powers.append(power)

    result = np.array(band_powers, dtype=np.float32)

    if squeeze_output:
        result = result.squeeze()

    return result


# ============================================================================
# Deferred implementations (Phase 3+)
# ============================================================================


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 256.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a band-pass filter to remove frequencies outside the specified range.

    Removes:
        - DC drift and slow artifacts (< lowcut Hz)
        - High-frequency noise and muscle artifacts (> highcut Hz)

    Args:
        signal: Input EEG signal (n_samples,) or (n_samples, n_channels)
        lowcut: Lower cutoff frequency in Hz (default: 0.5)
        highcut: Upper cutoff frequency in Hz (default: 45.0)
        fs: Sampling frequency in Hz (default: 256.0)
        order: Filter order (default: 4)

    Returns:
        Filtered signal with same shape as input

    Note:
        Implementation deferred to Phase 3.
        Current pipeline uses pre-computed band power from MindMonitor CSV.
    """
    # TODO: Implement using scipy.signal.butter and scipy.signal.filtfilt
    raise NotImplementedError(
        "bandpass_filter is deferred to Phase 3. "
        "Current pipeline uses pre-computed band power from CSV."
    )


def notch_filter(
    signal: np.ndarray,
    freq: float = 60.0,
    fs: float = 256.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply a notch filter to remove power line interference.

    Removes narrow-band noise at the specified frequency (50 Hz or 60 Hz
    depending on regional power grid).

    Args:
        signal: Input EEG signal (n_samples,) or (n_samples, n_channels)
        freq: Frequency to remove in Hz (default: 60.0 for Korea/US)
        fs: Sampling frequency in Hz (default: 256.0)
        quality_factor: Quality factor determining notch width (default: 30.0)
                        Higher values = narrower notch

    Returns:
        Filtered signal with same shape as input

    Note:
        Implementation deferred to Phase 3.
    """
    # TODO: Implement using scipy.signal.iirnotch and scipy.signal.filtfilt
    raise NotImplementedError("notch_filter is deferred to Phase 3.")
