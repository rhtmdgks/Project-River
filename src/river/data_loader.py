"""
Project River Data Loader

Functions for loading and combining EEG session data from MindMonitor/Muse CSV exports.
Supports windowing for temporal feature extraction.

Key Features:
    - Row-based loading: Each CSV row becomes a sample
    - Windowed loading: Sliding window creates (window_len, features) samples
    - Automatic label conversion: String Jamo → int via config
    - HeadBandOn filtering: Removes samples with poor electrode contact
    - NaN handling: Removes rows with missing EEG values before windowing

Unit Conversion:
    - window_size: Input in seconds, converted to samples via sampling_rate
    - stride: Input in seconds, converted to samples via sampling_rate
    - At 1 Hz (MindMonitor band power output), 1 second = 1 sample
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _get_eeg_cols() -> List[str]:
    """Get EEG column names from config (lazy import to avoid circular dependency)."""
    from .config import EEG_COLS

    return EEG_COLS


def _resolve_label(label: Union[int, str]) -> int:
    """
    Convert label to integer, supporting both int and Jamo string inputs.

    Args:
        label: Integer label (0-8) or Jamo string ('ㄱ', 'ㄴ', etc.)

    Returns:
        Integer label

    Raises:
        ValueError: If string label is not a valid Jamo character
        TypeError: If label is neither int nor str
    """
    from .config import default_config

    if isinstance(label, int):
        if label < 0 or label >= default_config.num_classes:
            raise ValueError(
                f"Label {label} out of range. Valid: 0-{default_config.num_classes - 1}"
            )
        return label

    if isinstance(label, str):
        # Try as Jamo character first
        if label in default_config.jamo_to_label:
            return default_config.jamo_to_label[label]

        # Try as numeric string (e.g., "0", "1")
        try:
            int_label = int(label)
            if 0 <= int_label < default_config.num_classes:
                return int_label
        except ValueError:
            pass

        raise ValueError(
            f"Invalid label '{label}'. "
            f"Use Jamo ({default_config.jamo_classes}) or int (0-{default_config.num_classes - 1})"
        )

    raise TypeError(f"label must be int or str, got {type(label).__name__}")


def load_session_csv(
    path: Union[str, Path],
    label: Union[int, str],
    window_size: Optional[float] = None,
    stride: Optional[float] = None,
    sampling_rate: Optional[float] = None,
    filter_headband: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single MindMonitor/Muse session CSV and extract EEG band power features.

    Supports two modes:
    1. Row-based (window_size=None): Each row is a sample
       Output: X.shape = (N, 20), y.shape = (N,)

    2. Windowed (window_size > 0): Sliding window over time
       Output: X.shape = (num_windows, window_len, 20), y.shape = (num_windows,)

    Processing Pipeline:
        1. Load CSV
        2. Filter HeadBandOn == 1.0 (if enabled and column exists)
        3. Extract EEG columns (20 band power features)
        4. Remove rows with NaN values
        5. Apply windowing (if window_size specified)

    Args:
        path: Path to the CSV file
        label: Class label. Accepts:
               - int: Direct label (0-8)
               - str: Jamo character ('ㄱ'-'ㅣ') or numeric string ('0'-'8')
        window_size: Window size in seconds. If None, returns row-based samples.
                     Converted to samples: window_len = int(window_size * sampling_rate)
        stride: Stride in seconds for sliding window. Defaults to window_size / 2.
                Converted to samples: stride_len = int(stride * sampling_rate)
        sampling_rate: Sampling rate in Hz. If None, uses config default (1.0 Hz).
                       MindMonitor outputs band power at ~1 Hz.
        filter_headband: If True, filter out rows where HeadBandOn != 1.0

    Returns:
        Tuple of (X, y):
            - X: np.ndarray, dtype=float32
            - y: np.ndarray, dtype=int64

    Raises:
        FileNotFoundError: If CSV file does not exist
        KeyError: If required EEG columns are missing
        ValueError: If label is invalid

    Example:
        >>> # Row-based loading
        >>> X, y = load_session_csv("data/raw/ㄱ/session.csv", label=0)
        >>> print(X.shape)  # (N, 20)

        >>> # Windowed loading with string label
        >>> X, y = load_session_csv("data/raw/ㄱ/session.csv", label="ㄱ",
        ...                         window_size=5.0, stride=2.0)
        >>> print(X.shape)  # (num_windows, 5, 20) at 1 Hz

        >>> # Using numeric string label
        >>> X, y = load_session_csv("data/raw/ㄱ/session.csv", label="0")
    """
    from .config import default_config

    # Resolve label to int
    label = _resolve_label(label)

    # Get config defaults
    eeg_cols = _get_eeg_cols()
    if sampling_rate is None:
        sampling_rate = default_config.default_sampling_rate

    path = Path(path)
    df = pd.read_csv(path)

    # Validate columns
    missing_cols = set(eeg_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing EEG columns in {path.name}: {missing_cols}")

    # Filter by HeadBandOn if enabled and column exists
    if filter_headband and "HeadBandOn" in df.columns:
        df = df[df["HeadBandOn"] == 1.0]

    # Extract EEG columns and drop NaN rows (before windowing)
    eeg_df = df[eeg_cols].dropna()
    data = eeg_df.values.astype(np.float32)

    n_samples = len(data)
    n_features = len(eeg_cols)

    # Handle empty data
    if n_samples == 0:
        if window_size is not None:
            window_len = int(window_size * sampling_rate)
            return (
                np.empty((0, window_len, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return (
            np.empty((0, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # Row-based mode
    if window_size is None:
        y = np.full(n_samples, label, dtype=np.int64)
        return data, y

    # Windowed mode
    if stride is None:
        stride = window_size / 2

    # Convert seconds to samples
    window_len = int(window_size * sampling_rate)
    stride_len = max(1, int(stride * sampling_rate))

    # Check if enough data for at least one window
    if window_len > n_samples:
        return (
            np.empty((0, window_len, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # Create windows using sliding window
    windows = []
    start = 0
    while start + window_len <= n_samples:
        windows.append(data[start : start + window_len])
        start += stride_len

    X = np.stack(windows, axis=0)
    y = np.full(len(windows), label, dtype=np.int64)

    return X, y


def load_multi_sessions(
    file_label_pairs: List[Tuple[Union[str, Path], Union[int, str]]],
    window_size: Optional[float] = None,
    stride: Optional[float] = None,
    sampling_rate: Optional[float] = None,
    filter_headband: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple session CSVs and combine into unified X, y arrays.

    Args:
        file_label_pairs: List of (file_path, label) tuples.
                          Labels can be int or str (Jamo/'0'-'8').
        window_size: Window size in seconds (None for row-based)
        stride: Stride in seconds for sliding window
        sampling_rate: Sampling rate in Hz
        filter_headband: If True, filter out rows where HeadBandOn != 1.0

    Returns:
        Tuple of (X, y):
            - Row mode: X.shape=(N_total, 20), y.shape=(N_total,)
            - Window mode: X.shape=(N_total, window_len, 20), y.shape=(N_total,)

    Raises:
        ValueError: If file_label_pairs is empty

    Example:
        >>> pairs = discover_dataset()
        >>> X, y = load_multi_sessions(pairs, window_size=5.0, stride=2.0)
    """
    if not file_label_pairs:
        raise ValueError("file_label_pairs cannot be empty")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for file_path, label in file_label_pairs:
        X_session, y_session = load_session_csv(
            file_path,
            label,
            window_size=window_size,
            stride=stride,
            sampling_rate=sampling_rate,
            filter_headband=filter_headband,
        )
        if len(X_session) > 0:
            X_list.append(X_session)
            y_list.append(y_session)

    if not X_list:
        from .config import default_config

        n_features = default_config.num_features
        if window_size is not None:
            sr = sampling_rate or default_config.default_sampling_rate
            window_len = int(window_size * sr)
            return (
                np.empty((0, window_len, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return (
            np.empty((0, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


def discover_dataset(
    dataset_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[Path, int]]:
    """
    Automatically discover CSV files in the dataset directory structure.

    Uses config.jamo_classes as the single source of truth for label mapping.
    Default dataset_dir is config.DATASET_DIR (data/raw/).

    Expected structure:
        data/raw/
        ├── ㄱ/
        │   └── *.csv
        ├── ㄴ/
        │   └── *.csv
        └── ...

    Args:
        dataset_dir: Root directory containing Jamo subdirectories.
                     If None, uses config.DATASET_DIR.

    Returns:
        List of (file_path, label) tuples ready for load_multi_sessions.
        Labels are integers (0-8) based on config.jamo_to_label mapping.

    Example:
        >>> pairs = discover_dataset()
        >>> print(f"Found {len(pairs)} files")

        >>> # Load all data with windowing
        >>> X, y = load_multi_sessions(pairs, window_size=5.0, stride=2.0)
    """
    from .config import DATASET_DIR, default_config

    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    dataset_dir = Path(dataset_dir)

    file_label_pairs: List[Tuple[Path, int]] = []

    # Use config's jamo_to_label mapping (single source of truth)
    for jamo, label in default_config.jamo_to_label.items():
        jamo_dir = dataset_dir / jamo
        if jamo_dir.exists():
            for csv_file in sorted(jamo_dir.glob("*.csv")):
                file_label_pairs.append((csv_file, label))

    return file_label_pairs


def get_min_samples_for_window(
    window_size: float,
    sampling_rate: Optional[float] = None,
) -> int:
    """
    Calculate minimum number of samples required to create at least one window.

    Args:
        window_size: Window size in seconds
        sampling_rate: Sampling rate in Hz. If None, uses config default.

    Returns:
        Minimum number of samples (rows) needed in CSV

    Example:
        >>> min_samples = get_min_samples_for_window(5.0, sampling_rate=1.0)
        >>> print(f"Need at least {min_samples} samples")
        Need at least 5 samples
    """
    from .config import default_config

    if sampling_rate is None:
        sampling_rate = default_config.default_sampling_rate

    return int(window_size * sampling_rate)
