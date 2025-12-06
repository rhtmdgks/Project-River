# Project River

> Research-driven EEG Jamo classification project using band-power features and a lightweight CNN (EEGNet) on Apple Silicon.

## Project Overview

Project River aims to classify Korean Jamo characters (ㄱ, ㄴ, ㄷ, ㄹ, ㅇ, ㅏ, ㅓ, ㅡ, ㅣ) from low-channel EEG signals captured via Muse 2 headband. The project leverages pre-computed band power features and a lightweight EEGNet-based CNN architecture optimized for Apple Silicon (M2).

## Data Description

- **Device**: Muse 2 (4-channel EEG)
- **Channels**: TP9, AF7, AF8, TP10
- **Features**: Band power values across 5 frequency bands
  - Delta (1-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-45 Hz)
- **Feature Dimension**: 20 (5 bands × 4 channels)
- **Classes**: 9 Korean Jamo characters

## Planned Architecture

```
Raw CSV (Band Power) → Preprocessing → Feature Selection → EEGNet (Lightweight CNN) → 9-class Classification
```

The first-generation model uses EEGNet, a compact CNN designed for EEG signal classification. Future iterations may explore RNN/CNN-RNN hybrids as data volume increases.

## Basic Setup

### Requirements

- Python 3.10
- macOS (Apple Silicon recommended)
- Virtual environment (recommended)

### Installation

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from src.river.data_loader import load_multi_sessions

# Define file-label pairs
file_label_pairs = [
    ("data/raw/session_ㄱ_001.csv", 0),
    ("data/raw/session_ㄴ_001.csv", 1),
    # ... more sessions
]

# Load data
X, y = load_multi_sessions(file_label_pairs)
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
```

## Project Structure

```
project-river/
├── README.md
├── requirements.txt
├── docs/
│   └── training_pipeline.md
├── src/
│   └── river/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       └── preprocessing.py
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
└── dataset/          # Original Muse CSV files
```

## License

Research use only.
