# Training Pipeline Documentation

## Overview

This document describes the end-to-end training pipeline for Project River, a Korean Jamo classification system using EEG band power features.

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌─────────┐    ┌────────────┐
│  Raw CSV    │ → │ Preprocessing │ → │ Feature Selection│ → │ Tensor Convert │ → │ EEGNet  │ → │ Evaluation │
│ (Muse Data) │    │   (Filter)    │    │  (Band Power)   │    │   (PyTorch)    │    │ Training│    │  (Metrics) │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────────┘    └─────────┘    └────────────┘
```

## 1. Data Input

### Source Format
- MindMonitor/Muse 2 exported CSV files
- Each file represents a single session for one Jamo character
- Sampling rate: ~256 Hz (approximate, varies by export settings)

### Key Columns
| Column Pattern | Description |
|----------------|-------------|
| `Delta_*`, `Theta_*`, `Alpha_*`, `Beta_*`, `Gamma_*` | Pre-computed band power values |
| `TP9`, `AF7`, `AF8`, `TP10` | 4 EEG channel locations |
| `HeadBandOn` | Device contact quality indicator |
| `TimeStamp` | Recording timestamp |

## 2. Preprocessing Stage

### Current Strategy (Phase 1)
The CSV files from MindMonitor already contain pre-computed band power features (Delta, Theta, Alpha, Beta, Gamma). For the initial version, we directly use these values as input features without additional signal processing.

**Rationale**: This approach allows rapid prototyping and baseline establishment before investing in raw signal processing infrastructure.

### Future Strategy (Phase 2+)
When raw EEG signal processing becomes necessary:

1. **Band-pass Filter**: 0.5–45 Hz
   - Removes DC drift (< 0.5 Hz)
   - Eliminates high-frequency noise (> 45 Hz)
   
2. **Notch Filter**: 60 Hz
   - Removes power line interference (60 Hz in Korea/US)
   
3. **Band Power Computation**
   - Apply FFT or Welch's method to compute spectral power
   - Extract power in each frequency band

### Data Quality Filtering
- Remove rows with NaN values in EEG columns
- Filter by `HeadBandOn == 1.0` when available (ensures good electrode contact)

## 3. Feature Construction

### Feature Vector Composition
Each sample consists of 20 features:

| Band | TP9 | AF7 | AF8 | TP10 |
|------|-----|-----|-----|------|
| Delta | ✓ | ✓ | ✓ | ✓ |
| Theta | ✓ | ✓ | ✓ | ✓ |
| Alpha | ✓ | ✓ | ✓ | ✓ |
| Beta | ✓ | ✓ | ✓ | ✓ |
| Gamma | ✓ | ✓ | ✓ | ✓ |

**Feature dimension**: 5 bands × 4 channels = **20 dimensions**

### Temporal Windowing (Optional)
For sequence-based models, data can be organized as:
- Window size: T time steps
- Input shape: `(T, 20)` or `(T, 5, 4)` for 2D convolution

## 4. Model Architecture

### Primary Model: EEGNet
EEGNet is selected as the first-generation model due to:
- Compact architecture suitable for limited training data
- Proven effectiveness on EEG classification tasks
- Efficient inference on edge devices

### Architecture Highlights
- Depthwise separable convolutions
- Temporal and spatial filtering
- Dropout for regularization
- Softmax output for 9-class classification

### Future Considerations
- **RNN/LSTM**: For capturing longer temporal dependencies
- **CNN-RNN Hybrid**: Combining spatial and temporal feature extraction
- **Transformer-based**: If data volume significantly increases

These alternatives are deferred to Phase 2 research pending sufficient data collection.

## 5. Training Configuration

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

### Hyperparameters (Initial)
- Optimizer: Adam
- Learning rate: 1e-3 (with scheduler)
- Batch size: 32
- Epochs: 100 (with early stopping)
- Loss function: Cross-entropy

### Regularization
- Dropout: 0.5
- Early stopping patience: 10 epochs
- Data augmentation: TBD (noise injection, time shifting)

## 6. Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro-averaged F1 for class imbalance handling
- **Confusion Matrix**: Per-class performance analysis

### Performance Targets

| Metric | Baseline (Random) | Phase 1 Target | Long-term Goal |
|--------|-------------------|----------------|----------------|
| Accuracy | 11.1% (1/9) | ≥ 50% | ≥ 70% |

**Phase 1 Target Rationale**: 
- 9-class random baseline is ~11%
- Achieving ≥50% demonstrates meaningful signal extraction
- Sets foundation for iterative improvement

## 7. Output Artifacts

### Model Checkpoints
- Best validation accuracy model
- Final epoch model
- Training history (loss, accuracy curves)

### Processed Data
- Numpy arrays: `X_train.npy`, `y_train.npy`, etc.
- Optional: Parquet format for larger datasets

### Reports
- Training logs
- Evaluation metrics
- Confusion matrix visualization

## References

- Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*.
- Muse 2 Technical Specifications: https://choosemuse.com/
