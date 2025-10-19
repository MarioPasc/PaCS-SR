# PaCS-SR: Patchwise Convex Stacking for 3D MRI Super-Resolution

Implements a patchwise convex stacker that learns one simplex-constrained weight vector per 3D tile and blends multiple SR experts linearly. See `configs/config.yaml` then drive the training/evaluation from your K-fold JSON manifest.

## Features

- **Patchwise Optimization**: Learns optimal blending weights per 3D tile
- **Multiple Expert Models**: Supports blending of any number of SR models (BSPLINE, SMORE, ECLARE, UNIRES, etc.)
- **Comprehensive Logging**: Real-time training progress, metrics, and session tracking
- **NPZ Weight Maps**: Efficient storage of weight distributions with full metadata
- **Edge-Aware Weighting**: Optional gradient-based voxel weighting
- **Laplacian Smoothing**: Spatial regularization across tile boundaries
- **Parallel Processing**: Multi-threaded patient evaluation

## New Enhancements (2025)

### 1. Enhanced Logging System
- **Session Tracking**: Timestamps, configuration display, experiment metadata
- **Training Progress**: Real-time patient processing and region optimization logging
- **Evaluation Metrics**: Per-patient and aggregate metrics for train/test splits
- **File Logging**: Optional log file output for post-analysis
- **Configurable Verbosity**: Adjust log level and frequency via config

### 2. NPZ Weight Map Storage
- **Efficient Format**: Compressed NPZ files with full metadata
- **4D Weight Maps**: Expanded weights matching volume dimensions (Z, Y, X, n_models)
- **Weight Analysis**: Entropy maps, statistics, and dominant model identification
- **Backward Compatible**: JSON weight files still created

### 3. Configuration Enhancements
- `log_level`: Control logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `log_to_file`: Enable/disable file logging
- `log_region_freq`: Configure region optimization logging frequency

See [ENHANCEMENTS_SUMMARY.md](ENHANCEMENTS_SUMMARY.md) for detailed documentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/PaCS-SR.git
cd PaCS-SR

# Install in editable mode
pip install -e .
```

## Quick Start (CLI)

PaCS-SR provides three command-line tools for the complete workflow:

### 1. Build K-Fold CV Manifest

```bash
# Scan filesystem and create K-fold cross-validation splits
pacs-sr-build-manifest --config configs/config.yaml
```

### 2. Train Model

```bash
# Train PaCS-SR for all folds, spacings, and pulses
pacs-sr-train --config configs/config.yaml

# Or train specific fold/spacing/pulse
pacs-sr-train --config configs/config.yaml --fold 1 --spacing 3mm --pulse t1c
```

### 3. Make Predictions

```bash
# Blend expert predictions using trained weights
pacs-sr-predict --config configs/config.yaml
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for detailed usage instructions and examples.

## Quick Start (Python API)

```python
from pathlib import Path
from pacs_sr.config.config import load_full_config
from pacs_sr.model.model import PatchwiseConvexStacker
import json

# Load configuration
config = load_full_config(Path("configs/config.yaml"))

# Load manifest
with open(config.data.out, "r") as f:
    manifest = json.load(f)

# Initialize model (logger created automatically)
model = PatchwiseConvexStacker(config.pacs_sr)

# Train and evaluate (using first fold)
fold = manifest["folds"][0]
weights = model.fit_one(fold, spacing="3mm", pulse="t1c")
results = model.evaluate_split(fold, spacing="3mm", pulse="t1c")
```

## Output Files

After training and evaluation:
```
{out_root}/{experiment_name}/{spacing}/{pulse}/
├── {experiment_name}_training.log           # Training log (if enabled)
├── metrics.json                             # Train/test metrics
├── weights.json                             # Region weights (JSON)
├── {patient}_blend_train.nii.gz            # Blended predictions (train)
├── {patient}_blend_test.nii.gz             # Blended predictions (test)
├── {patient}_weights_test.npz              # Weight maps (test, NPZ)
└── {patient}_weight_analysis_test.npz      # Weight analysis (test)
```

## Configuration

Edit `configs/config.yaml` to customize:
- Tile geometry (patch_size, stride)
- Optimization settings (simplex constraints, regularization)
- Edge-aware weighting parameters
- Logging verbosity and frequency
- Output options (blends, weight maps)

## Command-Line Tools

After installation, three CLI commands are available:

| Command | Purpose |
|---------|---------|
| `pacs-sr-build-manifest` | Scan filesystem and create K-fold CV manifest |
| `pacs-sr-train` | Train PaCS-SR model for all or specific folds/spacings/pulses |
| `pacs-sr-predict` | Generate predictions using trained weights |

**Example Workflow:**

```bash
# 1. Build manifest from your data
pacs-sr-build-manifest --config configs/config.yaml

# 2. Train model (all folds)
pacs-sr-train --config configs/config.yaml

# 3. Make predictions with trained weights
pacs-sr-predict --config configs/config.yaml \
  --weights output/fold_1/3mm/t1c/weights.json
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for comprehensive documentation.

## Dependencies

- numpy >= 1.23
- scipy >= 1.10 (optimization)
- nibabel >= 5.0 (NIfTI I/O)
- scikit-image >= 0.21 (metrics)
- joblib >= 1.3 (parallel processing)
- pyyaml >= 6.0 (configuration)
