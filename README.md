# Multi-output Gaussian Process for OSNR and Overlap Prediction

## Project Overview
This project implements a Multi-output Gaussian Process using GPyTorch to predict:
1. **Channel Overlap** (Binary) - Whether channels overlap (spacing <= 35.2 GHz)
2. **OSNR** (Continuous) - Optical Signal-to-Noise Ratio

## Dataset
- **Source**: Processed features from `processed_data/`
- **Size**: ~5,000 samples (reduced from 128M raw rows)
- **Features**: 20 extracted features per sample
- **Targets**:
    - Task 0: Overlap (Binary)
    - Task 1: OSNR (Continuous)

## Installation

### Using pip
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using uv (faster)
```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --epochs 50
```

**Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--smoke-test`: Run with a tiny subset for debugging

### Evaluation
```bash
python scripts/evaluate.py
```

## Methodology

### Multi-output Gaussian Process with Linear Model of Coregionalization (LMC)

This project uses a **Sparse Variational Gaussian Process (SVGP)** with **Linear Model of Coregionalization (LMC)** for scalable multi-output learning.

#### Why LMC for Correlated Outputs?

The features are predictive of both degradation sources. LMC models this by using **shared latent functions** that are linearly combined to produce both outputs.

#### LMC Architecture

**Latent Functions**: 3 shared Gaussian Processes model underlying patterns.

**Likelihoods**:
1. **Channel Overlap (Binary)**: Bernoulli likelihood (Task 0)
2. **OSNR (Continuous)**: Gaussian likelihood (Task 1)

Handled via `LikelihoodList` for mixed output types.

### Training Process

```
Loss = -ELBO = -(E[log p(y|f)] - KL[q(u)||p(u)])
```

## Project Structure
```
mogp_with_features/
├── checkpoints/                 # Saved model weights
│   └── processed/               # Models trained on processed data
├── configs/                     # Training configurations
│   └── processed/
├── processed_data/              # Input data (.npy files)
├── results/                     # Evaluation results
│   └── processed/
├── scripts/                     # Execution scripts
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── src/                         # Source code
│   ├── models/                 # Model definitions
│   └── utils/                  # Utilities (data loading)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Output
- **Checkpoints**: Saved to `checkpoints/processed/mixed_gp_model.pth`
- **Configs**: Saved to `configs/processed/training_config.json`
- **Results**: Saved to `results/processed/evaluation_results_*.json`

