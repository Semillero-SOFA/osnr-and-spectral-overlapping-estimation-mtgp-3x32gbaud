# Multi-output Gaussian Process for OSNR and Overlap Prediction

## Project Overview
This project implements a Multi-output Gaussian Process using GPyTorch to predict:
1. **Channel Overlap** (Binary) - Whether channels overlap (spacing is less than the channel bandwidth considering the roll-off factor)
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
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Methodology

### Feature extraction
The discussed topological features are extracted in the `01_ProbabilisticFeatures.ipynb` notebook, this notebook preprocesses the data by extracting a batch of 10k symbols from each different scenario, excess symbols are dropped. After that, a KDE PDF is generated for each feature, the peaks and valleys locations are extracted and used as features. The features are then saved in the `processed_data/` directory.

### Training and Evaluation
The training and evaluation of the multioutput GP is done in the `02_TrainAndEvaluate.ipynb` notebook. The notebook first loads the features and targets, then splits the data into training and validation sets. The multioutput GP is then trained on the training set and evaluated on the validation set. The specific implementation for the models in a lower level is done in the `src/` directory. The results are saved in the `results/` directory.

### Multi-output Gaussian Process with Linear Model of Coregionalization (LMC)

This project uses a **Sparse Variational Gaussian Process (SVGP)** with **Linear Model of Coregionalization (LMC)** for scalable multi-output learning. The features are predictive of both degradation sources, LMC models this by using **shared latent functions** that are linearly combined to produce both outputs.

#### LMC Architecture

**Latent Functions**: a default value of 3 shared Gaussian Processes model underlying patterns.

**Likelihoods**:
1. **Channel Overlap (Binary)**: Bernoulli likelihood (Task 0)
2. **OSNR (Continuous)**: Gaussian likelihood (Task 1)

Handled via `LikelihoodList` for mixed output types.

## Project Structure
```
mogp_with_features/
├── processed_data/                   # Input data (.npy files)
├── src/                              # Source code
│   ├── models/                       # Model definitions
│   └── utils/                        # Utilities (data loading)
├── 01_ProbabilisticFeatures.ipynb    # Feature extraction notebook
├── 02_MOGP_Training_Evaluation.ipynb # Training and evaluation notebook
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```