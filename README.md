# Multi-output Gaussian Process for OSNR and Overlap Prediction

This repository implements a Multi-output Gaussian Process (MOGP) model to simultaneously predict:
1.  **Channel Overlap** (Binary Classification)
2.  **Optical Signal-to-Noise Ratio (OSNR)** (Continuous Regression)

The project leverages **GPyTorch** for scalable Variational Inference using a **Linear Model of Coregionalization (LMC)**.

## 🚀 Getting Started

### Installation
1.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Repository Structure
The project is organized into two main self-contained notebooks:

*   **`01_ProbabilisticFeatures.ipynb`**: 
    *   **Purpose**: Feature extraction pipeline.
    *   **Input**: Raw I/Q symbols (downloaded automatically).
    *   **Output**: Topological feature vectors saved to `processed_data/`.
    
*   **`02_MOGP_Training_Evaluation.ipynb`**:
    *   **Purpose**: Model definition, training, and evaluation.
    *   **Input**: Processed features from the previous notebook.
    *   **Modules**: Contains the full `MixedGPModel` implementation and training logic.

## 🛠️ Usage
1.  **Step 1**: Run `01_ProbabilisticFeatures.ipynb` to generate the dataset.
2.  **Step 2**: Run `02_MOGP_Training_Evaluation.ipynb` to train the MOGP model and view performance metrics.

> **Note**: All mathematical methodology, architecture details, and feature engineering logic are documented directly within the notebooks.