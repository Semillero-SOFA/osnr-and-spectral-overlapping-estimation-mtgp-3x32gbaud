# OSNR and Spectral Overlap Estimation with MTGP

This repository contains the research code for a scientific publication developed within the **Semillero de Óptica y Fotónica Aplicada (SOFA)**, part of the Grupo de Investigación en Telecomunicaciones Aplicadas (GITA) at the Universidad de Antioquia. The work presents a probabilistic multi-output regression approach for joint estimation of OSNR and spectral channel spacing in gridless Nyquist-WDM optical communication systems.

## 🚀 Project Summary
This project develops a multi-task Gaussian Process (MTGP) solution to jointly predict two continuous outputs (double regression):

- **Channel Spectral Spacing** and **Optical Signal-to-Noise Ratio (OSNR)**, both continuous regression tasks.

### Key elements

- Two-stage, notebook-designed pipeline: (1) probabilistic feature extraction from raw I/Q symbols; (2) MTGP model training and evaluation using GPyTorch.
- Feature extraction converts scenario batches into 16-dimensional topological feature vectors (Counting Vectors).
- Modeling uses a multitask GP (LMC-style covariance / MultitaskKernel) and computes calibrated predictive intervals for both outputs.

For implementation details and mathematical derivations, see the notebooks in the repository root.

> **Note on dataset access:** the dataset used for this work is not publicly available. Researchers can request access from the Semillero de Óptica y Fotónica Aplicada (SOFA).

## 🛠️ How to use
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

2. Run the notebooks in order:
- `01_CountingVectorsFeatures.py` — extract features from CSV files and save to `processed_data/`.
- `02_MTGP_Training_Evaluation.py` — load features, train the general MTGP model, save checkpoint to `artifacts/`, and evaluate all data with granular precision plots.
- `02_Specific_MTGP_Training_Evaluation.py` — filter by Distance and Power, train independent scenario models, and compare them side-by-side.
- `02_General_GP_Training_Evaluation.py` — evaluate independent Gaussian Processes per output, dynamically appending True target variables.
- `03_Comparison.py` — consolidate results from all methods and export conference-ready figures to `paper/figures/`.

Processed data and artifacts:
- `processed_data/fcm/X_features.npy`, `processed_data/fcm/Y_targets.npy`, `processed_data/fcm/M_metadata.npy`
- `artifacts/multitask_gp_fcm.pt` (saved checkpoint containing model and preprocessing stats)

## 🔗 Fork Credit

This repository is a research extension forked from the author's own prior graduation project that first explored this MTGP formulation. This fork refines and extends it for scientific publication, including improved evaluation methodology, conference-ready visualizations, and expanded experimental analysis.

## 📚 References
- [1] A. Escobar P, N. Guerrero Gonzalez, and J. Granada Torres, “Spectral overlapping estimation based on machine learning for gridless Nyquist-wavelength division multiplexing systems,” Optical Engineering, vol. 59, p. 1, July 2020, doi: 10.1117/1.OE.59.7.076116.
- [2] J. J. G. Torres, A. M. C. Soto, and N. G. González, “A novel dispersion monitoring technique in W-band radio-over-fiber signals using clustering on asynchronous histograms,” Ingeniería e Investigación, vol. 34, no. 3, pp. 76–80, Sept. 2014, doi: 10.15446/ing.investig.v34n3.42902.
- [3] E. V. Bonilla, K. Chai, and C. Williams, “Multi-task Gaussian Process Prediction,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2007.
