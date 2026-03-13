# AGENT.md — Project Reference for AI Coding Agents

> This file gives AI agents (and new contributors) a complete mental model of the project's purpose, structure, methodology, and conventions. **Read this before touching any file.**

---

## 1. Project Purpose

This is a **scientific research project** developed within the **[Semillero de Óptica y Fotónica Aplicada (SOFA)](https://gita.udea.edu.co/)** at the *Universidad de Antioquia*, targeting peer-reviewed publication. It extends prior work on spectral overlapping estimation in gridless Nyquist-WDM systems with a full probabilistic, multi-output regression approach.

**Goal:** Jointly estimate two continuous physical quantities from optical fiber communication data:
- **Spectral Channel Spacing (GHz)** — how far apart optical channels are.
- **OSNR (dB)** — Optical Signal-to-Noise Ratio of the received signal.

This is a **double-regression** problem solved using probabilistic machine learning, specifically **Multitask Gaussian Processes (MTGPs)**.

The dataset originates from a **32 Gbaud, 3-channel Nyquist-WDM 16-QAM optical communication simulation** and was provided by SOFA. It is **not publicly available**; researchers must request access from SOFA.

---

## 2. Pipeline Overview

The project follows a clean **two-stage pipeline**:

```
Raw I/Q Symbols (CSV)
       │
       ▼
 [Stage 1] Feature Extraction
  01_CountingVectorsFeatures.py
  → Produces 16-dimensional Counting Vector features per scenario batch
  → Saves to processed_data/{fcm,gkm}/
       │
       ▼
 [Stage 2] Model Training + Evaluation
  02_General_MTGP_Training_Evaluation.py   ← All-data MTGP
  02_Specific_MTGP_Training_Evaluation.py  ← Per-scenario MTGP
  02_General_GP_Training_Evaluation.py     ← Single-output GP baseline
  03_Comparison.py                         ← Cross-method comparison & figure export
  → Saves model checkpoints to artifacts/
```

---

## 3. Dataset Details

### Raw Data (`data/`)
Two CSV files (sourced from Google Drive via `gdown` with IDs from `.env`):

| File | Feature method | Description |
|------|---------------|-------------|
| `FCM.csv` | Fuzzy C-Means (Euclidean distance) | Counting Vectors per scenario batch |
| `GKM.csv` | Gustafson-Kessel Means (Mahalanobis distance) | Counting Vectors per scenario batch |

Each CSV row contains 20 columns (no header):
- Columns 0–15: `cv_1` … `cv_16` — normalized probability counting vector features
- Column 16: `OSNR` (dB) — regression target
- Column 17: `Distance` (km) — metadata, NOT a feature
- Column 18: `Power` (dBm) — metadata, NOT a feature  
- Column 19: `Spacing` (GHz) — regression target

### Processed Data (`processed_data/{fcm,gkm}/`)
Three NumPy arrays extracted by `01_CountingVectorsFeatures.py`:
- `X_features.npy` — shape `(N, 16)` — float32 counting vector features
- `Y_targets.npy` — shape `(N, 2)` — float32 targets: `[Spacing, OSNR]`
- `M_metadata.npy` — shape `(N, 2)` — float32 metadata: `[Distance, Power]`

> **Column order for Y:** index 0 = Spacing, index 1 = OSNR. This ordering is maintained consistently across all notebooks.

### Scenarios (for specific MTGP)
Three distinct experimental conditions filtered from metadata:

| Scenario | Distance (km) | Power (dBm) |
|----------|--------------|------------|
| A | 0 | 0 |
| B | 270 | 0 |
| C | 270 | 9 |

### Outlier Removal
Applied in all model notebooks before train/test split:
- `OSNR > 45.0 dB` → removed
- `Spacing > 45.0 GHz` → removed

---

## 4. Methodology

### Feature Engineering: Counting Vectors
Counting Vectors (CVs) are a topological/probabilistic feature representation. Each CV is a 16-dimensional vector of normalized cluster membership probabilities computed from batches of I/Q symbols. Two variants are computed:
- **FCM** — Euclidean-distance-based Fuzzy C-Means clustering
- **GKM** — Mahalanobis-distance-based Gustafson-Kessel Means clustering

### Multitask Gaussian Process (MTGP) Model
Implemented using **GPyTorch** (`ExactGP` base class):

```python
class MultitaskGPModel(gpytorch.models.ExactGP):
    mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
    covar_module = MultitaskKernel(RBFKernel(), num_tasks=2, rank=1)
    # Output: MultitaskMultivariateNormal
```

- **Kernel:** `MultitaskKernel` (LMC-style, rank-1 inter-task correlation) wrapping an `RBFKernel`
- **Likelihood:** `MultitaskGaussianLikelihood`
- **Training loss:** Exact Marginal Log Likelihood (EMLL) via `ExactMarginalLogLikelihood`
- **Optimizer:** Adam, lr=0.01, 200 iterations
- **Max training points:** 5000 (randomly subsampled if dataset exceeds this)

### Single-Output GP Baseline
Implemented in `02_General_GP_Training_Evaluation.py` with `ExactGPModel`:
- `ConstantMean` + `ScaleKernel(RBFKernel())`
- `GaussianLikelihood`
- Trained for 4 configurations per dataset (FCM/GKM):
  1. **OSNR only** — 16 features → OSNR
  2. **OSNR + Spacing** — 17 features (16 + true Spacing) → OSNR
  3. **Spacing only** — 16 features → Spacing
  4. **Spacing + OSNR** — 17 features (16 + true OSNR) → Spacing

### Preprocessing (all model notebooks)
Applied consistently in every `load_dataset` / `load_scenario_dataset` / `load_dataset_gp` function:
1. **Train/test split** — 80/20, `random_state=42`, shuffle=True
2. **StandardScaler** on X (`sklearn.preprocessing.StandardScaler`, fit on train)
3. **Z-score normalization** on Y — `y_std = (y - y_mean) / y_std_dev` computed from training targets

Normalisation statistics (`scaler_x`, `y_mean`, `y_std`) are saved inside the checkpoint `.pt` file for proper denormalization at inference time.

---

## 5. Notebooks / Scripts

All Python files are **Marimo reactive notebooks** (`.py` format). Run with:
```bash
marimo run <notebook>.py        # non-interactive
marimo edit <notebook>.py       # interactive browser UI
```

### Execution Order

| Order | File | Purpose |
|-------|------|---------|
| 1 | `01_CountingVectorsFeatures.py` | Download raw CSVs (Google Drive), extract X/Y/M arrays, save to `processed_data/` |
| 2 | `02_General_MTGP_Training_Evaluation.py` | Train & evaluate general MTGP on full FCM + GKM datasets |
| 3 | `02_Specific_MTGP_Training_Evaluation.py` | Train & evaluate per-scenario MTGPs (3 scenarios × 2 datasets = 6 models) |
| 4 | `02_General_GP_Training_Evaluation.py` | Train & evaluate single-output GP baselines (4 configs × 2 datasets = 8 models) |
| 5 | `03_Comparison.py` | Cross-method comparison and publication-ready figure export |

Notebooks 2–5 are **independent** of each other (only depend on `processed_data/` and existing checkpoints).

---

## 6. Artifacts (Saved Checkpoints)

All model checkpoints are saved to `artifacts/` as PyTorch `.pt` files. Each checkpoint stores:
```python
{
  'model_state_dict': ...,
  'likelihood_state_dict': ...,
  'train_x_fit': Tensor,    # training data actually used (≤5000 points)
  'train_y_fit': Tensor,
  'scaler_x': StandardScaler,   # sklearn object
  'y_mean': np.ndarray,
  'y_std': np.ndarray
}
```

> **Checkpoint existence check:** all training functions validate existing checkpoints before retraining. If `required_keys` are all present, training is skipped. Delete the `.pt` file to force retraining.

### Naming Convention

| Pattern | Description |
|---------|-------------|
| `multitask_gp_{dataset}.pt` | General MTGP (e.g., `multitask_gp_fcm.pt`) |
| `mtgp_{dataset}_{dist}km_{pwr}dbm.pt` | Scenario-specific MTGP (e.g., `mtgp_fcm_270km_9dbm.pt`) |
| `gp_{config}_{dataset}.pt` | Single-output GP (e.g., `gp_osnr_only_fcm.pt`) |

---

## 7. Evaluation & Visualization Style

All evaluation functions share a consistent style optimized for **conference paper figures**:

### Metrics
- **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error), both in original units (GHz or dB), computed after denormalization.

### Plot Types
1. **Prediction scatter plot** — sorted by true label, with predicted line and 95% CI fill; black dots for true, blue line for predicted.
2. **Binned precision plot (boxplot + stripplot)** — uses Seaborn `boxplot` + `stripplot` overlaid:
   - Spacing: bins are the discrete true values (categorical x-axis)
   - OSNR: 10 equally-spaced bins, x-axis labels = bin center rounded to 1 decimal
   - Colors: `color='white'` for boxes, `color='black', alpha=0.3` for strip points
   - `showfliers=False` for cleaner plots
   - Y-axis: `MultipleLocator(0.5)` major, `MultipleLocator(0.1)` minor ticks

### Figure formatting
- No plot titles (removed for conference paper formatting)
- `fig.tight_layout()` applied to all figures
- Figures wrapped in `mo.ui.matplotlib()` for Marimo interactivity
- Layouts assembled with `mo.hstack(...)` and `mo.vstack(...)`

---

## 8. Environment & Dependencies

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| `marimo` | ≥ 0.20.4 |
| `torch` | ≥ 2.1.0 |
| `gpytorch` | ≥ 1.15.2 |
| `numpy` | ≥ 2.1.0 |
| `scikit-learn` | ≥ 1.5.0 |
| `pandas` | ≥ 2.2.0 |
| `matplotlib` | ≥ 3.10.8 |
| `seaborn` | ≥ 0.13.2 |
| `tqdm` | ≥ 4.66.0 |
| `gdown` | ≥ 5.0.0 |
| `python-dotenv` | ≥ 1.0.0 |

### Setup
```bash
# Using pip
python -m venv .venv && source .venv/bin/activate
pip install .

# Using uv (recommended, lockfile present)
uv sync
source .venv/bin/activate
```

### Environment Variables (`.env`)
```
GDRIVE_FILE_ID_FCM=<Google Drive file ID for FCM.csv>
GDRIVE_FILE_ID_GKM=<Google Drive file ID for GKM.csv>
```
Required only for Stage 1 data download. If `data/FCM.csv` and `data/GKM.csv` already exist, these can be omitted.

---

## 9. Reproducibility

All notebooks set a fixed global seed at the top of their computation graph:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```
Device selection is automatic (`cuda` if available, else `cpu`). The same seed and train/test split are used across all notebooks, so results are fully reproducible across runs (assuming checkpoints are deleted to retrain from scratch).

---

## 10. Key Design Decisions & Conventions

- **DRY functions:** each notebook defines reusable `train_*` / `evaluate_*` / `load_*` functions applied uniformly across both FCM and GKM datasets via a single call per cell.
- **Marimo reactivity:** each function definition lives in its own `@app.cell`. Cells that call training functions are separate from cells that call evaluation functions. This enables Marimo's reactive DAG to cache results properly.
- **No data leakage:** the `StandardScaler` and Y normalization statistics are always fitted on the training split only, then applied to the test split.
- **Checkpoint robustness:** a `required_keys` check is performed on every saved checkpoint before deciding to skip training. Corrupted or partial checkpoints trigger retraining.
- **Metadata is never a feature:** `Distance` and `Power` columns are stored in `M_metadata.npy` and used only for scenario filtering in `02_Specific_MTGP_Training_Evaluation.py`. They are never included in `X_features`.

---

## 11. References

- **[1]** A. Escobar P, N. Guerrero Gonzalez, and J. Granada Torres, "Spectral overlapping estimation based on machine learning for gridless Nyquist-WDM systems," *Optical Engineering*, vol. 59, 2020. doi: 10.1117/1.OE.59.7.076116
- **[2]** J. J. G. Torres et al., "A novel dispersion monitoring technique in W-band radio-over-fiber signals," *Ingeniería e Investigación*, vol. 34, no. 3, 2014. doi: 10.15446/ing.investig.v34n3.42902
- **[3]** E. V. Bonilla, K. Chai, and C. Williams, "Multi-task Gaussian Process Prediction," in *Advances in NIPS*, 2007.

---

## 12. Origin & Fork Credit

This repository is a research extension forked from the author's own prior graduation project:

> **osnr-and-spectral-overlapping-estimation-mtgp-3x32gbaud** (original graduation project)
> Semillero de Óptica y Fotónica Aplicada (SOFA) — Universidad de Antioquia

The original project explored the same MTGP formulation as a course deliverable. This fork extends and refines it for scientific publication, including improved evaluation methodology, conference-ready figures, and expanded experimental analysis.
