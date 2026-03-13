"""
Shared utilities for MTGP / GP training, evaluation, and plotting.

This module is imported by all 02_*.py Marimo notebooks to avoid code
duplication. Every function works in plain Python / PyTorch / Matplotlib.
"""

from __future__ import annotations

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SEED = 42
OUTLIER_THRESHOLD_OSNR = 45.0
OUTLIER_THRESHOLD_SPACING = 45.0
MAX_TRAIN_POINTS = 5000
TRAINING_ITERATIONS = 200

# ──────────────────────────────────────────────
# Reproducibility & device
# ──────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────

class MultitaskGPModel(gpytorch.models.ExactGP):
    """Exact multitask GP with LMC-style covariance (rank-1)."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=train_y.shape[1]
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=train_y.shape[1], rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    """Single-output exact GP with RBF kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
MTGP_REQUIRED_KEYS = frozenset({
    "model_state_dict", "likelihood_state_dict",
    "train_x_fit", "train_y_fit",
    "scaler_x", "y_mean", "y_std",
})

GP_REQUIRED_KEYS = frozenset({
    "model_state_dict", "likelihood_state_dict",
})


def validate_checkpoint(
    path: Path | str,
    required_keys: frozenset = MTGP_REQUIRED_KEYS,
) -> bool:
    """Return True if *path* exists and contains all *required_keys*."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return required_keys.issubset(ckpt.keys())
    except Exception as exc:
        print(f"Error reading checkpoint {path}: {exc}")
        return False


def save_checkpoint(path: Path | str, payload: dict) -> None:
    """Create parent dirs and save a PyTorch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Checkpoint saved to {path}")

# ──────────────────────────────────────────────
# Sub-sampling
# ──────────────────────────────────────────────

def _maybe_subsample(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    max_points: int = MAX_TRAIN_POINTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly subsample training data if it exceeds *max_points*."""
    if train_x.size(0) > max_points:
        sel = torch.randperm(train_x.size(0))[:max_points]
        return train_x[sel], train_y[sel]
    return train_x, train_y

# ──────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────

def fit_mtgp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    *,
    n_iter: int = TRAINING_ITERATIONS,
    lr: float = 0.01,
    desc: str = "Training MTGP",
) -> tuple[MultitaskGPModel, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
    """Train a MultitaskGPModel and return (model, likelihood)."""
    train_x_fit, train_y_fit = _maybe_subsample(train_x, train_y)
    train_x_fit = train_x_fit.to(device)
    train_y_fit = train_y_fit.to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=train_y_fit.shape[1]
    ).to(device)
    model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in tqdm(range(n_iter), desc=desc):
        optimizer.zero_grad()
        loss = -mll(model(train_x_fit), train_y_fit)
        loss.backward()
        optimizer.step()

    return model, likelihood


def fit_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    *,
    n_iter: int = TRAINING_ITERATIONS,
    lr: float = 0.01,
    desc: str = "Training GP",
) -> tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Train an ExactGPModel (single-output) and return (model, likelihood)."""
    train_x_fit, train_y_fit = _maybe_subsample(train_x, train_y)
    train_x_fit = train_x_fit.to(device)
    train_y_fit = train_y_fit.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x_fit, train_y_fit, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in tqdm(range(n_iter), desc=desc):
        optimizer.zero_grad()
        loss = -mll(model(train_x_fit), train_y_fit)
        loss.backward()
        optimizer.step()

    return model, likelihood

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def run_inference(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    test_x: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return (mean, lower, upper) as numpy arrays."""
    model.eval()
    likelihood.eval()
    test_x = test_x.to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean.cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
    return mean, lower, upper

# ──────────────────────────────────────────────
# Denormalisation & metrics
# ──────────────────────────────────────────────

def denormalize(arr: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Reverse z-score normalization."""
    return arr * y_std + y_mean


def compute_metrics_multitask(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mae_per_task, rmse_per_task) arrays of shape (num_tasks,)."""
    mae = np.mean(np.abs(y_pred - y_actual), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2, axis=0))
    return mae, rmse


def compute_metrics_single(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
) -> tuple[float, float]:
    """Return (mae, rmse) scalars for a single output."""
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
    return mae, rmse

# ──────────────────────────────────────────────
# Plotting — multitask (2-output) variants
# ──────────────────────────────────────────────

def plot_predictions_multitask(
    y_act: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    axs: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Scatter + 95 % CI for Spacing (idx 0) and OSNR (idx 1)."""
    labels = ["Spectral Spacing (GHz)", "OSNR (dB)"]
    if axs is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        axes = axs

    for col, label in enumerate(labels):
        ax = axes[col]
        sort_idx = np.argsort(y_act[:, col])
        x = np.arange(len(sort_idx))

        ax.plot(x, y_act[sort_idx, col], "k.", label="True", alpha=0.7)
        ax.plot(x, y_pred[sort_idx, col], "b-", label="Predicted", linewidth=2)
        ax.fill_between(
            x, lower[sort_idx, col], upper[sort_idx, col],
            alpha=0.3, label="95% CI",
        )
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if axs is None:
            ax.get_figure().tight_layout()

    return axes[0], axes[1]


def plot_violin_multitask(
    y_act: np.ndarray,
    y_pred: np.ndarray,
    *,
    axs: tuple[plt.Axes, plt.Axes] | None = None,
    num_bins_osnr: int = 10,
) -> tuple[plt.Axes, plt.Axes]:
    """Violin precision plots for Spacing (discrete) and OSNR (binned)."""
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    else:
        ax1, ax2 = axs

    # ── Spacing (discrete true values → categorical) ──
    abs_err_sp = np.abs(y_pred[:, 0] - y_act[:, 0])
    df_sp = pd.DataFrame({
        "Spectral Spacing (GHz)": y_act[:, 0],
        "Absolute Error": abs_err_sp,
    })

    sns.violinplot(
        data=df_sp, x="Spectral Spacing (GHz)", y="Absolute Error",
        color="lightgray", inner="box", linewidth=0.8, width=0.7, ax=ax1,
    )
    ax1.set_ylabel("Absolute Error")
    ax1.grid(True, which="both", alpha=0.3, axis="y")
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # ── OSNR (continuous → binned) ──
    abs_err_osnr = np.abs(y_pred[:, 1] - y_act[:, 1])
    bins = np.linspace(y_act[:, 1].min(), y_act[:, 1].max(), num_bins_osnr + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_labels = [f"{c:.1f}" for c in bin_centers]
    indices = np.clip(np.digitize(y_act[:, 1], bins) - 1, 0, num_bins_osnr - 1)

    df_osnr = pd.DataFrame({
        "OSNR (dB)": [bin_labels[i] for i in indices],
        "Absolute Error": abs_err_osnr,
    })
    df_osnr["sort_key"] = df_osnr["OSNR (dB)"].astype(float)
    df_osnr = df_osnr.sort_values("sort_key").drop("sort_key", axis=1)

    sns.violinplot(
        data=df_osnr, x="OSNR (dB)", y="Absolute Error",
        color="lightgray", inner="box", linewidth=0.8, width=0.7, ax=ax2,
    )
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True, which="both", alpha=0.3, axis="y")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    if axs is None:
        ax1.get_figure().tight_layout()

    return ax1, ax2

# ──────────────────────────────────────────────
# Plotting — single-output variants
# ──────────────────────────────────────────────

def plot_predictions_single(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_label: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter + 95 % CI for a single output."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    sort_idx = np.argsort(y_actual)
    x = np.arange(len(sort_idx))

    ax.plot(x, y_actual[sort_idx], "k.", label="True", alpha=0.7)
    ax.plot(x, y_pred[sort_idx], "b-", label="Predicted", linewidth=2)
    ax.fill_between(
        x, lower[sort_idx], upper[sort_idx],
        alpha=0.3, label="95% CI",
    )
    ax.set_ylabel(target_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ax.get_figure() and ax.get_figure().get_tight_layout() is False:
        ax.get_figure().tight_layout()
    return ax


def plot_violin_single(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    target_label: str,
    *,
    ax: plt.Axes | None = None,
    num_bins: int = 10,
) -> plt.Axes:
    """Violin precision plot for a single output."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    abs_errors = np.abs(y_pred - y_actual)

    if "Spacing" in target_label:
        df = pd.DataFrame({target_label: y_actual, "Absolute Error": abs_errors})
        sns.violinplot(
            data=df, x=target_label, y="Absolute Error",
            color="lightgray", inner="box", linewidth=0.8, width=0.7, ax=ax,
        )
    else:
        bins = np.linspace(y_actual.min(), y_actual.max(), num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_labels = [f"{c:.1f}" for c in bin_centers]
        indices = np.clip(np.digitize(y_actual, bins) - 1, 0, num_bins - 1)

        df = pd.DataFrame({
            target_label: [bin_labels[i] for i in indices],
            "Absolute Error": abs_errors,
        })
        df["sort_key"] = df[target_label].astype(float)
        df = df.sort_values("sort_key").drop("sort_key", axis=1)

        sns.violinplot(
            data=df, x=target_label, y="Absolute Error",
            color="lightgray", inner="box", linewidth=0.8, width=0.7, ax=ax,
        )

    ax.set_ylabel("Absolute Error")
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    if ax.get_figure() and ax.get_figure().get_tight_layout() is False:
        ax.get_figure().tight_layout()
    return ax

# ──────────────────────────────────────────────
# Data Loading & Preprocessing Helpers
# ──────────────────────────────────────────────

def load_processed_arrays(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, Y, and M arrays for a given dataset (fcm/gkm)."""
    base_path = Path(f"processed_data/{dataset_name.lower()}")
    X = np.load(base_path / "X_features.npy")
    Y = np.load(base_path / "Y_targets.npy")  # [Spacing, OSNR]
    M = np.load(base_path / "M_metadata.npy")  # [Distance, Power]
    return X, Y, M


def preprocess_data(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = SEED,
) -> dict:
    """
    Standardize X and Y, split into train/test, and return a dict of tensors and scalars.
    Handles multitask Y (2 cols).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Outlier removal
    mask = (Y[:, 0] <= OUTLIER_THRESHOLD_SPACING) & (Y[:, 1] <= OUTLIER_THRESHOLD_OSNR)
    X, Y = X[mask], Y[mask]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=True
    )

    scaler_x = StandardScaler()
    X_train_norm = scaler_x.fit_transform(X_train)
    X_test_norm = scaler_x.transform(X_test)

    y_mean = Y_train.mean(axis=0, keepdims=True)
    y_std = Y_train.std(axis=0, keepdims=True) + 1e-6
    Y_train_std = (Y_train - y_mean) / y_std
    Y_test_std = (Y_test - y_mean) / y_std

    return {
        "train_x": torch.tensor(X_train_norm, dtype=torch.float32),
        "train_y": torch.tensor(Y_train_std, dtype=torch.float32),
        "test_x": torch.tensor(X_test_norm, dtype=torch.float32),
        "test_y": torch.tensor(Y_test_std, dtype=torch.float32),
        "scaler_x": scaler_x,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_test_raw": Y_test,
    }


def save_plot(ax: plt.Axes | tuple[plt.Axes, ...], filename: str, formats: list[str] = ["pdf", "svg"]) -> None:
    """Save the figure(s) containing the given axes to the 'paper/figures' directory."""
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(ax, tuple):
        # Handle multiple axes (from multitask plots)
        for i, a in enumerate(ax):
            fig = a.get_figure()
            prefix = "spacing" if i == 0 else "osnr"
            for fmt in formats:
                save_path = out_dir / f"{filename}_{prefix}.{fmt}"
                fig.savefig(save_path, format=fmt, bbox_inches="tight")
                print(f"Plot saved: {save_path}")
    else:
        fig = ax.get_figure()
        for fmt in formats:
            save_path = out_dir / f"{filename}.{fmt}"
            fig.savefig(save_path, format=fmt, bbox_inches="tight")
            print(f"Plot saved: {save_path}")


# ──────────────────────────────────────────────
# Overlaid Comparison Plotting Helpers
# ──────────────────────────────────────────────

def plot_comparison_overlaid(
    results_dict: dict,
    target_idx: int,
    target_label: str,
    axs: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay multiple models' predictions on the same plot."""
    if axs is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        ax = axs

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["-", "--", "-.", ":"]
    
    first_method = list(results_dict.keys())[0]
    y_act = results_dict[first_method]["y_act"]
    # If multitasking, we need to extract the correct column
    if y_act.ndim > 1:
        y_act = y_act[:, target_idx]
    
    sort_idx = np.argsort(y_act)
    x = np.arange(len(sort_idx))
    
    # Plot True Values once
    ax.plot(x, y_act[sort_idx], "k.", label="True", alpha=0.4, markersize=4)

    for i, (name, res) in enumerate(results_dict.items()):
        if res is None: continue
        
        y_pred = res["y_pred"]
        y_low = res["y_low"]
        y_up = res["y_up"]
        
        if y_pred.ndim > 1:
            y_pred = y_pred[:, target_idx]
            y_low = y_low[:, target_idx]
            y_up = y_up[:, target_idx]
            
        color = colors[i % len(colors)]
        style = markers[i % len(markers)]
        
        ax.plot(x, y_pred[sort_idx], style, color=color, label=f"Pred: {name}", linewidth=1.5)
        ax.fill_between(x, y_low[sort_idx], y_up[sort_idx], color=color, alpha=0.1)

    ax.set_ylabel(target_label)
    ax.set_xlabel("Test Samples (Sorted by True Value)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    if axs is None:
        ax.get_figure().tight_layout()
    return ax


def plot_comparison_violins(
    results_dict: dict,
    target_idx: int,
    target_label: str,
    axs: plt.Axes | None = None,
    num_bins_osnr: int = 10,
) -> plt.Axes:
    """Create a grouped violin plot for multiple models' metrics."""
    if axs is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        ax = axs

    all_dfs = []
    for name, res in results_dict.items():
        if res is None: continue
        
        y_act = res["y_act"]
        y_pred = res["y_pred"]
        if y_act.ndim > 1:
            y_act = y_act[:, target_idx]
            y_pred = y_pred[:, target_idx]
            
        abs_err = np.abs(y_pred - y_act)
        
        if "Spacing" in target_label:
            df = pd.DataFrame({
                "Bin": y_act,
                "Absolute Error": abs_err,
                "Method": name
            })
        else:
            # OSNR Binning
            bins = np.linspace(y_act.min(), y_act.max(), num_bins_osnr + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_labels = [f"{c:.1f}" for c in bin_centers]
            indices = np.clip(np.digitize(y_act, bins) - 1, 0, num_bins_osnr - 1)
            df = pd.DataFrame({
                "Bin": [bin_labels[i] for i in indices],
                "Absolute Error": abs_err,
                "Method": name,
                "sort_key": [bin_centers[i] for i in indices]
            })
            
        all_dfs.append(df)

    big_df = pd.concat(all_dfs, ignore_index=True)
    if "OSNR" in target_label:
        big_df = big_df.sort_values("sort_key")

    sns.violinplot(
        data=big_df, x="Bin", y="Absolute Error", hue="Method",
        inner="box", linewidth=1, width=0.8, ax=ax,
        palette="muted", split=False
    )
    
    ax.set_xlabel(target_label)
    ax.set_ylabel("Absolute Error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Method", loc="upper right")
    
    if axs is None:
        ax.get_figure().tight_layout()
    return ax
