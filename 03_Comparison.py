# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gpytorch>=1.15.2",
#     "marimo>=0.20.4",
#     "matplotlib>=3.10.8",
#     "numpy>=2.1.0",
#     "pyzmq>=26.0.0",
#     "scikit-learn>=1.5.0",
#     "seaborn>=0.13.2",
#     "torch>=2.1.0",
#     "tqdm>=4.66.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fair Performance Comparison: MTGP vs Baseline GP (16 Features)

    This notebook provides a fair "apples-to-apples" comparison between Multitask Gaussian Processes (MTGP) and Independent Gaussian Processes (GP).

    **Key Scientific Constraint:**
    Both MTGP and the baseline GP models use the **exact same 16-feature input set**. We specifically exclude the "plus" GP variants (which utilize an additional feature) to evaluate the inherent benefit of multitask learning without external information advantages.

    Results are separated by feature extraction method (**FCM** and **GKM**) to maintain visual clarity.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import gpytorch
    from pathlib import Path
    import matplotlib.pyplot as plt
    import marimo as mo

    from utils import (
        get_device,
        MultitaskGPModel, ExactGPModel,
        load_processed_arrays, preprocess_data,
        run_inference, denormalize,
        compute_metrics_multitask, compute_metrics_single,
        plot_comparison_overlaid, plot_comparison_violins,
        save_plot
    )

    return (
        ExactGPModel,
        MultitaskGPModel,
        Path,
        compute_metrics_multitask,
        compute_metrics_single,
        denormalize,
        get_device,
        gpytorch,
        load_processed_arrays,
        mo,
        np,
        plot_comparison_overlaid,
        plot_comparison_violins,
        plt,
        preprocess_data,
        run_inference,
        torch,
    )


@app.cell
def _(get_device):
    device = get_device()
    return (device,)


@app.cell
def _(
    MultitaskGPModel,
    Path,
    compute_metrics_multitask,
    denormalize,
    device,
    gpytorch,
    load_processed_arrays,
    preprocess_data,
    run_inference,
    torch,
):
    def fetch_mtgp_data(dataset_name):
        X, Y, m = load_processed_arrays(dataset_name)
        data = preprocess_data(X, Y)
        test_x = data["test_x"].to(device)
        y_mean, y_std = data["y_mean"], data["y_std"]

        ckpt_path = Path("artifacts") / f"multitask_gp_{dataset_name.lower()}.pt"
        if not ckpt_path.exists():
            ckpt_path = Path("artifacts") / f"mtgp_{dataset_name.lower()}.pt"

        if not ckpt_path.exists():
            return None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt["train_x_fit"].to(device)
        train_y_fit = ckpt["train_y_fit"].to(device)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
        model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        likelihood.load_state_dict(ckpt["likelihood_state_dict"])

        mean, lower, upper = run_inference(model, likelihood, test_x, device)

        y_pred = denormalize(mean, y_mean, y_std)
        y_act = data["y_test_raw"]
        y_low = denormalize(lower, y_mean, y_std)
        y_up = denormalize(upper, y_mean, y_std)

        mae, rmse = compute_metrics_multitask(y_pred, y_act)
        from sklearn.metrics import r2_score
        r2 = r2_score(y_act, y_pred, multioutput='raw_values')
        return {
            "y_act": y_act, "y_pred": y_pred, "y_low": y_low, "y_up": y_up,
            "mae": mae, "rmse": rmse, "r2": r2
        }

    return (fetch_mtgp_data,)


@app.cell
def _(
    ExactGPModel,
    Path,
    compute_metrics_single,
    denormalize,
    device,
    gpytorch,
    load_processed_arrays,
    preprocess_data,
    run_inference,
    torch,
):
    def fetch_gp_data_fair(target_label, dataset_name):
        """Fetches the 16-feature baseline GP results (no extra features)."""
        X, Y, _ = load_processed_arrays(dataset_name)

        is_osnr = "OSNR" in target_label
        # Use ONLY version for fair comparison
        ckpt_name = f"gp_{'osnr_only' if is_osnr else 'spacing_only'}_{dataset_name}.pt"

        data = preprocess_data(X, Y)
        test_x = data["test_x"].to(device)
        y_test_raw_all = data["y_test_raw"]

        target_idx = 1 if is_osnr else 0
        y_test_raw = y_test_raw_all[:, target_idx]
        y_mean = data["y_mean"][0, target_idx]
        y_std = data["y_std"][0, target_idx]

        ckpt_path = Path("artifacts") / ckpt_name
        if not ckpt_path.exists():
            return None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt["train_x_fit"].to(device)
        train_y_fit = ckpt["train_y_fit"].to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x_fit, train_y_fit, likelihood).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        likelihood.load_state_dict(ckpt["likelihood_state_dict"])

        mean, lower, upper = run_inference(model, likelihood, test_x, device)

        y_pred = denormalize(mean, y_mean, y_std)
        y_low = denormalize(lower, y_mean, y_std)
        y_up = denormalize(upper, y_mean, y_std)

        mae, rmse = compute_metrics_single(y_pred, y_test_raw)
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test_raw, y_pred)
        return {
            "y_act": y_test_raw, "y_pred": y_pred, "y_low": y_low, "y_up": y_up,
            "mae": mae, "rmse": rmse, "r2": r2
        }

    return (fetch_gp_data_fair,)


@app.cell
def _(fetch_gp_data_fair, fetch_mtgp_data):
    # Load Fair Results (16 features only)

    # FCM Methods Cluster
    fcm_results = {
        "MTGP": fetch_mtgp_data("fcm"),
        "GP Spacing": fetch_gp_data_fair("Spectral Spacing (GHz)", "fcm"),
        "GP OSNR": fetch_gp_data_fair("OSNR (dB)", "fcm"),
    }

    # GKM Methods Cluster
    gkm_results = {
        "MTGP": fetch_mtgp_data("gkm"),
        "GP Spacing": fetch_gp_data_fair("Spectral Spacing (GHz)", "gkm"),
        "GP OSNR": fetch_gp_data_fair("OSNR (dB)", "gkm"),
    }
    return fcm_results, gkm_results


@app.cell
def _(Path, fcm_results, mo, plot_comparison_overlaid, plt):
    # Figure 1: FCM Fair Predictions (16 Features)
    _valid = {k: v for k, v in fcm_results.items() if v is not None}

    if "MTGP" not in _valid:
        fcm_pred_grid = mo.md("FCM data.")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 4))

        # Left: Spacing Comparison (MTGP vs 16-feat GP)
        _sp_cluster = {"MTGP": _valid["MTGP"], "GP": _valid.get("GP Spacing")}
        plot_comparison_overlaid({k: v for k, v in _sp_cluster.items() if v is not None}, target_idx=0, target_label="Spectral Spacing (GHz)", axs=_axs[0])
        # _axs[0].set_title("FCM - Spacing")

        # Right: OSNR Comparison (MTGP vs 16-feat GP)
        _osnr_cluster = {"MTGP": _valid["MTGP"], "GP": _valid.get("GP OSNR")}
        plot_comparison_overlaid({k: v for k, v in _osnr_cluster.items() if v is not None}, target_idx=1, target_label="OSNR (dB)", axs=_axs[1])
        # _axs[1].set_title("FCM - OSNR")

        _fig.tight_layout()
        Path("paper/figures").mkdir(parents=True, exist_ok=True)
        _fig.savefig("paper/figures/comparison_fcm_predictions.pdf", format="pdf", bbox_inches="tight")
        _fig.savefig("paper/figures/comparison_fcm_predictions.svg", format="svg", bbox_inches="tight")

        fcm_pred_grid = mo.ui.matplotlib(_axs[0])
    return (fcm_pred_grid,)


@app.cell
def _(gkm_results, mo, plot_comparison_overlaid, plt):
    # Figure 2: GKM Fair Predictions (16 Features)
    _valid = {k: v for k, v in gkm_results.items() if v is not None}

    if "MTGP" not in _valid:
        gkm_pred_grid = mo.md("GKM data.")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 4))

        # Left: Spacing Comparison (MTGP vs 16-feat GP)
        _sp_cluster = {"MTGP": _valid["MTGP"], "GP": _valid.get("GP Spacing")}
        plot_comparison_overlaid({k: v for k, v in _sp_cluster.items() if v is not None}, target_idx=0, target_label="Spectral Spacing (GHz)", axs=_axs[0])
        _axs[0].set_title("GKM - Spacing Prediction")

        # Right: OSNR Comparison (MTGP vs 16-feat GP)
        _osnr_cluster = {"MTGP": _valid["MTGP"], "GP": _valid.get("GP OSNR")}
        plot_comparison_overlaid({k: v for k, v in _osnr_cluster.items() if v is not None}, target_idx=1, target_label="OSNR (dB)", axs=_axs[1])
        _axs[1].set_title("GKM - OSNR Prediction")

        _fig.tight_layout()
        _fig.savefig("paper/figures/comparison_gkm_predictions.pdf", format="pdf", bbox_inches="tight")
        _fig.savefig("paper/figures/comparison_gkm_predictions.svg", format="svg", bbox_inches="tight")

        gkm_pred_grid = mo.ui.matplotlib(_axs[0])
    return (gkm_pred_grid,)


@app.cell
def _(fcm_results, mo, plot_comparison_violins, plt):
    # Figure 3: FCM Fair Violins
    _valid = {k: v for k, v in fcm_results.items() if v is not None}

    if "MTGP" not in _valid:
        fcm_violin_grid = mo.md("FCM data.")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 4))

        # Left: Spacing Errors
        _sp_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("GP Spacing")}
        plot_comparison_violins({k: v for k, v in _sp_cluster.items() if v is not None}, target_idx=0, target_label="Spectral Spacing (GHz)", axs=_axs[0])
        _axs[0].set_title("FCM - Spacing Error")

        # Right: OSNR Errors
        _osnr_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("GP OSNR")}
        plot_comparison_violins({k: v for k, v in _osnr_cluster.items() if v is not None}, target_idx=1, target_label="OSNR (dB)", axs=_axs[1])
        _axs[1].set_title("FCM - OSNR Error")

        _fig.tight_layout()
        _fig.savefig("paper/figures/comparison_fcm_violins.pdf", format="pdf", bbox_inches="tight")
        _fig.savefig("paper/figures/comparison_fcm_violins.svg", format="svg", bbox_inches="tight")

        fcm_violin_grid = mo.ui.matplotlib(_axs[0])
    return (fcm_violin_grid,)


@app.cell
def _(gkm_results, mo, plot_comparison_violins, plt):
    # Figure 4: GKM Violins
    _valid = {k: v for k, v in gkm_results.items() if v is not None}

    if "MTGP" not in _valid:
        gkm_violin_grid = mo.md("GKM data.")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 4))

        # Left: Spacing Errors
        _sp_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("GP Spacing")}
        plot_comparison_violins({k: v for k, v in _sp_cluster.items() if v is not None}, target_idx=0, target_label="Spectral Spacing (GHz)", axs=_axs[0])

        # Right: OSNR Errors
        _osnr_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("GP OSNR")}
        plot_comparison_violins({k: v for k, v in _osnr_cluster.items() if v is not None}, target_idx=1, target_label="OSNR (dB)", axs=_axs[1])
        _axs[1].set_ylabel("")

        _fig.tight_layout()
        _fig.savefig("paper/figures/comparison_gkm_violins.pdf", format="pdf", bbox_inches="tight")
        _fig.savefig("paper/figures/comparison_gkm_violins.svg", format="svg", bbox_inches="tight")

        gkm_violin_grid = mo.ui.matplotlib(_axs[0])
    return (gkm_violin_grid,)


@app.cell
def _(fcm_pred_grid, fcm_violin_grid, gkm_pred_grid, gkm_violin_grid, mo):
    mo.vstack([
        mo.md("## 1. FCM Comparison Study"),
        fcm_pred_grid,
        fcm_violin_grid,
        mo.md("---"),
        mo.md("## 2. GKM Comparison Study"),
        gkm_pred_grid,
        gkm_violin_grid
    ])
    return


@app.cell
def _(fcm_results, gkm_results, mo, np):
    def _fmt(m, key, idx=None):
        if m is None: return "N/A"
        val = m[key][idx] if (idx is not None and isinstance(m[key], (list, np.ndarray))) else m[key]
        return f"{val:.4f}"

    fair_comparison_table = mo.md(f"""
    ## Comparative Metrics (16-Feature Models Only)

    | Feature Method | Model | Spacing MAE | Spacing RMSE | Spacing R² | OSNR MAE | OSNR RMSE | OSNR R² |
    |---|---|---|---|---|---|---|---|
    | **FCM** | MTGP | `{_fmt(fcm_results['MTGP'], 'mae', 0)}` | `{_fmt(fcm_results['MTGP'], 'rmse', 0)}` | `{_fmt(fcm_results['MTGP'], 'r2', 0)}` | `{_fmt(fcm_results['MTGP'], 'mae', 1)}` | `{_fmt(fcm_results['MTGP'], 'rmse', 1)}` | `{_fmt(fcm_results['MTGP'], 'r2', 1)}` |
    | | GP (Baseline) | `{_fmt(fcm_results['GP Spacing'], 'mae')}` | `{_fmt(fcm_results['GP Spacing'], 'rmse')}` | `{_fmt(fcm_results['GP Spacing'], 'r2')}` | `{_fmt(fcm_results['GP OSNR'], 'mae')}` | `{_fmt(fcm_results['GP OSNR'], 'rmse')}` | `{_fmt(fcm_results['GP OSNR'], 'r2')}` |
    | **GKM** | MTGP | `{_fmt(gkm_results['MTGP'], 'mae', 0)}` | `{_fmt(gkm_results['MTGP'], 'rmse', 0)}` | `{_fmt(gkm_results['MTGP'], 'r2', 0)}` | `{_fmt(gkm_results['MTGP'], 'mae', 1)}` | `{_fmt(gkm_results['MTGP'], 'rmse', 1)}` | `{_fmt(gkm_results['MTGP'], 'r2', 1)}` |
    | | GP (Baseline) | `{_fmt(gkm_results['GP Spacing'], 'mae')}` | `{_fmt(gkm_results['GP Spacing'], 'rmse')}` | `{_fmt(gkm_results['GP Spacing'], 'r2')}` | `{_fmt(gkm_results['GP OSNR'], 'mae')}` | `{_fmt(gkm_results['GP OSNR'], 'rmse')}` | `{_fmt(gkm_results['GP OSNR'], 'r2')}` |
    """)

    fair_comparison_table
    return


if __name__ == "__main__":
    app.run()
