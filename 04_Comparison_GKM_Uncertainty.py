# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gpytorch>=1.15.2",
#     "marimo>=0.20.4",
#     "matplotlib>=3.10.8",
#     "numpy>=2.1.0",
#     "pandas>=2.2.0",
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
    mo.md(
        r"""
        # GKM Predictive Uncertainty: MTGP vs STGP

        This notebook compares predictive uncertainty across true value ranges for the GKM feature set.
        The uncertainty reference is taken as **2σ** from the predictive interval.
        """
    )
    return


@app.cell
def _():
    import gpytorch
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from pathlib import Path
    from scipy.interpolate import make_interp_spline
 
    from utils import (
        ExactGPModel,
        MultitaskGPModel,
        denormalize,
        get_device,
        load_processed_arrays,
        preprocess_data,
        run_inference,
    )
 
    return (
        ExactGPModel,
        MultitaskGPModel,
        Path,
        denormalize,
        get_device,
        gpytorch,
        load_processed_arrays,
        make_interp_spline,
        mo,
        np,
        pd,
        plt,
        preprocess_data,
        run_inference,
        sns,
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
    denormalize,
    device,
    gpytorch,
    load_processed_arrays,
    preprocess_data,
    run_inference,
    torch,
):
    def fetch_mtgp_data(dataset_name):
        X, Y, _ = load_processed_arrays(dataset_name)
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
        y_unc = 0.5 * (y_up - y_low)

        return {
            "y_act": y_act,
            "y_pred": y_pred,
            "y_low": y_low,
            "y_up": y_up,
            "y_unc": y_unc,
        }

    return (fetch_mtgp_data,)


@app.cell
def _(
    ExactGPModel,
    Path,
    denormalize,
    device,
    gpytorch,
    load_processed_arrays,
    preprocess_data,
    run_inference,
    torch,
):
    def fetch_stgp_data(target_label, dataset_name):
        X, Y, _ = load_processed_arrays(dataset_name)
        is_osnr = "OSNR" in target_label
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
        y_unc = 0.5 * (y_up - y_low)

        return {
            "y_act": y_test_raw,
            "y_pred": y_pred,
            "y_low": y_low,
            "y_up": y_up,
            "y_unc": y_unc,
        }

    return (fetch_stgp_data,)


@app.cell
def _(fetch_mtgp_data, fetch_stgp_data):
    gkm_results = {
        "MTGP": fetch_mtgp_data("gkm"),
        "STGP Spacing": fetch_stgp_data("Spectral Spacing (GHz)", "gkm"),
        "STGP OSNR": fetch_stgp_data("OSNR (dB)", "gkm"),
    }
    return (gkm_results,)


@app.cell
def _(np, pd, plt, sns, make_interp_spline):
    def plot_comparison_uncertainty(
        results_dict,
        target_idx,
        target_label,
        axs=None,
        num_bins=6,
    ):
        if axs is None:
            _, ax = plt.subplots(figsize=(14, 6))
        else:
            ax = axs
 
        keys = list(results_dict.keys())
        custom_colors = {
            "MTGP": "#0d4d2e",
            "STGP": "#2d8659",
        }
        custom_markers = {
            "MTGP": "o",
            "STGP": "s",
        }
        palette = {k: custom_colors.get(k, "#0d4d2e") for k in keys}
 
        for i, (name, res) in enumerate(results_dict.items()):
            if res is None:
                continue
 
            y_act = res["y_act"]
            y_unc = res["y_unc"]
            if y_act.ndim > 1:
                y_act = y_act[:, target_idx]
                y_unc = y_unc[:, target_idx]
 
            bins = np.linspace(y_act.min(), y_act.max(), num_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            indices = np.clip(np.digitize(y_act, bins) - 1, 0, num_bins - 1)
 
            means = []
            stds = []
            for bin_idx in range(num_bins):
                values = y_unc[indices == bin_idx]
                if values.size == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                means.append(np.mean(values))
                stds.append(np.std(values))
 
            means = np.asarray(means)
            stds = np.asarray(stds)
            valid = ~np.isnan(means)
 
            x_valid = bin_centers[valid]
            y_valid = means[valid]
            lower_valid = np.maximum(means[valid] - 2.0 * stds[valid], 0.0)
            upper_valid = means[valid] + 2.0 * stds[valid]
 
            if len(x_valid) > 3:
                spl = make_interp_spline(x_valid, y_valid, k=3)
                spl_lower = make_interp_spline(x_valid, lower_valid, k=3)
                spl_upper = make_interp_spline(x_valid, upper_valid, k=3)
 
                x_smooth = np.linspace(x_valid.min(), x_valid.max(), 300)
                y_smooth = spl(x_smooth)
                lower_smooth = spl_lower(x_smooth)
                upper_smooth = spl_upper(x_smooth)
            else:
                x_smooth = x_valid
                y_smooth = y_valid
                lower_smooth = lower_valid
                upper_smooth = upper_valid
 
            marker = custom_markers.get(name, "o")
            ax.plot(
                x_valid,
                y_valid,
                marker=marker,
                linewidth=0,
                markersize=5,
                color=palette[name],
                label=name,
                zorder=3,
            )
            ax.plot(
                x_smooth,
                y_smooth,
                linewidth=1.8,
                color=palette[name],
                zorder=2,
            )
            ax.fill_between(
                x_smooth,
                lower_smooth,
                upper_smooth,
                color=palette[name],
                alpha=0.2,
                zorder=1,
            )

        ax.set_xlabel(target_label)
        ax.set_ylabel("Mean 2σ Uncertainty")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Method", loc="upper right")

        if axs is None:
            ax.get_figure().tight_layout()
        return ax

    return (plot_comparison_uncertainty,)


@app.cell
def _(Path, gkm_results, mo, plot_comparison_uncertainty, plt):
    _valid = {k: v for k, v in gkm_results.items() if v is not None}

    if "MTGP" not in _valid:
        gkm_uncertainty_grid = mo.md("GKM data.")
    else:
        _fig, _axs = plt.subplots(1, 2, figsize=(9, 4))

        _sp_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("STGP Spacing")}
        plot_comparison_uncertainty(
            {k: v for k, v in _sp_cluster.items() if v is not None},
            target_idx=0,
            target_label="Spectral Spacing (GHz)",
            axs=_axs[0],
        )

        _osnr_cluster = {"MTGP": _valid["MTGP"], "STGP": _valid.get("STGP OSNR")}
        plot_comparison_uncertainty(
            {k: v for k, v in _osnr_cluster.items() if v is not None},
            target_idx=1,
            target_label="OSNR (dB)",
            axs=_axs[1],
        )
        _axs[1].set_ylabel("")

        _fig.tight_layout()
        Path("paper/figures").mkdir(parents=True, exist_ok=True)
        _fig.savefig("paper/figures/comparison_gkm_uncertainty.pdf", format="pdf", bbox_inches="tight")
        _fig.savefig("paper/figures/comparison_gkm_uncertainty.svg", format="svg", bbox_inches="tight")

        gkm_uncertainty_grid = mo.ui.matplotlib(_axs[0])
    return (gkm_uncertainty_grid,)


@app.cell
def _(gkm_uncertainty_grid, mo):
    mo.vstack(
        [
            mo.md("## GKM Predictive Uncertainty"),
            gkm_uncertainty_grid,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
