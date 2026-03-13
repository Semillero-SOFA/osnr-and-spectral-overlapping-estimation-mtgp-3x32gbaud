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
    # Multitask Gaussian Process: Training and Evaluation

    This notebook implements the training and evaluation pipeline for a Multitask (double regression) Gaussian Process (GP) model using the Counting Vectors features extracted in the previous step.

    **Objectives:**
    1.  **Load Processed Data:** Import the extracted counting vectors (FCM or GKM) and continuous targets.
    2.  **Define Model:** Initialize an Exact Multitask GP with `MultitaskMean` and `MultitaskKernel`.
    3.  **Evaluate Multiple Datasets:** We define DRY functions to run the entire pipeline for both FCM and GKM datasets, cleanly separating the training phase from evaluation.
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import gpytorch
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import marimo as mo

    from utils import (
        set_seed, get_device,
        MultitaskGPModel,
        fit_mtgp, run_inference,
        validate_checkpoint, save_checkpoint, MTGP_REQUIRED_KEYS,
        denormalize, compute_metrics_multitask,
        plot_predictions_multitask, plot_violin_multitask,
    )

    return (
        MTGP_REQUIRED_KEYS,
        MultitaskGPModel,
        Path,
        StandardScaler,
        compute_metrics_multitask,
        denormalize,
        fit_mtgp,
        get_device,
        gpytorch,
        mo,
        np,
        plot_predictions_multitask,
        plot_violin_multitask,
        run_inference,
        save_checkpoint,
        set_seed,
        torch,
        train_test_split,
        validate_checkpoint,
    )


@app.cell
def _(Path, StandardScaler, np, torch, train_test_split):
    def load_dataset(data_path, test_size=0.2, seed=42):
        path = Path(data_path)
        print(f"Loading data from {path}...")
        try:
            X = np.load(path / "X_features.npy")
            Y = np.load(path / "Y_targets.npy")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find data files in {path}. Ensure X_features.npy and Y_targets.npy exist."
            ) from e

        # Outlier removal
        OUTLIER_THRESHOLD_OSNR = 45.0
        OUTLIER_THRESHOLD_SPACING = 45.0
        mask = (Y[:, 0] <= OUTLIER_THRESHOLD_SPACING) & (Y[:, 1] <= OUTLIER_THRESHOLD_OSNR)
        X = X[mask]
        Y = Y[mask]

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

        return (
            torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(Y_train_std, dtype=torch.float32),
            torch.tensor(X_test_norm, dtype=torch.float32),
            torch.tensor(Y_test_std, dtype=torch.float32),
            scaler_x, y_mean, y_std
        )

    return (load_dataset,)


@app.cell
def _(get_device, set_seed):
    set_seed()
    device = get_device()
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Logic

    The function `train_mtgp` handles data loading, exact MLL optimization, and checkpoint saving. If a checkpoint already exists, it skips training to save time.
    """)
    return


@app.cell
def _(
    MTGP_REQUIRED_KEYS,
    Path,
    device,
    fit_mtgp,
    load_dataset,
    save_checkpoint,
    validate_checkpoint,
):
    def train_mtgp(dataset_name):
        print(f"\n{'='*50}")
        print(f"TRAINING PHASE: {dataset_name.upper()} Dataset")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        train_x, train_y, test_x, test_y, scaler_x, y_mean, y_std = load_dataset(data_path)

        _ARTIFACT_DIR = Path('artifacts')
        _ARTIFACT_DIR.mkdir(exist_ok=True)
        _ckpt_path = _ARTIFACT_DIR / f'multitask_gp_{dataset_name.lower()}.pt'

        if validate_checkpoint(_ckpt_path, MTGP_REQUIRED_KEYS):
            print(f'Valid checkpoint found at {_ckpt_path}. Skipping training.')
            return

        model, likelihood = fit_mtgp(
            train_x, train_y, device,
            desc=f"Training {dataset_name.upper()}",
        )

        # Retrieve the (possibly sub-sampled) training data from the model
        train_x_fit = model.train_inputs[0]
        train_y_fit = model.train_targets

        save_checkpoint(_ckpt_path, {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'train_x_fit': train_x_fit,
            'train_y_fit': train_y_fit,
            'scaler_x': scaler_x,
            'y_mean': y_mean,
            'y_std': y_std,
        })

    return (train_mtgp,)


@app.cell
def _(train_mtgp):
    train_mtgp("fcm")
    return


@app.cell
def _(train_mtgp):
    train_mtgp("gkm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation & Plotting Logic

    The `evaluate_mtgp` function restores the saved model checkpoint, runs inference on the test set, and visualizes the predictions using two routines:
    1. Overall raw prediction scatter plots with 95% CI (English labels).
    2. Violin precision plots across the true labels (English labels).
    """)
    return


@app.cell
def _(
    MultitaskGPModel,
    Path,
    compute_metrics_multitask,
    denormalize,
    device,
    gpytorch,
    load_dataset,
    mo,
    plot_predictions_multitask,
    plot_violin_multitask,
    run_inference,
    torch,
):
    def evaluate_mtgp(dataset_name):
        print(f"\n{'='*50}")
        print(f"EVALUATION PHASE: {dataset_name.upper()} Dataset")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        _, _, test_x, test_y, _, _, _ = load_dataset(data_path)

        _ckpt_path = Path('artifacts') / f'multitask_gp_{dataset_name.lower()}.pt'
        if not _ckpt_path.exists():
            print(f"Error: Model checkpoint {_ckpt_path} not found.")
            return None

        ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt['train_x_fit'].to(device)
        train_y_fit = ckpt['train_y_fit'].to(device)
        y_mean = ckpt['y_mean']
        y_std = ckpt['y_std']

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=test_y.shape[1]).to(device)
        model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])

        print("Running inference on test dataset...")
        mean, lower, upper = run_inference(model, likelihood, test_x, device)

        test_y_np = test_y.cpu().numpy()
        y_pred_denorm = denormalize(mean, y_mean, y_std)
        y_actual_denorm = denormalize(test_y_np, y_mean, y_std)
        lower_denorm = denormalize(lower, y_mean, y_std)
        upper_denorm = denormalize(upper, y_mean, y_std)

        mae, rmse = compute_metrics_multitask(y_pred_denorm, y_actual_denorm)

        print(f'\nMetrics ({dataset_name.upper()}):')
        print(f'MAE  - Spacing: {mae[0]:.3f} GHz | OSNR: {mae[1]:.3f} dB')
        print(f'RMSE - Spacing: {rmse[0]:.3f} GHz | OSNR: {rmse[1]:.3f} dB')

        metrics_table = mo.md(f"""
        ### Performance Metrics ({dataset_name.upper()})
        ---
        | Output | MAE | RMSE |
        |---|---|---|
        | **Spacing (GHz)** | `{mae[0]:.4f}` | `{rmse[0]:.4f}` |
        | **OSNR (dB)** | `{mae[1]:.4f}` | `{rmse[1]:.4f}` |
        """)

        ax_sp, ax_osnr = plot_predictions_multitask(
            y_actual_denorm, y_pred_denorm, lower_denorm, upper_denorm
        )
        ax_v_sp, ax_v_osnr = plot_violin_multitask(y_actual_denorm, y_pred_denorm)

        return mo.vstack([
            metrics_table,
            mo.md("#### Predictions and Confidence Intervals"),
            mo.hstack([mo.ui.matplotlib(ax_sp), mo.ui.matplotlib(ax_osnr)], justify="center"),
            mo.md("#### Precision Analysis (Violin)"),
            mo.hstack([mo.ui.matplotlib(ax_v_sp), mo.ui.matplotlib(ax_v_osnr)], justify="center"),
        ])

    return (evaluate_mtgp,)


@app.cell
def _(evaluate_mtgp):
    evaluate_mtgp("fcm")
    return


@app.cell
def _(evaluate_mtgp):
    evaluate_mtgp("gkm")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
