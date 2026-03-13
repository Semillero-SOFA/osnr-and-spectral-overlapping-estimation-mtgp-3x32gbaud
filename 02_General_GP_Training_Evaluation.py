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
    # General Gaussian Process: Training and Evaluation

    This notebook evaluates single-output Exact Gaussian Process (GP) models against the dataset. This explicitly drops the "multitask" nature of the MTGP.

    We train 4 specific configurations per dataset (FCM/GKM):
    1.  **OSNR ONLY:** Predicts OSNR using only the 16 features.
    2.  **OSNR + SPACING:** Predicts OSNR using the 16 features + the True Spacing.
    3.  **SPACING ONLY:** Predicts Spacing using only the 16 features.
    4.  **SPACING + OSNR:** Predicts Spacing using the 16 features + the True OSNR.
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
        ExactGPModel,
        fit_gp, run_inference,
        validate_checkpoint, save_checkpoint, GP_REQUIRED_KEYS,
        denormalize, compute_metrics_single,
        plot_predictions_single, plot_violin_single,
    )

    return (
        ExactGPModel,
        Path,
        StandardScaler,
        compute_metrics_single,
        denormalize,
        fit_gp,
        get_device,
        gpytorch,
        mo,
        np,
        plot_predictions_single,
        plot_violin_single,
        run_inference,
        save_checkpoint,
        set_seed,
        torch,
        train_test_split,
        validate_checkpoint,
        GP_REQUIRED_KEYS,
    )


@app.cell
def _(Path, StandardScaler, np, torch, train_test_split):
    def load_dataset_gp(data_path, test_size=0.2, seed=42):
        path = Path(data_path)
        print(f"Loading data from {path}...")
        try:
            X = np.load(path / "X_features.npy")
            Y = np.load(path / "Y_targets.npy")  # [Spacing, OSNR]
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

        # Individual per-output standardization
        y_mean_spacing = Y_train[:, 0].mean(axis=0)
        y_std_spacing = Y_train[:, 0].std(axis=0) + 1e-6
        Y_train_spacing_std = (Y_train[:, 0] - y_mean_spacing) / y_std_spacing
        Y_test_spacing_std = (Y_test[:, 0] - y_mean_spacing) / y_std_spacing

        y_mean_osnr = Y_train[:, 1].mean(axis=0)
        y_std_osnr = Y_train[:, 1].std(axis=0) + 1e-6
        Y_train_osnr_std = (Y_train[:, 1] - y_mean_osnr) / y_std_osnr
        Y_test_osnr_std = (Y_test[:, 1] - y_mean_osnr) / y_std_osnr

        # Augmented feature sets: 16 + True Spacing
        X_train_plus_spacing = np.column_stack((X_train, Y_train[:, 0]))
        X_test_plus_spacing = np.column_stack((X_test, Y_test[:, 0]))
        scaler_x_plus_spacing = StandardScaler()
        X_train_plus_spacing_norm = scaler_x_plus_spacing.fit_transform(X_train_plus_spacing)
        X_test_plus_spacing_norm = scaler_x_plus_spacing.transform(X_test_plus_spacing)

        # 16 + True OSNR
        X_train_plus_osnr = np.column_stack((X_train, Y_train[:, 1]))
        X_test_plus_osnr = np.column_stack((X_test, Y_test[:, 1]))
        scaler_x_plus_osnr = StandardScaler()
        X_train_plus_osnr_norm = scaler_x_plus_osnr.fit_transform(X_train_plus_osnr)
        X_test_plus_osnr_norm = scaler_x_plus_osnr.transform(X_test_plus_osnr)

        return {
            'x_norm': (torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(X_test_norm, dtype=torch.float32)),
            'x_plus_spacing_norm': (torch.tensor(X_train_plus_spacing_norm, dtype=torch.float32), torch.tensor(X_test_plus_spacing_norm, dtype=torch.float32)),
            'x_plus_osnr_norm': (torch.tensor(X_train_plus_osnr_norm, dtype=torch.float32), torch.tensor(X_test_plus_osnr_norm, dtype=torch.float32)),
            'y_spacing_std': (torch.tensor(Y_train_spacing_std, dtype=torch.float32), torch.tensor(Y_test_spacing_std, dtype=torch.float32)),
            'y_osnr_std': (torch.tensor(Y_train_osnr_std, dtype=torch.float32), torch.tensor(Y_test_osnr_std, dtype=torch.float32)),
            'y_true_raw': torch.tensor(Y_test, dtype=torch.float32),
            'scalars': {
                'x': scaler_x,
                'x_plus_spacing': scaler_x_plus_spacing,
                'x_plus_osnr': scaler_x_plus_osnr,
                'mean_spacing': y_mean_spacing,
                'std_spacing': y_std_spacing,
                'mean_osnr': y_mean_osnr,
                'std_osnr': y_std_osnr
            }
        }

    return (load_dataset_gp,)


@app.cell
def _(set_seed, get_device):
    set_seed()
    device = get_device()
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Logic

    We define a reusable loop that takes a specific config tag (e.g. `osnr_only`, `osnr_plus_spacing`) and trains the exact GP.
    """)
    return


@app.cell
def _(
    Path,
    device,
    fit_gp,
    save_checkpoint,
    validate_checkpoint,
    GP_REQUIRED_KEYS,
):
    def train_gp_config(dataset_name, config_name, train_x, train_y, scalars):
        print(f"\n{'='*50}")
        print(f"TRAINING PHASE: {dataset_name.upper()} | {config_name}")
        print(f"{'='*50}")

        _ARTIFACT_DIR = Path('artifacts')
        _ARTIFACT_DIR.mkdir(exist_ok=True)
        _ckpt_path = _ARTIFACT_DIR / f'gp_{config_name}_{dataset_name.lower()}.pt'

        if validate_checkpoint(_ckpt_path, GP_REQUIRED_KEYS):
            print(f'Valid checkpoint found at {_ckpt_path}. Skipping training.')
            return

        model, likelihood = fit_gp(
            train_x, train_y, device,
            desc=f"Training {config_name}",
        )

        train_x_fit = model.train_inputs[0]
        train_y_fit = model.train_targets

        save_checkpoint(_ckpt_path, {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'train_x_fit': train_x_fit,
            'train_y_fit': train_y_fit,
            'scalars': scalars,
        })

    return (train_gp_config,)


@app.cell
def _(load_dataset_gp, train_gp_config):
    def run_training_suite(dataset_name):
        data_path = f"processed_data/{dataset_name.lower()}"
        data = load_dataset_gp(data_path)

        x_tr, _ = data['x_norm']
        x_plus_sp_tr, _ = data['x_plus_spacing_norm']
        x_plus_osnr_tr, _ = data['x_plus_osnr_norm']

        y_sp_tr, _ = data['y_spacing_std']
        y_osnr_tr, _ = data['y_osnr_std']

        sc = data['scalars']

        # 1. OSNR ONLY
        train_gp_config(dataset_name, 'osnr_only', x_tr, y_osnr_tr, sc)
        # 2. OSNR + SPACING
        train_gp_config(dataset_name, 'osnr_plus_spacing', x_plus_sp_tr, y_osnr_tr, sc)
        # 3. SPACING ONLY
        train_gp_config(dataset_name, 'spacing_only', x_tr, y_sp_tr, sc)
        # 4. SPACING + OSNR
        train_gp_config(dataset_name, 'spacing_plus_osnr', x_plus_osnr_tr, y_sp_tr, sc)

        return "Training Suite Complete."

    return (run_training_suite,)


@app.cell
def _(run_training_suite):
    run_training_suite("fcm")
    return


@app.cell
def _(run_training_suite):
    run_training_suite("gkm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation & Plotting Logic

    We evaluate the models one-by-one and plot them using violin plots for precision analysis.
    """)
    return


@app.cell
def _(
    ExactGPModel,
    Path,
    compute_metrics_single,
    denormalize,
    device,
    gpytorch,
    load_dataset_gp,
    mo,
    plot_predictions_single,
    plot_violin_single,
    run_inference,
    torch,
):
    def evaluate_gp_config(dataset_name, config_name, test_x, test_y, y_mean, y_std, target_label):
        _ckpt_path = Path('artifacts') / f'gp_{config_name}_{dataset_name.lower()}.pt'
        if not _ckpt_path.exists():
            return mo.md(f"Error: Model checkpoint {_ckpt_path} not found."), None

        ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt['train_x_fit'].to(device)
        train_y_fit = ckpt['train_y_fit'].to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x_fit, train_y_fit, likelihood).to(device)
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])

        mean, lower, upper = run_inference(model, likelihood, test_x, device)

        y_pred_denorm = denormalize(mean, y_mean, y_std)
        y_actual_denorm = denormalize(test_y, y_mean, y_std)
        lower_denorm = denormalize(lower, y_mean, y_std)
        upper_denorm = denormalize(upper, y_mean, y_std)

        mae, rmse = compute_metrics_single(y_pred_denorm, y_actual_denorm)

        ax_pred = plot_predictions_single(y_actual_denorm, y_pred_denorm, lower_denorm, upper_denorm, target_label)
        ax_violin = plot_violin_single(y_actual_denorm, y_pred_denorm, target_label)

        ui_component = mo.vstack([
            mo.md(f"### {config_name.upper()}"),
            mo.md(f"**MAE:** `{mae:.4f}` | **RMSE:** `{rmse:.4f}`"),
            mo.hstack([mo.ui.matplotlib(ax_pred), mo.ui.matplotlib(ax_violin)])
        ])

        return ui_component, {'mae': mae, 'rmse': rmse}

    def evaluate_all_gps(dataset_name):
        data_path = f"processed_data/{dataset_name.lower()}"
        data = load_dataset_gp(data_path)

        _, x_te = data['x_norm']
        _, x_plus_sp_te = data['x_plus_spacing_norm']
        _, x_plus_osnr_te = data['x_plus_osnr_norm']

        _, y_sp_te = data['y_spacing_std']
        _, y_osnr_te = data['y_osnr_std']

        sc = data['scalars']
        y_mean_osnr = sc['mean_osnr']
        y_std_osnr = sc['std_osnr']
        y_mean_sp = sc['mean_spacing']
        y_std_sp = sc['std_spacing']

        # OSNR Only
        ui_o_only, m_o_only = evaluate_gp_config(dataset_name, 'osnr_only', x_te, y_osnr_te.cpu().numpy(), y_mean_osnr, y_std_osnr, 'OSNR (dB)')
        # OSNR + Spacing
        ui_o_sp, m_o_sp = evaluate_gp_config(dataset_name, 'osnr_plus_spacing', x_plus_sp_te, y_osnr_te.cpu().numpy(), y_mean_osnr, y_std_osnr, 'OSNR (dB)')
        # Spacing Only
        ui_s_only, m_s_only = evaluate_gp_config(dataset_name, 'spacing_only', x_te, y_sp_te.cpu().numpy(), y_mean_sp, y_std_sp, 'Spectral Spacing (GHz)')
        # Spacing + OSNR
        ui_s_o, m_s_o = evaluate_gp_config(dataset_name, 'spacing_plus_osnr', x_plus_osnr_te, y_sp_te.cpu().numpy(), y_mean_sp, y_std_sp, 'Spectral Spacing (GHz)')

        table = mo.md(f"""
        ## Comparison Metrics ({dataset_name.upper()})

        | Configuration | MAE | RMSE |
        |---|---|---|
        | **OSNR ONLY** | `{m_o_only['mae']:.4f}` | `{m_o_only['rmse']:.4f}` |
        | **OSNR + SPACING** | `{m_o_sp['mae']:.4f}` | `{m_o_sp['rmse']:.4f}` |
        | **SPACING ONLY** | `{m_s_only['mae']:.4f}` | `{m_s_only['rmse']:.4f}` |
        | **SPACING + OSNR** | `{m_s_o['mae']:.4f}` | `{m_s_o['rmse']:.4f}` |
        """)

        return mo.vstack([
            mo.md(f"# {dataset_name.upper()} Independent GP Evaluations"),
            table,
            ui_o_only,
            ui_o_sp,
            ui_s_only,
            ui_s_o,
            mo.md("---")
        ])

    return (evaluate_all_gps,)


@app.cell
def _(evaluate_all_gps):
    fcm_report = evaluate_all_gps("fcm")
    fcm_report
    return


@app.cell
def _(evaluate_all_gps):
    gkm_report = evaluate_all_gps("gkm")
    gkm_report
    return


if __name__ == "__main__":
    app.run()
