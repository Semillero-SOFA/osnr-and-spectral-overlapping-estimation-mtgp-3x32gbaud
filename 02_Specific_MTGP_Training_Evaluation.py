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
    # Specific Scenario MTGP Evaluation

    This notebook trains and evaluates Multitask Gaussian Process (MTGP) models individually for three distinct scenarios defined by `[Distance (km), Power (dBm)]`:
    1.  **Scenario A:** 0 km, 0 dBm
    2.  **Scenario B:** 270 km, 0 dBm
    3.  **Scenario C:** 270 km, 9 dBm

    **Objectives:**
    - Load the processed counting vectors and filter them by the specific `Distance` and `Power` metadata.
    - Train individual MTGP models for each scenario.
    - Evaluate and compare the performance across the scenarios.
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
        MTGP_REQUIRED_KEYS,
    )


@app.cell
def _(Path, StandardScaler, np, torch, train_test_split):
    def load_scenario_dataset(data_path, target_distance, target_power, test_size=0.2, seed=42):
        path = Path(data_path)
        try:
            X_all = np.load(path / "X_features.npy")
            Y_all = np.load(path / "Y_targets.npy")
            M_all = np.load(path / "M_metadata.npy")  # [Distance, Power]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find data files in {path}. Ensure X, Y, and M arrays exist."
            ) from e

        # Filter by scenario
        scenario_mask = np.isclose(M_all[:, 0], target_distance) & np.isclose(M_all[:, 1], target_power)
        X = X_all[scenario_mask]
        Y = Y_all[scenario_mask]

        if len(X) == 0:
            raise ValueError(f"No samples found for scenario: {target_distance} km, {target_power} dBm")

        print(f"Loaded {len(X)} samples for {target_distance} km, {target_power} dBm from {data_path}")

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

    return (load_scenario_dataset,)


@app.cell
def _(set_seed, get_device):
    set_seed()
    device = get_device()
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scenario Specific Training Logic

    The `train_scenario_mtgp` function will check for specifically named checkpoints indicating both the dataset algorithm (FCM/GKM) and the scenario parameters.
    """)
    return


@app.cell
def _(
    Path,
    device,
    fit_mtgp,
    load_scenario_dataset,
    save_checkpoint,
    validate_checkpoint,
    MTGP_REQUIRED_KEYS,
):
    def train_scenario_mtgp(dataset_name, dist, pwr):
        print(f"\n{'='*50}")
        print(f"TRAINING PHASE: {dataset_name.upper()} Dataset | {dist} km | {pwr} dBm")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        train_x, train_y, test_x, test_y, scaler_x, y_mean, y_std = load_scenario_dataset(data_path, dist, pwr)

        _ARTIFACT_DIR = Path('artifacts')
        _ARTIFACT_DIR.mkdir(exist_ok=True)
        ckpt_name = f'mtgp_{dataset_name.lower()}_{dist}km_{pwr}dbm.pt'
        _ckpt_path = _ARTIFACT_DIR / ckpt_name

        if validate_checkpoint(_ckpt_path, MTGP_REQUIRED_KEYS):
            print(f'Valid checkpoint found at {_ckpt_path}. Skipping training.')
            return

        model, likelihood = fit_mtgp(
            train_x, train_y, device,
            desc=f"Training {ckpt_name}",
        )

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

    return (train_scenario_mtgp,)


@app.cell
def _(train_scenario_mtgp):
    # Train FCM Models
    train_scenario_mtgp("fcm", 0, 0)
    train_scenario_mtgp("fcm", 270, 0)
    train_scenario_mtgp("fcm", 270, 9)
    return


@app.cell
def _(train_scenario_mtgp):
    # Train GKM Models
    train_scenario_mtgp("gkm", 0, 0)
    train_scenario_mtgp("gkm", 270, 0)
    train_scenario_mtgp("gkm", 270, 9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation per Scenario
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
    load_scenario_dataset,
    mo,
    np,
    plot_predictions_multitask,
    plot_violin_multitask,
    run_inference,
    torch,
):
    def evaluate_scenario_mtgp(dataset_name, dist, pwr):
        data_path = f"processed_data/{dataset_name.lower()}"
        _, _, test_x, test_y, _, _, _ = load_scenario_dataset(data_path, dist, pwr)

        ckpt_name = f'mtgp_{dataset_name.lower()}_{dist}km_{pwr}dbm.pt'
        _ckpt_path = Path('artifacts') / ckpt_name

        if not _ckpt_path.exists():
            return mo.md(f"**Error:** Model checkpoint {ckpt_name} not found."), None

        ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt['train_x_fit'].to(device)
        train_y_fit = ckpt['train_y_fit'].to(device)
        y_mean = ckpt['y_mean']
        y_std = ckpt['y_std']

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=test_y.shape[1]).to(device)
        model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])

        mean, lower, upper = run_inference(model, likelihood, test_x, device)

        test_y_np = test_y.cpu().numpy()
        y_pred_denorm = denormalize(mean, y_mean, y_std)
        y_actual_denorm = denormalize(test_y_np, y_mean, y_std)
        lower_denorm = denormalize(lower, y_mean, y_std)
        upper_denorm = denormalize(upper, y_mean, y_std)

        mae, rmse = compute_metrics_multitask(y_pred_denorm, y_actual_denorm)

        ax_sp, ax_osnr = plot_predictions_multitask(
            y_actual_denorm, y_pred_denorm, lower_denorm, upper_denorm
        )
        ax_v_sp, ax_v_osnr = plot_violin_multitask(y_actual_denorm, y_pred_denorm)

        metrics = {"mae": mae.tolist(), "rmse": rmse.tolist()}

        ui_component = mo.vstack([
            mo.md(f"### Scenario: {dist} km, {pwr} dBm"),
            mo.md(f"""
            | Output | MAE | RMSE |
            |---|---|---|
            | **Spacing (GHz)** | `{mae[0]:.4f}` | `{rmse[0]:.4f}` |
            | **OSNR (dB)** | `{mae[1]:.4f}` | `{rmse[1]:.4f}` |
            """),
            mo.md("#### Predictions and Confidence Intervals"),
            mo.hstack([mo.ui.matplotlib(ax_sp), mo.ui.matplotlib(ax_osnr)], justify="center"),
            mo.md("#### Precision Analysis (Violin)"),
            mo.hstack([mo.ui.matplotlib(ax_v_sp), mo.ui.matplotlib(ax_v_osnr)], justify="center"),
            mo.md("---")
        ])

        return ui_component, metrics

    return (evaluate_scenario_mtgp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FCM Evaluation & Comparison
    """)
    return


@app.cell
def _(evaluate_scenario_mtgp, mo):
    fcm_ui_1, m_fcm_1 = evaluate_scenario_mtgp("fcm", 0, 0)
    fcm_ui_2, m_fcm_2 = evaluate_scenario_mtgp("fcm", 270, 0)
    fcm_ui_3, m_fcm_3 = evaluate_scenario_mtgp("fcm", 270, 9)

    fcm_comparison = mo.md(f"""
    ## FCM Overall Comparison

    | Scenario | Spacing MAE | Spacing RMSE | OSNR MAE | OSNR RMSE |
    |---|---|---|---|---|
    | **0 km, 0 dBm**   | `{m_fcm_1['mae'][0]:.4f}` | `{m_fcm_1['rmse'][0]:.4f}` | `{m_fcm_1['mae'][1]:.4f}` | `{m_fcm_1['rmse'][1]:.4f}` |
    | **270 km, 0 dBm** | `{m_fcm_2['mae'][0]:.4f}` | `{m_fcm_2['rmse'][0]:.4f}` | `{m_fcm_2['mae'][1]:.4f}` | `{m_fcm_2['rmse'][1]:.4f}` |
    | **270 km, 9 dBm** | `{m_fcm_3['mae'][0]:.4f}` | `{m_fcm_3['rmse'][0]:.4f}` | `{m_fcm_3['mae'][1]:.4f}` | `{m_fcm_3['rmse'][1]:.4f}` |
    """)

    mo.vstack([fcm_ui_1, fcm_ui_2, fcm_ui_3, fcm_comparison])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GKM Evaluation & Comparison
    """)
    return


@app.cell
def _(evaluate_scenario_mtgp, mo):
    gkm_ui_1, m_gkm_1 = evaluate_scenario_mtgp("gkm", 0, 0)
    gkm_ui_2, m_gkm_2 = evaluate_scenario_mtgp("gkm", 270, 0)
    gkm_ui_3, m_gkm_3 = evaluate_scenario_mtgp("gkm", 270, 9)

    gkm_comparison = mo.md(f"""
    ## GKM Overall Comparison

    | Scenario | Spacing MAE | Spacing RMSE | OSNR MAE | OSNR RMSE |
    |---|---|---|---|---|
    | **0 km, 0 dBm**   | `{m_gkm_1['mae'][0]:.4f}` | `{m_gkm_1['rmse'][0]:.4f}` | `{m_gkm_1['mae'][1]:.4f}` | `{m_gkm_1['rmse'][1]:.4f}` |
    | **270 km, 0 dBm** | `{m_gkm_2['mae'][0]:.4f}` | `{m_gkm_2['rmse'][0]:.4f}` | `{m_gkm_2['mae'][1]:.4f}` | `{m_gkm_2['rmse'][1]:.4f}` |
    | **270 km, 9 dBm** | `{m_gkm_3['mae'][0]:.4f}` | `{m_gkm_3['rmse'][0]:.4f}` | `{m_gkm_3['mae'][1]:.4f}` | `{m_gkm_3['rmse'][1]:.4f}` |
    """)

    mo.vstack([gkm_ui_1, gkm_ui_2, gkm_ui_3, gkm_comparison])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
