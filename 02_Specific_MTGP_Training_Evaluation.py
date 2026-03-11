# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gpytorch>=1.15.2",
#     "marimo>=0.20.4",
#     "matplotlib>=3.10.8",
#     "numpy>=2.1.0",
#     "pyzmq>=26.0.0",
#     "scikit-learn>=1.5.0",
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
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from tqdm.notebook import tqdm
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import marimo as mo

    return (
        Path,
        StandardScaler,
        gpytorch,
        mo,
        np,
        plt,
        ticker,
        torch,
        tqdm,
        train_test_split,
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
def _(gpytorch):
    class MultitaskGPModel(gpytorch.models.ExactGP):
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

    return (MultitaskGPModel,)


@app.cell
def _(np, torch):
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    MultitaskGPModel,
    Path,
    device,
    gpytorch,
    load_scenario_dataset,
    torch,
    tqdm,
):
    def train_scenario_mtgp(dataset_name, dist, pwr):
        print(f"\n{'='*50}")
        print(f"TRAINING PHASE: {dataset_name.upper()} Dataset | {dist} km | {pwr} dBm")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        train_x, train_y, test_x, test_y, scaler_x, y_mean, y_std = load_scenario_dataset(data_path, dist, pwr)

        train_x = train_x.to(device)
        train_y = train_y.to(device)

        max_train_points = 5000
        if train_x.size(0) > max_train_points:
            sel = torch.randperm(train_x.size(0))[:max_train_points]
            train_x_fit = train_x[sel]
            train_y_fit = train_y[sel]
        else:
            train_x_fit = train_x
            train_y_fit = train_y

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)
        model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)

        _ARTIFACT_DIR = Path('artifacts')
        _ARTIFACT_DIR.mkdir(exist_ok=True)
        ckpt_name = f'mtgp_{dataset_name.lower()}_{dist}km_{pwr}dbm.pt'
        _ckpt_path = _ARTIFACT_DIR / ckpt_name

        if _ckpt_path.exists():
            try:
                ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                required_keys = {'model_state_dict', 'likelihood_state_dict', 'train_x_fit', 'train_y_fit', 'scaler_x', 'y_mean', 'y_std'}
                if required_keys.issubset(ckpt.keys()):
                    print(f'Valid checkpoint found at {_ckpt_path}. Skipping training.')
                    return
            except Exception as e:
                print(f"Error reading checkpoint: {e}. Retraining...")

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        print(f"Training Model ({ckpt_name})...")
        training_iterations = 200 # Adjust as needed
        for i in tqdm(range(training_iterations), desc=f"Training {ckpt_name}"):
            optimizer.zero_grad()
            output = model(train_x_fit)
            loss = -mll(output, train_y_fit)
            loss.backward()
            optimizer.step()

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'train_x_fit': train_x_fit,
            'train_y_fit': train_y_fit,
            'scaler_x': scaler_x,
            'y_mean': y_mean,
            'y_std': y_std
        }
        torch.save(checkpoint, _ckpt_path)
        print(f'Training complete. Checkpoint saved to {_ckpt_path}')

    return (train_scenario_mtgp,)


@app.cell
def _(train_scenario_mtgp):
    # Train FMC Models
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
    device,
    gpytorch,
    load_scenario_dataset,
    mo,
    np,
    plt,
    ticker,
    torch,
):
    def plot_predictions(y_act, y_pred, lower_denorm, upper_denorm, title_suffix):
        sort_indices_spacing = np.argsort(y_act[:, 0])
        sort_indices_osnr = np.argsort(y_act[:, 1])

        # Spacing Plot
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        x_spacing = np.arange(len(sort_indices_spacing))
        ax1.plot(x_spacing, y_act[sort_indices_spacing, 0], 'k.', label='True', alpha=0.7)
        ax1.plot(x_spacing, y_pred[sort_indices_spacing, 0], 'b-', label='Predicted', linewidth=2)
        ax1.fill_between(x_spacing, lower_denorm[sort_indices_spacing, 0], upper_denorm[sort_indices_spacing, 0], alpha=0.3, label='95% CI')
        ax1.set_ylabel('Spectral Spacing (GHz)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        ui_ax1 = mo.ui.matplotlib(ax1)

        # OSNR Plot
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        x_osnr = np.arange(len(sort_indices_osnr))
        ax2.plot(x_osnr, y_act[sort_indices_osnr, 1], 'k.', label='True', alpha=0.7)
        ax2.plot(x_osnr, y_pred[sort_indices_osnr, 1], 'b-', label='Predicted', linewidth=2)
        ax2.fill_between(x_osnr, lower_denorm[sort_indices_osnr, 1], upper_denorm[sort_indices_osnr, 1], alpha=0.3, label='95% CI')
        ax2.set_ylabel('OSNR (dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        ui_ax2 = mo.ui.matplotlib(ax2)

        return mo.hstack([ui_ax1, ui_ax2], justify="center")

    def plot_binned_metrics(y_act, y_pred, title_suffix):
        # 1. Spacing (discrete)
        y_true_spacing = y_act[:, 0]
        y_pred_spacing = y_pred[:, 0]
        abs_errors_spacing = np.abs(y_pred_spacing - y_true_spacing)

        unique_spacing = np.unique(y_true_spacing)
        spacing_means = []
        spacing_stds = []
        for val in unique_spacing:
            mask = np.isclose(y_true_spacing, val)
            spacing_means.append(np.mean(abs_errors_spacing[mask]))
            spacing_stds.append(np.std(abs_errors_spacing[mask]))

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.errorbar(unique_spacing, spacing_means, yerr=spacing_stds, fmt='o', capsize=4, mfc='white')
        ax1.set_xlabel('Spectral Spacing (GHz)')
        ax1.set_ylabel('MAE ± STD')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        fig1.tight_layout()
        ui_ax1 = mo.ui.matplotlib(ax1)

        # 2. OSNR (continuous -> binned)
        y_true_osnr = y_act[:, 1]
        y_pred_osnr = y_pred[:, 1]
        abs_errors_osnr = np.abs(y_true_osnr - y_pred_osnr)

        num_bins = 10
        bins = np.linspace(np.min(y_true_osnr), np.max(y_true_osnr), num_bins + 1)
        indices = np.digitize(y_true_osnr, bins) - 1
        indices = np.clip(indices, 0, num_bins - 1)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        osnr_means = []
        osnr_stds = []
        valid_centers = []
        for b in range(num_bins):
            mask = (indices == b)
            if np.any(mask):
                osnr_means.append(np.mean(abs_errors_osnr[mask]))
                osnr_stds.append(np.std(abs_errors_osnr[mask]))
                valid_centers.append(bin_centers[b])

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.errorbar(valid_centers, osnr_means, yerr=osnr_stds, fmt='s', capsize=4, mfc='white')
        ax2.set_xlabel('OSNR (dB)')
        ax2.set_ylabel('MAE ± STD')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        fig2.tight_layout()
        ui_ax2 = mo.ui.matplotlib(ax2)

        return mo.hstack([ui_ax1, ui_ax2], justify="center")

    def evaluate_scenario_mtgp(dataset_name, dist, pwr):
        data_path = f"processed_data/{dataset_name.lower()}"
        _, _, test_x, test_y, _, _, _ = load_scenario_dataset(data_path, dist, pwr)
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        ckpt_name = f'mtgp_{dataset_name.lower()}_{dist}km_{pwr}dbm.pt'
        _ckpt_path = Path('artifacts') / ckpt_name

        if not _ckpt_path.exists():
            return mo.md(f"**Error:** Model checkpoint {ckpt_name} not found."), None, None

        ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        train_x_fit = ckpt['train_x_fit'].to(device)
        train_y_fit = ckpt['train_y_fit'].to(device)
        y_mean = ckpt['y_mean']
        y_std = ckpt['y_std']

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=test_y.shape[1]).to(device)
        model = MultitaskGPModel(train_x_fit, train_y_fit, likelihood).to(device)

        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            mean = observed_pred.mean.cpu().numpy()
            lower, upper = observed_pred.confidence_region()
            lower = lower.cpu().numpy()
            upper = upper.cpu().numpy()

        test_y_np = test_y.cpu().numpy()
        y_pred_denorm = mean * y_std + y_mean
        y_actual_denorm = test_y_np * y_std + y_mean
        lower_denorm = lower * y_std + y_mean
        upper_denorm = upper * y_std + y_mean

        mae = np.mean(np.abs(y_pred_denorm - y_actual_denorm), axis=0)
        rmse = np.sqrt(np.mean((y_pred_denorm - y_actual_denorm) ** 2, axis=0))

        title_suffix = f"({dataset_name.upper()} | {dist}km | {pwr}dBm)"
        plots1 = plot_predictions(y_actual_denorm, y_pred_denorm, lower_denorm, upper_denorm, title_suffix)
        plots2 = plot_binned_metrics(y_actual_denorm, y_pred_denorm, title_suffix)

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
            plots1,
            mo.md("#### Precision Analysis (Binned MAE)"),
            plots2,
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
