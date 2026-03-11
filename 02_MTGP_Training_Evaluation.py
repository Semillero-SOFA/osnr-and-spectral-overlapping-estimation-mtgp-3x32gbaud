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
    import matplotlib.pyplot as plt
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
        torch,
        tqdm,
        train_test_split,
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
    return SEED, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Logic

    The function `train_mtgp` handles data loading, exact MLL optimization, and checkpoint saving. If a checkpoint already exists, it skips training to save time.
    """)
    return


@app.cell
def _(
    MultitaskGPModel,
    Path,
    device,
    gpytorch,
    load_dataset,
    torch,
    tqdm,
):
    def train_mtgp(dataset_name):
        print(f"\n{'='*50}")
        print(f"TRAINING PHASE: {dataset_name.upper()} Dataset")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        train_x, train_y, test_x, test_y, scaler_x, y_mean, y_std = load_dataset(data_path)

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
        _ckpt_path = _ARTIFACT_DIR / f'multitask_gp_{dataset_name.lower()}.pt'
        
        # Validates checkpoint is robust (has all required keys)
        if _ckpt_path.exists():
            try:
                ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                required_keys = {'model_state_dict', 'likelihood_state_dict', 'train_x_fit', 'train_y_fit', 'scaler_x', 'y_mean', 'y_std'}
                if required_keys.issubset(ckpt.keys()):
                    print(f'Valid checkpoint found at {_ckpt_path}. Skipping training. Use evaluate_mtgp directly.')
                    return
                else:
                    print(f'Legacy checkpoint at {_ckpt_path} missing new metadata. Retraining...')
            except Exception as e:
                print(f"Error reading checkpoint: {e}. Retraining...")

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        print(f"Training Model ({dataset_name.upper()})...")
        training_iterations = 200 # Adjust as needed
        for i in tqdm(range(training_iterations), desc=f"Training {dataset_name}"):
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
    2. Binned MAE and STD precision plots across the true labels (English labels).
    """)
    return


@app.cell
def _(
    MultitaskGPModel,
    Path,
    device,
    gpytorch,
    load_dataset,
    mo,
    np,
    plt,
    torch,
):
    def plot_predictions(y_act, y_pred, lower_denorm, upper_denorm, dataset_name):
        sort_indices_spacing = np.argsort(y_act[:, 0])
        sort_indices_osnr = np.argsort(y_act[:, 1])

        # Spacing Plot
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        x_spacing = np.arange(len(sort_indices_spacing))
        ax1.plot(x_spacing, y_act[sort_indices_spacing, 0], 'k.', label='True', alpha=0.7)
        ax1.plot(x_spacing, y_pred[sort_indices_spacing, 0], 'b-', label='Predicted', linewidth=2)
        ax1.fill_between(x_spacing, lower_denorm[sort_indices_spacing, 0], upper_denorm[sort_indices_spacing, 0], alpha=0.3, label='95% CI')
        ax1.set_xlabel('Sample Index (sorted)')
        ax1.set_ylabel('Spectral Spacing (GHz)')
        ax1.set_title(f'Prediction - Spacing ({dataset_name.upper()})')
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
        ax2.set_xlabel('Sample Index (sorted)')
        ax2.set_ylabel('OSNR (dB)')
        ax2.set_title(f'Prediction - OSNR ({dataset_name.upper()})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        ui_ax2 = mo.ui.matplotlib(ax2)

        return mo.hstack([ui_ax1, ui_ax2], justify="center")

    def plot_binned_metrics(y_act, y_pred, dataset_name):
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
        ax1.errorbar(unique_spacing, spacing_means, yerr=spacing_stds, fmt='o-', capsize=4, mfc='white')
        ax1.set_xlabel('Spectral Spacing (GHz)')
        ax1.set_ylabel('MAE ± STD')
        ax1.set_title(f'Precision by Spacing ({dataset_name.upper()})')
        ax1.grid(True, alpha=0.3)
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
        ax2.errorbar(valid_centers, osnr_means, yerr=osnr_stds, fmt='s-', capsize=4, mfc='white')
        ax2.set_xlabel('OSNR (dB) [Binned]')
        ax2.set_ylabel('MAE ± STD')
        ax2.set_title(f'Precision by OSNR ({dataset_name.upper()})')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        ui_ax2 = mo.ui.matplotlib(ax2)
        
        return mo.hstack([ui_ax1, ui_ax2], justify="center")

    def evaluate_mtgp(dataset_name):
        print(f"\n{'='*50}")
        print(f"EVALUATION PHASE: {dataset_name.upper()} Dataset")
        print(f"{'='*50}")

        data_path = f"processed_data/{dataset_name.lower()}"
        _, _, test_x, test_y, _, _, _ = load_dataset(data_path)
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        _ckpt_path = Path('artifacts') / f'multitask_gp_{dataset_name.lower()}.pt'
        if not _ckpt_path.exists():
            print(f"Error: Model checkpoint {_ckpt_path} not found. Ensure training is complete first.")
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
        
        model.eval()
        likelihood.eval()

        print("Running inference on test dataset...")
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

        print(f'\nMetrics ({dataset_name.upper()}):')
        print(f'MAE  - Spacing: {mae[0]:.3f} GHz | OSNR: {mae[1]:.3f} dB')
        print(f'RMSE - Spacing: {rmse[0]:.3f} GHz | OSNR: {rmse[1]:.3f} dB')

        # Formatting metrics table
        metrics_table = mo.md(f"""
        ### Performance Metrics ({dataset_name.upper()})
        ---
        | Output | MAE | RMSE |
        |---|---|---|
        | **Spacing (GHz)** | `{mae[0]:.4f}` | `{rmse[0]:.4f}` |
        | **OSNR (dB)** | `{mae[1]:.4f}` | `{rmse[1]:.4f}` |
        """)

        # Generating interactive plots
        plots1 = plot_predictions(y_actual_denorm, y_pred_denorm, lower_denorm, upper_denorm, dataset_name)
        plots2 = plot_binned_metrics(y_actual_denorm, y_pred_denorm, dataset_name)
        
        # Displaying structured output
        return mo.vstack([
            metrics_table,
            mo.md("#### Predictions and Confidence Intervals"),
            plots1,
            mo.md("#### Precision Analysis (Binned MAE)"),
            plots2
        ])

    return evaluate_mtgp, plot_binned_metrics, plot_predictions


@app.cell
def _(evaluate_mtgp):
    # Call evaluation for FCM
    # If this fails, ensure you have run the training cells above!
    return evaluate_mtgp("fcm")


@app.cell
def _(evaluate_mtgp):
    # Call evaluation for GKM
    # If this fails, ensure you have run the training cells above!
    return evaluate_mtgp("gkm")


if __name__ == "__main__":
    app.run()
