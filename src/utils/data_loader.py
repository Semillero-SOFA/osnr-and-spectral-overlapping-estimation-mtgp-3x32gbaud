import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_dataset(data_path="processed_data", test_size=0.2, seed=42):
    """
    Load processed dataset from .npy files.

    Args:
        data_path (str): Path to directory containing .npy files.
        test_size (float): Fraction of data to use for testing.
        seed (int): Random seed.

    Returns:
        train_x, train_y, test_x, test_y, scaler_x, scaler_y_osnr
    """
    path = Path(data_path)

    print(f"Loading data from {path}...")
    try:
        X = np.load(path / "X_features.npy")
        Y = np.load(path / "Y_targets.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find data files in {path}. Ensure X_features.npy and Y_targets.npy exist."
        ) from e

    print(f"Loaded X: {X.shape}, Y: {Y.shape}")

    # Targets Processing
    # Column 0: Spacing (Continuous) -> Overlap (Binary)
    # Threshold: <= 35.2 is Overlap (1), > 35.2 is No Overlap (0)
    BANDWIDTH = 32 # GHz
    ROLLOFF_FACTOR = 0.1
    spacing = Y[:, 0]
    y_overlap = (spacing <= BANDWIDTH * (1 + ROLLOFF_FACTOR)).astype(float)

    # Column 1: OSNR (Continuous)
    y_osnr = Y[:, 1]

    # Stack targets: [Overlap, OSNR]
    # Task 0: Overlap (Binary)
    # Task 1: OSNR (Continuous)
    y = np.stack([y_overlap, y_osnr], axis=1)

    # Split
    # Stratify by overlap to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y_overlap
    )

    # Normalize Inputs
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    # Normalize OSNR (Target 1)
    # We don't normalize Binary target (Target 0)
    scaler_y_osnr = StandardScaler()
    y_train_osnr = scaler_y_osnr.fit_transform(y_train[:, 1].reshape(-1, 1)).flatten()
    y_test_osnr = scaler_y_osnr.transform(y_test[:, 1].reshape(-1, 1)).flatten()

    # Reassemble y
    y_train[:, 1] = y_train_osnr
    y_test[:, 1] = y_test_osnr

    # Convert to Tensor
    train_x = torch.tensor(X_train).float()
    train_y = torch.tensor(y_train).float()
    test_x = torch.tensor(X_test).float()
    test_y = torch.tensor(y_test).float()

    return train_x, train_y, test_x, test_y, scaler_x, scaler_y_osnr
