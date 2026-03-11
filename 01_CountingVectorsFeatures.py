# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gdown>=5.0.0",
#     "marimo>=0.20.4",
#     "numpy>=2.1.0",
#     "pandas>=2.2.0",
#     "python-dotenv>=1.0.0",
#     "pyzmq>=26.0.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Counting Vectors for 32 Gbaud Nyquist-WDM 16-QAM Optical Communication System
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Libraries and modules
    We import standard libraries for data manipulation (`pandas`, `numpy`) and file I/O (`os`, `gdown`).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import os
    import warnings
    import gdown
    from dotenv import load_dotenv

    return gdown, load_dotenv, np, os, pd, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Database loading

    We load two pre-computed Counting Vector datasets from Google Drive (if not already present locally):

    - **FCM.csv** — Counting vectors computed using **Euclidean distance** (Fuzzy C-Means).
    - **GKM.csv** — Counting vectors computed using **Mahalanobis distance** (Gustafson-Kessel Means).

    Each file contains one row per scenario batch. The columns are structured as:

    | Columns | Description |
    |---------|-------------|
    | `cv_1` … `cv_16` | The 16 counting vector features (normalized probabilities) |
    | `OSNR` | Optical Signal-to-Noise Ratio target |
    | `Distance` | Transmission distance (metadata) |
    | `Power` | Launch power (metadata) |
    | `Spacing` | Spectral channel spacing target |

    The complete dataset is described in detail by Pérez et al. [1].

    **Reference:**
    [1] A. Escobar P, N. Guerrero Gonzalez, and J. Granada Torres, "Spectral overlapping estimation based on machine learning for gridless Nyquist-wavelength division multiplexing systems," Optical Engineering, vol. 59, p. 1, July 2020, doi: 10.1117/1.OE.59.7.076116.
    """)
    return


@app.cell
def _(gdown, load_dotenv, os, warnings):
    load_dotenv()
    FILE_ID_FCM = os.getenv("GDRIVE_FILE_ID_FCM")
    FILE_ID_GKM = os.getenv("GDRIVE_FILE_ID_GKM")
    FCM_PATH = "./data/FCM.csv"
    GKM_PATH = "./data/GKM.csv"

    if not FILE_ID_FCM or not FILE_ID_GKM:
        warnings.warn("GDRIVE_FILE_ID_FCM or GDRIVE_FILE_ID_GKM not set in .env — skipping download.")

    os.makedirs("./data", exist_ok=True)

    if not os.path.exists(FCM_PATH) and FILE_ID_FCM:
        print("Downloading FCM data from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID_FCM}", FCM_PATH, quiet=False)
        print("FCM download successful!")
    else:
        print("FCM data already present — skipping download.")

    if not os.path.exists(GKM_PATH) and FILE_ID_GKM:
        print("Downloading GKM data from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID_GKM}", GKM_PATH, quiet=False)
        print("GKM download successful!")
    else:
        print("GKM data already present — skipping download.")
    return FCM_PATH, GKM_PATH


@app.cell
def _(FCM_PATH, GKM_PATH, pd):
    print("Loading FCM data...")
    df_fcm = pd.read_csv(FCM_PATH, header=None)
    print(f"  Shape: {df_fcm.shape}")

    print("\nLoading GKM data...")
    df_gkm = pd.read_csv(GKM_PATH, header=None)
    print(f"  Shape: {df_gkm.shape}")
    return df_fcm, df_gkm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preparation

    In this step we extract the three arrays required by the downstream MTGP model:

    - **`X_features`** — shape `(N, 16)` — the 16 counting-vector probabilities.
    - **`Y_targets`** — shape `(N, 2)` — the regression targets: `[Spacing, OSNR]`.
    - **`M_metadata`** — shape `(N, 2)` — auxiliary scenario info: `[Distance, Power]`.

    We process both the FCM and GKM datasets separately, producing one set of arrays for each.
    """)
    return


@app.cell
def _(df_fcm, df_gkm, np):
    def extract_arrays(df):
        """Extract X, Y, M arrays from a headerless counting-vector dataframe."""
        X = df.iloc[:, 0:16].to_numpy(dtype=np.float32)       # features
        Y = df.iloc[:, [19, 16]].to_numpy(dtype=np.float32)   # [Spacing, OSNR]
        M = df.iloc[:, [17, 18]].to_numpy(dtype=np.float32)   # [Distance, Power]
        return X, Y, M

    X_fcm, Y_fcm, M_fcm = extract_arrays(df_fcm)
    X_gkm, Y_gkm, M_gkm = extract_arrays(df_gkm)

    print("FCM arrays:")
    print(f"  X_features : {X_fcm.shape}")
    print(f"  Y_targets  : {Y_fcm.shape}")
    print(f"  M_metadata : {M_fcm.shape}")

    print("\nGKM arrays:")
    print(f"  X_features : {X_gkm.shape}")
    print(f"  Y_targets  : {Y_gkm.shape}")
    print(f"  M_metadata : {M_gkm.shape}")
    return M_fcm, M_gkm, X_fcm, X_gkm, Y_fcm, Y_gkm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save processed datasets

    We save the extracted arrays to `processed_data/fcm/` and `processed_data/gkm/` respectively, in NumPy's `.npy` format for fast loading in downstream model training notebooks.
    """)
    return


@app.cell
def _(M_fcm, M_gkm, X_fcm, X_gkm, Y_fcm, Y_gkm, np, os):
    def save_arrays(X, Y, M, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X_features.npy"), X)
        np.save(os.path.join(output_dir, "Y_targets.npy"),  Y)
        np.save(os.path.join(output_dir, "M_metadata.npy"), M)
        print(f"Saved to '{output_dir}/':")
        print(f"  X_features.npy  shape={X.shape}  ({X.nbytes / 1e6:.2f} MB)")
        print(f"  Y_targets.npy   shape={Y.shape}  ({Y.nbytes / 1e6:.2f} MB)")
        print(f"  M_metadata.npy  shape={M.shape}  ({M.nbytes / 1e6:.2f} MB)")

    save_arrays(X_fcm, Y_fcm, M_fcm, "processed_data/fcm")
    print()
    save_arrays(X_gkm, Y_gkm, M_gkm, "processed_data/gkm")

    print("\nAll datasets saved successfully.")
    return


if __name__ == "__main__":
    app.run()
