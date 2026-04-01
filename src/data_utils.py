"""
data_utils.py — Data utilities for HGA-LSTM

Contains:
    - Synthetic pulp data generator (physics-based, for testing/verification)
    - CSV loader for real operational data
    - Sequential train/val/test split (no shuffling — preserves time order)

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2026
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("HGA-LSTM.data")


def generate_synthetic_pulp_data(
    n_samples: int = 2000,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic time-series data simulating Derrick Stack Sizer operation.

    Physics-based models:
        Pulp density:  rho = 1 / (1 - c_v)  where c_v = solid_vol_fraction
                       c_v = solid_pct/100 * (1/rho_solid), rho_solid=4.5 t/m3
        Efficiency:    eta = 81.4 * G_amp * G_freq * G_angle  (Chi et al., 2021)
                       where G_x = exp(-(x - x_opt)^2 / (2*sigma_x^2))
                       Optimal: A=3.7mm, f=13.4Hz, beta=40.9 deg

    Feature ranges (realistic operational parameters):
        amplitude_mm  : 3.0 -- 4.0 mm  (vibration amplitude)
        frequency_hz  : 12.0 -- 14.0 Hz (vibration frequency)
        angle_deg     : 35.0 -- 45.0 deg (vibration angle)
        pulp_flow     : 100 -- 200 m3/h  (feed flow rate)
        solid_pct     : 35 -- 55 %       (solid content by mass)

    Target ranges:
        density       : 1.05 -- 1.85 t/m3 (pulp density)
        efficiency    : 40 -- 97 %         (classification efficiency)

    Args:
        n_samples: Number of time steps to generate
        noise_std: Gaussian noise standard deviation (relative scale)
        seed:      Random seed for reproducibility

    Returns:
        DataFrame with columns: amplitude_mm, frequency_hz, angle_deg,
                                 pulp_flow, solid_pct, density, efficiency
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, n_samples)

    # Autocorrelated operational parameters (alpha=0.95 smoothing)
    amplitude = 3.5 + 0.5 * np.sin(t * 0.3)  + rng.normal(0, 0.10, n_samples)
    frequency = 13.0 + 1.0 * np.cos(t * 0.2)  + rng.normal(0, 0.20, n_samples)
    angle     = 40.0 + 5.0 * np.sin(t * 0.15) + rng.normal(0, 0.50, n_samples)
    pulp_flow = 150.0 + 30.0 * np.sin(t * 0.1) + rng.normal(0, 5.00, n_samples)
    solid_pct = 45.0 + 10.0 * np.sin(t * 0.25 + 1) + rng.normal(0, 2.0, n_samples)

    # Pulp density model: rho = 1 / (1 - c_v)
    # c_v = solid mass fraction / solid density (rho_solid ~ 4.5 t/m3 for magnetite)
    c_v = (solid_pct / 100.0) / 4.5
    density = (1.0 / (1.0 - c_v + 1e-6)
               + 0.05 * np.sin(t * 0.5)
               + noise_std * rng.normal(0, 1, n_samples))
    density = np.clip(density, 1.05, 1.85)

    # Screening efficiency model (Gaussian response surface)
    G_amp  = np.exp(-((amplitude - 3.7) ** 2) / (2 * 0.5**2))
    G_freq = np.exp(-((frequency - 13.4) ** 2) / (2 * 0.7**2))
    G_ang  = np.exp(-((angle - 40.9)    ** 2) / (2 * 4.0**2))
    efficiency = (81.4 * G_amp * G_freq * G_ang
                  * (1.0 - 0.001 * np.abs(solid_pct - 40.0))
                  + noise_std * 10 * rng.normal(0, 1, n_samples))
    efficiency = np.clip(efficiency, 40.0, 97.0)

    df = pd.DataFrame({
        "amplitude_mm": np.clip(amplitude, 1.0, 6.0),
        "frequency_hz": np.clip(frequency, 8.0, 20.0),
        "angle_deg":    np.clip(angle,     20.0, 60.0),
        "pulp_flow":    np.clip(pulp_flow,  50.0, 300.0),
        "solid_pct":    np.clip(solid_pct,  10.0, 70.0),
        "density":      density,
        "efficiency":   efficiency,
    })

    log.info(f"Synthetic data: {len(df)} rows")
    log.info(f"  density   : {df.density.min():.3f} -- {df.density.max():.3f} t/m3")
    log.info(f"  efficiency: {df.efficiency.min():.1f} -- {df.efficiency.max():.1f} %")
    return df


def load_csv(
    path: str | Path,
    feature_cols: list[str],
    target_col: str,
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load operational data from CSV file.

    Supported CSV formats (Derrick Stack Sizer dataset):
        input_data_large.csv:
            Timestamp, Feed_Flowrate_tph, Feed_Fe_percent,
            Feed_Size_minus150um_percent, Pulp_Density_t_m3,
            Water_Consumption_m3_per_t, Equipment_Status
            NOTE: Do NOT include Equipment_Status in feature_cols
                  (categorical string, requires encoding first)

        output_data_large.csv:
            Timestamp, Fines_Product_tph, Fines_Fe_percent,
            Fines_Recovery_percent, Classification_Efficiency_percent

        equipment_parameters_large.csv:
            Timestamp, Equipment_ID, Vibration_Frequency_Hz,
            Amplitude_mm, Eccentric_Angle_degrees,
            Screen_Downtime_hours, Motor_Current_A, Screen_Vibration_G

    Args:
        path:         Path to CSV file
        feature_cols: List of numeric column names to use as features
                      (Timestamp column is automatically excluded)
        target_col:   Name of the target/output column
        dropna:       Drop rows with missing values in selected columns

    Returns:
        X: float32 array [N, len(feature_cols)]
        y: float32 array [N]

    Raises:
        FileNotFoundError: if path does not exist
        KeyError:          if any column name is not found in CSV
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Validate columns
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        available = [c for c in df.columns if c != "Timestamp"]
        raise KeyError(
            f"Columns not found: {missing}\n"
            f"Available columns: {available}"
        )

    if dropna:
        df = df.dropna(subset=feature_cols + [target_col])

    # Filter out non-operational rows (if Equipment_Status exists)
    if "Equipment_Status" in df.columns:
        n_before = len(df)
        df = df[df["Equipment_Status"] == "operational"]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            log.info(f"  Filtered out {n_dropped} non-operational rows")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    log.info(f"CSV loaded: {path.name} | X={X.shape} | y={y.shape}")
    log.info(f"  target '{target_col}': "
             f"min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Sequential (non-shuffled) train/val/test split for time series.

    IMPORTANT: Do NOT shuffle time-series data before splitting.
    Shuffling causes data leakage: future observations can appear
    in the training set, producing unrealistically optimistic metrics.

    Split: |--- train (70%) ---|--- val (15%) ---|--- test (15%) ---|

    Args:
        X:          Input features [N, features]
        y:          Target values  [N]
        val_ratio:  Fraction for validation set
        test_ratio: Fraction for test set

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n       = len(X)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    log.info(f"Split: train={n_train} | val={n_val} | test={n_test} "
             f"(total={n})")

    return (
        X[:n_train],                    y[:n_train],
        X[n_train: n_train + n_val],    y[n_train: n_train + n_val],
        X[n_train + n_val:],            y[n_train + n_val:],
    )