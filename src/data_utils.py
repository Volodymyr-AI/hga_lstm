"""
data_utils.py — Data utilities for HGA-LSTM (v2: multi-task ready)

CHANGES v2 (for Section 3, recommendation subsystem β):
    + Added synthesis of motor_current (A) — vibrating motor current
    + Added synthesis of screen_runtime (h) — screen operating time since last maintenance
    + Eccentric angle (angle_deg) now depends on runtime and current:
      optimal β drifts with screen wear; current introduces fluctuations
      due to uneven loading.

Contains:
    - Synthetic pulp data generator (physics-based, з motor_current/runtime)
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

    v2:
        Motor current:    I = I_0 + k_A * amplitude + k_w * wear + noise
                          I_0=8.0 A, k_A=0.6 A/mm, k_w=0.5 A
        Screen runtime:   runtime = linspace(0, 200, n) — сита працюють до 200 год
        Optimal angle drift:
                          beta_opt(t) = 40.9 + drift * (runtime/200)
                          where drift = +3° (зі зношеністю опт. кут зростає)

    Feature ranges:
        amplitude_mm   : 3.0 -- 4.0 mm
        frequency_hz   : 12.0 -- 14.0 Hz
        angle_deg      : 35.0 -- 45.0 deg
        pulp_flow      : 100 -- 200 m3/h
        solid_pct      : 35 -- 55 %
        motor_current  : 8 -- 16 A           [v2]
        screen_runtime : 0 -- 200 h          [v2]

    Targets:
        density        : 1.05 -- 1.85 t/m3
        efficiency     : 40 -- 97 %

    Args:
        n_samples: Number of time steps to generate
        noise_std: Gaussian noise standard deviation (relative scale)
        seed:      Random seed for reproducibility

    Returns:
        DataFrame with columns:
            amplitude_mm, frequency_hz, angle_deg, pulp_flow, solid_pct,
            motor_current, screen_runtime,       [v2 inputs]
            density, efficiency                  [targets]
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 4 * np.pi, n_samples)

    # ── Operational parameters (v2.1: expanded ranges for η variability) ──
    # Previously, the ranges were too narrow around the Gaussian optima in the η formula,
    # causing η to barely vary and the network to learn to output a constant value.
    # Now, the parameters cover the full operating range from constraint section 2.6.4:
    #   amplitude: 2–8 mm (MPC MV(2) constraint)
    #   frequency: 12–25 Hz (MPC MV(1) constraint)
    amplitude = 4.5 + 2.0 * np.sin(t * 0.3) + rng.normal(0, 0.30, n_samples)
    amplitude = np.clip(amplitude, 2.0, 8.0)

    frequency = 17.0 + 5.0 * np.cos(t * 0.2) + rng.normal(0, 0.5, n_samples)
    frequency = np.clip(frequency, 12.0, 25.0)

    pulp_flow = 150.0 + 30.0 * np.sin(t * 0.1) + rng.normal(0, 5.00, n_samples)
    solid_pct = 45.0 + 10.0 * np.sin(t * 0.25 + 1) + rng.normal(0, 2.0, n_samples)

    # ──v2: screen_runtime (CYCLIC — realistic maintenance cycles) ──
    # In a real factory, screens are replaced every 1–2 weeks. Each cycle:
    # a fresh screen (runtime=0) gradually wears down over ~200 hours, then is replaced.
    # Multiple full cycles are included in the dataset so that the train, validation,
    # and test sets share the same runtime distributions, allowing the model
    # to interpolate rather than extrapolate.
    n_cycles = 10
    samples_per_cycle = n_samples // n_cycles
    screen_runtime = np.zeros(n_samples)
    for i in range(n_cycles):
        start = i * samples_per_cycle
        end   = start + samples_per_cycle if i < n_cycles - 1 else n_samples
        screen_runtime[start:end] = np.linspace(0.0, 200.0, end - start)

    # ── NEW v2: motor_current ──
    # I = I_0 + k_A * amplitude + k_wear * (runtime/200) + noise
    # При amplitude=4 мм, runtime=200 год: I ≈ 8 + 2.4 + 0.5 = 10.9 А
    I_0 = 8.0       # базовий струм холостого ходу, А
    k_A = 0.6       # А на 1 мм амплітуди
    k_w = 0.5       # А додаткового навантаження при повному зносі
    motor_current = (I_0
                     + k_A * amplitude
                     + k_w * (screen_runtime / 200.0)
                     + rng.normal(0, 0.15, n_samples))
    motor_current = np.clip(motor_current, 6.0, 20.0)

    # ── NEW v2: angle_deg тепер залежить від runtime ──
    # Оптимальний β зміщується зі зношеністю сита: 40.9° -> ~44° при повному ресурсі.
    # У реальному житті оператор підбирає β інтуїтивно — наша задача навчити
    # LSTM знаходити цю залежність із runtime та motor_current.
    beta_drift = 3.0 * (screen_runtime / 200.0)              # дрейф оптимуму
    beta_noise = 0.5 * np.sin(t * 0.4)                       # повільні коливання
    beta_current_corr = 0.3 * (motor_current - 10.5)         # коригування за струмом
    angle = (40.9 + beta_drift + beta_noise + beta_current_corr
             + rng.normal(0, 0.5, n_samples))

    # ── Pulp density (як раніше) ──
    c_v = (solid_pct / 100.0) / 4.5
    density = (1.0 / (1.0 - c_v + 1e-6)
               + 0.05 * np.sin(t * 0.5)
               + noise_std * rng.normal(0, 1, n_samples))
    density = np.clip(density, 1.05, 1.85)

    G_amp  = np.exp(-((amplitude - 3.7) ** 2) / (2 * 1.5**2))
    G_freq = np.exp(-((frequency - 13.4) ** 2) / (2 * 3.0**2))
    optimal_beta = 40.9 + beta_drift
    G_ang  = np.exp(-((angle - optimal_beta) ** 2) / (2 * 4.0**2))
    
    efficiency = (81.4 * G_amp * G_freq * G_ang
                  * (1.0 - 0.001 * np.abs(solid_pct - 40.0))
                  + noise_std * 2 * rng.normal(0, 1, n_samples))
    efficiency = np.clip(efficiency, 40.0, 97.0)

    df = pd.DataFrame({
        # Старі стовпці (для зворотної сумісності зі single-task v1)
        "amplitude_mm":   np.clip(amplitude, 1.0, 6.0),
        "frequency_hz":   np.clip(frequency, 8.0, 20.0),
        "angle_deg":      np.clip(angle,     20.0, 60.0),
        "pulp_flow":      np.clip(pulp_flow,  50.0, 300.0),
        "solid_pct":      np.clip(solid_pct,  10.0, 70.0),
        # NEW v2: входи для β-рекомендатора
        "motor_current":  motor_current,
        "screen_runtime": screen_runtime,
        # Таргети
        "density":        density,
        "efficiency":     efficiency,
    })

    log.info(f"Synthetic data: {len(df)} rows  (v2: with motor_current, screen_runtime)")
    log.info(f"  density        : {df.density.min():.3f} -- {df.density.max():.3f} t/m3")
    log.info(f"  efficiency     : {df.efficiency.min():.1f} -- {df.efficiency.max():.1f} %")
    log.info(f"  angle_deg      : {df.angle_deg.min():.2f} -- {df.angle_deg.max():.2f} °")
    log.info(f"  motor_current  : {df.motor_current.min():.2f} -- {df.motor_current.max():.2f} A")
    log.info(f"  screen_runtime : {df.screen_runtime.min():.1f} -- {df.screen_runtime.max():.1f} h")
    return df


def load_csv(
    path: str | Path,
    feature_cols: list[str],
    target_col: str | list[str],
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load operational data from CSV file.

    v2: target_col тепер може бути str (single-task) АБО list[str] (multi-task).

    Supported CSV formats (Derrick Stack Sizer dataset):
        input_data_large.csv, output_data_large.csv, equipment_parameters_large.csv
        (структура — як у v1)

    Args:
        path:         Path to CSV file
        feature_cols: List of numeric column names to use as features
        target_col:   Target column name (str) АБО список таргетів (list[str])
        dropna:       Drop rows with missing values in selected columns

    Returns:
        X: float32 array [N, len(feature_cols)]
        y: float32 array [N] (single-task) АБО [N, n_targets] (multi-task)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Нормалізуємо target_col до списку для уніфікованої обробки
    targets = [target_col] if isinstance(target_col, str) else list(target_col)

    # Validate columns
    missing = [c for c in feature_cols + targets if c not in df.columns]
    if missing:
        available = [c for c in df.columns if c != "Timestamp"]
        raise KeyError(
            f"Columns not found: {missing}\n"
            f"Available columns: {available}"
        )

    if dropna:
        df = df.dropna(subset=feature_cols + targets)

    if "Equipment_Status" in df.columns:
        n_before = len(df)
        df = df[df["Equipment_Status"] == "operational"]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            log.info(f"  Filtered out {n_dropped} non-operational rows")

    X = df[feature_cols].values.astype(np.float32)
    y = df[targets].values.astype(np.float32)
    # Якщо single-task — повертаємо одновимірний вектор для сумісності з v1
    if len(targets) == 1:
        y = y.flatten()

    log.info(f"CSV loaded: {path.name} | X={X.shape} | y={y.shape}")
    if y.ndim == 1:
        log.info(f"  target '{targets[0]}': "
                 f"min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    else:
        for i, name in enumerate(targets):
            col = y[:, i]
            log.info(f"  target '{name}': "
                     f"min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")
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

    Працює і для single-task (y: [N]) і для multi-task (y: [N, n_targets]) —
    в обох випадках просто розрізає по першій осі.

    Split: |--- train (70%) ---|--- val (15%) ---|--- test (15%) ---|

    Args:
        X:          Input features [N, features]
        y:          Target values  [N]   (single-task)
                    АБО            [N, n_targets] (multi-task)
        val_ratio:  Fraction for validation set
        test_ratio: Fraction for test set

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n       = len(X)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    log.info(f"Split: train={n_train} | val={n_val} | test={n_test} (total={n})")

    return (
        X[:n_train],                    y[:n_train],
        X[n_train: n_train + n_val],    y[n_train: n_train + n_val],
        X[n_train + n_val:],            y[n_train + n_val:],
    )
