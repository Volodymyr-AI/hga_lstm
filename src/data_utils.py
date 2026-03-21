"""
data_utils.py — Утиліти для підготовки даних HGA-LSTM

Містить:
  - Генератор синтетичних даних пульпи (для тестування)
  - Функції завантаження CSV
  - Розбиття на train/val/test

Автор: [Ваше ім'я]
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
    Генерує синтетичні дані, що імітують роботу вібраційного грохота
    для класифікації залізорудної пульпи.

    Вхідні ознаки (технологічні параметри):
      - amplitude_mm:    Амплітуда вібрації [мм]    — оптимум 3-4 мм
      - frequency_hz:    Частота вібрації [Гц]       — оптимум 12-14 Гц
      - angle_deg:       Кут вібрації [°]            — оптимум 35-45°
      - pulp_flow:       Витрата пульпи [м³/год]
      - solid_content:   Вміст твердого [%]          — до 70%

    Цільова змінна:
      - density:         Щільність пульпи [г/см³]    — 1.2–1.8
      - efficiency:      Ефективність грохочення [%] — 70–97%

    Args:
        n_samples: Кількість точок
        noise_std: Стандартне відхилення шуму
        seed:      Seed для відтворюваності

    Returns:
        DataFrame з усіма колонками
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_samples)

    # Технологічні параметри з реалістичними діапазонами
    amplitude  = 3.5 + 0.5 * np.sin(t * 0.3) + rng.normal(0, 0.1, n_samples)
    frequency  = 13.0 + 1.0 * np.cos(t * 0.2) + rng.normal(0, 0.2, n_samples)
    angle      = 40.0 + 5.0 * np.sin(t * 0.15) + rng.normal(0, 0.5, n_samples)
    pulp_flow  = 150.0 + 30.0 * np.sin(t * 0.1) + rng.normal(0, 5.0, n_samples)
    solid_pct  = 45.0 + 10.0 * np.sin(t * 0.25 + 1) + rng.normal(0, 2.0, n_samples)

    # Фізична модель щільності пульпи
    # ρ_пульпи ≈ ρ_рідини / (1 - c_v) де c_v — об'ємна концентрація
    solid_vol_frac = solid_pct / 100.0 * (1.0 / 4.5)  # ρ_solid ≈ 4.5 г/см³ для залізної руди
    density = (
        1.0 / (1 - solid_vol_frac + 1e-6)
        + 0.05 * np.sin(t * 0.5)
        + noise_std * rng.normal(0, 1, n_samples)
    )
    density = np.clip(density, 1.05, 1.85)

    # Фізична модель ефективності грохочення
    # Оптимальна ефективність при A=3.7мм, f=13.4Гц, β=40.9°
    eff_amp  = np.exp(-((amplitude - 3.7) ** 2) / (2 * 0.5**2))
    eff_freq = np.exp(-((frequency - 13.4) ** 2) / (2 * 0.7**2))
    eff_ang  = np.exp(-((angle - 40.9) ** 2) / (2 * 4.0**2))
    efficiency = 81.4 * eff_amp * eff_freq * eff_ang * (1 - 0.001 * np.abs(solid_pct - 40))
    efficiency += noise_std * 10 * rng.normal(0, 1, n_samples)
    efficiency = np.clip(efficiency, 40.0, 97.0)

    df = pd.DataFrame({
        "amplitude_mm": np.clip(amplitude, 1.0, 6.0),
        "frequency_hz": np.clip(frequency, 8.0, 20.0),
        "angle_deg":    np.clip(angle, 20.0, 60.0),
        "pulp_flow":    np.clip(pulp_flow, 50.0, 300.0),
        "solid_pct":    np.clip(solid_pct, 10.0, 70.0),
        "density":      density,
        "efficiency":   efficiency,
    })

    log.info(f"Синтетичні дані згенеровано: {len(df)} рядків")
    log.info(f"  density  : {df.density.min():.3f} – {df.density.max():.3f} г/см³")
    log.info(f"  efficiency: {df.efficiency.min():.1f} – {df.efficiency.max():.1f} %")
    return df


def load_csv(
    path: str | Path,
    feature_cols: list[str],
    target_col: str,
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Завантажує дані з CSV файлу.

    Args:
        path:         Шлях до CSV
        feature_cols: Список колонок-ознак
        target_col:   Назва цільової колонки
        dropna:       Видаляти рядки з NaN

    Returns:
        X: [N, features]
        y: [N]
    """
    df = pd.read_csv(path)
    if dropna:
        df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    log.info(f"CSV завантажено: {path} | shape={X.shape}")
    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Послідовне (не перемішане) розбиття часових рядів.

    Args:
        X, y:       Масиви даних
        val_ratio:  Частка для валідації
        test_ratio: Частка для тесту

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_val - n_test

    return (
        X[:n_train],      y[:n_train],
        X[n_train:n_train+n_val], y[n_train:n_train+n_val],
        X[n_train+n_val:], y[n_train+n_val:],
    )
