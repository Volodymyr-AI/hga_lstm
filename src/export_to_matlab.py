"""
export_to_matlab.py — Експорт прогнозів HGA-LSTM у Simulink-сумісний .mat

Готує файл eta_pred.mat для замкненої моделі MPC у Simulink (Розділ 3).

ФОРМАТ ДАНИХ для блоку From File:
    Простий 2D-масив [N_channels+1 x N_samples]:
        перший рядок  -> час (с)
        наступні рядки -> значення сигналів
    Це найбезпечніший формат, що приймається блоком From File
    у будь-якій версії Simulink (R2020+, включно з R2026a).

Записується через scipy.io.savemat у MAT v5 — підтримується 2D double-масивами.

Змінні у файлі:
    d_pred    — [2 x N]: рядок 1 = час, рядок 2 = прогноз d^ (для md MPC)
    eta_pred  — [2 x N]: час + прогноз η^
    eta_true  — [2 x N]: час + факт η
    Ts        — крок дискретизації (с)
    metric_*  — RMSE / MAE / ARGE / R²

Запуск:
    python src/export_to_matlab.py --self-test
    python src/export_to_matlab.py --ts 5

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HGA-LSTM.export")


def _time_array(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Формує масив для блоку From File у форматі 'time + data rows':
        row 0       -> time
        rows 1..ch  -> data channels
    Результат [(1+ch) x N], double.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.shape[0] != len(time):
        values = values.T
    else:
        values = values.T
    # Тепер values: [ch x N], stack з часом зверху -> [(ch+1) x N]
    return np.vstack([time.reshape(1, -1), values])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-9
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE":  float(np.mean(np.abs(err))),
        "ARGE": float(np.mean(np.abs(err) / (np.abs(y_true) + eps))),
        "R2":   float(1.0 - np.sum(err**2)
                      / (np.sum((y_true - y_true.mean())**2) + eps)),
    }


def export(model_path: str, out_path: str, ts: float,
           n_samples: int, seed: int, self_test: bool) -> None:
    if self_test:
        log.warning("SELF-TEST: генерую фіктивні прогнози (без моделі/torch)")
        rng = np.random.default_rng(seed)
        n = max(50, n_samples // 8)
        eta_true = 80.0 + 8.0 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 1.0, n)
        eta_pred = eta_true + rng.normal(0, 1.2, n)
        d_pred   = 60.0 + 3.0 * np.cos(np.linspace(0, 5, n)) + rng.normal(0, 0.4, n)
    else:
        import torch  # noqa: F401
        from hga_lstm import HGALSTM
        from data_utils import generate_synthetic_pulp_data, train_val_test_split

        mp = Path(model_path)
        if not mp.exists():
            log.error(f"Модель не знайдено: {mp}")
            log.error("Спершу навчіть модель:  python src/train.py")
            sys.exit(1)

        log.info(f"Завантаження моделі: {mp}")
        model = HGALSTM.load(str(mp))

        log.info("Генерація тестових (синтетичних) даних...")
        df = generate_synthetic_pulp_data(n_samples=n_samples, seed=seed)
        features = ["amplitude_mm", "frequency_hz", "angle_deg",
                    "pulp_flow", "solid_pct"]
        X = df[features].values.astype("float32")
        y = df["efficiency"].values.astype("float32")

        _, _, _, _, X_test, y_test = train_val_test_split(X, y)

        eta_pred = model.predict(X_test)
        seq = model.best_hp.seq_len
        eta_true = y_test[seq:]

        d_full = df["solid_pct"].values.astype("float32")
        d_test = d_full[-len(y_test):]
        d_pred = d_test[seq:]

    n = min(len(eta_true), len(eta_pred), len(d_pred))
    eta_true = np.asarray(eta_true[:n], dtype=np.float64)
    eta_pred = np.asarray(eta_pred[:n], dtype=np.float64)
    d_pred   = np.asarray(d_pred[:n],   dtype=np.float64)
    t = np.arange(n, dtype=np.float64) * ts

    log.info(f"Довжина сигналів: {n} | Ts={ts}s | T_кінц={t[-1]:.0f}s")

    m = _metrics(eta_true, eta_pred)
    log.info(f"Метрики η: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
             f"ARGE={m['ARGE']:.4f}  R2={m['R2']:.4f}")

    data = {
        "d_pred":   _time_array(t, d_pred),
        "eta_pred": _time_array(t, eta_pred),
        "eta_true": _time_array(t, eta_true),
        "Ts":       np.array([[float(ts)]], dtype=np.float64),
    }
    for k, v in m.items():
        data[f"metric_{k}"] = np.array([[float(v)]], dtype=np.float64)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out), data, do_compression=True, oned_as="column")

    log.info(f"Збережено: {out}  ({out.stat().st_size/1024:.1f} KB)")
    log.info("У Simulink From File:")
    log.info(f"  File name      = {out.name}")
    log.info(f"  Variable name  = d_pred  (формат: рядок 1 = час, рядок 2 = d^)")
    log.info(f"  Sample time    = -1 (inherited) або {ts}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Експорт HGA-LSTM -> Simulink .mat")
    p.add_argument("--model",     default="outputs/hga_lstm_model.pt")
    p.add_argument("--out",       default="outputs/eta_pred.mat")
    p.add_argument("--ts",        type=float, default=5.0)
    p.add_argument("--n-samples", type=int,   default=2000)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    export(a.model, a.out, a.ts, a.n_samples, a.seed, a.self_test)
