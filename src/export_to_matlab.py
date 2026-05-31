"""
export_to_matlab.py — Експорт прогнозів HGA-LSTM у Simulink-сумісний .mat v7.3

Готує файл eta_pred.mat для замкненої моделі MPC у Simulink (Розділ 3).

ВАЖЛИВО про формат:
    Simulink-блок From File для timeseries-даних вимагає MAT v7.3 (HDF5).
    scipy.io.savemat пише лише v5/v7 і не підходить.
    Тут використовується пакет hdf5storage, який створює коректний v7.3-MAT
    із правильним заголовком 'MATLAB 7.3 MAT-file'.

Що пишеться у файл (читається у MATLAB як звичайні змінні):
    d_pred    — структура Simulink 'Structure with time' для порту md MPC
    eta_pred  — структура з прогнозом η^ (верифікація)
    eta_true  — структура з фактом η
    Ts        — крок дискретизації (с)
    metric_*  — RMSE / MAE / ARGE / R²  (скалярні змінні)

Структура 'Structure with time' (саме її розпізнає блок From File):
    sig.time            — [N x 1]
    sig.signals.values  — [N x ch]
    sig.signals.dimensions — ch

Залежність:
    pip install hdf5storage

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
import hdf5storage

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HGA-LSTM.export")


# ---------------------------------------------------------------------------
# Simulink 'Structure with time' builder
# ---------------------------------------------------------------------------

def _sim_struct(time: np.ndarray, values: np.ndarray) -> dict:
    """
    Формує структуру Simulink 'Structure with time' як вкладений dict.
    hdf5storage запише її у MAT v7.3 так, що Simulink розпізнає поля:
        s.time              -> [N x 1]
        s.signals.values    -> [N x ch]
        s.signals.dimensions -> ch
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1, 1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    ch = values.shape[1]
    return {
        "time": time,
        "signals": {
            "values": values,
            "dimensions": np.array([[ch]], dtype=np.float64),
        },
    }


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Основна логіка
# ---------------------------------------------------------------------------

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

    # Вирівнювання довжин і час
    n = min(len(eta_true), len(eta_pred), len(d_pred))
    eta_true = np.asarray(eta_true[:n], dtype=np.float64)
    eta_pred = np.asarray(eta_pred[:n], dtype=np.float64)
    d_pred   = np.asarray(d_pred[:n],   dtype=np.float64)
    t = np.arange(n, dtype=np.float64) * ts

    log.info(f"Довжина сигналів: {n} | Ts={ts}s | T_кінц={t[-1]:.0f}s")

    m = _metrics(eta_true, eta_pred)
    log.info(f"Метрики η: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
             f"ARGE={m['ARGE']:.4f}  R2={m['R2']:.4f}")

    # ── Запис у MAT v7.3 через hdf5storage ─────────────────────
    data = {
        "d_pred":   _sim_struct(t, d_pred),
        "eta_pred": _sim_struct(t, eta_pred),
        "eta_true": _sim_struct(t, eta_true),
        "Ts":       np.array([[float(ts)]], dtype=np.float64),
    }
    for k, v in m.items():
        data[f"metric_{k}"] = np.array([[float(v)]], dtype=np.float64)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdf5storage.write(data, filename=str(out), matlab_compatible=True,
                      store_python_metadata=False)

    log.info(f"Збережено (MAT v7.3 / HDF5): {out}  "
             f"({out.stat().st_size/1024:.1f} KB)")
    log.info("У Simulink: блок From File -> File name: eta_pred.mat, "
             "Variable name: d_pred")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Експорт HGA-LSTM -> Simulink .mat v7.3")
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
