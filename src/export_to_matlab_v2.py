"""
export_to_matlab_v2.py — Експорт multi-task HGA-LSTM у Simulink-сумісний .mat

Готує файл eta_pred.mat для замкненої моделі MPC у Simulink (Розділ 3).
Використовує навчену модель HGA-LSTM v2, яка одночасно прогнозує:
    - η (ефективність класифікації, %) — головна керована величина
    - β (рекомендація кута ексцентрика, °) — рекомендаційний сигнал

ФОРМАТ .mat (для блоку From File):
    Простий 2D-масив [2 x N], MAT v5.
    Рядок 1 = час (с)
    Рядок 2 = значення сигналу

Змінні у файлі:
    d_pred     — [2 x N]: час + прогноз d̂ = η̂ (для порту md MPC)
    eta_pred   — [2 x N]: час + прогноз η̂
    beta_pred  — [2 x N]: час + рекомендація β̂ (для окремого Display у Simulink)
    eta_true   — [2 x N]: час + фактичне η (з тестової вибірки)
    beta_true  — [2 x N]: час + фактичне β
    Ts         — крок дискретизації (с)
    metric_eta_*  — RMSE / MAE / ARGE / R² для η
    metric_beta_* — RMSE / MAE / ARGE / R² для β

Запуск (з кореня проєкту):
    python src/export_to_matlab_v2.py                    # синтетика v2, seed 42
    python src/export_to_matlab_v2.py --seed 100         # інша вибірка
    python src/export_to_matlab_v2.py --self-test        # без моделі (фіктивні прогнози)

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
log = logging.getLogger("HGA-LSTM-v2.export")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_array(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Формує масив для блоку From File у форматі 'time + data':
        row 0 -> time (с)
        row 1 -> сигнал
    Результат [2 x N], double.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    return np.vstack([time.reshape(1, -1), values.reshape(1, -1)])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / ARGE / R² для одного сигналу."""
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
# Main export
# ---------------------------------------------------------------------------

def export(model_path: str, out_path: str, ts: float,
           n_samples: int, seed: int, self_test: bool) -> None:
    """
    Головна функція експорту.

    Args:
        model_path: шлях до hga_lstm_v2_model.pt
        out_path:   шлях, куди зберегти .mat
        ts:         крок дискретизації (с)
        n_samples:  розмір синтетичного датасету
        seed:       seed для відтворюваності
        self_test:  якщо True — не завантажує модель, генерує фіктивні прогнози
    """
    if self_test:
        log.warning("SELF-TEST: генерую фіктивні прогнози (без моделі/torch)")
        rng = np.random.default_rng(seed)
        n = max(50, n_samples // 8)
        eta_true = 80.0 + 8.0 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 1.0, n)
        eta_pred = eta_true + rng.normal(0, 1.2, n)
        beta_true = 40.9 + 2.5 * np.sin(np.linspace(0, 4, n)) + rng.normal(0, 0.3, n)
        beta_pred = beta_true + rng.normal(0, 0.4, n)
    else:
        import torch  # noqa: F401
        from hga_lstm_v2 import HGALSTMv2
        from data_utils import generate_synthetic_pulp_data, train_val_test_split

        mp = Path(model_path)
        if not mp.exists():
            log.error(f"Модель не знайдено: {mp}")
            log.error("Спершу навчіть модель:  python src/train_v2.py")
            sys.exit(1)

        log.info(f"Завантаження моделі: {mp}")
        model = HGALSTMv2.load(str(mp))
        log.info(f"  input_size:   {model.input_size}")
        log.info(f"  target_names: {model.target_names}")
        log.info(f"  seq_len:      {model.best_hp.seq_len}")

        # Перевіримо, що модель v2 (multi-task для η + β)
        if len(model.target_names) != 2:
            log.warning(f"Модель має {len(model.target_names)} таргетів, "
                        "очікується 2 (efficiency, angle_deg)")

        log.info(f"Генерація тестових (синтетичних) даних n={n_samples}, seed={seed}...")
        df = generate_synthetic_pulp_data(n_samples=n_samples, seed=seed)

        # ВАЖЛИВО: список фічей має ТОЧНО збігатися з тим, на чому модель навчали
        features = ["amplitude_mm", "frequency_hz", "pulp_flow", "solid_pct",
                    "motor_current", "screen_runtime"]

        # Перевірка розмірностей
        if model.input_size != len(features):
            log.error(f"Модель очікує {model.input_size} фічей, а ми даємо {len(features)}")
            log.error(f"Фічі: {features}")
            sys.exit(1)

        X = df[features].values.astype("float32")
        y = df[model.target_names].values.astype("float32")   # [N, 2]

        # Той самий split, що при навчанні (train / val / test = 70 / 15 / 15)
        _, _, _, _, X_test, y_test = train_val_test_split(X, y)

        # Прогноз на тестовій вибірці
        log.info(f"Прогноз на X_test shape={X_test.shape}...")
        preds = model.predict(X_test)                # [N-seq_len, 2]

        seq = model.best_hp.seq_len
        y_test_aligned = y_test[seq:]                # [N-seq_len, 2]

        # Розкладаємо на η і β
        eta_pred  = preds[:, 0]
        beta_pred = preds[:, 1]
        eta_true  = y_test_aligned[:, 0]
        beta_true = y_test_aligned[:, 1]

    # ── Формуємо часову вісь ──────────────────────────────
    n = min(len(eta_true), len(eta_pred), len(beta_true), len(beta_pred))
    eta_true  = np.asarray(eta_true[:n],  dtype=np.float64)
    eta_pred  = np.asarray(eta_pred[:n],  dtype=np.float64)
    beta_true = np.asarray(beta_true[:n], dtype=np.float64)
    beta_pred = np.asarray(beta_pred[:n], dtype=np.float64)
    t = np.arange(n, dtype=np.float64) * ts

    log.info(f"Довжина сигналів: {n} | Ts={ts}s | T_кінц={t[-1]:.0f}s")

    # ── Метрики окремо для кожного таргета ────────────────
    m_eta  = _metrics(eta_true,  eta_pred)
    m_beta = _metrics(beta_true, beta_pred)

    log.info("Метрики η:    " +
             "  ".join(f"{k}={v:.4f}" for k, v in m_eta.items()))
    log.info("Метрики β:    " +
             "  ".join(f"{k}={v:.4f}" for k, v in m_beta.items()))

    # ── Формуємо словник змінних для .mat ─────────────────
    data = {
        # d_pred = прогноз збурення для порту md MPC.
        # За методологією Розділу 3, d̂ ототожнюється з прогнозом ефективності:
        # тобто d_pred = eta_pred (для сумісності з Simulink-моделями,
        # у яких порт md чекає сигнал у форматі d̂).
        "d_pred":    _time_array(t, eta_pred),
        "eta_pred":  _time_array(t, eta_pred),
        "beta_pred": _time_array(t, beta_pred),
        "eta_true":  _time_array(t, eta_true),
        "beta_true": _time_array(t, beta_true),
        "Ts":        np.array([[float(ts)]], dtype=np.float64),
    }

    # Метрики — окремо з префіксами
    for k, v in m_eta.items():
        data[f"metric_eta_{k}"] = np.array([[float(v)]], dtype=np.float64)
    for k, v in m_beta.items():
        data[f"metric_beta_{k}"] = np.array([[float(v)]], dtype=np.float64)

    # ── Збереження ────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out), data, do_compression=True, oned_as="column")

    log.info(f"Збережено: {out}  ({out.stat().st_size/1024:.1f} KB)")
    log.info("=" * 55)
    log.info("У Simulink From File блоку вкажіть:")
    log.info(f"  File name      = {out.name}")
    log.info(f"  Variable name  = d_pred   (для порту md MPC)")
    log.info(f"  Sample time    = -1 (inherited) або {ts}")
    log.info("Для β-рекомендації (окремий Display):")
    log.info(f"  Variable name  = beta_pred")
    log.info("=" * 55)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task HGA-LSTM v2 -> Simulink .mat експорт",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",     default="outputs/hga_lstm_v2_model.pt",
                   help="Шлях до навченої multi-task моделі")
    p.add_argument("--out",       default="outputs/eta_pred.mat",
                   help="Куди зберегти .mat")
    p.add_argument("--ts",        type=float, default=5.0,
                   help="Крок дискретизації Simulink (с). За замовч. 5")
    p.add_argument("--n-samples", type=int,   default=2000,
                   help="Розмір синтетичного датасету")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed (для відтворюваності)")
    p.add_argument("--self-test", action="store_true",
                   help="Не завантажувати модель, генерувати фіктивні прогнози")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    export(a.model, a.out, a.ts, a.n_samples, a.seed, a.self_test)