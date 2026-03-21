"""
visualize.py — Візуалізація результатів HGA-LSTM

Містить:
  - Графік конвергенції GA
  - Графік прогнозування vs реальні дані
  - Графік похибок
  - Зведена таблиця метрик

Автор: [Ваше ім'я]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger("HGA-LSTM.viz")

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "font.family": "DejaVu Sans",
}


def plot_ga_convergence(
    history: list[dict],
    save_path: str = "plots/ga_convergence.png",
) -> None:
    """Графік конвергенції генетичного алгоритму."""
    gens  = [h["generation"] for h in history]
    best  = [h["best_rmse"] for h in history]
    avg   = [h["avg_rmse"]  for h in history]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(gens, best, color="#58a6ff", label="Найкраще RMSE", linewidth=2)
        ax.plot(gens, avg,  color="#f0883e", label="Середнє RMSE",
                linestyle="--", alpha=0.7)
        ax.fill_between(gens, best, avg, alpha=0.08, color="#58a6ff")
        ax.set_xlabel("Покоління")
        ax.set_ylabel("RMSE (нормалізований)")
        ax.set_title("Конвергенція Генетичного Алгоритму — HGA-LSTM")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, save_path)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Щільність пульпи [г/см³]",
    save_path: str = "plots/predictions.png",
) -> None:
    """Графік прогнозування vs реальні дані + scatter plot."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

        # Часовий ряд
        ax1 = fig.add_subplot(gs[0])
        t = np.arange(len(y_true))
        ax1.plot(t, y_true, color="#3fb950", label="Реальні", alpha=0.8)
        ax1.plot(t, y_pred, color="#58a6ff", label="HGA-LSTM", alpha=0.8)
        ax1.fill_between(t, y_true, y_pred, alpha=0.1, color="#f78166")
        ax1.set_xlabel("Часовий крок")
        ax1.set_ylabel(target_name)
        ax1.set_title("Прогноз HGA-LSTM")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(y_true, y_pred, alpha=0.4, s=8, color="#58a6ff")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax2.plot(lims, lims, color="#f0883e", linewidth=1.5, label="Ідеал")
        ax2.set_xlabel("Реальні")
        ax2.set_ylabel("Прогноз")
        ax2.set_title("Діаграма розсіювання")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        _save(fig, save_path)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "plots/errors.png",
) -> None:
    """Розподіл похибок прогнозування."""
    errors = y_pred - y_true
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Часовий ряд похибок
        ax1.plot(errors, color="#f78166", alpha=0.7, linewidth=1)
        ax1.axhline(0, color="#ffffff", linewidth=0.8, linestyle="--")
        ax1.fill_between(range(len(errors)), errors, alpha=0.2, color="#f78166")
        ax1.set_xlabel("Часовий крок")
        ax1.set_ylabel("Похибка")
        ax1.set_title("Похибки прогнозування")
        ax1.grid(True, alpha=0.3)

        # Гістограма
        ax2.hist(errors, bins=40, color="#58a6ff", alpha=0.8, edgecolor="#0d1117")
        ax2.axvline(errors.mean(), color="#f0883e", linewidth=1.5,
                    label=f"Середня: {errors.mean():.4f}")
        ax2.axvline(errors.mean() + errors.std(), color="#3fb950", linewidth=1,
                    linestyle="--", label=f"±σ: {errors.std():.4f}")
        ax2.axvline(errors.mean() - errors.std(), color="#3fb950", linewidth=1, linestyle="--")
        ax2.set_xlabel("Похибка")
        ax2.set_ylabel("Частота")
        ax2.set_title("Розподіл похибок")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        _save(fig, save_path)


def print_metrics_table(metrics: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    """Виводить красиву таблицю метрик з порівнянням."""
    print("\n" + "=" * 55)
    print("  РЕЗУЛЬТАТИ HGA-LSTM")
    print("=" * 55)
    if baseline:
        print(f"  {'Метрика':<10} {'Базова':>10} {'HGA-LSTM':>10} {'Покращ.':>10}")
        print("-" * 55)
        for k in metrics:
            bv = baseline.get(k, float("nan"))
            mv = metrics[k]
            if bv != 0 and not np.isnan(bv):
                imp = (bv - mv) / abs(bv) * 100 if k != "R2" else (mv - bv) / abs(bv) * 100
                sign = "+" if imp > 0 else ""
                print(f"  {k:<10} {bv:>10.4f} {mv:>10.4f} {sign}{imp:>8.1f}%")
            else:
                print(f"  {k:<10} {'N/A':>10} {mv:>10.4f}")
    else:
        print(f"  {'Метрика':<15} {'Значення':>15}")
        print("-" * 55)
        for k, v in metrics.items():
            print(f"  {k:<15} {v:>15.4f}")
    print("=" * 55)

    # Порівняння з результатами Zou et al.
    print("\n  Порівняння з Zou et al. (2023):")
    print(f"  {'RMSE до GA':>20}: 3.8300")
    print(f"  {'RMSE після HGA-LSTM':>20}: 3.0800")
    print(f"  {'Ваш RMSE':>20}: {metrics.get('RMSE', 'N/A'):.4f}")
    print("=" * 55 + "\n")


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Графік збережено: {path}")
