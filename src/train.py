"""
train.py — Головний скрипт запуску HGA-LSTM

Використання:
    # Синтетичні дані (для тестування):
    python train.py

    # Власні CSV дані:
    python train.py --csv data/pulp_data.csv --features amplitude_mm frequency_hz angle_deg pulp_flow solid_pct --target density

    # Швидкий тест:
    python train.py --fast

Автор: [Ваше ім'я]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hga_lstm import HGALSTM, GAConfig, TrainConfig
from data_utils import generate_synthetic_pulp_data, load_csv, train_val_test_split
from visualize import (
    plot_ga_convergence,
    plot_predictions,
    plot_error_distribution,
    print_metrics_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HGA-LSTM.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HGA-LSTM: Hybrid Genetic Algorithm + LSTM для прогнозування параметрів пульпи",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
 
    # ── Дані ──
    data = p.add_argument_group("Дані")
    data.add_argument("--csv", type=str, default=None,
                      help="Шлях до CSV (якщо не вказано — синтетичні дані)")
    data.add_argument("--features", nargs="+",
                      default=["amplitude_mm", "frequency_hz", "angle_deg",
                               "pulp_flow", "solid_pct"],
                      help="Колонки-ознаки")
    data.add_argument("--target", type=str, default="density",
                      help="Цільова колонка")
    data.add_argument("--n-samples", type=int, default=2000,
                      help="К-сть синтетичних даних (ігнорується при --csv)")
 
    # ── GA параметри ──
    ga = p.add_argument_group("Генетичний алгоритм")
    ga.add_argument("--ga-pop",  type=int, default=20, help="Розмір популяції")
    ga.add_argument("--ga-gen",  type=int, default=30, help="К-сть поколінь")
    ga.add_argument("--no-sqp",  action="store_true",  help="Вимкнути SQP уточнення")
 
    # ── Навчання ──
    tr = p.add_argument_group("Навчання")
    tr.add_argument("--epochs",  type=int, default=100)
    tr.add_argument("--seed",    type=int, default=42)
    tr.add_argument("--device",  type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    tr.add_argument("--fast",    action="store_true",
                    help="Швидкий тест: pop=6, gen=5, epochs=20")
 
    # ── Вивід ──
    out = p.add_argument_group("Вивід")
    out.add_argument("--save-dir", type=str, default="outputs",
                     help="Папка для збереження результатів")
 
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Конфігурація ──
    if args.fast:
        log.info("FAST MODE: зменшені параметри для тесту")
        ga_cfg = GAConfig(population_size=6, n_generations=5)
        train_cfg = TrainConfig(epochs=20, patience=5, device=args.device, seed=args.seed)
    else:
        ga_cfg = GAConfig(population_size=args.ga_pop, n_generations=args.ga_gen)
        train_cfg = TrainConfig(
            epochs=args.epochs,
            patience=15,
            device=args.device,
            seed=args.seed,
            sqp_refine=not args.no_sqp,
        )

    # ── Дані ──
    if args.csv:
        log.info(f"Завантаження даних з {args.csv}")
        X, y = load_csv(args.csv, args.features, args.target)
    else:
        log.info("Генерація синтетичних даних пульпи...")
        df = generate_synthetic_pulp_data(n_samples=args.n_samples, seed=args.seed)
        X = df[args.features].values.astype("float32")
        y = df[args.target].values.astype("float32")

    log.info(f"Дані: X={X.shape}, y={y.shape}")
    log.info(f"Ціль '{args.target}': min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    log.info(f"Розбиття: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Навчання HGA-LSTM ──
    model = HGALSTM(input_size=X.shape[1], ga_cfg=ga_cfg, train_cfg=train_cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Оцінка ──
    metrics = model.evaluate(X_test, y_test)
    print_metrics_table(metrics)

    # ── Прогноз для візуалізації ──
    y_pred = model.predict(X_test)
    seq_len = model.best_hp.seq_len
    y_test_aligned = y_test[seq_len:]

    # ── Збереження ──
    model.save(str(save_dir / "hga_lstm_model.pt"))
    model.save_ga_history(str(save_dir / "ga_history.json"))

    # ── Графіки ──
    plot_dir = str(save_dir / "plots")
    plot_ga_convergence(model.ga_history, f"{plot_dir}/ga_convergence.png")
    plot_predictions(y_test_aligned, y_pred, args.target, f"{plot_dir}/predictions.png")
    plot_error_distribution(y_test_aligned, y_pred, f"{plot_dir}/errors.png")

    log.info("=" * 60)
    log.info(f"✅ Готово! Результати збережені в: {save_dir}/")
    log.info(f"   - Модель:   {save_dir}/hga_lstm_model.pt")
    log.info(f"   - Графіки:  {save_dir}/plots/")
    log.info(f"   - GA лог:   {save_dir}/ga_history.json")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
