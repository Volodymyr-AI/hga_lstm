"""
train_v2.py — Multi-task HGA-LSTM training entry point

Тренує одну модель з двома таргетами:
    - efficiency (η, головна метрика, вага 0.7)
    - angle_deg  (β-рекомендація, вага 0.3)

ВАЖЛИВО: v2 НЕ замінює v1, працює паралельно.
Зберігає модель у outputs/hga_lstm_v2_model.pt (інший шлях, ніж v1).

Запуск з кореня проєкту:

    # Quick smoke test (~2 хв на CPU, ~30 с на GPU)
    python src/train_v2.py --fast

    # Повний прогін (~30 хв, GA=20×30 + SQP + final)
    python src/train_v2.py

    # Кастомні ваги завдань (η=0.6, β=0.4)
    python src/train_v2.py --weights 0.6 0.4

    # Кастомні таргети
    python src/train_v2.py --targets efficiency density

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Шлях
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hga_lstm_v2 import HGALSTMv2, GAConfig, TrainConfig
from data_utils import generate_synthetic_pulp_data, load_csv, train_val_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HGA-LSTM-v2.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HGA-LSTM v2: Multi-task (η + β) training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("Data")
    data.add_argument("--csv",       default=None,
                      help="Path to CSV (omit -> synthetic data)")
    data.add_argument("--features",  nargs="+",
                      default=["amplitude_mm", "frequency_hz", "pulp_flow",
                               "solid_pct", "motor_current", "screen_runtime"],
                      help="Numeric feature column names (6 default)")
    data.add_argument("--targets",   nargs="+",
                      default=["efficiency", "angle_deg"],
                      help="Target columns (multi-task, 2 за замовч.)")
    data.add_argument("--n-samples", type=int, default=2000)

    ga = p.add_argument_group("Genetic Algorithm")
    ga.add_argument("--ga-pop", type=int, default=20)
    ga.add_argument("--ga-gen", type=int, default=30)
    ga.add_argument("--no-sqp", action="store_true")

    tr = p.add_argument_group("Training")
    tr.add_argument("--epochs",  type=int, default=100)
    tr.add_argument("--seed",    type=int, default=42)
    tr.add_argument("--device",  default="auto", choices=["auto","cuda","cpu"])
    tr.add_argument("--weights", nargs="+", type=float, default=[0.9, 0.1],
                    help="Loss weights per task (must match --targets length)")
    tr.add_argument("--fast",    action="store_true",
                    help="Quick smoke test: pop=6, gen=5, epochs=20")

    out = p.add_argument_group("Output")
    out.add_argument("--save-dir", default="outputs",
                     help="Output directory for model and history")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("  HGA-LSTM v2 Training (Multi-Task)")
    log.info("=" * 55)
    log.info(f"  Features ({len(args.features)}): {args.features}")
    log.info(f"  Targets  ({len(args.targets)}):  {args.targets}")
    log.info(f"  Weights  : {args.weights}")
    log.info("=" * 55)

    if len(args.weights) != len(args.targets):
        log.error(f"--weights ({len(args.weights)}) must match "
                  f"--targets ({len(args.targets)}) length")
        sys.exit(1)

    # ── Configuration ───────────────────────────────────────
    if args.fast:
        log.info("FAST MODE — reduced parameters for smoke test")
        ga_cfg    = GAConfig(population_size=6, n_generations=5)
        train_cfg = TrainConfig(epochs=20, patience=5,
                                device=args.device, seed=args.seed,
                                task_weights=args.weights,
                                sqp_refine=not args.no_sqp)
    else:
        ga_cfg    = GAConfig(population_size=args.ga_pop,
                             n_generations=args.ga_gen)
        train_cfg = TrainConfig(epochs=args.epochs, patience=15,
                                device=args.device, seed=args.seed,
                                task_weights=args.weights,
                                sqp_refine=not args.no_sqp)

    # ── Data loading ────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            csv_path = Path(__file__).parent.parent / args.csv
        log.info(f"Loading data from {csv_path}")
        X, y = load_csv(str(csv_path), args.features, args.targets)
    else:
        log.info("Generating synthetic pulp data (v2 with motor_current/runtime)...")
        df = generate_synthetic_pulp_data(n_samples=args.n_samples,
                                          seed=args.seed)
        X = df[args.features].values.astype("float32")
        y = df[args.targets].values.astype("float32")

    log.info(f"Data: X={X.shape}, y={y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # ── Train ───────────────────────────────────────────────
    model = HGALSTMv2(input_size=X.shape[1], target_names=args.targets,
                      ga_cfg=ga_cfg, train_cfg=train_cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ────────────────────────────────────────────
    log.info("=" * 55)
    log.info("  EVALUATION ON TEST SET")
    log.info("=" * 55)
    metrics = model.evaluate(X_test, y_test)

    print()
    print("=" * 60)
    print("  HGA-LSTM v2 RESULTS (per task)")
    print("=" * 60)
    for task, m in metrics.items():
        print(f"\n  [{task}]")
        print(f"    RMSE = {m['RMSE']:.4f}")
        print(f"    MAE  = {m['MAE']:.4f}")
        print(f"    ARGE = {m['ARGE']:.4f}")
        print(f"    R²   = {m['R2']:.4f}")
    print("=" * 60 + "\n")

    # ── Save ────────────────────────────────────────────────
    model_path = save_dir / "hga_lstm_v2_model.pt"
    model.save(str(model_path))
    model.save_ga_history(str(save_dir / "ga_history_v2.json"))

    log.info("=" * 55)
    log.info("Training complete!")
    log.info(f"  Model:    {model_path}")
    log.info(f"  History:  {save_dir}/ga_history_v2.json")
    for task, m in metrics.items():
        log.info(f"  [{task}] R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
