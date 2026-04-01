"""
train.py — HGA-LSTM training entry point

Run from project root (train-model-hga-lstm/):

    # Quick smoke test (~2 min, synthetic data)
    python src/train.py --fast

    # Full run, synthetic data (~30 min)
    python src/train.py

    # Real data: predict pulp density
    python src/train.py \
        --csv data/data_optimized/input_data_large.csv \
        --features Feed_Flowrate_tph Feed_Fe_percent \
                   Feed_Size_minus150um_percent Water_Consumption_m3_per_t \
        --target Pulp_Density_t_m3

    # Real data: predict screening efficiency
    python src/train.py \
        --csv data//data_optimized/output_data_large.csv \
        --features Fines_Product_tph Fines_Fe_percent Fines_Recovery_percent \
        --target Classification_Efficiency_percent

NOTE: Do NOT include 'Equipment_Status' in --features (categorical string).
NOTE: Do NOT include 'Timestamp' in --features (auto-excluded).

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path fix: allows running as  python src/train.py  from project root
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hga_lstm import HGALSTM, GAConfig, TrainConfig
from data_utils import generate_synthetic_pulp_data, load_csv, train_val_test_split
from visualize import (
    plot_ga_convergence,
    plot_predictions,
    plot_error_distribution,
    plot_feature_correlation,
    plot_model_comparison,
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
        description="HGA-LSTM: Hybrid GA + LSTM for iron ore pulp parameter prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("Data")
    data.add_argument("--csv",       default=None,
                      help="Path to CSV (omit to use synthetic data)")
    data.add_argument("--features",  nargs="+",
                      default=["amplitude_mm", "frequency_hz", "angle_deg",
                               "pulp_flow", "solid_pct"],
                      help="Numeric feature column names")
    data.add_argument("--target",    default="density",
                      help="Target column name")
    data.add_argument("--n-samples", type=int, default=2000,
                      help="Number of synthetic samples (ignored with --csv)")

    ga = p.add_argument_group("Genetic Algorithm")
    ga.add_argument("--ga-pop",  type=int,   default=20)
    ga.add_argument("--ga-gen",  type=int,   default=30)
    ga.add_argument("--no-sqp",  action="store_true",
                    help="Disable SQP local refinement after GA")

    tr = p.add_argument_group("Training")
    tr.add_argument("--epochs",  type=int,   default=100)
    tr.add_argument("--seed",    type=int,   default=42)
    tr.add_argument("--device",  default="auto", choices=["auto", "cuda", "cpu"])
    tr.add_argument("--fast",    action="store_true",
                    help="Quick test: pop=6, gen=5, epochs=20 (~2 min)")

    out = p.add_argument_group("Output")
    out.add_argument("--save-dir", default="outputs",
                     help="Directory for model, history, and plots")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)

    log.info("=" * 55)
    log.info("  HGA-LSTM Training")
    log.info("=" * 55)

    # ── Configuration ────────────────────────────────────────
    if args.fast:
        log.info("FAST MODE — reduced parameters for smoke test")
        ga_cfg    = GAConfig(population_size=6, n_generations=5)
        train_cfg = TrainConfig(epochs=20, patience=5,
                                device=args.device, seed=args.seed)
    else:
        ga_cfg    = GAConfig(population_size=args.ga_pop,
                             n_generations=args.ga_gen)
        train_cfg = TrainConfig(
            epochs=args.epochs, patience=15,
            device=args.device, seed=args.seed,
            sqp_refine=not args.no_sqp,
        )

    # ── Data loading ─────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            # Try relative to project root
            csv_path = Path(__file__).parent.parent / args.csv
        log.info(f"Loading data from {csv_path}")
        X, y = load_csv(str(csv_path), args.features, args.target)
    else:
        log.info("Generating synthetic pulp data...")
        df = generate_synthetic_pulp_data(
            n_samples=args.n_samples, seed=args.seed
        )
        X = df[args.features].values.astype("float32")
        y = df[args.target].values.astype("float32")

    log.info(f"Data: X={X.shape}, y={y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # ── Train HGA-LSTM ───────────────────────────────────────
    model = HGALSTM(input_size=X.shape[1], ga_cfg=ga_cfg, train_cfg=train_cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────
    metrics = model.evaluate(X_test, y_test)
    print_metrics_table(metrics)

    y_pred         = model.predict(X_test)
    seq_len        = model.best_hp.seq_len
    y_test_aligned = y_test[seq_len:]

    # ── Save model and GA history ────────────────────────────
    model.save(str(save_dir / "hga_lstm_model.pt"))
    model.save_ga_history(str(save_dir / "ga_history.json"))

    # ── Figures ──────────────────────────────────────────────
    plot_dir = str(save_dir / "plots")
    log.info(f"Generating figures -> {plot_dir}/")

    # Fig 1: GA convergence
    plot_ga_convergence(
        model.ga_history,
        f"{plot_dir}/ga_convergence.png",
    )

    # Fig 2: Predictions vs ground truth
    plot_predictions(
        y_test_aligned, y_pred,
        target_name=args.target,
        save_path=f"{plot_dir}/predictions.png",
    )

    # Fig 3: Error distribution
    plot_error_distribution(
        y_test_aligned, y_pred,
        save_path=f"{plot_dir}/errors.png",
    )

    # Fig 4: Feature correlation heatmap
    plot_feature_correlation(
        X_test, y_test,
        feature_names=args.features,
        target_name=args.target,
        save_path=f"{plot_dir}/correlation.png",
    )

    # Fig 5: Model comparison (uses paper Table 2 reference values)
    plot_model_comparison(
        metrics={
            "Baseline LSTM": {"RMSE": 3.830, "MAE": 2.941, "R2": 0.871},
            "LSTM+PSO":      {"RMSE": 3.420, "MAE": 2.680, "R2": 0.902},
            "HGA-LSTM":      {"RMSE": metrics["RMSE"],
                              "MAE":  metrics["MAE"],
                              "R2":   metrics["R2"]},
        },
        save_path=f"{plot_dir}/comparison.png",
    )

    # Paper-style (white background) versions of key figures
    plot_predictions(
        y_test_aligned, y_pred,
        target_name=args.target,
        save_path=f"{plot_dir}/predictions_paper.png",
        paper_style=True,
    )
    plot_model_comparison(
        metrics={
            "Baseline LSTM": {"RMSE": 3.830, "MAE": 2.941, "R2": 0.871},
            "LSTM+PSO":      {"RMSE": 3.420, "MAE": 2.680, "R2": 0.902},
            "HGA-LSTM":      {"RMSE": metrics["RMSE"],
                              "MAE":  metrics["MAE"],
                              "R2":   metrics["R2"]},
        },
        save_path=f"{plot_dir}/comparison_paper.png",
        paper_style=True,
    )

    log.info("=" * 55)
    log.info("Training complete!")
    log.info(f"  Model:    {save_dir}/hga_lstm_model.pt")
    log.info(f"  Figures:  {save_dir}/plots/")
    log.info(f"  RMSE:     {metrics['RMSE']:.4f}")
    log.info(f"  R2:       {metrics['R2']:.4f}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()