"""
visualize.py — Visualization utilities for HGA-LSTM results

Generates publication-ready figures for the HGA-LSTM paper:
    Fig 1. GA convergence curve
    Fig 2. Time-series predictions vs ground truth + scatter plot
    Fig 3. Prediction error distribution + histogram
    Fig 4. Feature correlation heatmap (new)
    Fig 5. Multi-model comparison bar chart (new)

All figures use a dark theme compatible with academic paper exports.
For paper submission: use plot_for_paper() variants (white background, 300 dpi).

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2025
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

log = logging.getLogger("HGA-LSTM.viz")

# ---------------------------------------------------------------------------
# Dark theme (for screen/presentation)
# ---------------------------------------------------------------------------
DARK_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}

# Light theme (for paper submission — white background, print-friendly)
PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "black",
    "text.color":       "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "grid.color":       "#cccccc",
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

# Color palette
C_TRUE   = "#3fb950"   # green  — ground truth
C_PRED   = "#58a6ff"   # blue   — HGA-LSTM prediction
C_LSTM   = "#f0883e"   # orange — baseline LSTM
C_ERR    = "#f78166"   # red    — error fill
C_IDEAL  = "#e3b341"   # yellow — ideal line (scatter)


# ---------------------------------------------------------------------------
# Figure 1: GA convergence
# ---------------------------------------------------------------------------

def plot_ga_convergence(
    history: list[dict],
    save_path: str = "plots/ga_convergence.png",
    paper_style: bool = False,
) -> None:
    """
    Plot GA convergence: best RMSE and population average RMSE per generation.

    Shows:
        - Blue solid line:  best individual RMSE (monotonically non-increasing)
        - Orange dashed:    population average RMSE
        - Shaded area:      gap between best and average (diversity indicator)
        - Vertical dashed:  convergence generation (where best stabilizes)

    Args:
        history:     List of dicts from GeneticAlgorithm.history
        save_path:   Output file path (.png)
        paper_style: Use white background for paper submission
    """
    gens = [h["generation"] for h in history]
    best = [h["best_rmse"]  for h in history]
    avg  = [h["avg_rmse"]   for h in history]

    # Detect convergence: first gen where best stops improving (>1% change)
    conv_gen = len(gens)
    for i in range(1, len(best)):
        if abs(best[i] - best[i-1]) / (abs(best[i-1]) + 1e-9) < 0.01:
            conv_gen = gens[i]
            break

    style = PAPER_STYLE if paper_style else DARK_STYLE
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(gens, best, color=C_PRED,  lw=2.0, label="Best individual RMSE",  zorder=3)
        ax.plot(gens, avg,  color=C_LSTM,  lw=1.5, ls="--", alpha=0.75,
                label="Population average RMSE", zorder=2)
        ax.fill_between(gens, best, avg, alpha=0.10, color=C_PRED)

        ax.axvline(conv_gen, color="#8b949e", lw=1.2, ls=":",
                   label=f"Convergence (gen {conv_gen})")

        ax.set_xlabel("Generation")
        ax.set_ylabel("RMSE (normalized)")
        ax.set_title("Genetic Algorithm Convergence — HGA-LSTM")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(gens))

        _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2: Predictions vs ground truth
# ---------------------------------------------------------------------------

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Pulp Density [t/m3]",
    save_path: str = "plots/predictions.png",
    paper_style: bool = False,
) -> None:
    """
    Two-panel figure: time-series comparison + scatter plot.

    Left panel:  Time series of true vs predicted values with error fill
    Right panel: Scatter plot (true vs predicted) with ideal y=x line
                 and linear regression line + R2 annotation

    Args:
        y_true:      Ground truth values [N]
        y_pred:      Model predictions   [N]
        target_name: Y-axis label
        save_path:   Output file path
        paper_style: White background for paper
    """
    style = PAPER_STYLE if paper_style else DARK_STYLE
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 6))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.30)

        # ── Left: time series ──
        ax1 = fig.add_subplot(gs[0])
        t = np.arange(len(y_true))
        ax1.plot(t, y_true, color=C_TRUE, lw=1.5, label="Ground truth", zorder=3)
        ax1.plot(t, y_pred, color=C_PRED, lw=1.5, label="HGA-LSTM",     zorder=2, alpha=0.9)
        ax1.fill_between(t, y_true, y_pred, alpha=0.12, color=C_ERR)

        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        ax1.set_title(f"HGA-LSTM Prediction  |  RMSE={rmse:.4f}  MAE={mae:.4f}")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel(target_name)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ── Right: scatter ──
        ax2 = fig.add_subplot(gs[1])
        ax2.scatter(y_true, y_pred, alpha=0.30, s=6, color=C_PRED, zorder=2)

        lims = [min(y_true.min(), y_pred.min()) * 0.98,
                max(y_true.max(), y_pred.max()) * 1.02]
        ax2.plot(lims, lims, color=C_IDEAL, lw=1.5, ls="--", label="Ideal", zorder=3)

        # Regression line + R2
        slope, intercept, r_val, _, _ = stats.linregress(y_true, y_pred)
        x_fit = np.array(lims)
        ax2.plot(x_fit, slope * x_fit + intercept, color=C_ERR, lw=1.5,
                 label=f"Fit (R²={r_val**2:.3f})", zorder=4)

        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel("Ground truth")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Scatter plot")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3: Error distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "plots/errors.png",
    paper_style: bool = False,
) -> None:
    """
    Two-panel error analysis: residual time series + histogram with normal fit.

    Left panel:  Residuals over time (e_t = y_pred_t - y_true_t)
    Right panel: Error histogram with fitted normal distribution N(mu, sigma)
                 and +/-sigma markers

    Note: Residuals should be approximately normally distributed (N(0, sigma))
    for a well-calibrated model. Systematic bias (mu != 0) indicates underfitting.

    Args:
        y_true, y_pred: Arrays of same length [N]
        save_path:      Output file path
        paper_style:    White background for paper
    """
    errors = y_pred - y_true
    mu, sigma = float(errors.mean()), float(errors.std())

    style = PAPER_STYLE if paper_style else DARK_STYLE
    with plt.rc_context(style):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: residual time series ──
        ax1.plot(errors, color=C_ERR, alpha=0.70, lw=1.0)
        ax1.axhline(0,  color="#ffffff" if not paper_style else "black",
                    lw=0.8, ls="--", alpha=0.6)
        ax1.axhline(mu, color=C_LSTM, lw=1.2, ls="-",
                    label=f"Mean bias: {mu:.4f}")
        ax1.fill_between(range(len(errors)), errors, alpha=0.18, color=C_ERR)
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Prediction error")
        ax1.set_title("Residuals over time")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ── Right: histogram + normal fit ──
        ax2.hist(errors, bins=50, color=C_PRED, alpha=0.75,
                 edgecolor="#0d1117" if not paper_style else "white",
                 density=True, label="Error distribution")

        x_norm = np.linspace(errors.min(), errors.max(), 300)
        ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                 color="white" if not paper_style else "black",
                 lw=2.0, label=f"N(μ={mu:.4f}, σ={sigma:.4f})")

        ax2.axvline(mu,         color=C_LSTM, lw=1.5, ls="--",
                    label=f"μ = {mu:.4f}")
        ax2.axvline(mu + sigma, color=C_TRUE, lw=1.2, ls=":",
                    label=f"+σ = {sigma:.4f}")
        ax2.axvline(mu - sigma, color=C_TRUE, lw=1.2, ls=":")

        ax2.set_xlabel("Prediction error")
        ax2.set_ylabel("Probability density")
        ax2.set_title("Error histogram")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("HGA-LSTM Prediction Error Analysis", fontweight="bold")
        plt.tight_layout()
        _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 4: Feature correlation heatmap (NEW — useful for paper)
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str = "Target",
    save_path: str = "plots/correlation.png",
    paper_style: bool = False,
) -> None:
    """
    Correlation heatmap between all features and the target variable.

    Shows Pearson correlation coefficients as color-coded matrix.
    Helps identify which features are most predictive of the target.

    Args:
        X:             Feature matrix [N, F]
        y:             Target array   [N]
        feature_names: List of F feature names
        target_name:   Name of target variable
        save_path:     Output file path
        paper_style:   White background for paper
    """
    # Build combined data matrix (features + target)
    data = np.column_stack([X, y])
    labels = feature_names + [target_name]
    corr = np.corrcoef(data.T)

    style = PAPER_STYLE if paper_style else DARK_STYLE
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))

        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pearson correlation")

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="black" if abs(corr[i, j]) < 0.6 else "white")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title("Feature-Target Correlation Matrix", fontweight="bold")

        plt.tight_layout()
        _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 5: Multi-model comparison (NEW — required for paper Table 2)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str = "plots/comparison.png",
    paper_style: bool = False,
) -> None:
    """
    Bar chart comparing multiple models across RMSE, MAE, and R2 metrics.

    Args:
        metrics: Dict of {model_name: {metric_name: value}}
                 Example:
                     {
                         "Baseline LSTM": {"RMSE": 3.83, "MAE": 2.94, "R2": 0.871},
                         "LSTM+PSO":      {"RMSE": 3.42, "MAE": 2.68, "R2": 0.902},
                         "HGA-LSTM":      {"RMSE": 3.08, "MAE": 2.39, "R2": 0.935},
                     }
        save_path:   Output file path
        paper_style: White background for paper
    """
    model_names  = list(metrics.keys())
    metric_names = ["RMSE", "MAE", "R2"]
    lower_better = {"RMSE": True, "MAE": True, "R2": False}

    n_models  = len(model_names)
    colors    = [C_LSTM, "#8b949e", C_PRED][:n_models]

    style = PAPER_STYLE if paper_style else DARK_STYLE
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for ax, metric in zip(axes, metric_names):
            values = [metrics[m].get(metric, 0.0) for m in model_names]
            bars = ax.bar(model_names, values, color=colors,
                          edgecolor="white" if not paper_style else "#333",
                          linewidth=1.2, width=0.5)

            # Highlight the best bar
            best_idx = (np.argmin(values) if lower_better[metric]
                        else np.argmax(values))
            bars[best_idx].set_edgecolor("white" if not paper_style else "black")
            bars[best_idx].set_linewidth(2.5)

            # Value labels on top of bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003 * max(values),
                        f"{val:.3f}", ha="center", va="bottom",
                        fontweight="bold", fontsize=10)

            # Improvement vs first model
            base = values[0]
            best = values[best_idx]
            pct  = ((base - best) / abs(base) * 100 if lower_better[metric]
                    else (best - base) / abs(base) * 100)
            ax.text(0.5, 0.97, f"Improvement: {pct:.1f}%",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color=C_TRUE, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="none",
                              edgecolor=C_TRUE, alpha=0.8))

            label = f"{metric} ({'lower' if lower_better[metric] else 'higher'} is better)"
            ax.set_title(label, fontweight="bold")
            ax.set_ylabel(metric)
            ypad = 0.08 * (max(values) - min(values))
            ax.set_ylim(min(values) - ypad, max(values) + ypad * 2.5)

        fig.suptitle(
            "Model Comparison: Baseline LSTM / LSTM+PSO / HGA-LSTM\n"
            "Pulp density prediction on Derrick Stack Sizer dataset",
            fontweight="bold",
        )
        plt.tight_layout()
        _save(fig, save_path)


# ---------------------------------------------------------------------------
# Console metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(
    metrics: dict[str, float],
    baseline: dict[str, float] | None = None,
) -> None:
    """
    Print formatted metrics table to console.

    If baseline is provided, shows percentage improvement for each metric.
    Also prints comparison with Zou et al. (2022) reference values.

    Args:
        metrics:  Dict with RMSE, MAE, ARGE, R2 from HGA-LSTM
        baseline: Optional dict with same keys for baseline model
    """
    print("\n" + "=" * 58)
    print("  HGA-LSTM RESULTS")
    print("=" * 58)

    if baseline:
        print(f"  {'Metric':<8} {'Baseline':>10} {'HGA-LSTM':>10} {'Improv.':>10}")
        print("-" * 58)
        for k in metrics:
            bv, mv = baseline.get(k, float("nan")), metrics[k]
            if not np.isnan(bv) and bv != 0:
                # For R2: higher is better; for others: lower is better
                imp  = (mv - bv) / abs(bv) * 100 if k == "R2" else (bv - mv) / abs(bv) * 100
                sign = "+" if imp > 0 else ""
                print(f"  {k:<8} {bv:>10.4f} {mv:>10.4f} {sign}{imp:>8.1f}%")
            else:
                print(f"  {k:<8} {'N/A':>10} {mv:>10.4f}")
    else:
        print(f"  {'Metric':<12} {'Value':>12}")
        print("-" * 58)
        for k, v in metrics.items():
            print(f"  {k:<12} {v:>12.4f}")

    print("=" * 58)
    print("\n  Reference — Zou et al. (2022):")
    print(f"  {'RMSE baseline':>22}: 3.8300")
    print(f"  {'RMSE HGA-LSTM (ref)':>22}: 3.0800   (-19.5%)")
    print(f"  {'RMSE this model':>22}: {metrics.get('RMSE', float('nan')):>8.4f}")
    print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure and close it to free memory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {path}")