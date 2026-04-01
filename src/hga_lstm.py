"""
hga_lstm.py — Hybrid Genetic Algorithm + LSTM (HGA-LSTM)

Implementation based on Zou et al. methodology for predicting
iron ore pulp density and screening efficiency.

Reference:
    Zou G. et al. An HGA-LSTM-Based Intelligent Model for Ore Pulp Density
    in the Hydrometallurgical Process. Materials. 2022. Vol. 15, No. 21. Article 7586.
    Results: RMSE 3.83 -> 3.08 (-19.5%), ARGE 0.119 -> 0.0752 (-36.8%)

Architecture:
    Phase 1 - Genetic Algorithm: global search over 6D hyperparameter space
    Phase 2 - SQP (L-BFGS-B):   local refinement of best GA solution
    Phase 3 - LSTM training:     final model with optimal hyperparameters

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2025
"""

from __future__ import annotations

import json
import logging
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

log = logging.getLogger("HGA-LSTM")


# ---------------------------------------------------------------------------
# Hyperparameter search space
# Defined as module-level constant (NOT inside dataclass) to allow
# access as HP_BOUNDS without instantiating HyperParams()
# ---------------------------------------------------------------------------
HP_BOUNDS: list[tuple[float, float]] = [
    (16.0,  256.0),   # hidden_size:    LSTM hidden units
    (1.0,   4.0),     # num_layers:     stacked LSTM layers
    (0.0,   0.5),     # dropout:        regularization rate
    (1e-5,  0.1),     # learning_rate:  Adam optimizer LR
    (8.0,   128.0),   # batch_size:     mini-batch size
    (5.0,   50.0),    # seq_len:        lookback window length
]


@dataclass
class HyperParams:
    """
    LSTM hyperparameters optimized by GA + SQP.
    6D search space: theta = {hidden_size, num_layers, dropout, lr, batch_size, seq_len}
    """
    hidden_size:   int   = 64
    num_layers:    int   = 2
    dropout:       float = 0.2
    learning_rate: float = 1e-3
    batch_size:    int   = 32
    seq_len:       int   = 10

    def to_vector(self) -> np.ndarray:
        """Encode as float vector for GA/SQP."""
        return np.array([
            self.hidden_size, self.num_layers, self.dropout,
            self.learning_rate, self.batch_size, self.seq_len,
        ], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "HyperParams":
        """Decode float vector back to HyperParams (rounds integer fields)."""
        return cls(
            hidden_size=max(8,   int(round(v[0]))),
            num_layers= max(1,   min(4,   int(round(v[1])))),
            dropout=    float(np.clip(v[2], 0.0, 0.5)),
            learning_rate=float(np.clip(v[3], 1e-5, 0.1)),
            batch_size= max(4,   int(round(v[4]))),
            seq_len=    max(3,   min(100, int(round(v[5])))),
        )


@dataclass
class GAConfig:
    """Genetic Algorithm configuration."""
    population_size: int   = 20
    n_generations:   int   = 30
    crossover_prob:  float = 0.8
    mutation_prob:   float = 0.15
    mutation_scale:  float = 0.1
    elitism_count:   int   = 2
    tournament_k:    int   = 3


@dataclass
class TrainConfig:
    """LSTM training configuration."""
    epochs:     int   = 100
    patience:   int   = 15
    min_delta:  float = 1e-6
    device:     str   = "auto"
    seed:       int   = 42
    sqp_refine: bool  = True


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    """
    Multi-layer LSTM for time-series regression.

    Architecture: Input [B,T,F] -> LSTM x L -> Dropout -> FC -> Output [B,1]

    LSTM equations (Hochreiter & Schmidhuber, 1997):
        f_t = sigma(W_f [h_{t-1}, x_t] + b_f)       forget gate
        i_t = sigma(W_i [h_{t-1}, x_t] + b_i)       input gate
        C_t_tilde = tanh(W_C [h_{t-1}, x_t] + b_C)  cell candidate
        C_t = f_t * C_{t-1} + i_t * C_t_tilde        cell state update
        o_t = sigma(W_o [h_{t-1}, x_t] + b_o)        output gate
        h_t = o_t * tanh(C_t)                         hidden state output
    """

    def __init__(self, input_size: int, output_size: int, hp: HyperParams):
        super().__init__()
        self.hp = hp
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            batch_first=True,
            dropout=hp.dropout if hp.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=hp.dropout)
        self.fc = nn.Linear(hp.hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]     # use only last time step
        return self.fc(self.dropout(last_step))


# ---------------------------------------------------------------------------
# Min-Max scaler
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Feature-wise Min-Max normalization to [0, 1].
    Epsilon avoids division by zero for constant features.
    """
    EPS = 1e-9

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self._scale(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Scaler not fitted")
        return self._scale(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Scaler not fitted")
        return X * (self.max_ - self.min_ + self.EPS) + self.min_

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_ + self.EPS)


# ---------------------------------------------------------------------------
# Sliding-window sequence builder
# ---------------------------------------------------------------------------

def make_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping sliding-window sequences for LSTM input.

    For each index i in [0, N-seq_len):
        X[i] = data[i : i+seq_len]     shape [seq_len, features]
        y[i] = targets[i + seq_len]    next-step target

    Returns:
        X: [N-seq_len, seq_len, features]
        y: [N-seq_len]
    """
    X_list, y_list = [], []
    for i in range(len(data) - seq_len):
        X_list.append(data[i: i + seq_len])
        y_list.append(targets[i + seq_len])
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_evaluate(
    hp: HyperParams,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> tuple[float, LSTMRegressor | None]:
    """
    Train LSTM with given hyperparameters, return (val_rmse, model).

    Adam optimizer update rule:
        m_t = beta1*m_{t-1} + (1-beta1)*grad_L
        v_t = beta2*v_{t-1} + (1-beta2)*grad_L^2
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
    with beta1=0.9, beta2=0.999, eps=1e-8.

    Gradient clipping (max_norm=1.0) prevents exploding gradients.
    ReduceLROnPlateau halves LR when val_rmse stalls for 5 epochs.

    Returns:
        (inf, None) if training failed (e.g. insufficient data for batch)
        (best_val_rmse, model) on success
    """
    torch.manual_seed(cfg.seed)
    input_size = X_train.shape[-1]

    X_seq_tr,  y_seq_tr  = make_sequences(X_train, y_train, hp.seq_len)
    X_seq_val, y_seq_val = make_sequences(X_val,   y_val,   hp.seq_len)

    if len(X_seq_tr) < hp.batch_size:
        return float("inf"), None

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_seq_tr).to(device),
            torch.from_numpy(y_seq_tr).unsqueeze(1).to(device),
        ),
        batch_size=hp.batch_size, shuffle=True, drop_last=False,
    )

    model = LSTMRegressor(input_size, 1, hp).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp.learning_rate,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=False,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_cnt  = 0
    best_state:   dict | None = None

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_seq_val).to(device))
            val_rmse = float(np.sqrt(np.mean(
                (preds.cpu().numpy().flatten() - y_seq_val) ** 2
            )))

        scheduler.step(val_rmse)

        if verbose and (epoch + 1) % 20 == 0:
            log.info(f"  Epoch {epoch+1:4d}/{cfg.epochs} | val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse - cfg.min_delta:
            best_val_rmse = val_rmse
            patience_cnt  = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if verbose:
                    log.info(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse, model


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    GA for LSTM hyperparameter optimization.

    Operators:
        Selection : Tournament (size k=3)
        Crossover : Simulated Binary Crossover — SBX (eta=2)
        Mutation  : Gaussian with adaptive scale
        Elitism   : Top-2 preserved unconditionally each generation

    SBX spread factor beta:
        if u <= 0.5: beta = (2u)^{1/(eta+1)}
        else:        beta = (1/(2*(1-u)))^{1/(eta+1)}
        c1 = 0.5 * [(1+beta)*p1 + (1-beta)*p2]
        c2 = 0.5 * [(1-beta)*p1 + (1+beta)*p2]
    """

    def __init__(self, bounds: list[tuple[float, float]], ga_cfg: GAConfig):
        self.bounds = np.array(bounds)
        self.cfg    = ga_cfg
        self.dim    = len(bounds)
        self._best_vector:  np.ndarray | None = None
        self._best_fitness: float = float("inf")
        self.history: list[dict] = []

    def _init_population(self) -> np.ndarray:
        return np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.cfg.population_size, self.dim),
        )

    def _tournament_select(self, fitness: np.ndarray) -> int:
        idx = np.random.choice(len(fitness), self.cfg.tournament_k, replace=False)
        return int(idx[np.argmin(fitness[idx])])

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray):
        eta = 2.0
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                u = random.random()
                b = ((2*u)**(1/(eta+1)) if u <= 0.5
                     else (1/(2*(1-u)))**(1/(eta+1)))
                c1[i] = 0.5*((1+b)*p1[i] + (1-b)*p2[i])
                c2[i] = 0.5*((1-b)*p1[i] + (1+b)*p2[i])
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mut = ind.copy()
        for i in range(self.dim):
            if random.random() < self.cfg.mutation_prob:
                scale = self.cfg.mutation_scale * (self.bounds[i,1] - self.bounds[i,0])
                mut[i] = np.clip(mut[i] + np.random.normal(0, scale),
                                 self.bounds[i,0], self.bounds[i,1])
        return mut

    def _clip(self, v: np.ndarray) -> np.ndarray:
        return np.clip(v, self.bounds[:, 0], self.bounds[:, 1])

    def run(self, fitness_fn) -> np.ndarray:
        """
        Run GA evolution.
        fitness_fn(vector) -> float, lower is better.
        Returns best vector found.
        """
        pop     = self._init_population()
        fitness = np.array([fitness_fn(ind) for ind in pop])
        log.info(f"GA start | initial best RMSE: {fitness.min():.4f}")

        for gen in range(self.cfg.n_generations):
            t0 = time.time()
            next_pop: list[np.ndarray] = []

            # Elitism
            elite_idx = np.argsort(fitness)[: self.cfg.elitism_count]
            next_pop.extend(pop[i].copy() for i in elite_idx)

            # Reproduction
            while len(next_pop) < self.cfg.population_size:
                p1 = pop[self._tournament_select(fitness)]
                p2 = pop[self._tournament_select(fitness)]
                c1, c2 = (self._sbx_crossover(p1, p2)
                          if random.random() < self.cfg.crossover_prob
                          else (p1.copy(), p2.copy()))
                next_pop.append(self._clip(self._mutate(c1)))
                if len(next_pop) < self.cfg.population_size:
                    next_pop.append(self._clip(self._mutate(c2)))

            pop     = np.array(next_pop)
            fitness = np.array([fitness_fn(ind) for ind in pop])

            gen_best = float(fitness.min())
            gen_avg  = float(fitness.mean())
            elapsed  = time.time() - t0

            if gen_best < self._best_fitness:
                self._best_fitness = gen_best
                self._best_vector  = pop[int(fitness.argmin())].copy()

            self.history.append({
                "generation": gen + 1,
                "best_rmse":  gen_best,
                "avg_rmse":   gen_avg,
                "elapsed_s":  round(elapsed, 2),
            })
            log.info(f"Gen {gen+1:3d}/{self.cfg.n_generations} | "
                     f"Best: {gen_best:.4f} | Avg: {gen_avg:.4f} | {elapsed:.1f}s")

        return self._best_vector


# ---------------------------------------------------------------------------
# SQP local refinement
# ---------------------------------------------------------------------------

def sqp_refine(
    initial_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    fitness_fn,
    maxiter: int = 30,
) -> np.ndarray:
    """
    L-BFGS-B local refinement of the GA solution.

    L-BFGS-B is functionally equivalent to SQP for box-constrained problems
    without nonlinear constraints. It uses a limited-memory quasi-Newton
    approximation of the inverse Hessian.

    Returns refined vector (or initial_vector if refinement does not improve).
    """
    log.info("SQP refinement (L-BFGS-B)...")
    initial_rmse = fitness_fn(initial_vector)

    result = minimize(
        fitness_fn, x0=initial_vector, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-9},
    )

    if result.fun < initial_rmse:
        log.info(f"SQP improved: {initial_rmse:.4f} -> {result.fun:.4f}")
        return result.x

    log.warning("SQP did not improve solution — keeping GA result")
    return initial_vector


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression quality metrics.

        RMSE = sqrt(mean((y_pred - y_true)^2))
        MAE  = mean(|y_pred - y_true|)
        ARGE = mean(|y_pred - y_true| / (|y_true| + eps))
        R2   = 1 - SS_res / SS_tot
    """
    eps = 1e-9
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE":  float(np.mean(np.abs(err))),
        "ARGE": float(np.mean(np.abs(err) / (np.abs(y_true) + eps))),
        "R2":   float(1.0 - np.sum(err**2) / (np.sum((y_true - y_true.mean())**2) + eps)),
    }


# ---------------------------------------------------------------------------
# Main HGA-LSTM class
# ---------------------------------------------------------------------------

class HGALSTM:
    """
    HGA-LSTM: Hybrid Genetic Algorithm + LSTM.

    Training pipeline:
        1. GA:          Evolve population to find near-optimal hyperparameters
        2. SQP:         L-BFGS-B local refinement of best GA solution
        3. Final train: LSTM with optimal hyperparameters (2x epochs/patience)

    All features and targets are normalized to [0,1] (no data leakage:
    scaler is fit on training set only, then applied to val/test).

    Usage:
        model = HGALSTM(input_size=5)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        model.save("outputs/model.pt")
    """

    def __init__(
        self,
        input_size: int,
        ga_cfg:    GAConfig    | None = None,
        train_cfg: TrainConfig | None = None,
    ):
        self.input_size = input_size
        self.ga_cfg     = ga_cfg    or GAConfig()
        self.train_cfg  = train_cfg or TrainConfig()
        self.device     = self._resolve_device()

        self.best_hp:    HyperParams   | None = None
        self.best_model: LSTMRegressor | None = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.ga_history: list[dict] = []

        log.info(f"HGA-LSTM | device={self.device} | input_size={input_size}")

    def _resolve_device(self) -> torch.device:
        if self.train_cfg.device == "auto":
            if torch.cuda.is_available():
                log.info(f"GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            log.warning("No GPU — using CPU")
            return torch.device("cpu")
        return torch.device(self.train_cfg.device)

    def fit(self, X_train, y_train, X_val, y_val) -> "HGALSTM":
        """Full training pipeline: GA -> SQP -> Final LSTM."""
        torch.manual_seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        random.seed(self.train_cfg.seed)

        # Normalize: fit on train only (no data leakage)
        X_tr_s  = self.scaler_X.fit_transform(X_train)
        y_tr_s  = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_val_s = self.scaler_X.transform(X_val)
        y_val_s = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        def fitness_fn(vector: np.ndarray) -> float:
            hp = HyperParams.from_vector(vector)
            try:
                rmse, _ = train_evaluate(
                    hp, X_tr_s, y_tr_s, X_val_s, y_val_s,
                    self.train_cfg, self.device,
                )
            except Exception as exc:
                log.debug(f"fitness_fn error: {exc}")
                return float("inf")
            return rmse

        # Phase 1: GA
        log.info("=" * 55)
        log.info("PHASE 1: Genetic Algorithm")
        log.info("=" * 55)
        ga = GeneticAlgorithm(HP_BOUNDS, self.ga_cfg)
        best_vector = ga.run(fitness_fn)
        self.ga_history = ga.history

        # Phase 2: SQP
        if self.train_cfg.sqp_refine:
            log.info("=" * 55)
            log.info("PHASE 2: SQP Refinement")
            log.info("=" * 55)
            best_vector = sqp_refine(best_vector, HP_BOUNDS, fitness_fn)

        self.best_hp = HyperParams.from_vector(best_vector)
        log.info(f"Optimal hyperparameters: {asdict(self.best_hp)}")

        # Phase 3: Final training with 2x epochs
        log.info("=" * 55)
        log.info("PHASE 3: Final LSTM Training")
        log.info("=" * 55)
        final_cfg = TrainConfig(
            epochs=self.train_cfg.epochs * 2,
            patience=self.train_cfg.patience * 2,
            device=self.train_cfg.device,
            seed=self.train_cfg.seed,
        )
        rmse, model = train_evaluate(
            self.best_hp, X_tr_s, y_tr_s, X_val_s, y_val_s,
            final_cfg, self.device, verbose=True,
        )

        if model is None:
            raise RuntimeError(
                "Final LSTM training failed — check data size / hyperparameter bounds"
            )

        self.best_model = model
        log.info(f"Final val RMSE (normalized): {rmse:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for X. Returns array of length len(X) - seq_len."""
        if self.best_model is None:
            raise RuntimeError("Model not trained — call .fit() first")
        self.best_model.eval()

        X_s = self.scaler_X.transform(X)
        dummy = np.zeros(len(X_s), dtype=np.float32)
        X_seq, _ = make_sequences(X_s, dummy, self.best_hp.seq_len)

        with torch.no_grad():
            preds_norm = (self.best_model(torch.from_numpy(X_seq).to(self.device))
                          .cpu().numpy().flatten())

        return self.scaler_y.inverse_transform(preds_norm.reshape(-1, 1)).flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute RMSE/MAE/ARGE/R2 on test data."""
        preds     = self.predict(X)
        y_aligned = y[self.best_hp.seq_len:]
        metrics   = compute_metrics(y_aligned, preds)
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")
        return metrics

    def save(self, path: str = "hga_lstm_model.pt") -> None:
        """Save model, hyperparameters, and scalers to .pt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":  self.best_model.state_dict(),
            "hp":           asdict(self.best_hp),
            "input_size":   self.input_size,
            "scaler_X_min": self.scaler_X.min_,
            "scaler_X_max": self.scaler_X.max_,
            "scaler_y_min": self.scaler_y.min_,
            "scaler_y_max": self.scaler_y.max_,
            "ga_history":   self.ga_history,
        }, path)
        log.info(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "HGALSTM":
        """Load model from .pt file for inference."""
        ckpt = torch.load(path, map_location="cpu")
        hp   = HyperParams(**ckpt["hp"])
        obj  = cls(input_size=ckpt["input_size"])
        obj.best_hp    = hp
        obj.best_model = LSTMRegressor(ckpt["input_size"], 1, hp)
        obj.best_model.load_state_dict(ckpt["model_state"])
        obj.best_model.to(obj.device).eval()
        obj.scaler_X.min_ = ckpt["scaler_X_min"]
        obj.scaler_X.max_ = ckpt["scaler_X_max"]
        obj.scaler_y.min_ = ckpt["scaler_y_min"]
        obj.scaler_y.max_ = ckpt["scaler_y_max"]
        obj.ga_history    = ckpt.get("ga_history", [])
        log.info(f"Model loaded: {path}")
        return obj

    def save_ga_history(self, path: str = "ga_history.json") -> None:
        """Save GA convergence history to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ga_history, f, indent=2, ensure_ascii=False)
        log.info(f"GA history saved: {path}")