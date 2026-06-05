"""
hga_lstm_v2.py — Multi-task HGA-LSTM (η + β-recommendation)

ЗМІНИ vs v1 (для Розділу 3, рекомендаційна підсистема β):
    + MultiTaskLSTM: спільний LSTM-енкодер + 2 task-specific голови
    + Зважена втрата: L = w_eta * MSE(η) + w_beta * MSE(β)
      (за замовчуванням w_eta=0.7, w_beta=0.3 — η важливіша)
    + Два MinMaxScaler для y: окрема нормалізація для η і β
    + Метрики рахуються окремо для кожного таргета

Архітектура:
    Phase 1 - Genetic Algorithm: пошук гіперпараметрів (як v1)
    Phase 2 - SQP (L-BFGS-B):   локальне уточнення
    Phase 3 - LSTM training:     фінальне навчання з оптимальними HP

Reference (для архітектури):
    Caruana R. Multitask Learning. Machine Learning, 1997.
    Hard parameter sharing — спільні приховані шари, окремі голови.

Authors: Moiseichenko V.V., Savytskyi O.I.
         Kryvyi Rih National University, 2026
"""

from __future__ import annotations

import json
import logging
import random
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

log = logging.getLogger("HGA-LSTM-v2")


# ---------------------------------------------------------------------------
# Hyperparameter search space (як v1)
# ---------------------------------------------------------------------------
HP_BOUNDS: list[tuple[float, float]] = [
    (16.0,  256.0),
    (1.0,   4.0),
    (0.0,   0.5),
    (1e-5,  0.1),
    (8.0,   128.0),
    (5.0,   50.0),
]


@dataclass
class HyperParams:
    hidden_size:   int   = 64
    num_layers:    int   = 2
    dropout:       float = 0.2
    learning_rate: float = 1e-3
    batch_size:    int   = 32
    seq_len:       int   = 10

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.hidden_size, self.num_layers, self.dropout,
            self.learning_rate, self.batch_size, self.seq_len,
        ], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "HyperParams":
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
    population_size: int   = 20
    n_generations:   int   = 30
    crossover_prob:  float = 0.8
    mutation_prob:   float = 0.15
    mutation_scale:  float = 0.1
    elitism_count:   int   = 2
    tournament_k:    int   = 3


@dataclass
class TrainConfig:
    epochs:     int   = 100
    patience:   int   = 15
    min_delta:  float = 1e-6
    device:     str   = "auto"
    seed:       int   = 42
    sqp_refine: bool  = True
    # NEW v2: ваги втрат для multi-task (за замовч.: η важливіша)
    task_weights: list[float] = field(default_factory=lambda: [0.7, 0.3])


# ---------------------------------------------------------------------------
# Multi-task LSTM: спільний енкодер + 2 голови
# ---------------------------------------------------------------------------

class MultiTaskLSTM(nn.Module):
    """
    Multi-task LSTM з hard parameter sharing.

    Архітектура:
        x [B, T, F] -> LSTM -> last_step [B, H]
                                  |
                                  +-> dropout -> FC_eta  [B, 1]  -> eta
                                  |
                                  +-> dropout -> FC_beta [B, 1]  -> beta

    Loss:
        L = w_eta * MSE(eta_pred, eta_true) + w_beta * MSE(beta_pred, beta_true)

    LSTM-рівняння — як у v1 (Hochreiter & Schmidhuber, 1997):
        f_t = sigma(W_f [h_{t-1}, x_t] + b_f)
        i_t = sigma(W_i [h_{t-1}, x_t] + b_i)
        C_t = f_t * C_{t-1} + i_t * tanh(W_C [h_{t-1}, x_t] + b_C)
        o_t = sigma(W_o [h_{t-1}, x_t] + b_o)
        h_t = o_t * tanh(C_t)
    """

    def __init__(self, input_size: int, hp: HyperParams, n_tasks: int = 2):
        super().__init__()
        self.hp = hp
        self.n_tasks = n_tasks

        # Спільний енкодер
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hp.hidden_size,
            num_layers=hp.num_layers,
            batch_first=True,
            dropout=hp.dropout if hp.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=hp.dropout)

        # Окремі голови (по одній на таргет)
        self.heads = nn.ModuleList([
            nn.Linear(hp.hidden_size, 1) for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B, n_tasks]  — кожен стовпець = окремий таргет.
        """
        lstm_out, _ = self.lstm(x)
        h = self.dropout(lstm_out[:, -1, :])     # [B, hidden]
        # Конкатенуємо виходи всіх голів у [B, n_tasks]
        outs = [head(h) for head in self.heads]  # список з n_tasks тензорів [B,1]
        return torch.cat(outs, dim=1)            # [B, n_tasks]


# ---------------------------------------------------------------------------
# MinMaxScaler — як у v1
# ---------------------------------------------------------------------------

class MinMaxScaler:
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
# Sliding-window для multi-task
# ---------------------------------------------------------------------------

def make_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-task sliding window:
        data:    [N, F]
        targets: [N] (single)  АБО  [N, T] (multi)
        returns: X [N-seq_len, seq_len, F], y [N-seq_len] АБО [N-seq_len, T]
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
# Multi-task training loop
# ---------------------------------------------------------------------------

def train_evaluate(
    hp: HyperParams,
    X_train: np.ndarray,
    y_train: np.ndarray,    # [N, T]
    X_val: np.ndarray,
    y_val: np.ndarray,      # [N, T]
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> tuple[float, MultiTaskLSTM | None]:
    """
    Train multi-task LSTM, return (weighted_val_rmse, model).

    Validation metric:
        weighted_rmse = sqrt(w_eta * MSE(eta) + w_beta * MSE(beta))
    Це фітнес для GA — він керує оптимізацією гіперпараметрів.
    """
    torch.manual_seed(cfg.seed)
    input_size = X_train.shape[-1]
    n_tasks = y_train.shape[1]
    weights = torch.tensor(cfg.task_weights[:n_tasks], dtype=torch.float32,
                           device=device)

    X_seq_tr,  y_seq_tr  = make_sequences(X_train, y_train, hp.seq_len)
    X_seq_val, y_seq_val = make_sequences(X_val,   y_val,   hp.seq_len)

    if len(X_seq_tr) < hp.batch_size:
        return float("inf"), None

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_seq_tr).to(device),
            torch.from_numpy(y_seq_tr).to(device),    # [B, T]
        ),
        batch_size=hp.batch_size, shuffle=True, drop_last=False,
    )

    model = MultiTaskLSTM(input_size, hp, n_tasks=n_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    def weighted_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [B, T]
        weighted_loss = sum_t( w_t * MSE_t )
        """
        per_task_mse = ((pred - true) ** 2).mean(dim=0)        # [T]
        return (per_task_mse * weights).sum()

    best_val = float("inf")
    patience_cnt = 0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Валідація
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_seq_val).to(device)).cpu().numpy()
            # weighted RMSE як фітнес для GA
            per_task_mse = ((preds - y_seq_val) ** 2).mean(axis=0)
            w_np = np.asarray(cfg.task_weights[:n_tasks])
            val_rmse = float(np.sqrt((per_task_mse * w_np).sum()))

        scheduler.step(val_rmse)

        if verbose and (epoch + 1) % 20 == 0:
            log.info(f"  Epoch {epoch+1:4d}/{cfg.epochs} | val_w_rmse={val_rmse:.4f}")

        if val_rmse < best_val - cfg.min_delta:
            best_val = val_rmse
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if verbose:
                    log.info(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val, model


# ---------------------------------------------------------------------------
# Genetic Algorithm — без змін vs v1
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """GA (SBX + tournament + Gaussian mutation + elitism). Як v1."""

    def __init__(self, bounds: list[tuple[float, float]], ga_cfg: GAConfig):
        self.bounds = np.array(bounds)
        self.cfg    = ga_cfg
        self.dim    = len(bounds)
        self._best_vector:  np.ndarray | None = None
        self._best_fitness: float = float("inf")
        self.history: list[dict] = []

    def _init_population(self) -> np.ndarray:
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                 size=(self.cfg.population_size, self.dim))

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
        pop = self._init_population()
        fitness = np.array([fitness_fn(ind) for ind in pop])
        log.info(f"GA start | initial best weighted RMSE: {fitness.min():.4f}")

        for gen in range(self.cfg.n_generations):
            t0 = time.time()
            next_pop: list[np.ndarray] = []

            elite_idx = np.argsort(fitness)[: self.cfg.elitism_count]
            next_pop.extend(pop[i].copy() for i in elite_idx)

            while len(next_pop) < self.cfg.population_size:
                p1 = pop[self._tournament_select(fitness)]
                p2 = pop[self._tournament_select(fitness)]
                c1, c2 = (self._sbx_crossover(p1, p2)
                          if random.random() < self.cfg.crossover_prob
                          else (p1.copy(), p2.copy()))
                next_pop.append(self._clip(self._mutate(c1)))
                if len(next_pop) < self.cfg.population_size:
                    next_pop.append(self._clip(self._mutate(c2)))

            pop = np.array(next_pop)
            fitness = np.array([fitness_fn(ind) for ind in pop])

            gen_best = float(fitness.min())
            gen_avg  = float(fitness.mean())
            elapsed  = time.time() - t0

            if gen_best < self._best_fitness:
                self._best_fitness = gen_best
                self._best_vector  = pop[int(fitness.argmin())].copy()

            self.history.append({
                "generation": gen + 1, "best_rmse": gen_best,
                "avg_rmse": gen_avg, "elapsed_s": round(elapsed, 2),
            })
            log.info(f"Gen {gen+1:3d}/{self.cfg.n_generations} | "
                     f"Best: {gen_best:.4f} | Avg: {gen_avg:.4f} | {elapsed:.1f}s")

        return self._best_vector


def sqp_refine(initial_vector, bounds, fitness_fn, maxiter: int = 30):
    """L-BFGS-B refinement. Без змін vs v1."""
    log.info("SQP refinement (L-BFGS-B)...")
    initial = fitness_fn(initial_vector)
    result = minimize(fitness_fn, x0=initial_vector, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-9})
    if result.fun < initial:
        log.info(f"SQP improved: {initial:.4f} -> {result.fun:.4f}")
        return result.x
    log.warning("SQP did not improve — keeping GA result")
    return initial_vector


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    target_names: list[str] | None = None) -> dict[str, dict[str, float]]:
    """
    Compute metrics per task.

    y_true, y_pred: [N] (single) АБО [N, T] (multi)
    Returns: dict { task_name: {RMSE, MAE, ARGE, R2} }
    """
    eps = 1e-9
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    n_tasks = y_true.shape[1]
    if target_names is None:
        target_names = [f"task_{i}" for i in range(n_tasks)]

    out = {}
    for i, name in enumerate(target_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        err = yp - yt
        out[name] = {
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE":  float(np.mean(np.abs(err))),
            "ARGE": float(np.mean(np.abs(err) / (np.abs(yt) + eps))),
            "R2":   float(1.0 - np.sum(err**2) /
                          (np.sum((yt - yt.mean())**2) + eps)),
        }
    return out


# ---------------------------------------------------------------------------
# Main HGA-LSTM class — v2 multi-task
# ---------------------------------------------------------------------------

class HGALSTMv2:
    """
    HGA-LSTM v2 — багатоцільова версія для прогнозу η + рекомендації β.

    Pipeline:
        1. GA:          еволюція гіперпараметрів (зважена RMSE як фітнес)
        2. SQP:         L-BFGS-B уточнення
        3. Final train: фінальне навчання з 2× epochs/patience

    Usage:
        model = HGALSTMv2(input_size=7, target_names=['efficiency','angle_deg'])
        model.fit(X_train, y_train, X_val, y_val)   # y_*: [N, 2]
        preds = model.predict(X_test)                # [N-seq_len, 2]
        metrics = model.evaluate(X_test, y_test)     # per-task dict
        model.save("outputs/hga_lstm_v2_model.pt")
    """

    def __init__(
        self,
        input_size: int,
        target_names: list[str],
        ga_cfg:    GAConfig    | None = None,
        train_cfg: TrainConfig | None = None,
    ):
        self.input_size   = input_size
        self.target_names = list(target_names)
        self.n_tasks      = len(target_names)
        self.ga_cfg       = ga_cfg    or GAConfig()
        self.train_cfg    = train_cfg or TrainConfig()
        self.device       = self._resolve_device()

        self.best_hp:    HyperParams    | None = None
        self.best_model: MultiTaskLSTM  | None = None
        self.scaler_X = MinMaxScaler()
        # Окремий scaler для кожного таргета — щоб різні діапазони (η: 40-97,
        # β: 35-45) не змішувались у нормалізації
        self.scalers_y = [MinMaxScaler() for _ in range(self.n_tasks)]
        self.ga_history: list[dict] = []

        log.info(f"HGA-LSTM v2 | device={self.device} | "
                 f"input_size={input_size} | tasks={target_names} | "
                 f"weights={self.train_cfg.task_weights}")

    def _resolve_device(self) -> torch.device:
        if self.train_cfg.device == "auto":
            if torch.cuda.is_available():
                log.info(f"GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            log.warning("No GPU — using CPU")
            return torch.device("cpu")
        return torch.device(self.train_cfg.device)

    def _scale_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Per-column normalization (each target uses its own scaler)."""
        out = np.zeros_like(y, dtype=np.float32)
        for i in range(self.n_tasks):
            col = y[:, i].reshape(-1, 1)
            if fit:
                out[:, i] = self.scalers_y[i].fit_transform(col).flatten()
            else:
                out[:, i] = self.scalers_y[i].transform(col).flatten()
        return out

    def _unscale_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse per-column normalization."""
        out = np.zeros_like(y_norm, dtype=np.float32)
        for i in range(self.n_tasks):
            col = y_norm[:, i].reshape(-1, 1)
            out[:, i] = self.scalers_y[i].inverse_transform(col).flatten()
        return out

    def fit(self, X_train, y_train, X_val, y_val) -> "HGALSTMv2":
        """
        Full pipeline. y_train/y_val має бути shape [N, n_tasks].
        """
        if y_train.ndim != 2 or y_train.shape[1] != self.n_tasks:
            raise ValueError(
                f"y_train shape {y_train.shape} != [N, {self.n_tasks}]"
            )

        torch.manual_seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        random.seed(self.train_cfg.seed)

        # Нормування X — fit on train only
        X_tr_s  = self.scaler_X.fit_transform(X_train)
        X_val_s = self.scaler_X.transform(X_val)
        # Нормування Y — по таргетах окремо
        y_tr_s  = self._scale_y(y_train, fit=True)
        y_val_s = self._scale_y(y_val,   fit=False)

        def fitness_fn(vector: np.ndarray) -> float:
            hp = HyperParams.from_vector(vector)
            try:
                rmse, _ = train_evaluate(hp, X_tr_s, y_tr_s,
                                         X_val_s, y_val_s,
                                         self.train_cfg, self.device)
            except Exception as exc:
                log.debug(f"fitness_fn error: {exc}")
                return float("inf")
            return rmse

        log.info("=" * 55)
        log.info("PHASE 1: Genetic Algorithm (multi-task)")
        log.info("=" * 55)
        ga = GeneticAlgorithm(HP_BOUNDS, self.ga_cfg)
        best_vector = ga.run(fitness_fn)
        self.ga_history = ga.history

        if self.train_cfg.sqp_refine:
            log.info("=" * 55)
            log.info("PHASE 2: SQP Refinement")
            log.info("=" * 55)
            best_vector = sqp_refine(best_vector, HP_BOUNDS, fitness_fn)

        self.best_hp = HyperParams.from_vector(best_vector)
        log.info(f"Optimal hyperparameters: {asdict(self.best_hp)}")

        log.info("=" * 55)
        log.info("PHASE 3: Final Multi-Task LSTM Training")
        log.info("=" * 55)
        final_cfg = TrainConfig(
            epochs=self.train_cfg.epochs * 2,
            patience=self.train_cfg.patience * 2,
            device=self.train_cfg.device,
            seed=self.train_cfg.seed,
            task_weights=self.train_cfg.task_weights,
        )
        rmse, model = train_evaluate(self.best_hp, X_tr_s, y_tr_s,
                                     X_val_s, y_val_s,
                                     final_cfg, self.device, verbose=True)
        if model is None:
            raise RuntimeError("Final training failed — check data/HP bounds")

        self.best_model = model
        log.info(f"Final val weighted RMSE (normalized): {rmse:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict — returns [N-seq_len, n_tasks] in original scale."""
        if self.best_model is None:
            raise RuntimeError("Model not trained")
        self.best_model.eval()

        X_s = self.scaler_X.transform(X)
        dummy = np.zeros((len(X_s), self.n_tasks), dtype=np.float32)
        X_seq, _ = make_sequences(X_s, dummy, self.best_hp.seq_len)

        with torch.no_grad():
            preds_norm = (self.best_model(torch.from_numpy(X_seq).to(self.device))
                          .cpu().numpy())   # [N, n_tasks]
        return self._unscale_y(preds_norm)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
        """Compute metrics per task. y: [N, n_tasks]."""
        preds = self.predict(X)
        y_aligned = y[self.best_hp.seq_len:]
        metrics = compute_metrics(y_aligned, preds, self.target_names)
        for task, m in metrics.items():
            log.info(f"  [{task}]  " +
                     "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        return metrics

    def save(self, path: str = "hga_lstm_v2_model.pt") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version":      "v2_multitask",
            "model_state":  self.best_model.state_dict(),
            "hp":           asdict(self.best_hp),
            "input_size":   self.input_size,
            "target_names": self.target_names,
            "task_weights": self.train_cfg.task_weights,
            "scaler_X_min": self.scaler_X.min_,
            "scaler_X_max": self.scaler_X.max_,
            "scalers_y_min": [s.min_ for s in self.scalers_y],
            "scalers_y_max": [s.max_ for s in self.scalers_y],
            "ga_history":   self.ga_history,
        }, path)
        log.info(f"Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "HGALSTMv2":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("version") != "v2_multitask":
            log.warning("Checkpoint is not v2_multitask — proceeding cautiously")
        hp = HyperParams(**ckpt["hp"])
        obj = cls(input_size=ckpt["input_size"],
                  target_names=ckpt["target_names"])
        obj.best_hp = hp
        obj.best_model = MultiTaskLSTM(ckpt["input_size"], hp,
                                       n_tasks=len(ckpt["target_names"]))
        obj.best_model.load_state_dict(ckpt["model_state"])
        obj.best_model.to(obj.device).eval()
        obj.scaler_X.min_ = ckpt["scaler_X_min"]
        obj.scaler_X.max_ = ckpt["scaler_X_max"]
        for i, s in enumerate(obj.scalers_y):
            s.min_ = ckpt["scalers_y_min"][i]
            s.max_ = ckpt["scalers_y_max"][i]
        obj.ga_history = ckpt.get("ga_history", [])
        log.info(f"Model loaded: {path}")
        return obj

    def save_ga_history(self, path: str = "ga_history_v2.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ga_history, f, indent=2, ensure_ascii=False)
        log.info(f"GA history saved: {path}")
