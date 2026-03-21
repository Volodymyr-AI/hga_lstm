"""
HGA-LSTM: Hybrid Genetic Algorithm + LSTM для прогнозування щільності пульпи
та ефективності грохочення залізорудної пульпи.

Реалізація на основі методології Zou et al., поєднує:
  - Генетичний алгоритм (GA) для глобального пошуку гіперпараметрів
  - Sequential Quadratic Programming (SQP) для локального уточнення
  - LSTM мережу для часових рядів

Референс: Zou et al. - HGA-LSTM model for pulp density prediction
          RMSE: 3.83 → 3.08 (-19.5%), ARGE: 0.119 → 0.0752 (-36.8%)

Автор: [Ваше ім'я]
Дата: 2025
"""

from __future__ import annotations

import json
import logging
import random
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─── Логування ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("HGA-LSTM")


# ─── Конфігурація ─────────────────────────────────────────────────────────────

@dataclass
class HyperParams:
    """Гіперпараметри LSTM мережі, що оптимізуються GA+SQP."""
    hidden_size: int = 64           # Розмір прихованого шару [32, 256]
    num_layers: int = 2             # Кількість LSTM шарів [1, 4]
    dropout: float = 0.2            # Dropout між шарами [0.0, 0.5]
    learning_rate: float = 1e-3     # Швидкість навчання [1e-5, 1e-1]
    batch_size: int = 32            # Розмір батчу [8, 128]
    seq_len: int = 10               # Довжина вхідної послідовності [5, 50]

    def to_vector(self) -> np.ndarray:
        """Кодує гіперпараметри у вектор для GA/SQP."""
        return np.array([
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.learning_rate,
            self.batch_size,
            self.seq_len,
        ], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "HyperParams":
        """Декодує вектор у гіперпараметри з округленням цілих значень."""
        return cls(
            hidden_size=max(8, int(round(v[0]))),
            num_layers=max(1, min(4, int(round(v[1])))),
            dropout=float(np.clip(v[2], 0.0, 0.5)),
            learning_rate=float(np.clip(v[3], 1e-5, 0.1)),
            batch_size=max(4, int(round(v[4]))),
            seq_len=max(3, min(100, int(round(v[5])))),
        )

    # ── Межі для GA ──
    BOUNDS: list[tuple[float, float]] = field(default_factory=lambda: [
        (16.0, 256.0),   # hidden_size
        (1.0,  4.0),     # num_layers
        (0.0,  0.5),     # dropout
        (1e-5, 0.1),     # learning_rate
        (8.0,  128.0),   # batch_size
        (5.0,  50.0),    # seq_len
    ])


@dataclass
class GAConfig:
    """Конфігурація генетичного алгоритму."""
    population_size: int = 20       # Розмір популяції
    n_generations: int = 30         # Кількість поколінь
    crossover_prob: float = 0.8     # Ймовірність схрещування
    mutation_prob: float = 0.15     # Ймовірність мутації
    mutation_scale: float = 0.1     # Масштаб мутації (відносний)
    elitism_count: int = 2          # Кількість елітних особин
    tournament_k: int = 3           # Розмір турніру


@dataclass
class TrainConfig:
    """Конфігурація навчання LSTM."""
    epochs: int = 100
    patience: int = 15              # Early stopping patience
    min_delta: float = 1e-6
    device: str = "auto"            # "auto", "cuda", "cpu"
    seed: int = 42
    save_dir: str = "checkpoints"
    sqp_refine: bool = True         # Уточнення SQP після GA


# ─── LSTM модель ──────────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """
    Багатошарова LSTM мережа для прогнозування часових рядів.

    Архітектура:
        Input → LSTM × num_layers → Dropout → FC → Output

    Рівняння (згідно з дисертацією):
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   — вентиль забування
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   — вхідний вентиль
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) — кандидат стану
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       — стан комірки
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   — вихідний вентиль
        h_t = o_t * tanh(C_t)                  — прихований стан

    Args:
        input_size:  Кількість вхідних ознак
        output_size: Кількість виходів (1 для щільності пульпи)
        hp:          Гіперпараметри (HyperParams)
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
        """
        Args:
            x: Tensor [batch, seq_len, input_size]
        Returns:
            out: Tensor [batch, output_size]
        """
        lstm_out, _ = self.lstm(x)          # [B, T, H]
        last_step = lstm_out[:, -1, :]      # [B, H]
        out = self.fc(self.dropout(last_step))
        return out


# ─── Утиліти даних ────────────────────────────────────────────────────────────

class MinMaxScaler:
    """Нормалізація Min-Max з підтримкою inverse_transform."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self._scale(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "Scaler not fitted"
        return self._scale(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "Scaler not fitted"
        return X * (self.max_ - self.min_ + 1e-9) + self.min_

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_ + 1e-9)


def make_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Створює ковзні вікна для часових рядів.

    Args:
        data:    [N, features] — нормалізовані вхідні дані
        targets: [N]           — цільова змінна
        seq_len: довжина вікна

    Returns:
        X: [N-seq_len, seq_len, features]
        y: [N-seq_len]
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── Навчання та оцінка ───────────────────────────────────────────────────────

def train_evaluate(
    hp: HyperParams,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = False,
) -> tuple[float, LSTMRegressor]:
    """
    Навчає LSTM з даними гіперпараметрами та повертає (val_rmse, модель).

    Оптимізатор Adam:
        m_t = β₁·m_{t-1} + (1-β₁)·∇L
        v_t = β₂·v_{t-1} + (1-β₂)·∇L²
        θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
        де β₁=0.9, β₂=0.999, ε=1e-8

    Returns:
        val_rmse: RMSE на валідаційній вибірці
        model:    Навчена модель
    """
    torch.manual_seed(cfg.seed)
    input_size = X_train.shape[-1]

    # Побудова послідовностей
    Xs_tr, ys_tr = make_sequences(X_train, y_train, hp.seq_len)
    Xs_val, ys_val = make_sequences(X_val, y_val, hp.seq_len)

    if len(Xs_tr) < hp.batch_size:
        return float("inf"), None

    train_ds = TensorDataset(
        torch.from_numpy(Xs_tr).to(device),
        torch.from_numpy(ys_tr).unsqueeze(1).to(device),
    )
    loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)

    model = LSTMRegressor(input_size, 1, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_cnt = 0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Валідація
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xs_val).to(device)
            pv = model(xv).cpu().numpy().flatten()
            val_rmse = float(np.sqrt(np.mean((pv - ys_val) ** 2)))

        scheduler.step(val_rmse)

        if val_rmse < best_val_loss - cfg.min_delta:
            best_val_loss = val_rmse
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if verbose:
                    log.debug(f"  Early stop @ epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss, model


# ─── Генетичний алгоритм ──────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Генетичний алгоритм для оптимізації гіперпараметрів LSTM.

    Оператори:
      - Турнірна селекція
      - SBX схрещування (Simulated Binary Crossover)
      - Гаусівська мутація
      - Елітизм

    Args:
        bounds: Список кортежів (min, max) для кожного параметра
        ga_cfg: Конфігурація GA
    """

    def __init__(self, bounds: list[tuple[float, float]], ga_cfg: GAConfig):
        self.bounds = np.array(bounds)
        self.cfg = ga_cfg
        self.dim = len(bounds)
        self._best_individual: np.ndarray | None = None
        self._best_fitness: float = float("inf")
        self.history: list[dict] = []

    def _init_population(self) -> np.ndarray:
        """Ініціалізує популяцію рівномірно в межах bounds."""
        pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.cfg.population_size, self.dim),
        )
        return pop

    def _tournament_select(self, fitness: np.ndarray) -> np.ndarray:
        """Турнірна селекція: обирає кращого з k випадкових."""
        idx = np.random.choice(len(fitness), self.cfg.tournament_k, replace=False)
        best = idx[np.argmin(fitness[idx])]
        return best

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SBX схрещування (η=2)."""
        eta = 2.0
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return c1, c2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Гаусівська мутація з адаптивним масштабом."""
        mutant = individual.copy()
        for i in range(self.dim):
            if random.random() < self.cfg.mutation_prob:
                scale = self.cfg.mutation_scale * (self.bounds[i, 1] - self.bounds[i, 0])
                mutant[i] += np.random.normal(0, scale)
                mutant[i] = np.clip(mutant[i], self.bounds[i, 0], self.bounds[i, 1])
        return mutant

    def _clip(self, individual: np.ndarray) -> np.ndarray:
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    def run(self, fitness_fn) -> np.ndarray:
        """
        Запускає GA оптимізацію.

        Args:
            fitness_fn: callable(vector) → float (менше = краще)

        Returns:
            best_vector: Найкращий знайдений вектор гіперпараметрів
        """
        population = self._init_population()
        fitness = np.array([fitness_fn(ind) for ind in population])

        log.info(f"GA початок: найкраще RMSE = {fitness.min():.4f}")

        for gen in range(self.cfg.n_generations):
            t0 = time.time()
            new_pop = []

            # Елітизм
            elite_idx = np.argsort(fitness)[: self.cfg.elitism_count]
            for idx in elite_idx:
                new_pop.append(population[idx].copy())

            # Відтворення
            while len(new_pop) < self.cfg.population_size:
                p1_idx = self._tournament_select(fitness)
                p2_idx = self._tournament_select(fitness)
                p1, p2 = population[p1_idx], population[p2_idx]

                if random.random() < self.cfg.crossover_prob:
                    c1, c2 = self._sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                new_pop.append(self._clip(self._mutate(c1)))
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(self._clip(self._mutate(c2)))

            population = np.array(new_pop)
            fitness = np.array([fitness_fn(ind) for ind in population])

            gen_best = fitness.min()
            gen_avg = fitness.mean()
            elapsed = time.time() - t0

            if gen_best < self._best_fitness:
                self._best_fitness = gen_best
                self._best_individual = population[fitness.argmin()].copy()

            self.history.append({
                "generation": gen + 1,
                "best_rmse": float(gen_best),
                "avg_rmse": float(gen_avg),
                "elapsed_s": round(elapsed, 2),
            })

            log.info(
                f"Gen {gen+1:3d}/{self.cfg.n_generations} | "
                f"Best RMSE: {gen_best:.4f} | Avg: {gen_avg:.4f} | "
                f"{elapsed:.1f}s"
            )

        return self._best_individual


# ─── SQP Уточнення ────────────────────────────────────────────────────────────

def sqp_refine(
    initial_vector: np.ndarray,
    bounds: list[tuple[float, float]],
    fitness_fn,
    maxiter: int = 30,
) -> np.ndarray:
    """
    Уточнення рішення GA методом SQP (Sequential Quadratic Programming).

    Використовує scipy L-BFGS-B (квазі-Ньютонівський метод, аналогічний SQP
    для обмежених задач без нелінійних обмежень).

    Args:
        initial_vector: Стартова точка від GA
        bounds:         Межі параметрів
        fitness_fn:     Цільова функція
        maxiter:        Максимум ітерацій локальної оптимізації

    Returns:
        Уточнений вектор гіперпараметрів
    """
    log.info("SQP уточнення (L-BFGS-B)...")

    # Плавна обгортка для неперервної оптимізації
    def smooth_fitness(v):
        hp = HyperParams.from_vector(v)
        return fitness_fn(hp.to_vector())

    result = minimize(
        smooth_fitness,
        x0=initial_vector,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-9},
    )
    log.info(f"SQP завершено: RMSE = {result.fun:.4f} | success={result.success}")
    return result.x


# ─── Метрики ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Обчислює метрики якості моделі.

    RMSE = √(1/n · Σ(y_pred - y_true)²)
    MAE  = 1/n · Σ|y_pred - y_true|
    ARGE = 1/n · Σ|y_pred - y_true| / max(|y_true|, ε)
    R²   = 1 - SS_res / SS_tot
    """
    eps = 1e-9
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    arge = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + eps))
    return {"RMSE": rmse, "MAE": mae, "ARGE": arge, "R2": r2}


# ─── Головний клас HGA-LSTM ───────────────────────────────────────────────────

class HGALSTM:
    """
    Головний клас HGA-LSTM моделі.

    Поєднує:
      1. Генетичний алгоритм для пошуку гіперпараметрів
      2. SQP для локального уточнення
      3. LSTM для прогнозування

    Args:
        input_size:  Кількість вхідних ознак
        ga_cfg:      Конфігурація GA
        train_cfg:   Конфігурація навчання

    Example::

        model = HGALSTM(input_size=5)
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        input_size: int,
        ga_cfg: GAConfig | None = None,
        train_cfg: TrainConfig | None = None,
    ):
        self.input_size = input_size
        self.ga_cfg = ga_cfg or GAConfig()
        self.train_cfg = train_cfg or TrainConfig()
        self.device = self._resolve_device()

        self.best_hp: HyperParams | None = None
        self.best_model: LSTMRegressor | None = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.ga_history: list[dict] = []

        log.info(f"HGA-LSTM ініціалізовано | device={self.device} | input_size={input_size}")

    def _resolve_device(self) -> torch.device:
        if self.train_cfg.device == "auto":
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                log.info(f"GPU знайдено: {name}")
                return torch.device("cuda")
            log.warning("GPU не знайдено, використовується CPU")
            return torch.device("cpu")
        return torch.device(self.train_cfg.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HGALSTM":
        """
        Повний цикл навчання: GA → SQP → Фінальне навчання LSTM.

        Args:
            X_train: [N_train, features]
            y_train: [N_train]
            X_val:   [N_val, features]
            y_val:   [N_val]

        Returns:
            self
        """
        torch.manual_seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)

        # Нормалізація
        X_tr_s = self.scaler_X.fit_transform(X_train)
        y_tr_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_val_s = self.scaler_X.transform(X_val)
        y_val_s = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        # Функція фітнесу для GA
        def fitness_fn(vector: np.ndarray) -> float:
            hp = HyperParams.from_vector(vector)
            try:
                rmse, _ = train_evaluate(
                    hp, X_tr_s, y_tr_s, X_val_s, y_val_s,
                    self.train_cfg, self.device, verbose=False,
                )
            except Exception as e:
                log.debug(f"fitness_fn error: {e}")
                return float("inf")
            return rmse

        # ── Фаза 1: GA ──
        log.info("=" * 60)
        log.info("ФАЗА 1: Генетичний алгоритм")
        log.info("=" * 60)
        hp_bounds = HyperParams().BOUNDS
        ga = GeneticAlgorithm(hp_bounds, self.ga_cfg)
        best_vector = ga.run(fitness_fn)
        self.ga_history = ga.history

        # ── Фаза 2: SQP ──
        if self.train_cfg.sqp_refine:
            log.info("=" * 60)
            log.info("ФАЗА 2: SQP уточнення")
            log.info("=" * 60)
            best_vector = sqp_refine(best_vector, hp_bounds, fitness_fn)

        self.best_hp = HyperParams.from_vector(best_vector)
        log.info(f"Оптимальні гіперпараметри: {asdict(self.best_hp)}")

        # ── Фаза 3: Фінальне навчання ──
        log.info("=" * 60)
        log.info("ФАЗА 3: Фінальне навчання LSTM")
        log.info("=" * 60)
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
        self.best_model = model
        log.info(f"Фінальний val RMSE (нормалізований): {rmse:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогнозує значення для нових даних.

        Args:
            X: [N, features]

        Returns:
            predictions: [N - seq_len] — у вихідних одиницях
        """
        assert self.best_model is not None, "Спочатку виклич .fit()"
        self.best_model.eval()

        X_s = self.scaler_X.transform(X)
        dummy_y = np.zeros(len(X_s))
        Xs, _ = make_sequences(X_s, dummy_y, self.best_hp.seq_len)

        with torch.no_grad():
            xv = torch.from_numpy(Xs).to(self.device)
            preds_s = self.best_model(xv).cpu().numpy().flatten()

        return self.scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """
        Обчислює метрики на тестовій вибірці.

        Args:
            X: [N, features]
            y: [N] — справжні значення

        Returns:
            dict з RMSE, MAE, ARGE, R²
        """
        preds = self.predict(X)
        y_aligned = y[self.best_hp.seq_len :]
        metrics = compute_metrics(y_aligned, preds)
        log.info("Метрики на тестовій вибірці:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")
        return metrics

    def save(self, path: str = "hga_lstm_model.pt") -> None:
        """Зберігає модель, гіперпараметри та скейлери."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.best_model.state_dict(),
            "hp": asdict(self.best_hp),
            "input_size": self.input_size,
            "scaler_X_min": self.scaler_X.min_,
            "scaler_X_max": self.scaler_X.max_,
            "scaler_y_min": self.scaler_y.min_,
            "scaler_y_max": self.scaler_y.max_,
            "ga_history": self.ga_history,
        }, path)
        log.info(f"Модель збережена: {path}")

    @classmethod
    def load(cls, path: str) -> "HGALSTM":
        """Завантажує збережену модель."""
        ckpt = torch.load(path, map_location="cpu")
        hp = HyperParams(**ckpt["hp"])
        input_size = ckpt["input_size"]

        obj = cls(input_size=input_size)
        obj.best_hp = hp
        obj.best_model = LSTMRegressor(input_size, 1, hp)
        obj.best_model.load_state_dict(ckpt["model_state"])
        obj.best_model.to(obj.device)

        obj.scaler_X.min_ = ckpt["scaler_X_min"]
        obj.scaler_X.max_ = ckpt["scaler_X_max"]
        obj.scaler_y.min_ = ckpt["scaler_y_min"]
        obj.scaler_y.max_ = ckpt["scaler_y_max"]
        obj.ga_history = ckpt.get("ga_history", [])
        log.info(f"Модель завантажена з: {path}")
        return obj

    def save_ga_history(self, path: str = "ga_history.json") -> None:
        """Зберігає історію GA для аналізу."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ga_history, f, indent=2, ensure_ascii=False)
        log.info(f"Історія GA збережена: {path}")
