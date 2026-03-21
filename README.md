# HGA-LSTM — Гібридний Генетичний Алгоритм + LSTM
### Прогнозування щільності пульпи та ефективності грохочення

Реалізація на основі методології **Zou et al.**:
- RMSE: 3.83 → 3.08 (-19.5%)
- ARGE: 0.119 → 0.0752 (-36.8%)

---

## Структура проекту
```
hga_lstm/
├── hga_lstm.py      # Головна модель (GA + SQP + LSTM)
├── data_utils.py    # Утиліти даних
├── visualize.py     # Графіки
├── train.py         # Скрипт запуску
└── requirements.txt
```

---

## Встановлення

### 1. PyTorch для RTX 5070 (CUDA 12.4)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Інші залежності
```bash
pip install numpy pandas scipy matplotlib
```

---

## Запуск

### Тест на синтетичних даних (швидко ~2 хв)
```bash
python train.py --fast
```

### Повний запуск (GA: 20 особин × 30 поколінь)
```bash
python train.py
```

### Власні дані (CSV)
```bash
python train.py \
  --csv data/pulp_data.csv \
  --features amplitude_mm frequency_hz angle_deg pulp_flow solid_pct \
  --target density
```

### Параметри
| Параметр    | За замовч. | Опис                         |
|-------------|-----------|------------------------------|
| --ga-pop    | 20        | Розмір популяції GA          |
| --ga-gen    | 30        | Кількість поколінь GA        |
| --epochs    | 100       | Епох навчання LSTM           |
| --no-sqp    | False     | Вимкнути SQP уточнення       |
| --device    | auto      | auto / cuda / cpu            |
| --save-dir  | outputs   | Папка для збереження         |

---

## Архітектура HGA-LSTM

```
Вхідні параметри (амплітуда, частота, кут, витрата, вміст твердого)
        ↓
[Генетичний Алгоритм] → пошук гіперпараметрів (20 особин × 30 поколінь)
        ↓
[SQP (L-BFGS-B)]     → локальне уточнення
        ↓
[LSTM мережа]        → навчання з оптимальними гіперпараметрами
        ↓
Прогноз щільності пульпи / ефективності грохочення
```

### Гіперпараметри що оптимізуються GA+SQP:
- `hidden_size`   — розмір прихованого шару [16–256]
- `num_layers`    — кількість LSTM шарів [1–4]
- `dropout`       — дроп-аут [0.0–0.5]
- `learning_rate` — швидкість навчання [1e-5–0.1]
- `batch_size`    — розмір батчу [8–128]
- `seq_len`       — довжина вхідної послідовності [5–50]

---

## Виходи
```
outputs/
├── hga_lstm_model.pt      # Збережена модель
├── ga_history.json        # Лог конвергенції GA
└── plots/
    ├── ga_convergence.png  # Конвергенція GA
    ├── predictions.png     # Прогноз vs реальні
    └── errors.png          # Розподіл похибок
```
