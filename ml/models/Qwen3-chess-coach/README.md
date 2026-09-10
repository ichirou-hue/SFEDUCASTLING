# Qwen3-Chess-Coach ♞

Дообучение (SFT) модели **Qwen/Qwen3-8B** под роль **шахматного тренера** — модель объясняет сделанный ход игрока, сравнивает его с лучшим ходом движка и даёт практические советы на русском языке.

Подход: **4-bit квантование (NF4) + LoRA (QLoRA)**, обучение через `trl.SFTTrainer`.

---

## 📂 Файлы

| Файл | Назначение |
|------|-----------|
| `config.py` | Все гиперпараметры: базовая модель, датасет, квантование, LoRA, SFT. |
| `train.py` | Запуск QLoRA SFT-обучения, сохранение LoRA-адаптера в `./outputs`. |
| `merge_lora.py` | Слияние LoRA-адаптера с базовой моделью → `./outputs/merged`. |
| `inference.py` | Инференс из `./outputs/merged`: тренерский разбор хода по FEN + движковой оценке. |
| `accelerate_config.yaml` | Конфигурация multi-GPU (2 процесса, bf16). |
| `requirements.txt` | Зависимости (torch, transformers, datasets, accelerate, peft, trl, bitsandbytes). |

---

## ⚙️ Конфигурация (config.py)

| Параметр | Значение |
|----------|----------|
| Базовая модель | `Qwen/Qwen3-8B` |
| Датасет | `khoilamalphaai/chess-coach-move-review` (HuggingFace) |
| Квантование | 4-bit NF4 + double quant, bf16 |
| LoRA | `r=64`, `α=128`, `dropout=0.05`, targets: все `*_proj` |
| Обучение | 3 эпохи, batch 2, grad-accum 8, LR 2e-4, cosine, warmup 3% |
| Параметры SFT | `max_seq_length=4096`, save/eval каждые 100 шагов |

---

## 🚀 Рабочий процесс

### 1. Обучение

```bash
pip install -r requirements.txt

# Одна GPU
python train.py

# Multi-GPU (2 процесса)
accelerate launch --config_file accelerate_config.yaml train.py
```

Результат: LoRA-адаптер и токенизатор в `./outputs/`.

### 2. Слияние адаптера

```bash
python merge_lora.py   # → ./outputs/merged
```

### 3. Инференс

```bash
python inference.py
```

Функция `run_inference(fen, move, eval_score, best_move, show_thinking=False)`:

- системная инструкция — строгие правила русской шахматной терминологии (не путать бьющую/взятую фигуру, корректные названия фигур);
- генерирует объяснение почему ход `move` слабее лучшего хода `best_move` и как искать подобные решения;
- `temperature=0.3`, `max_new_tokens=512`;
- тег `thinking` отрезается, если `show_thinking=False`.

---

## 📈 Кто использует модель

- [`ml/chess_benchmark`](../chess_benchmark) — оценивает модель (локально через `transformers` + `peft`, включая merged-чекпоинты).
- [`ml/hypotheses`](../hypotheses) — сравнивает `./outputs/merged` с облачным GigaChess.
- `backend/` — сервис Qwen3-8B разворачивается через vLLM (`start_llm.sh`) и используется как LLM-бэкенд.

> Для бенчмарка и гипотез сначала выполните `train.py` + `merge_lora.py`, см. [`ml/README.md`](../README.md).