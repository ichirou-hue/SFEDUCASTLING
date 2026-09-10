# Synthesis Pipeline 🛠️

Конвейер автоматической генерации **тренерских SFT-датасетов** для шахматных LLM.

Из гроссмейстерских партий извлекаются позиции, анализируются движком **Stockfish**, а LLM генерирует методические объяснения, которые после детерминированной валидации превращаются в ChatML-датасет для дообучения.

Полная целевая структура (каталоги `data/` по шагам) описана в [`PROJECT_STRUCTURE.txt`](PROJECT_STRUCTURE.txt).

---

## 📂 Структура

| Путь | Назначение |
|------|-----------|
| `src/schemas.py` | Pydantic-схемы: `RawPosition`, `MultiPVLine`, `AnalyzedPosition`, `CoachExplanation`, `SimilarExample`. |
| `src/prompts.py` | `SYSTEM_PROMPT` тренера + `build_user_prompt()` (FEN, данные движка, RAG-контекст). |
| `src/llm_api.py` | `ChessLLMClient` — OpenAI-совместимый клиент (OpenRouter/Groq/OpenAI). |
| `src/chess_engine.py` | `StockfishAnalyzer` — анализ позиции (глубина 22, MultiPV=3, centipawn loss). |
| `src/board_features.py` | Детерминированные факты о позиции: висячие фигуры, нападения, атаки. |
| `scripts/` | 5 шагов конвейера + обогащение признаков (`extra_features/`). |
| `rag_module_chessclip/` | Векторный поиск шахматных аналогий на эмбеддингах ChessCLIP. |
| `reasoning_draft_model/` | Подготовка «reasoning draft»-датасета из человеческих комментариев + SFT. |
| `tests/` | Тесты генерации, векторного хранилища и энкодера. |

---

## 🔀 Конвейер данных

```
PGN гроссмейстеров (Elo > 2400)
        │
        ▼  1. 01_extract_master_positions.py   (ходы 8–35, только позиционные моменты)
data/01_raw/master_positions.jsonl
        │
        ▼  2. 02_run_stockfish.py              (глубина 22, MultiPV=3, centipawn loss)
data/02_engine_analyzed/master_analyzed.jsonl
        │
        ▼  3. 03_generate.py                   (LLM + RAG-контекст по схожим позициям)
data/03_generated/master_explanations.jsonl
        │
        ▼  4. 04_validate_examples.py          (легальность ходов FIDE, нотация → rejected.jsonl)
data/04_validated/master_validated.jsonl
        │
        ▼  5. 05_prepare_sft_dataset.py        (ChatML-диалоги, сплит 80/10/10)
data/08_final_split/{train,val,test}.jsonl
```

Зарезервированные слоты расширения: `05_llm_judged` (LLM-as-a-Judge), `06_deduplicated`, `07_balanced` — см. `PROJECT_STRUCTURE.txt`.

---

## 🧩 RAG-модуль ChessCLIP (`rag_module_chessclip/`)

Векторный индекс шахматных позиций для подстановки похожих примеров в промпт:

| Файл | Назначение |
|------|-----------|
| `board_converter.py` | FEN → память из **112 бинарных плоскостей** LCZero `(1, 112, 8, 8)`. |
| `clip_encoder.py` | `ChessPositionEncoder` — FiLM-ResNet → **512-d** вектор (L2-нормализация). Конфиг `chessclip-quickgelu.json`. |
| `vector_store.py` | `ChessVectorStore` — обёртка над ChromaDB (upsert/query, cosine). |
| `chessclip/` | Копия форка OpenCLIP (Waterhorse) с реализацией ChessCLIP (`src/open_clip`, `src/training`). |

> Веса энкодера ожидаются в `~/SFEDUCASTLING/ml/models/chessclip/chessclip.pt`.

---

## 🧩 Reasoning Draft Model (`reasoning_draft_model/`)

Превращение реальных комментариев людей к партиям в «черновые рассуждения»:

| Скрипт | Назначение |
|--------|-----------|
| `clean_and_filter_pairs.py` | Regex-фильтрация пар ход/комментарий (GameKnot). |
| `llm_audit_and_clean.py` | Пакетная LLM-курация (метки CLEANED / REJECT). |
| `fast_llm_post_audit.py` | Ускоренный повторный проход (resume-safe). |
| `sft_train_reasoning_draft_model.py` | QLoRA SFT Qwen2.5-7B на итоговом датасете. |

---

## 🚀 Запуск

```bash
pip install -r requirements.txt

# Шаг 1: извлечь позиции мастеров из PGN
python scripts/01_extract_master_positions.py --help

# Шаг 2: проанализировать Stockfish (пакетно, с resume)
python scripts/02_run_stockfish.py

# Шаг 3: сгенерировать объяснения LLM (нужны ключи OpenAI-совместимого API)
python scripts/03_generate.py

# Шаг 4: валидация
python scripts/04_validate_examples.py

# Шаг 5: подготовить SFT-датасет
python scripts/05_prepare_sft_dataset.py

# Тесты
python -m pytest tests/
```

Ключи API задаются переменными окружения проекта `src/llm_api.py` (OpenRouter / Groq / OpenAI).

---

## 📄 См. также

- [`PROJECT_STRUCTURE.txt`](PROJECT_STRUCTURE.txt) — полная целевая схема каталогов и номеров шагов.
- [`scripts/extra_features/README.md`](scripts/extra_features/README.md) — обогащение позиций доп. признаками (NAG, флаги, оценка путей).
- [`rag_module_chessclip/chessclip/README.md`](rag_module_chessclip/chessclip/README.md) — upstream-документация ChessCLIP/OpenCLIP.