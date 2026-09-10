# ML Research Stack 🧠

Исследовательская часть платформы **SFEDUCASTLING**: обучение и оценка шахматных LLM, RAG по шахматной литературе и синтез тренерских датасетов.

Здесь решается цепочка: **измерить (бенчмарк) → дотюнить (SFT/QLoRA) → обогатить знаниями (RAG) → синтезировать данные (pipeline) → проверить гипотезы**.

---

## 📂 Структура

| Папка | Назначение | Подробнее |
|-------|-----------|-----------|
| [`chess_benchmark/`](chess_benchmark) | Модульный бенчмарк шахматных LLM — 5 независимых модулей, до **2500 баллов**. Поддерживает локальные transformers/peft-модели и облачные API (в т.ч. GigaChess), синхронизация в W&B. | [`chess_benchmark/README.md`](chess_benchmark/README.md) |
| [`gigachess/`](gigachess) | HTTP/1.1-клиент облачного инференса **GigaChess** (Cloud.ru): прогрев инстанса (`/ping`) и запросы `/chat` с передачей FEN в attachments. | [`gigachess/README.md`](gigachess/README.md) |
| [`hypotheses/`](hypotheses) | Эксперименты-проверки гипотез о качестве тренерских ответов (структурированный вывод H1.1, точность цитирования оценки H1.3). Сравнение локальной Qwen3 и облачного GigaChess, метрики в Prometheus/Grafana. | [`hypotheses/README.md`](hypotheses/README.md) |
| [`models/`](models) | Обучаемые модели: **Qwen3-chess-coach** — QLoRA SFT Qwen3-8B под роль шахматного тренера (train / inference / merge). | [`models/README.md`](models/README.md) |
| [`rag_system_v1/`](rag_system_v1) | RAG-система по шахматной литературе: парсинг (VLM + OCR) и краулинг, чанкинг, векторная база ChromaDB, динамический ретривер со стадийным фильтром, конструктор RAG-промптов. | [`rag_system_v1/README.md`](rag_system_v1/README.md) |
| [`synthesis_pipeline/`](synthesis_pipeline) | Конвейер синтеза тренерских SFT-данных: извлечение позиций мастеров → анализ Stockfish → генерация объяснений LLM → валидация → ChatML-датасет. Включает RAG-модуль **ChessCLIP** (эмбеддинги позиций) и подготовку reasoning-датасета. | [`synthesis_pipeline/README.md`](synthesis_pipeline/README.md) |

---

## 🔗 Общий поток

```
chess_benchmark            синтез данных
   │  (оценка качества)          │
   ▼                            ▼
rag_system_v1 ──► Qwen3-chess-coach (SFT) ──► hypotheses (A/B-проверка)
   (RAG-контекст)                       │
                                        ▼
                                 production: backend/llm
                                 (chess_explainer.py, gigachess.py)
```

---

## 🚀 Быстрый старт

Каждый модуль имеет собственные зависимости (`requirements.txt`) и документацию. Базовая установка:

```bash
pip install -r requirements.txt          # общие (образовательные) зависимости
pip install -r rag_system_v1/requirements.txt
pip install -r synthesis_pipeline/requirements.txt
pip install -r models/Qwen3-chess-coach/requirements.txt
```

### Основные сценарии

```bash
# 1. Оценить модель на бенчмарке
python chess_benchmark/run_benchmark.py --help

# 2. Прогреть и проверить GigaChess
python gigachess/sber_client.py

# 3. Проверить гипотезу H1.1 (структурированный вывод)
python hypotheses/scripts/eval_h1_1.py

# 4. Дообучить тренера (QLoRA)
python models/Qwen3-chess-coach/train.py

# 5. Прогнать RAG-тест на 20 верифицированных позициях
python rag_system_v1/test_rag.py
```

> Бенчмарк, гипотезы и RAG-тесты ссылаются на merged-модель `models/Qwen3-chess-coach/outputs/merged` — создаётся после `train.py` + `merge_lora.py`.