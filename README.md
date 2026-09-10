# SFEDUCASTLING ♟️

**AI Chess Platform** — интеллектуальная шахматная платформа для обучения пользователей любого уровня.

Платформа объединяет веб-интерфейс (React + Vite), backend-шлюз (FastAPI), классические шахматные движки (Stockfish), человекоподобный движок **Maia3** и стек **ML-исследований** (QLoRA-тюнинг, RAG, бенчмарки LLM, синтез датасетов).

Подробное техническое задание — в [`TZ_CHESSAGINE.md`](TZ_CHESSAGINE.md).

---

## 🎯 Функциональность

- **Анализ позиции** — Stockfish (глубина 18, MultiPV 5), информация о дебюте из Lichess Masters Explorer.
- **Человекоподобная игра** — движок Maia3 с настраиваемым рейтингом (Elo), UCI-обёртка.
- **Объяснение ходов** — LLM (GigaChess / Qwen3-8B) генерирует тренерские разборы, тональность и стиль.
- **Паззлы и уровень** — тест определения уровня (Elo-бакеты), тактические задачи с гибридной проверкой (эталон + Stockfish).
- **Обучение** — структурированный курс по правилам и базовым понятиям (модули → уроки → задания).
- **Чат с ИИ** — диалоговый помощник с учётом текущей позиции и истории ходов.
- **Регистрация / профиль** — JWT-авторизация, шахматный профиль.

---

## 🛠 Стек

| Компонент | Технология |
|-----------|------------|
| Backend | Python ≥3.14, FastAPI, Uvicorn, SQLAlchemy 2 (async), Alembic |
| БД | PostgreSQL (asyncpg) |
| Frontend | React 19, Vite, react-router-dom, react-chessboard, chess.js, axios |
| Движки | Stockfish (local exe), Maia3 (git submodule, UCI) |
| LLM | GigaChess (Cloud.ru), Qwen3-8B (vLLM, порт 8000), LLaVA (Vision, отключена) |
| ML | PyTorch, transformers, PEFT/TRL (QLoRA), ChromaDB, Prometheus |
| Пакеты | uv (`pyproject.toml` + `uv.lock`) |

---

## 🏗 Архитектура

```plaintext
┌──────────────────────┐     ┌──────────────────────────────────────────────┐
│      Frontend        │     │                     Backend                   │
│  React 19 + Vite     │────▶│  FastAPI :8005 (грузит frontend/dist)        │
│  (frontend/src)      │     │  └─ api_gateway/routes — 13 роутеров:        │
│                      │     │     auth, game, analysis, analyze, knowledge,│
│                      │     │     data, chat, vision, chess_profile,       │
│                      │     │     explanation, learning, training           │
└──────────────────────┘     │  └─ services — training_checker/service/seed  │
                             │  └─ llm — gigachess.py, chess_explainer.py    │
                             │  └─ state.py — singleton-менеджеры движков    │
                             └──────────┬───────────────────────────────────┘
                                        │
        ┌───────────────────┬───────────┼───────────────┬──────────────────┐
        ▼                   ▼           ▼               ▼                  ▼
   ┌─────────────┐   ┌─────────────┐ ┌──────────┐  ┌─────────────┐  ┌──────────────┐
   │  Stockfish  │   │   Maia3     │ │ GigaChess│  │ Qwen3-8B    │  │  PostgreSQL  │
   │  (local)    │   │ (submodule) │ │ (Cloud)  │  │ (vLLM :8000)│  │  (async DB)  │
   └─────────────┘   └─────────────┘ └──────────┘  └─────────────┘  └──────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────────────────────┐
                          │            ML Research (ml/)              │
                          │  chess_benchmark · gigachess · hypotheses │
                          │  models/Qwen3-chess-coach · rag_system_v1 │
                          │  synthesis_pipeline                       │
                          └───────────────────────────────────────────┘
```

---

## 📁 Структура проекта

### Реальная структура (текущее состояние)

```plaintext
├── backend/                 # FastAPI backend
│   ├── api_gateway/
│   │   ├── routes/          #   13 роутеров: auth, game, analysis, analyze,
│   │   │                    #   knowledge, data, chat, vision, chess_profile,
│   │   │                    #   explanation, learning, training
│   │   ├── state.py         #   singleton-менеджеры: Stockfish, Maia3, GigaChess,
│   │   │                    #   базы дебютов и паззлов, LLaVA (отключена)
│   │   ├── models.py        #   Pydantic-модели запросов/ответов
│   │   ├── security.py      #   JWT-авторизация
│   │   └── sanitize.py      #   санитизация ввода
│   ├── services/            #   training_service, training_checker,
│   │                        #   training_seed, game_recorder
│   ├── models/              #   SQLAlchemy: user, game, chat_message,
│   │                        #   dataset_move, training_{module,lesson,task,attempt}
│   ├── db/                  #   async SQLAlchemy + asyncpg (session, base)
│   ├── config/              #   settings.py (pydantic-settings)
│   ├── llm/                 #   gigachess.py (HTTP-клиент), chess_explainer.py
│   ├── knowledge/           #   openings.json, puzzles.json + билдеры
│   ├── analysis/            #   classifier.py (классификация ошибок)
│   ├── app.py               #   точка входа (uvicorn :8005)
│   ├── async_queue/         #   Celery (stub)
│   └── monitoring/          #   Prometheus (stub)
├── frontend/                # React 19 + Vite
│   ├── src/
│   │   ├── components/      #   Chessboard, MoveHistory, ChatPanel, EvalBar,
│   │   │                    #   PuzzlesPage, TrainingPage, Sidebar, TopBar,
│   │   │                    #   MiniBoard, FenBar, RegisterModal
│   │   ├── app.jsx          #   корневой компонент (роутинг)
│   │   ├── main.jsx         #   точка входа
│   │   └── api.js           #   axios + JWT (localStorage)
│   ├── public/              #   статика, фигуры, шрифты
│   └── dist/                #   собранный бандл (раздаётся backend'ом)
├── ml/                      # ML-исследования и тренировка (см. ml/README.md)
│   ├── chess_benchmark/     #   бенчмарк LLM до 2500 баллов
│   ├── gigachess/           #   HTTP-клиент GigaChess (Cloud.ru)
│   ├── hypotheses/          #   проверка гипотез H1.1 / H1.3
│   ├── models/              #   Qwen3-chess-coach (QLoRA SFT)
│   ├── rag_system_v1/       #   RAG по шахматной литературе (ChromaDB)
│   └── synthesis_pipeline/  #   синтез тренерских SFT-датасетов
├── maia3/                   # git submodule — человекоподобный движок (UCI)
├── alembic/                 # миграции (5 версий)
├── scripts/                 # утилиты (create_admin.py и др.)
├── tests/                   # pytest: routes/, training/, analysis/, models/
├── training_data/           # артефакты датасетов (parquet/jsonl)
├── analysis/                # классификатор ошибок, NAG-аннотаторы (stub)
├── input_gateway/           # мультимодальный ввод (stub)
├── perception/              # CV: BoardToFEN-проект, эмбеддер позиций
├── reasoning/               # генерация объяснений (stub)
├── memory/                  # RAG поверх БД (stub)
├── evaluation/              # метрики и арена агентов (stub)
├── docs/                    # user_guide.md (пусто)
├── TZ_CHESSAGINE.md         # техническое задание
├── start_llm.sh             # запуск Qwen3-8B через vLLM (порт 8000)
├── attach_llm.sh            # подключение LLM-сервиса
├── run_server.sh            # установка и запуск backend
├── requirements.txt         # зависимости Python
├── pyproject.toml           # ruff, mypy, pytest (coverage)
└── uv.lock                  # lock-файл uv
```

> Часть модулей (`analysis/`, `input_gateway/`, `reasoning/`, `memory/`, `evaluation/`, `backend/async_queue/`, `backend/monitoring/`) пока содержит пустые заглушки — рабочий контур: **backend + frontend + ml/**.

### ML Research Stack (`ml/`)

| Папка | Назначение |
|-------|------------|
| [`chess_benchmark`](ml/chess_benchmark) | Модульный бенчмарк шахматных LLM на 2500 баллов (5 модулей по 500). Полная документация — `ml/chess_benchmark/README.md`. |
| [`gigachess`](ml/gigachess) | HTTP/1.1-клиент облачного инференса GigaChess (ping + `/chat` с FEN). |
| [`hypotheses`](ml/hypotheses) | A/B-проверка гипотез (структурированный вывод, точность цитирования eval) против Qwen3 и GigaChess, метрики в Prometheus. |
| [`models/Qwen3-chess-coach`](ml/models/Qwen3-chess-coach) | QLoRA SFT модели Qwen3-8B под роль шахматного тренера + инференс. |
| [`rag_system_v1`](ml/rag_system_v1) | RAG по шахматной литературе: парсинг PDF/текстов, чанкинг, ChromaDB, динамический ретривер. |
| [`synthesis_pipeline`](ml/synthesis_pipeline) | Конвейер генерации синтетических тренерских объяснений (Stockfish → LLM → валидация → SFT-датасет), ChessCLIP RAG. |

---

## 🚀 Запуск

### Backend (Windows / 127.0.0.1:8005)

```bash
# 1. Python-окружение и зависимости
pip install -r requirements.txt

# 2. Конфигурация (в .env)
#    DATABASE_URL, JWT_SECRET, MAIA3_*, GIGACHESS_BASE_URL и т.д. — см. backend/config/settings.py

# 3. Миграции БД
alembic upgrade head

# 4. Запуск API (одновременно раздаёт собранный frontend/dist)
python backend/app.py
```

### Frontend (dev-режим)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, /api проксируется на :8005
npm run build      # сборка в dist/
```

### LLM-сервис (Qwen3-8B через vLLM, порт 8000)

```bash
bash start_llm.sh
```

### Linux

```bash
bash run_server.sh
```

---

## 🧪 Тесты

```bash
pytest tests/            # API-роуты, model, training, classifier
```

Покрытие настраивается в `pyproject.toml` (ruff + mypy + pytest).

---

## 👥 Команда

* **Егор** — Project Lead (архитектура, LLM/GigaChess)
* **Митя** — Backend Developer (Python, API)
* **Илья Бабченков** — Frontend Developer
* **Илья Слынько** — ML Engineer (Maia, датасеты)
* **Даниил** — ML Engineer (LLM Tuning, промпты)
* **Тима** — UI/UX Designer