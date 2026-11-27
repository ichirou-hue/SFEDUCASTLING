# SFEDUCASTLING ♟️

**AI Chess Platform** - интеллектуальная шахматная платформа для обучения пользователей любого уровня.

Платформа объединяет веб-интерфейс (React + Vite), backend-шлюз (FastAPI), классические шахматные движки (Stockfish), человекоподобный движок **Maia3** и стек **ML-исследований** (QLoRA-тюнинг, RAG, бенчмарки LLM, синтез датасетов).

## 🎯 Функциональность

- **Анализ позиции** - Stockfish (глубина 20, MultiPV 5, настраивается), информация о дебюте из Lichess Masters Explorer.
- **Человекоподобная игра** - движок Maia3 с настраиваемым рейтингом (Elo), UCI-обёртка, 7 уровней сложности.
- **Объяснение ходов** - LLM (GigaChess / Qwen3-8B) генерирует тренерские разборы; пакет `backend/llm/explainer/` (факты позиции и хода, граундинг, валидация, автоматический ремонт ответа).
- **Паззлы и уровень** - персональный level-test из 20 задач, тактические задачи с гибридной проверкой (эталон + Stockfish), интервальные повторения ошибок.
- **Обучение** - структурированный курс (модули → уроки → задания), адаптивная и динамическая сложность.
- **Онбординг** - анкета при первом входе, по ней подбираются уровень и персональные задачи.
- **Чат с ИИ** - диалоговый помощник с учётом текущей позиции и истории ходов.
- **Регистрация / профиль** - JWT-авторизация, шахматный профиль, аккаунты Lichess / Chess.com, виджет прогресса.
- **Админ-панель** - список пользователей, профили и статистика (`/admin`).

## 🛠 Стек

| Компонент | Технология |
|-----------|------------|
| Backend | Python ≥3.14, FastAPI, Uvicorn, SQLAlchemy 2 (async), Alembic |
| БД | SQLite (aiosqlite, по умолчанию) или PostgreSQL (asyncpg) |
| Frontend | React 19, Vite, react-router-dom, react-chessboard, chess.js, axios |
| Движки | Stockfish (local exe), Maia3 (git submodule, UCI) |
| LLM | GigaChess (Cloud.ru), Qwen3-8B (vLLM, порт 8000), LLaVA (Vision, отключена) |
| ML | PyTorch, transformers, PEFT/TRL (QLoRA), ChromaDB, Prometheus |
| Пакеты | uv (`pyproject.toml` + `uv.lock`) |

## 🏗 Архитектура

```plaintext
┌──────────────────────┐     ┌──────────────────────────────────────────────┐
│      Frontend        │     │                     Backend                   │
│  React 19 + Vite     │────▶│  FastAPI :8005 (грузит frontend/dist)        │
│  (frontend/src)      │     │  └─ api_gateway/routes: 13 роутеров           │
│                      │     │     admin, analysis, analyze, auth, chat,    │
│                      │     │     chess_profile, data, explanation, game,  │
│                      │     │     knowledge, learning, training, vision    │
└──────────────────────┘ │  └─ services: adaptive/assessment/training_*    │
                             │  └─ llm: gigachess.py, explainer/             │
                             │  └─ state.py: singleton-менеджеры движков     │
                             └──────────┬───────────────────────────────────┘
                                        │
        ┌───────────────────┬───────────┼───────────────┬──────────────────┐
        ▼                   ▼           ▼               ▼                  ▼
   ┌─────────────┐   ┌─────────────┐ ┌──────────┐  ┌─────────────┐  ┌──────────────┐
   │  Stockfish  │   │   Maia3     │ │ GigaChess│  │ Qwen3-8B    │  │ SQLite /     │
   │  (local)    │   │ (submodule) │ │ (Cloud)  │  │ (vLLM :8000)│  │ PostgreSQL   │
   └─────────────┘   └─────────────┘ └──────────┘  └─────────────┘  └──────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────────────────────┐
                          │            ML Research (ml/)              │
                          │  benchmark · training · hypotheses │
                          │  models/Qwen3-chess-coach · rag_system │
                          │  pipelines · datasets                   │
                          └───────────────────────────────────────────┘
```

## 📁 Структура проекта

```plaintext
├── backend/                 # FastAPI backend
│   ├── api_gateway/
│   │   ├── routes/          #   13 роутеров: admin, analysis, analyze, auth,
│   │   │                    #   chat, chess_profile, data, explanation, game,
│   │   │                    #   knowledge, learning, training, vision
│   │   ├── state.py         #   singleton-менеджеры: Stockfish, Maia3, GigaChess,
│   │   │                    #   базы дебютов и паззлов, LLaVA (отключена)
│   │   ├── models.py        #   Pydantic-модели запросов/ответов
│   │   ├── dependecies.py   #   зависимости: текущий пользователь, RBAC-роли
│   │   ├── security.py      #   JWT-токены, bcrypt (cost 12)
│   │   └── sanitize.py      #   санитизация ввода
│   ├── services/            #   adaptive_logic/adaptive_training, assessment,
│   │                        #   dynamic_difficulty, course_topics, review_logic,
│   │                        #   training_service/checker/seed/reviews/course_data,
│   │                        #   game_recorder
│   ├── models/              #   SQLAlchemy: user (+ chess_account, theme_difficulty),
│   │                        #   game, chat_message, dataset_move, level_test,
│   │                        #   puzzle_attempt, training_{module,lesson,task,attempt,review}
│   ├── db/                  #   async SQLAlchemy (session, base)
│   ├── config/              #   settings.py (pydantic-settings)
│   ├── llm/                 #   gigachess.py (HTTP-клиент), chess_explainer.py (фасад),
│   │                        #   explainer/ (пакет: факты, граундинг, валидация, ремонт)
│   ├── knowledge/           #   openings_ru.py, puzzles.json + билдеры
│   ├── analysis/            #   classifier.py (классификация ошибок)
│   ├── app.py               #   точка входа (uvicorn :8005)
│   ├── async_queue/         #   Celery (stub)
│   └── monitoring/          #   Prometheus (stub)
├── frontend/                # React 19 + Vite
│   ├── src/
│   │   ├── components/      #   Chessboard, MoveHistory, ChatPanel, EvalBar, FenBar,
│   │   │                    #   PuzzlesPage, TrainingPage, LevelTestPage, AdminPage,
│   │   │                    #   UserProfilePage, OnboardingModal, MiniProfilePopover,
│   │   │                    #   ProgressWidget, PuzzleRegisterGate, Sidebar, TopBar,
│   │   │                    #   MiniBoard, RegisterModal
│   │   ├── app.jsx          #   корневой компонент (роутинг)
│   │   ├── main.jsx         #   точка входа
│   │   └── api.js           #   axios + JWT (localStorage)
│   ├── public/              #   статика, фигуры, шрифты
│   └── dist/                #   собранный бандл (раздаётся backend'ом)
├── ml/                      # ML-исследования и оценка
│   ├── datasets/            #   данные и манифесты
│   ├── pipelines/           #   генерация и сборка датасетов
│   ├── training/            #   обучение и LoRA
│   ├── benchmark/           #   GigaBench: задания, раннеры и скоринг
│   ├── models/              #   модельные реализации
│   ├── hypotheses/          #   исследовательские проверки
│   └── rag_system/          #   RAG по шахматной литературе
├── maia3/                   # git submodule: человекоподобный движок (UCI)
├── alembic/                 # миграции (21 версия)
├── scripts/                 # create_admin.py, init_db.py
├── tests/                   # pytest: routes/, training/, learning/, auth/,
│                            # knowledge/, analysis/, models/
├── training_data/           # артефакты датасетов (parquet/jsonl)
├── analysis/                # классификатор ошибок, NAG-аннотаторы (stub)
├── input_gateway/           # мультимодальный ввод (stub)
├── perception/              # CV: BoardToFEN-проект, эмбеддер позиций
├── reasoning/               # генерация объяснений (stub)
├── memory/                  # RAG поверх БД (stub)
├── evaluation/              # метрики и арена агентов (stub)
├── docs/                    # user_guide.md, api/endpoints.md
├── DESIGN.md                # дизайн-документ
├── .env.example             # шаблон конфигурации (копируется в .env)
├── start_llm.sh             # запуск Qwen3-8B через vLLM (порт 8000)
├── attach_llm.sh            # подключение LLM-сервиса
├── run.bat / run.sh         # установка и запуск (Windows / Linux)
├── requirements.txt         # зависимости Python
├── pyproject.toml           # ruff, mypy, pytest (coverage)
└── uv.lock                  # lock-файл uv
```

> Часть модулей (`analysis/`, `input_gateway/`, `reasoning/`, `memory/`, `evaluation/`, `backend/async_queue/`, `backend/monitoring/`) пока содержит пустые заглушки; рабочий контур: **backend + frontend + ml/**.

### ML (`ml/`)

| Каталог | Назначение |
|---|---|
| [`datasets`](ml/datasets/) | Наборы комментариев, клеток, смешанный SFT-набор и манифесты. |
| [`pipelines`](ml/pipelines/) | Генерация, подготовка, проверка и сборка датасетов. |
| [`training`](ml/training/) | Обучение Qwen и локальные эксперименты с SFT. |
| [`benchmark`](ml/benchmark/) | GigaBench: подготовка, оценка и сравнение моделей. |
| [`models`](ml/models/) | Реализации моделей и инференса. |

[Документация ML и команды запуска](docs/ml/README.md) · [Результаты четырёх Qwen-моделей](docs/ml/2026-11-27-qwen-four-models-results.md).

## 🚀 Запуск

### Быстрый старт

```bash
# Windows
run.bat

# Linux / macOS
bash run.sh
```

Скрипт создаст venv, установит зависимости, скопирует `.env.example` в `.env`, прогонит миграции и запустит сервер на http://127.0.0.1:8005.

### Backend (вручную)

```bash
# 1. Python-окружение и зависимости
pip install -r requirements.txt

# 2. Конфигурация
cp .env.example .env    # Windows: copy .env.example .env
# DATABASE_URL (по умолчанию SQLite), JWT_SECRET_KEY, MAIA3_*,
# GIGACHESS_BASE_URL и т.д.; см. backend/config/settings.py

# 3. Миграции БД + наполнение учебного курса
python -m scripts.init_db
python -m backend.services.training_seed   # идемпотентно

# 4. Запуск API (одновременно раздаёт собранный frontend/dist)
python -m backend.app
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

### Администратор

```bash
python -m scripts.create_admin <login> <password> [--email me@example.com]
# или: python -m scripts.init_db --admin <login> <password>
```

## 🧪 Тесты

```bash
pytest tests/            # routes/, training/, learning/, auth/, knowledge/, analysis/, models/
```

Покрытие настраивается в `pyproject.toml` (ruff + mypy + pytest).

## 👥 Команда

* **Егор**: Project Lead (архитектура, LLM/GigaChess)
* **Митя**: Backend Developer (Python, API)
* **Илья Бабченков**: Frontend Developer
* **Илья Слынько**: ML Engineer (Maia, датасеты)
* **Даниил**: ML Engineer (LLM Tuning, промпты)

## Общие файлы проекта

- Документация и иллюстрации находятся в корневом `docs/`, ML-материалы — в `docs/ml/`.
- Рабочие настройки всех модулей хранятся в корневом `.env`, шаблон — в `.env.example`.
- Зависимости собраны в корневом `requirements.txt`. Команды установки выполняются из корня репозитория.
- Списки `requirements-ci.txt` и `requirements-runtime.txt` обслуживают существующие CI и backend. ML-зависимости собраны в корневом `requirements.txt`.
- Архив прежнего окружения датасета сохранён локально в `docs/ml/private/dataset-env-backup.txt`.

Версии из прежних списков сохранены. Полная установка объединённого набора требует отдельной проверки на целевой ОС и версии Python.

## Структура тестов

Общий каталог ML-тестов — `tests/ml/`.
Ресурсоёмкие ML-проверки находятся в `tests/ml/integration/`.

```bash
python -m unittest discover -s tests/ml/dataset_gen -v
python -m unittest discover -s tests/ml/gigabench -v
```

Для ресурсоёмких ML-проверок используйте `python -m pytest tests/ml/integration --run-ml-integration`. Они требуют весов, движков или внешних сервисов.
