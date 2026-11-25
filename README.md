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
                          │  gigabench · gigachess · hypotheses │
                          │  models/Qwen3-chess-coach · rag_system_v1 │
                          │  synthesis_pipeline                       │
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
├── ml/                      # ML-исследования и тренировка (см. раздел ML)
│   ├── gigabench/     #   GigaBench v1.1 для оценки шахматных моделей
│   ├── gigachess/           #   HTTP-клиент GigaChess (Cloud.ru)
│   ├── hypotheses/          #   проверка гипотез H1.1 / H1.3
│   ├── models/              #   Qwen3-chess-coach (QLoRA SFT)
│   ├── rag_system_v1/       #   RAG по шахматной литературе (ChromaDB)
│   └── synthesis_pipeline/  #   синтез тренерских SFT-датасетов
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

### ML Research Stack (`ml/`)

| Папка | Назначение |
|-------|------------|
| [`gigabench`](ml/gigabench) | GigaBench v1.1 для проверки доски, правил, тактики и объяснений. Полная документация: [документация](#manual-ml-gigabench). |
| [`gigachess`](ml/gigachess) | HTTP/1.1-клиент облачного инференса GigaChess (ping + `/chat` с FEN). |
| [`hypotheses`](ml/hypotheses) | A/B-проверка гипотез (структурированный вывод, точность цитирования eval) против Qwen3 и GigaChess, метрики в Prometheus. |
| [`models/Qwen3-chess-coach`](ml/models/Qwen3-chess-coach) | QLoRA SFT модели Qwen3-8B под роль шахматного тренера + инференс. |
| [`rag_system_v1`](ml/rag_system_v1) | RAG по шахматной литературе: парсинг PDF/текстов, чанкинг, ChromaDB, динамический ретривер. |
| [`synthesis_pipeline`](ml/synthesis_pipeline) | Конвейер генерации синтетических тренерских объяснений (Stockfish → LLM → валидация → SFT-датасет), ChessCLIP RAG. |

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

## Руководства ML

Все руководства ML собраны здесь. В примерах каждого модуля сохранён его рабочий каталог.

- [docs/ml](#manual-docs-ml)
- [ml](#manual-ml)
- [ml/dataset_gen](#manual-ml-dataset-gen)
- [ml/dataset_gen/gigachess_data](#manual-ml-dataset-gen-gigachess-data)
- [ml/gigabench](#manual-ml-gigabench)
- [ml/gigachess](#manual-ml-gigachess)
- [ml/hypotheses](#manual-ml-hypotheses)
- [ml/models/Qwen3-chess-coach](#manual-ml-models-Qwen3-chess-coach)
- [ml/models](#manual-ml-models)
- [ml/rag_system_v1](#manual-ml-rag-system-v1)
- [ml/synthesis_pipeline](#manual-ml-synthesis-pipeline)
- [ml/synthesis_pipeline/rag_module_chessclip/chessclip](#manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip)
- [ml/synthesis_pipeline/scripts/extra_features](#manual-ml-synthesis-pipeline-scripts-extra-features)

<a id="manual-docs-ml"></a>

### docs/ml

<a id="manual-docs-ml--документация-ml"></a>

### Документация ML

| Каталог | Содержание |
|---|---|
| [pipeline](docs/ml/pipeline) | Схемы GigaChess и Reasoning Draft Model, исходники draw.io и отчёт |
| [tools](docs/ml/tools) | Спецификации python-chess, Stockfish и NAG |
| [references/sber](docs/ml/references/sber) | Исходные материалы и отчёт SFT |
| [datasets](docs/ml/datasets) | Отчёт по генерации описаний ходов |

Файлы доступа к серверам и VPN перенесены в локальный `private/`, исключённый
из Git. Для настройки нового сервера используйте собственные параметры доступа.

Код и инструкции запуска:

- [Подготовка датасетов](#manual-ml-dataset-gen)
- [GigaBench v1.1](#manual-ml-gigabench)
- [Обучение Qwen3](#manual-ml-models-Qwen3-chess-coach)
- [Обзор ML](#manual-ml)

Текущий целевой стенд — терминальный Linux с SSH, RTX 4060 Ti 8 ГБ,
16 ГБ ОЗУ и Qwen3-1.7B для QLoRA. В репозитории остаются отдельные
экспериментальные скрипты Qwen3-8B и T-lite. GigaBench v1.1 содержит адаптер
runtime GigaChess и mock-проверку. Для оценки Qwen3-1.7B потребуется отдельный
адаптер инференса.


<a id="manual-ml"></a>

### ml

<a id="manual-ml--ml-research-stack-"></a>

### ML Research Stack 🧠

Исследовательская часть платформы **SFEDUCASTLING**: обучение и оценка шахматных LLM, RAG по шахматной литературе и синтез тренерских датасетов.

Здесь решается цепочка: **измерить (бенчмарк) → дотюнить (SFT/QLoRA) → обогатить знаниями (RAG) → синтезировать данные (pipeline) → проверить гипотезы**.

---

Документация и отчёты: [docs/ml](#manual-docs-ml).
Подготовка данных: [dataset_gen](#manual-ml-dataset-gen).


<a id="manual-ml---структура"></a>

#### 📂 Структура

| Папка | Назначение | Подробнее |
|-------|-----------|-----------|
| [`gigabench/`](ml/gigabench) | GigaBench v1.1: доска, правила, тактика, метрики и сравнение прогонов. Текущий адаптер использует runtime GigaChess. | [`gigabench/README.md`](#manual-ml-gigabench) |
| [`gigachess/`](ml/gigachess) | HTTP/1.1-клиент облачного инференса **GigaChess** (Cloud.ru): прогрев инстанса (`/ping`) и запросы `/chat` с передачей FEN в attachments. | [`gigachess/README.md`](#manual-ml-gigachess) |
| [`hypotheses/`](ml/hypotheses) | Эксперименты-проверки гипотез о качестве тренерских ответов (структурированный вывод H1.1, точность цитирования оценки H1.3). Сравнение локальной Qwen3 и облачного GigaChess, метрики в Prometheus/Grafana. | [`hypotheses/README.md`](#manual-ml-hypotheses) |
| [`models/`](ml/models) | Обучаемые модели: **Qwen3-chess-coach** - QLoRA SFT Qwen3-8B под роль шахматного тренера (train / inference / merge). | [`models/README.md`](#manual-ml-models) |
| [`rag_system_v1/`](ml/rag_system_v1) | RAG-система по шахматной литературе: парсинг (VLM + OCR) и краулинг, чанкинг, векторная база ChromaDB, динамический ретривер со стадийным фильтром, конструктор RAG-промптов. | [`rag_system_v1/README.md`](#manual-ml-rag-system-v1) |
| [`synthesis_pipeline/`](ml/synthesis_pipeline) | Конвейер синтеза тренерских SFT-данных: извлечение позиций мастеров → анализ Stockfish → генерация объяснений LLM → валидация → ChatML-датасет. Включает RAG-модуль **ChessCLIP** (эмбеддинги позиций) и подготовку reasoning-датасета. | [`synthesis_pipeline/README.md`](#manual-ml-synthesis-pipeline) |

---


<a id="manual-ml---общий-поток"></a>

#### 🔗 Общий поток

```
gigabench            синтез данных
   │  (оценка качества)          │
   ▼                            ▼
rag_system_v1 ──► Qwen3-chess-coach (SFT) ──► hypotheses (A/B-проверка)
   (RAG-контекст)                       │
                                        ▼
                                 production: backend/llm
                                 (chess_explainer.py, gigachess.py)
```

---


<a id="manual-ml---быстрый-старт"></a>

#### 🚀 Быстрый старт

Зависимости всех модулей собраны в корневом `requirements.txt`. Установка выполняется из корня репозитория:

```bash
pip install -r requirements.txt          # общие (образовательные) зависимости
```


<a id="manual-ml--основные-сценарии"></a>

##### Основные сценарии

```bash
# 1. Оценить модель на бенчмарке
python gigabench/run.py --help

# 2. Прогреть и проверить GigaChess
python gigachess/sber_client.py

# 3. Проверить гипотезу H1.1 (структурированный вывод)
python hypotheses/scripts/eval_h1_1.py

# 4. Дообучить тренера (QLoRA)
python models/Qwen3-chess-coach/train.py

# 5. Прогнать RAG-тест на 20 верифицированных позициях
python ../tests/ml/integration/rag/test_rag.py
```

> Бенчмарк, гипотезы и RAG-тесты ссылаются на merged-модель `models/Qwen3-chess-coach/outputs/merged` - создаётся после `train.py` + `merge_lora.py`.


<a id="manual-ml-dataset-gen"></a>

### ml/dataset_gen

<a id="manual-ml-dataset-gen--подготовка-датасетов"></a>

### Подготовка датасетов

Команды ниже выполняются из корня репозитория. Генератор описаний ходов и
вопросов по доске расположен в `gigachess_data/`.


<a id="manual-ml-dataset-gen--структура"></a>

#### Структура

| Путь | Назначение |
|---|---|
| `gigachess_data/gcdata/` | Факты позиции, термины, шаблоны и проверки |
| `gigachess_data/s*.py` | Последовательные этапы конвейера |
| `gigachess_data/tools/` | Отбор данных, диагностика и сборка комплектов |
| `tests/ml/dataset_gen/` | Проверки генератора и шахматных ключей |
| `gigachess_data/train_local/` | Локальные обучение и инференс T-lite |
| `docs/ml/datasets/contracts/` | Контракты и описание терминов, путь от корня проекта |

Партии, движки, веса, `work*`, `out*` и `handoff*` хранятся локально и
исключены из Git. Старые пути сохраняются для повторного запуска экспериментов.
Локальные скрипты обучения пока используют T-lite и свои пути к данным.


<a id="manual-ml-dataset-gen--запуск"></a>

#### Запуск

Существующие команды и контракты описаны в [руководстве](#manual-ml-dataset-gen-gigachess-data).
Скрипты запускаются из `gigachess_data/`.


<a id="manual-ml-dataset-gen--независимая-оценка"></a>

#### Независимая оценка

Для подготовки контрольных позиций используется [GigaBench v1.1](#manual-ml-gigabench).
После `prepare` экспортируйте позиции через `export-heldout` и передайте файл
в `build --heldout`. Исключение остальных позиций той же исходной партии
проверяется отдельно по источникам и группам.

Отчёт по собранному датасету находится в [документации](#manual-docs-ml).


<a id="manual-ml-dataset-gen-gigachess-data"></a>

### ml/dataset_gen/gigachess_data

Рабочий каталог для команд этого руководства — `ml/dataset_gen/gigachess_data`.

<a id="manual-ml-dataset-gen-gigachess-data--gigachess-внутренний-контур-подготовки-sft-датасета"></a>

### GigaChess: внутренний контур подготовки SFT-датасета

Сырые партии → факты (python-chess + Stockfish + NAG) → несвязный черновик → LLM связывает слова → валидатор → SFT JSONL.
Reasoning Draft Model не обучается: её роль (синтаксическая склейка) выполняет LLM по API, а скрипт проверяет, что она ничего не добавила.


<a id="manual-ml-dataset-gen-gigachess-data--что-модель-ждёт-на-входе-из-паспорта-и-контракта-02"></a>

#### Что модель ждёт на входе (из паспорта и контракта 02)

| Ограничение модели | Как учтено |
|---|---|
| Одна доска FEN на пример, 69 векторов энкодера строятся из того же FEN | `fen` = позиция **до** хода; ход, позиция после и все факты лежат в тексте и в `facts_v2.jsonl` |
| Контекст ≤ 512 позиций, включая 69 шахматных слотов и ответ | черновик урезается по приоритету фактов, ответ ≤ 700 символов, итоговая длина проверяется в шаге 6 |
| Координаты в вопросе/ответе заменяются на gigatoken | клетки только как `e4` отдельным словом; SAN (`Nxe5`, `O-O`) и латиница в ответе запрещены валидатором |
| Вопрос/ответ — обычный русский текст, gigatoken не писать | так и есть |
| Контракт v1 знает только `free_square` / `piece_on_square`, лишние поля отклоняет | разборы ходов идут в `commentary_v2.jsonl` (`gigachess-dataset-v2-draft`, задача `move_commentary`) — **нужно согласовать расширение контракта с командой обучения**; `tqu_v1.jsonl` строго по v1 |
| Сплит по партиям, одинаковые расстановки не пересекают сплиты, heldout-доски исключены | union-find по расстановкам + `--heldout evidence/heldout_boards.json` |
| FEN как `Board.fen(en_passant='fen')` | везде так |

Вопрос к модели: `Ход белых: конь с f3 на e5 со взятием.\n[Замысел игрока: …]\nПрокомментируй этот ход.`
Строку «Замысел игрока» получают ~15% примеров (dropout 85/15 из пайплайна). Без неё мотив выводится из геометрии хода.


<a id="manual-ml-dataset-gen-gigachess-data--шаги"></a>

#### Шаги

```bash
cp ../../../.env.example ../../../.env    # из ml/dataset_gen/gigachess_data, вписать LLM_API_KEY
python s1_extract.py --chessgpt annotated_pgn-data.jsonl-0000*-of-00002 --pgn raw/*.pgn --out work/moves.jsonl
python s2_intent.py  --inp work/moves.jsonl --out work/moves_intent.jsonl
python s3_enrich.py  --inp work/moves_intent.jsonl --out work/enriched.jsonl --engine /usr/bin/stockfish --depth 18 --workers 8
python s4_draft.py   --inp work/enriched.jsonl --out work/drafts.jsonl --tokenizer path/to/tokenizer.json
python s5_polish.py  --inp work/drafts.jsonl --out work/polished.jsonl --dry-run   # сначала посмотреть промпт
python s5_polish.py  --inp work/drafts.jsonl --out work/polished.jsonl --limit 50  # пилот, затем без --limit
python s6_build.py   --drafts work/drafts.jsonl --polished work/polished.jsonl --out-dir out \
                     --heldout evidence/heldout_boards.json --tokenizer path/to/tokenizer.json
python ../../../tests/ml/dataset_gen/test_pipeline.py
```

Шаги 2, 3 и 5 можно продолжить после остановки: готовые `id` пропускаются.

1. **s1_extract**: PGN / ChessGPT `{"text": PGN}` / готовые ходы `{"fen_before", "move", "game_id", "human_intent"?}`. Только главная линия: комментарии в вариантах относятся не к сыгранному ходу.
2. **s2_intent**: замысел берётся из входных партий, без LLM. Скрипт чистит комментарий: убирает `[%eval]` и ссылки, отбрасывает мусор (приветствия, «поставьте лайк», автоаннотации lichess, результат партии) и оставляет только предложения с маркерами намерения, без вердиктов, оценок и вариантов. Цитата сохраняется как есть, на языке оригинала. Явный `human_intent` от разметчика проходит те же правила. В черновике цитата заменяет мотив по геометрии в корзине «Замысел»; LLM передаёт её по-русски как мнение игрока («по замыслу игрока…»), остальные три корзины — только факты.
3. **s3_enrich**: python-chess (связки, висячие, шах/мат/пат, рокировка, материал; дельта хода) + Stockfish (оценка до/после, лучший ход и его линия, CPL, PV) + шахматный смысл хода ([gcdata/motifs.py](ml/dataset_gen/gigachess_data/gcdata/motifs.py)): угроза хода и угроза соперника до хода через нулевой ход (отражена / не отражена), тип взятия (размен / выигрыш материала / жертва), двойной удар, фигура под боем, проходная, открытая вертикаль. Своя угроза — только тактическая (взятие, шах, мат); угроза соперника — тактическая или тихая от 3 пешек. NAG по падению шансов на выигрыш (`?!` ≥ 10%, `?` ≥ 20%, `??` ≥ 30%, `!` — единственный нефорсированный ход); при мате — «допускает мат в N» / «упущен мат».
4. **s4_draft**: план описания хода. Три слоя: факты (python-chess, Stockfish, [gcdata/positional.py](ml/dataset_gen/gigachess_data/gcdata/positional.py)) → словарь терминов ([gcdata/concepts.py](ml/dataset_gen/gigachess_data/gcdata/concepts.py)): тактика и маты — ключи по варианту движка ([gcdata/keys.py](ml/dataset_gen/gigachess_data/gcdata/keys.py), правила как у разметчика задач lichess, сверка — `tools/validate_keys.py`), приём засчитывается, только если начинается сыгранным ходом; позиционные термины — по изменению, которое сделал ход → планировщик ([gcdata/draft.py](ml/dataset_gen/gigachess_data/gcdata/draft.py)): одна главная история. План — 3 пункта: что делает ход (главный термин + второй, только если весомый); итог — вердикт, перевес словами, причина (при ошибке) или главное последствие, материал словами — только при смене от фигуры; что было сильнее (только при ошибке). Чисел в плане нет, кроме «мат в N». Термины помечены `[вилка]`. Формат согласован 2026-10-29.
5. **s5_polish**: LLM превращает план в описание хода — один абзац, 2–4 предложения, без чисел и пересказа варианта ход за ходом. Валидатор №2: клетки совпадают с планом; чисел нет, кроме «мат в N» из плана; все термины из `[ ]` названы, а терминов не из плана нет; метки связей переданы союзами; вердикт передан, при ошибке нет хвалебного тона; один абзац; без нотации, латиницы, первого лица и штампов; лимит 600 символов. При нарушении — до 3 повторов, отклонённые варианты идут в `*.rejected.jsonl`.
6. **s6_build**: `commentary_v2.jsonl`, `tqu_v1.jsonl` (70/30), `facts_v2.jsonl`, `dpo_pairs.jsonl` (только train), `stats.json` с SHA256.


<a id="manual-ml-dataset-gen-gigachess-data--llm"></a>

#### LLM

- По умолчанию `backend=openai`: `POST {LLM_BASE_URL}/chat/completions` (anymodel.org), без внешних зависимостей.
- `--backend anthropic`: официальный SDK, `claude-opus-5-5`, `effort=low`, серверный fallback при отказе.
- `temperature` не передаётся: Opus 5.5 отклоняет параметры сэмплинга. Детерминизм дают жёсткий промпт и валидатор.


<a id="manual-ml-dataset-gen-gigachess-data--важно"></a>

#### Важно

- `tests/ml/dataset_gen/fake_uci.py` — игрушка для smoke-теста. Для данных нужен настоящий Stockfish 16+.
- Без `--tokenizer` длина оценивается по символам с запасом. Для приёмки нужен `tokenizer.json` модели.
- `ru.BLACK_ROOT = "чёрн"` должен совпадать с canonical в `dataset_tools.py`, иначе TQU не пройдёт валидатор v1.
- Права на ChessGPT (lichess studies, gameknot и др.) надо подтвердить. Документ 01 этого прямо требует.


<a id="manual-ml-gigabench"></a>

### ml/gigabench

<a id="manual-ml-gigabench--единый-бенчмарк-gigachess"></a>

### Единый бенчмарк GigaChess

Проверки понимания доски, правил, короткой тактики, выбора хода и объяснений
для нашей реконструкции GigaChess. Один запуск готовит замороженный набор,
загружает модель один раз, сохраняет ответы, считает автоматические метрики
и формирует общий REPORT_RU.md и summary.json.

Это самостоятельный адаптированный набор по идеям ChessQA, Random-Chess и
BIG-bench Checkmate in One. Это не официальный прогон этих бенчмарков.
Исторический пилот `benchmarks/lichess89_v1` относится к исходному проекту GigaChess и в SFEDUCASTLING отсутствует.

Источник — `origin/feature/gigabench-v1.1`. Команды выполняются из корня
SFEDUCASTLING. Адаптер использует специальный runtime GigaChess.
Для Qwen3-1.7B требуется отдельная интеграция инференса.


<a id="manual-ml-gigabench--что-входит"></a>

#### Что входит

| Блок | Проверки |
|---|---|
| Structural | фигура и цвет, пустые клетки, шах, легальность, атаки и защита, число ходов, изменение доски после последовательности, материальный баланс |
| Rules | рокировка через атакованное поле и из-под шаха, взятие на проходе с открытием короля, превращение, различение пата и мата |
| Motifs | абсолютная связка, атаки связанной фигуры, двойная атака; есть положительные и отрицательные случаи |
| Random-Chess inspired | полный список легальных ходов на небольших множествах; precision, recall, F1 и точное совпадение |
| Tactics | все допустимые маты в один; форсированный мат не более чем в два собственных хода, включая проверенные отрицательные примеры |
| Comment facts | выбор фактически верного комментария к известному ходу из четырёх вариантов |
| Наш пилот | 178 вопросов М1, 79 основных ходов PGN для М2, 89 рекомендаций М3 и 89 записей для экспертных М4–М7 |

Синтетические позиции формируются детерминированно по seed. Случайные партии
проигрываются легальными ходами от начальной позиции; исходная последовательность
сохраняется и проверяется тестами. Для отдельных учебных правил и KQK используются
явно обозначенные искусственные позиции. Не утверждается, что все они происходят
из человеческих партий.

Список ходов ограничен восемью ходами в полном списке или в списке одной фигуры:
такой ответ помещается в текущий потолок генерации GigaChess в 64 новых токена.
Это ограничение адаптации, а не результат оригинального Random-Chess.

Comment facts проверяет факты о позиции; не заменяет качество педагогического
объяснения. Блок М4–М7 остаётся самостоятельным.


<a id="manual-ml-gigabench--что-нового-в-11"></a>

#### Что нового в 1.1

Полный список — в CHANGELOG.md. Коротко:

    # пересчитать метрики готового прогона новым кодом (без сервера)
    python ml/gigabench/run.py rescore --source /path/to/completed-run --out ml/gigabench/runs/rescored

    # подготовить полный набор: позиции из партий, задачи Lichess и оценка позиции
    # (замороженные выборки Lichess уже лежат в data/sources)
    python ml/gigabench/run.py prepare --out ml/gigabench/runs/v1_1 --profile standard --games ml/gigabench/data/sources/games_v1_1.json --puzzles ml/gigabench/data/sources/puzzles_v1_1.json --with-judgement --stockfish /path/to/stockfish

    # пересобрать выборки из исходных выгрузок Lichess и скачать Stockfish 19 (обычно не нужно)
    python ml/gigabench/tools/fetch_sources.py --work /tmp/gigabench_sources

    # сгенерировать ответы с нужным профилем входа
    python ml/gigabench/run.py generate --out ml/gigabench/runs/v1_1 --project-root /home/USER/gigachess --input-profile question_only

    # список позиций для исключения из обучения
    python ml/gigabench/run.py export-heldout --run ml/gigabench/runs/v1_1 --out heldout_boards.json

Каждый прогон пишет report.html: итоги по блокам, графики по задачам и все ответы с фильтрами.


<a id="manual-ml-gigabench--установка"></a>

#### Установка

Для подготовки и оценки достаточно Python 3.11+ и зависимости правил:

    python -m pip install -r requirements.txt

Для запуска самой модели используйте существующее CUDA-окружение GigaChess.
Повторная установка PyTorch и скачивание весов бенчмарком не выполняются.
Для М3 нужен отдельный официальный Stockfish 19 с одной NNUE-сетью.
Бинарный файл движка и веса модели в репозиторий бенчмарка не входят.

После публикации этой папки как корня отдельного репозитория уберите префикс
ml/gigabench/ из команд.


<a id="manual-ml-gigabench--подключение-наших-89-записей"></a>

#### Подключение наших 89 записей

Приватный набор в эту копию не включён. Импортируйте свой исходный файл:

    python ml/gigabench/run.py import-legacy --source datasets/lichess_pilot_20261101/dataset_all_89_review.jsonl

Бенчмарк не изменяет эти комментарии и не признаёт их автоматически верными.
Сохраняются все 89 записей; 10 аналитических вариантов исключаются только из М2.
Подлинность исторических основных линий PGN отдельно не устанавливается.

Публичный архив намеренно не содержит авторских комментариев. При отсутствии
локального файла обычный запуск выдаёт понятную ошибку. Для публичной
синтетической части можно явно указать --without-legacy.


<a id="manual-ml-gigabench--один-полный-автоматический-запуск"></a>

#### Один полный автоматический запуск

Из корня проекта на сервере с подготовленным окружением:

    python ml/gigabench/run.py all --project-root /home/USER/gigachess --stockfish /path/to/stockfish --out ml/gigabench/runs/baseline --profile standard

Пример пути к Stockfish в исходном проекте:

    datasets/lichess_pilot_20261101/engine/stockfish/stockfish-windows-x86-64-universal.exe

На Linux укажите Linux-бинарник Stockfish 19. Модель должна находиться в
--project-root с каталогами releases/gigachess-control-v1 и
code/interface_resolution_v1_20261018. Поддерживается текущий BF16-путь.

Профили smoke / quick / standard различаются числом синтетических позиций;
89 исходных записей при подключении сохраняются во всех профилях.
--adapter PATH подключает дообученный адаптер формата gigachess-adapter-pilot-v1;
совместимость и SHA256 проверяет загрузчик модели. Без этого параметра
запускается исходный checkpoint. Обучение бенчмарк не выполняет.

После прерывания тот же набор и модель можно продолжить:

    python ml/gigabench/run.py all --project-root /home/USER/gigachess --stockfish /path/to/stockfish --out ml/gigabench/runs/baseline --resume

Для технической проверки без GPU:

    python ml/gigabench/run.py all --backend mock --without-legacy --profile smoke --out ml/gigabench/runs/mock

Mock всегда отвечает MOCK_INVALID и не имеет доступа к эталонам.
Его отчёт явно помечен и никогда не является оценкой нейросети.


<a id="manual-ml-gigabench--поэтапный-запуск"></a>

#### Поэтапный запуск

    python ml/gigabench/run.py prepare --out ml/gigabench/runs/baseline --profile standard
    python ml/gigabench/run.py generate --out ml/gigabench/runs/baseline --project-root /home/USER/gigachess
    python ml/gigabench/run.py score --out ml/gigabench/runs/baseline --stockfish /path/to/stockfish

После prepare данные, вопросы, конфигурация и исходники фиксируются хешами.
Изменение кода или данных требует нового прогона, не тихой подмены протокола.
Эталоны не передаются в адаптер модели: он получает только id, fen, prompt
и max_new_tokens. В предсказании и рекомендации хода нет человеческого ответа
или позиции после хода.


<a id="manual-ml-gigabench--объяснения-и-тренеры"></a>

#### Объяснения и тренеры

Один автоматический запуск не может сам получить независимые оценки тренеров.
В coach_rubrics_pending.jsonl создаются шаблоны на все 89 объяснений.
До генерации тренер задаёт уровень ученика, 2–5 обязательных идей, допустимые
факты, критические ошибки, подписывает и датирует каждую рубрику с часовым поясом.

    python ml/gigabench/run.py generate --out ml/gigabench/runs/baseline --project-root /home/USER/gigachess --stage explanations --rubrics approved_rubrics.jsonl

Хеш рубрик сохраняется до генерации. Без утверждения этот этап не запускается.
Затем собираются разметка утверждений, покрытия идей, критических ошибок и оценки
двух тренеров. Оценка привязывается к digest точного текста ответа
через response_sha256; формула digest реализована в gigabench/io.py.
При расхождении итоговых баллов М7 нужен третий тренер.

    python ml/gigabench/run.py grade --out ml/gigabench/runs/baseline --rubrics approved_rubrics.jsonl --ratings ratings.jsonl

Команда дополняет общий отчёт М4–М7 и долей принятых объяснений. До этого они
имеют значение null и статус pending, а не фиктивный ноль или оценку сходства.
Формат ratings и формулы описаны в METRICS.md.


<a id="manual-ml-gigabench--сравнение-версий"></a>

#### Сравнение версий

    python ml/gigabench/run.py compare --before ml/gigabench/runs/baseline --after ml/gigabench/runs/finetuned --out ml/gigabench/runs/comparison.json

Сравнение требует одинаковых данных, вопросов, кода, конфигурации и движка,
полных ответов и отсутствия mock-режима. Разница точности считается парно,
с bootstrap по группам партий/синтетических исходных позиций. Матовые
исключения М3 показаны отдельно. Для экспертной части сравниваются одинаковые
утверждённые рубрики; исходные оценки сохраняются.


<a id="manual-ml-gigabench--публикация"></a>

#### Публикация

    python ml/gigabench/run.py package --out ml/gigabench/dist/GigaChessBenchmark-public.zip

Архив содержит код, конфигурацию, тесты, документацию и сведения об источниках.
Он НЕ содержит data/private, результаты, Word-документы, ключи, модельные веса,
бинарники и NNUE. Внутри есть PUBLIC_MANIFEST.json с SHA256 каждого файла.
Команда только создаёт локальный архив; ничего не публикует на GitHub.

Исторические результаты пилота в эту копию не включены.


<a id="manual-ml-gigabench--проверки"></a>

#### Проверки

    python -m unittest discover -s tests/ml/gigabench -v

При переменной STOCKFISH_EXECUTABLE дополнительно выполняется реальный
интеграционный тест движка. Обычные тесты работают без GPU и сети.
Для сохранения независимости проверки весь тестовый набор, связанные партии,
варианты и дубли из других источников должны оставаться вне обучения.


<a id="manual-ml-gigachess"></a>

### ml/gigachess

Рабочий каталог для команд этого руководства — `ml/gigachess`.

<a id="manual-ml-gigachess--gigachess-client-"></a>

### GigaChess Client 🌐

HTTP-клиент для облачного инференса **GigaChess** (Cloud.ru). Используется как бэкенд-движок LLM для шахматных запросов.

---


<a id="manual-ml-gigachess---файлы"></a>

#### 📄 Файлы

| Файл | Назначение |
|------|-----------|
| `sber_client.py` | Основной клиент: `HTTP11Adapter`, `get_session()`, `ping_server()`, `ask_gigachess(prompt, fen)`, `extract_uci_move(text)`. |

---


<a id="manual-ml-gigachess---как-работает"></a>

#### 🔧 Как работает

1. **`ping_server()`** - GET-запрос к `<BASE_URL>/ping`, проверяет доступность инстанса (прогрев после cold start).
2. **`ask_gigachess(prompt, fen="")`** - POST-запрос к `<BASE_URL>/chat`:
   - промпт передаётся в `messages[0].content`;
   - если задан FEN - передаётся в `attachments` (модель понимает позицию по нотации);
   - параметры инференса: `temperature=0.2`, `top_p=0.95`, `max_tokens=128`, `model="gigachess"`;
   - таймаут **900 секунд** - на случай прогрева инстанса.
3. **`extract_uci_move(text)`** - извлекает первый UCI-ход (`e2e4`, `e7e8q`) из ответа модели.


<a id="manual-ml-gigachess--принудительный-http11"></a>

##### Принудительный HTTP/1.1

`HTTP11Adapter` принудительно использует HTTP/1.1 вместо HTTP/2 - решает проблему зависания запросов на h2-endpoint'ах.

> Внимание: `verify=False` отключает проверку TLS-сертификата (для сервисов Cloud.ru с самоподписанными/недоверенными сертификатами). Используется только для исследовательских целей.

---


<a id="manual-ml-gigachess---использование"></a>

#### 🚀 Использование


<a id="manual-ml-gigachess--проверка-и-ручной-запрос"></a>

##### Проверка и ручной запрос

```bash
python sber_client.py
```

Скрипт: ждёт прогрева инстанса (до 60 попыток по 10 сек), отправляет тестовый запрос со стартовой позицией и выводит ответ + извлечённый UCI-ход.


<a id="manual-ml-gigachess--как-библиотека"></a>

##### Как библиотека

```python
from sber_client import ask_gigachess, ping_server, extract_uci_move

if ping_server():
    answer = ask_gigachess("Какой лучший ход?", fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(answer)
    print(extract_uci_move(answer))
```

---


<a id="manual-ml-gigachess---конфигурация"></a>

#### ⚙️ Конфигурация

`BASE_URL` захардкожен в начале файла и указывает на персональный Model RUN инстанс:

```
https://<uuid>.modelrun.inference.cloud.ru
```

Для смены инстанса отредактируйте `BASE_URL` в `sber_client.py`.

> В production-backend также используется клиент `backend/llm/gigachess.py` (`GigachessClient`), настраиваемый через `GIGACHESS_BASE_URL` в `.env`.


<a id="manual-ml-hypotheses"></a>

### ml/hypotheses

Рабочий каталог для команд этого руководства — `ml/hypotheses`.

<a id="manual-ml-hypotheses--hypotheses-"></a>

### Hypotheses 🧪

Эксперименты по проверке гипотез о качестве тренерских ответов шахматных LLM. Сравнивают локальную дообученную модель **Qwen3-chess-coach** и облачный **GigaChess**, метрики публикуются в **Prometheus** для дашбордов Grafana.

---


<a id="manual-ml-hypotheses---структура"></a>

#### 📂 Структура

| Путь | Назначение |
|------|-----------|
| `scripts/eval_h1_1.py` | Гипотеза **H1.1**: модель выдаёт **структурированный ответ** (ход + оценка + шахматный термин). |
| `scripts/eval_h1_3.py` | Гипотеза **H1.3**: модель точно **цитирует оценку позиции** (в пределах ±0.5 пешки от Stockfish). |
| `data/dataset_h1_1.json` | Датасет позиций для H1.1 (FEN, сделанный ход, ожидание Stockfish). |
| `data/dataset_h1_3.json` | Датасет позиций для H1.3 (FEN, сделанный ход, эталонная оценка Stockfish). |

---


<a id="manual-ml-hypotheses---гипотезы"></a>

#### 🧩 Гипотезы


<a id="manual-ml-hypotheses--h11---структурированный-тренерский-вывод"></a>

##### H1.1 - Структурированный тренерский вывод

Одиночный системный промпт `"Ты - шахматный аналитик и тренер."`. Ответ модели должен содержать три компонента:

- **Move** - ход в любой нотации (UCI/SAN/русская);
- **Eval** - оценка позиции;
- **Term** - хотя бы один стратегический термин (центр, связка, вилка, развитие, рокировка, проходная и т.д. - словарь `CHESS_STRATEGIC_TERMS`).

Параметры вывода отсекаются через `clean_response()` (режет теги `thinking`/`response`).


<a id="manual-ml-hypotheses--h13---точность-цитирования-оценки"></a>

##### H1.3 - Точность цитирования оценки

Оценка в ответе модели сверяется с эталонной оценкой Stockfish: ход засчитывается, если попадает в допуск **±0.5 пешки**.


<a id="manual-ml-hypotheses--метрики-prometheus"></a>

##### Метрики Prometheus

| Gauge | Смысл |
|-------|-------|
| `chess_h1_1_structured_pct` | Итоговый % ответов со всеми тремя компонентами |
| `chess_h1_1_has_move_count` | Кол-во ответов с ходом |
| `chess_h1_1_has_eval_count` | Кол-во ответов с оценкой |
| `chess_h1_1_has_term_count` | Кол-во ответов с термином |
| `chess_h1_3_eval_accuracy_pct` | % ответов в пределах ±0.5 пешки |
| `chess_h1_3_within_tolerance_count` | Кол-во ответов в допуске |
| `chess_h1_3_has_eval_count` | Кол-во ответов с какой-либо оценкой |

Все метрики имеют label `model` (например `qwen-merged` / `gigachess`) - для stacked-диаграмм в Grafana.

---


<a id="manual-ml-hypotheses---запуск"></a>

#### 🚀 Запуск

```bash
# H1.1 - структурированный вывод
python scripts/eval_h1_1.py

# H1.3 - точность цитирования eval
python scripts/eval_h1_3.py
```


<a id="manual-ml-hypotheses--зависимости-от-соседних-модулей"></a>

##### Зависимости от соседних модулей

- Скрипты импортируют `from sber_client import ask_gigachess` из `../gigachess/`.
- Для локальной модели ищется merged-чекпоинт `../Qwen3-chess-coach/outputs/merged` (создаётся через `Qwen3-chess-coach/merge_lora.py`); имя базовой модели читается из `outputs/adapter_config.json`.
- Сценарий по умолчанию читает датасет из текущей директории (`benchmark.json` / `dataset_h1_3.json`), запускайте скрипты из `hypotheses/scripts/`.


<a id="manual-ml-models-Qwen3-chess-coach"></a>

### ml/models/Qwen3-chess-coach

Рабочий каталог для команд этого руководства — `ml/models/Qwen3-chess-coach`.

<a id="manual-ml-models-Qwen3-chess-coach--qwen3-chess-coach-"></a>

### Qwen3-Chess-Coach ♞

Дообучение (SFT) модели **Qwen/Qwen3-8B** под роль **шахматного тренера** - модель объясняет сделанный ход игрока, сравнивает его с лучшим ходом движка и даёт практические советы на русском языке.

Подход: **4-bit квантование (NF4) + LoRA (QLoRA)**, обучение через `trl.SFTTrainer`.

---


<a id="manual-ml-models-Qwen3-chess-coach---файлы"></a>

#### 📂 Файлы

| Файл | Назначение |
|------|-----------|
| `config.py` | Все гиперпараметры: базовая модель, датасет, квантование, LoRA, SFT. |
| `train.py` | Запуск QLoRA SFT-обучения, сохранение LoRA-адаптера в `./outputs`. |
| `merge_lora.py` | Слияние LoRA-адаптера с базовой моделью → `./outputs/merged`. |
| `inference.py` | Инференс из `./outputs/merged`: тренерский разбор хода по FEN + движковой оценке. |
| `accelerate_config.yaml` | Конфигурация multi-GPU (2 процесса, bf16). |
| `requirements.txt` | Зависимости (torch, transformers, datasets, accelerate, peft, trl, bitsandbytes). |

---


<a id="manual-ml-models-Qwen3-chess-coach---конфигурация-configpy"></a>

#### ⚙️ Конфигурация (config.py)

| Параметр | Значение |
|----------|----------|
| Базовая модель | `Qwen/Qwen3-8B` |
| Датасет | `khoilamalphaai/chess-coach-move-review` (HuggingFace) |
| Квантование | 4-bit NF4 + double quant, bf16 |
| LoRA | `r=64`, `α=128`, `dropout=0.05`, targets: все `*_proj` |
| Обучение | 3 эпохи, batch 2, grad-accum 8, LR 2e-4, cosine, warmup 3% |
| Параметры SFT | `max_seq_length=4096`, save/eval каждые 100 шагов |

---


<a id="manual-ml-models-Qwen3-chess-coach---рабочий-процесс"></a>

#### 🚀 Рабочий процесс


<a id="manual-ml-models-Qwen3-chess-coach--1-обучение"></a>

##### 1. Обучение

```bash
pip install -r requirements.txt

# Одна GPU
python train.py

# Multi-GPU (2 процесса)
accelerate launch --config_file accelerate_config.yaml train.py
```

Результат: LoRA-адаптер и токенизатор в `./outputs/`.


<a id="manual-ml-models-Qwen3-chess-coach--2-слияние-адаптера"></a>

##### 2. Слияние адаптера

```bash
python merge_lora.py   # → ./outputs/merged
```


<a id="manual-ml-models-Qwen3-chess-coach--3-инференс"></a>

##### 3. Инференс

```bash
python inference.py
```

Функция `run_inference(fen, move, eval_score, best_move, show_thinking=False)`:

- системная инструкция - строгие правила русской шахматной терминологии (не путать бьющую/взятую фигуру, корректные названия фигур);
- генерирует объяснение почему ход `move` слабее лучшего хода `best_move` и как искать подобные решения;
- `temperature=0.3`, `max_new_tokens=512`;
- тег `thinking` отрезается, если `show_thinking=False`.

---


<a id="manual-ml-models-Qwen3-chess-coach---кто-использует-модель"></a>

#### 📈 Кто использует модель

- [`ml/gigabench`](ml/gigabench) - GigaBench v1.1; текущий адаптер рассчитан на runtime GigaChess.
- [`ml/hypotheses`](ml/models/hypotheses) - сравнивает `./outputs/merged` с облачным GigaChess.
- `backend/` - сервис Qwen3-8B разворачивается через vLLM (`start_llm.sh`) и используется как LLM-бэкенд.

> Для бенчмарка и гипотез сначала выполните `train.py` + `merge_lora.py`, см. [`ml/README.md`](#manual-ml-models).


<a id="manual-ml-models"></a>

### ml/models

Рабочий каталог для команд этого руководства — `ml/models`.

<a id="manual-ml-models--ml-models-"></a>

### ML Models 🧠

Каталог обучаемых моделей платформы. Сейчас содержит один проект.

| Папка | Модель | Задача |
|-------|--------|--------|
| [`Qwen3-chess-coach/`](ml/models/Qwen3-chess-coach) | **Qwen3-8B** (QLoRA SFT) | Генерация тренерских объяснений ходов на русском языке |

Подробнее - [`Qwen3-chess-coach/README.md`](#manual-ml-models-Qwen3-chess-coach).


<a id="manual-ml-rag-system-v1"></a>

### ml/rag_system_v1

Рабочий каталог для команд этого руководства — `ml/rag_system_v1`.

<a id="manual-ml-rag-system-v1--rag-system-v1-"></a>

### RAG System v1 📚

Retrieval-Augmented Generation для шахматного тренера: собирает знания из шахматной литературы и статей, чанкит их в векторную базу **ChromaDB** и подмешивает релевантный контекст в промпты LLM.

---


<a id="manual-ml-rag-system-v1---структура"></a>

#### 📂 Структура

| Путь | Назначение |
|------|-----------|
| `parser_chunker.py` | Классификатор-чанкер: PDF/текст → чанки → ChromaDB (коллекция `chess_knowledge_base`). |
| `dynamic_retriever.py` | `DynamicChessRetriever` - умный запрос и поиск по стадии игры (opening/middlegame/endgame). |
| `chess_analysis/engine_analyzer.py` | `ChessEngineAnalyzer` - анализ позиции через python-chess + Stockfish (глубина 18). |
| `chess_analysis/prompt_builder.py` | `ChessPromptBuilder` - сборка строго структурированных RAG-промптов с few-shot защитой. |
| `parsers/auto_crawler.py` | Краулер категорий Wikipedia (RU/EN) по шахматной теории → `data/parsed_texts/articles_parsed.txt`. |
| `parsers/vlm_parser.py` | VLM-парсер (Qwen2-VL-2B) книг в PDF → чистый текст (`tarrasch_vlm_clean.txt`). |
| `data/raw_books/` | Исходные PDF-книги (например «300 шахматных партий» Тарраша). |
| `data/parsed_texts/` | Распарсенные тексты (OCR-сырьё и VLM-чистые версии). |
| `test_rag.py` | Тест на **20 верифицированных позициях** (дебют/миттельшпиль/эндшпиль) с эталонными ответами Stockfish. |
| `test_batch_coach.py` | Пакетная проверка тренерских ответов через облачную модель. |

---


<a id="manual-ml-rag-system-v1---как-устроен-pipeline"></a>

#### 🔧 Как устроен pipeline

```
     сбор источников                     индексация                     генерация
┌──────────────────────┐      ┌───────────────────────┐      ┌───────────────────────────┐
│ PDF-книги ──VLM/OCR──┤      │                       │      │                           │
│ Wikipedia ──краулер──┼─────▶│  parser_chunker.py    │─────▶│  DynamicChessRetriever    │
│ автостатьи           │      │  (чанки + тысячи стадий│      │  → релевантные чанки      │
└──────────────────────┘      │   → ChromaDB)          │      │  ChessPromptBuilder       │
                              └───────────────────────┘      │  → промпт для LLM          │
                                                       └──────┴───────────────────────────┘
```

1. **Парсинг**: книги распознаются через VLM (`Qwen2-VL-2B`) или OCR, тексты по шахматной теории собираются краулером из Wikipedia.
2. **Чанкинг и индексация**: `ChessChunker` разбивает текст на смысловые фрагменты, классифицирует по стадии игры и складывает в ChromaDB с эмбеддерами `paraphrase-multilingual-MiniLM-L12-v2`.
3. **Ретрив**: `DynamicChessRetriever.build_smart_query()` собирает запрос из типа ошибки, описаний ходов и стадии; `retrieve_context()` фильтрует по `stage` (для opening/endgame) и возвращает top-k чанков.
4. **Промпт**: `ChessPromptBuilder` объединяет анализ движка (Stockfish), извлечённый контекст и вопрос ученика в структурированный промпт для LLM.

---


<a id="manual-ml-rag-system-v1---запуск"></a>

#### 🚀 Запуск

```bash
pip install -r requirements.txt

# Построить индексы из data/parsed_texts
python parser_chunker.py

# Проверить RAG на 20 верифицированных позициях
python ../../tests/ml/integration/rag/test_rag.py

# Пакетная проверка тренера через облачную модель
python ../../tests/ml/integration/rag/test_batch_coach.py
```

> Путь к Stockfish настраивается переменной окружения `STOCKFISH_PATH` (по умолчанию `stockfish` - бинарник должен быть в PATH).

---


<a id="manual-ml-rag-system-v1---особенности-импортов"></a>

#### ⚠️ Особенности импортов

Модули ссылаются на пакет `knowledge_base` (например `from knowledge_base.dynamic_retriever import build_llm_rag_prompt`). Для этого в начале каждого файла выполняется вставка базовой директории в `sys.path`. При переносе кода убедитесь, что `BASE_DIR` указывает на корень, где доступен пакет `knowledge_base` (в текущей схеме - `ml/`).


<a id="manual-ml-synthesis-pipeline"></a>

### ml/synthesis_pipeline

Рабочий каталог для команд этого руководства — `ml/synthesis_pipeline`.

<a id="manual-ml-synthesis-pipeline--synthesis-pipeline-"></a>

### Synthesis Pipeline 🛠️

Конвейер автоматической генерации **тренерских SFT-датасетов** для шахматных LLM.

Из гроссмейстерских партий извлекаются позиции, анализируются движком **Stockfish**, а LLM генерирует методические объяснения, которые после детерминированной валидации превращаются в ChatML-датасет для дообучения.

Полная целевая структура (каталоги `data/` по шагам) описана в [`PROJECT_STRUCTURE.txt`](ml/synthesis_pipeline/PROJECT_STRUCTURE.txt).

---


<a id="manual-ml-synthesis-pipeline---структура"></a>

#### 📂 Структура

| Путь | Назначение |
|------|-----------|
| `src/schemas.py` | Pydantic-схемы: `RawPosition`, `MultiPVLine`, `AnalyzedPosition`, `CoachExplanation`, `SimilarExample`. |
| `src/prompts.py` | `SYSTEM_PROMPT` тренера + `build_user_prompt()` (FEN, данные движка, RAG-контекст). |
| `src/llm_api.py` | `ChessLLMClient` - OpenAI-совместимый клиент (OpenRouter/Groq/OpenAI). |
| `src/chess_engine.py` | `StockfishAnalyzer` - анализ позиции (глубина 22, MultiPV=3, centipawn loss). |
| `src/board_features.py` | Детерминированные факты о позиции: висячие фигуры, нападения, атаки. |
| `scripts/` | 5 шагов конвейера + обогащение признаков (`extra_features/`). |
| `rag_module_chessclip/` | Векторный поиск шахматных аналогий на эмбеддингах ChessCLIP. |
| `reasoning_draft_model/` | Подготовка «reasoning draft»-датасета из человеческих комментариев + SFT. |
| `tests/ml/integration/synthesis_pipeline/` | Тесты генерации, векторного хранилища и энкодера, путь от корня проекта. |

---


<a id="manual-ml-synthesis-pipeline---конвейер-данных"></a>

#### 🔀 Конвейер данных

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

Зарезервированные слоты расширения: `05_llm_judged` (LLM-as-a-Judge), `06_deduplicated`, `07_balanced` - см. `PROJECT_STRUCTURE.txt`.

---


<a id="manual-ml-synthesis-pipeline---rag-модуль-chessclip-rag_module_chessclip"></a>

#### 🧩 RAG-модуль ChessCLIP (`rag_module_chessclip/`)

Векторный индекс шахматных позиций для подстановки похожих примеров в промпт:

| Файл | Назначение |
|------|-----------|
| `board_converter.py` | FEN → память из **112 бинарных плоскостей** LCZero `(1, 112, 8, 8)`. |
| `clip_encoder.py` | `ChessPositionEncoder` - FiLM-ResNet → **512-d** вектор (L2-нормализация). Конфиг `chessclip-quickgelu.json`. |
| `vector_store.py` | `ChessVectorStore` - обёртка над ChromaDB (upsert/query, cosine). |
| `chessclip/` | Копия форка OpenCLIP (Waterhorse) с реализацией ChessCLIP (`src/open_clip`, `src/training`). |

> Веса энкодера ожидаются в `~/SFEDUCASTLING/ml/models/chessclip/chessclip.pt`.

---


<a id="manual-ml-synthesis-pipeline---reasoning-draft-model-reasoning_draft_model"></a>

#### 🧩 Reasoning Draft Model (`reasoning_draft_model/`)

Превращение реальных комментариев людей к партиям в «черновые рассуждения»:

| Скрипт | Назначение |
|--------|-----------|
| `clean_and_filter_pairs.py` | Regex-фильтрация пар ход/комментарий (GameKnot). |
| `llm_audit_and_clean.py` | Пакетная LLM-курация (метки CLEANED / REJECT). |
| `fast_llm_post_audit.py` | Ускоренный повторный проход (resume-safe). |
| `sft_train_reasoning_draft_model.py` | QLoRA SFT Qwen2.5-7B на итоговом датасете. |

---


<a id="manual-ml-synthesis-pipeline---запуск"></a>

#### 🚀 Запуск

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
python -m pytest ../../tests/ml/integration/synthesis_pipeline --run-ml-integration
```

Ключи API задаются переменными окружения проекта `src/llm_api.py` (OpenRouter / Groq / OpenAI).

---


<a id="manual-ml-synthesis-pipeline---см-также"></a>

#### 📄 См. также

- [`PROJECT_STRUCTURE.txt`](ml/synthesis_pipeline/PROJECT_STRUCTURE.txt) - полная целевая схема каталогов и номеров шагов.
- [`scripts/extra_features/README.md`](#manual-ml-synthesis-pipeline-scripts-extra-features) - обогащение позиций доп. признаками (NAG, флаги, оценка путей).
- [`rag_module_chessclip/chessclip/README.md`](#manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip) - upstream-документация ChessCLIP/OpenCLIP.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip"></a>

### ml/synthesis_pipeline/rag_module_chessclip/chessclip

Рабочий каталог для команд этого руководства — `ml/synthesis_pipeline/rag_module_chessclip/chessclip`.

<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--openclip"></a>

### OpenCLIP

[[Paper]](https://arxiv.org/abs/2212.07143) [[Colab]](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb)
[![pypi](https://img.shields.io/pypi/v/open_clip_torch.svg)](https://pypi.python.org/pypi/open_clip_torch)

Welcome to an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training).

The goal of this repository is to enable training models with contrastive image-text supervision, and to investigate their properties such as robustness to distribution shift. Our starting point is an implementation of CLIP that matches the accuracy of the original CLIP models when trained on the same dataset.
Specifically, a ResNet-50 model trained with our codebase on OpenAI's [15 million image subset of YFCC](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md) achieves **32.7%** top-1 accuracy on ImageNet. OpenAI's CLIP model reaches **31.3%** when trained on the same subset of YFCC. For ease of experimentation, we also provide code for training on the 3 million images in the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) dataset, where a ResNet-50x4 trained with our codebase reaches 22.2% top-1 ImageNet accuracy.

We further this with a replication study on a dataset of comparable size to OpenAI's, [LAION-400M](https://arxiv.org/abs/2111.02114), and with the larger [LAION-2B](https://laion.ai/blog/laion-5b/) superset. In addition, we study scaling behavior in a paper on [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).

We have trained:
  * ViT-B/32 on LAION-400M with a accuracy of **62.9%**, comparable to OpenAI's **63.2%**, zero-shot top-1 on ImageNet1k
  * ViT-B/32 on LAION-2B with a accuracy of **66.6%**.
  * ViT-B/16 on LAION-400M achieving an accuracy of **67.1%**, lower than OpenAI's **68.3%** (as measured here, 68.6% in paper)
  * ViT-B/16+ 240x240 (~50% more FLOPS than B/16 224x224) on LAION-400M achieving an accuracy of **69.2%**
  * ViT-L/14 on LAION-400M with an accuracy of **72.77%**, vs OpenAI's **75.5%** (as measured here, 75.3% in paper)
  * ViT-L/14 on LAION-2B with an accuracy of **75.3%**, vs OpenAI's **75.5%** (as measured here, 75.3% in paper)
  * ViT-H/14 on LAION-2B with an accuracy of **78.0**. The best in1k zero-shot for released, open-source weights thus far.
  * ViT-g/14 on LAION-2B with an accuracy of **76.6**. This was trained on reduced schedule, same samples seen as 400M models.

As we describe in more detail [below](#manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--why-are-low-accuracy-clip-models-interesting), CLIP models in a medium accuracy regime already allow us to draw conclusions about the robustness of larger CLIP models since the models follow [reliable scaling laws](https://arxiv.org/abs/2107.04649).

This codebase is work in progress, and we invite all to contribute in making it more accessible and useful. In the future, we plan to add support for TPU training and release larger models. We hope this codebase facilitates and promotes further research in contrastive image-text learning. Please submit an issue or send an email if you have any other requests or suggestions.

Note that portions of `src/open_clip/` modelling and tokenizer code are adaptations of OpenAI's official [repository](https://github.com/openai/CLIP).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--approach"></a>

#### Approach

| ![CLIP](https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/CLIP.png) |
|:--:|
| Image Credit: https://github.com/openai/CLIP |


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--usage"></a>

#### Usage

```
pip install open_clip_torch
```

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

To compute billions of embeddings efficiently, you can use [clip-retrieval](https://github.com/rom1504/clip-retrieval) which has openclip support.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--fine-tuning-on-classification-tasks"></a>

#### Fine-tuning on classification tasks

This repository is focused on training CLIP models. To fine-tune a *trained* zero-shot model on a downstream classification task such as ImageNet, please see [our other repository: WiSE-FT](https://github.com/mlfoundations/wise-ft). The [WiSE-FT repository](https://github.com/mlfoundations/wise-ft) contains code for our paper on [Robust Fine-tuning of Zero-shot Models](https://arxiv.org/abs/2109.01903), in which we introduce a technique for fine-tuning zero-shot models while preserving robustness under distribution shift.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--data"></a>

#### Data

To download datasets as webdataset, we recommend [img2dataset](https://github.com/rom1504/img2dataset)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--conceptual-captions"></a>

##### Conceptual Captions

See [cc3m img2dataset example](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--yfcc-and-other-datasets"></a>

##### YFCC and other datasets

In addition to specifying the training data via CSV files as mentioned above, our codebase also supports [webdataset](https://github.com/webdataset/webdataset), which is recommended for larger scale datasets. The expected format is a series of `.tar` files. Each of these `.tar` files should contain two files for each training example, one for the image and one for the corresponding text. Both files should have the same name but different extensions. For instance, `shard_001.tar` could contain files such as `abc.jpg` and `abc.txt`. You can learn more about `webdataset` at [https://github.com/webdataset/webdataset](https://github.com/webdataset/webdataset). We use `.tar` files with 1,000 data points each, which we create using [tarp](https://github.com/webdataset/tarp).

You can download the YFCC dataset from [Multimedia Commons](http://mmcommons.org/).
Similar to OpenAI, we used a subset of YFCC to reach the aforementioned accuracy numbers.
The indices of images in this subset are in [OpenAI's CLIP repository](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md).



<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--training-clip"></a>

#### Training CLIP


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--install"></a>

##### Install

We advise you first create a virtual environment with:

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
```

You can then install openclip for training with `pip install 'open_clip_torch[training]'`.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--development"></a>

###### Development

If you want to make changes to contribute code, you can close openclip then run `make install` in openclip folder (after creating a virtualenv)

Install pip PyTorch as per https://pytorch.org/get-started/locally/

You may run `make install-training` to install training deps


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--testing"></a>

###### Testing

Test can be run with `make install-test` then `make test`

`python -m pytest -x -s -v tests -k "training"` to run a specific test

Running regression tests against a specific git revision or tag:
1. Generate testing data
    ```sh
    python ../../../../tests/ml/integration/chessclip/util_test.py --model RN50 RN101 --save_model_list models.txt --git_revision 9d31b2ec4df6d8228f370ff20c8267ec6ba39383
    ```
    **_WARNING_: This will invoke git and modify your working tree, but will reset it to the current state after data has been generated! \
    Don't modify your working tree while test data is being generated this way.**

2. Run regression tests
    ```sh
    OPEN_CLIP_TEST_REG_MODELS=models.txt python -m pytest -x -s -v -m regression_test
    ```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--sample-single-process-running-code"></a>

##### Sample single-process running code:

```bash
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```

Note: `imagenet-val` is the path to the *validation* set of ImageNet for zero-shot evaluation, not the training set!
You can remove this argument if you do not want to perform zero-shot evaluation on ImageNet throughout training. Note that the `val` folder should contain subfolders. If it doest not, please use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--multi-gpu-and-beyond"></a>

##### Multi-GPU and Beyond

This code has been battle tested up to 1024 A100s and offers a variety of solutions
for distributed training. We include native support for SLURM clusters.

As the number of devices used to train increases, so does the space complexity of
the the logit matrix. Using a naïve all-gather scheme, space complexity will be
`O(n^2)`. Instead, complexity may become effectively linear if the flags
`--gather-with-grad` and `--local-loss` are used. This alteration results in one-to-one
numerical results as the naïve method.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--epochs"></a>

###### Epochs

For larger datasets (eg Laion2B), we recommend setting --train-num-samples to a lower value than the full epoch, for example `--train-num-samples 135646078` to 1/16 of an epoch in conjunction with --dataset-resampled to do sampling with replacement. This allows having frequent checkpoints to evaluate more often.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--patch-dropout"></a>

###### Patch Dropout

<a href="https://arxiv.org/abs/2212.00794">Recent research</a> has shown that one can dropout half to three-quarters of the visual tokens, leading to up to 2-3x training speeds without loss of accuracy.

You can set this on your visual transformer config with the key `patch_dropout`.

In the paper, they also finetuned without the patch dropout at the end. You can do this with the command-line argument `--force-patch-dropout 0.`


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--single-node"></a>

###### Single-Node

We make use of `torchrun` to launch distributed jobs. The following launches a
a job on a node of 4 GPUs:

```bash
cd open_clip/src
torchrun --nproc_per_node 4 -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--multi-node"></a>

###### Multi-Node

The same script above works, so long as users include information about the number
of nodes and host node.

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--slurm"></a>

###### SLURM

This is likely the easiest solution to utilize. The following script was used to
train our largest models:

```bash
#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition PARTITION_NAME

eval "$(/path/to/conda/bin/conda shell.bash hook)" # init conda
conda activate open_clip
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /shared/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="/data/LAION-400M/{00000..41455}.tar" \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=8 \
    --model ViT-B-32 \
    --name "ViT-B-32-Vanilla" \
    --seed 0 \
    --local-loss \
    --gather-with-grad
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--resuming-from-a-checkpoint"></a>

##### Resuming from a checkpoint:

```bash
python -m training.main \
    --train-data="/path/to/train_data.csv" \
    --val-data="/path/to/validation_data.csv"  \
    --resume /path/to/checkpoints/epoch_K.pt
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--training-with-pre-trained-language-models-as-text-encoder"></a>

##### Training with pre-trained language models as text encoder:

If you wish to use different language models as the text encoder for CLIP you can do so by using one of the Hugging Face model configs in ```src/open_clip/model_configs``` and passing in it's tokenizer as the ```--model``` and ```--hf-tokenizer-name``` parameters respectively. Currently we only support RoBERTa ("test-roberta" config), however adding new models should be trivial. You can also determine how many layers, from the end, to leave unfrozen with the ```--lock-text-unlocked-layers``` parameter. Here's an example command to train CLIP with the RoBERTa LM that has it's last 10 layers unfrozen:
```bash
python -m training.main \
         --train-data="pipe:aws s3 cp s3://s-mas/cc3m/{00000..00329}.tar -" \
         --train-num-samples 3000000 \
         --val-data="pipe:aws s3 cp s3://s-mas/cc3m/{00330..00331}.tar -" \
         --val-num-samples 10000 \
         --dataset-type webdataset \
         --batch-size 256 \
         --warmup 2000 \
         --epochs 10 \
         --lr 5e-4 \
         --precision amp \
         --workers 6 \
         --model "roberta-ViT-B-32" \
         --lock-text \
         --lock-text-unlocked-layers 10 \
         --name "10_unfrozen" \
         --report-to "tensorboard" \
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--loss-curves"></a>

##### Loss Curves

When run on a machine with 8 GPUs the command should produce the following training curve for Conceptual Captions:

![CLIP zero shot training curve](https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/clip_zeroshot.png)

More detailed curves for Conceptual Captions are given at [кривые обучения](docs/ml/chessclip/clip_conceptual_captions.md).

When training a RN50 on YFCC the same hyperparameters as above are used, with the exception of `lr=5e-4` and `epochs=32`.

Note that to use another model, like `ViT-B/32` or `RN50x4` or `RN50x16` or `ViT-B/16`, specify with `--model RN50x4`.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--launch-tensorboard"></a>

##### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--evaluation--zero-shot"></a>

#### Evaluation / Zero-Shot


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--evaluating-local-checkpoint"></a>

##### Evaluating local checkpoint:

```bash
python -m training.main \
    --val-data="/path/to/validation_data.csv"  \
    --model RN101 \
    --pretrained /path/to/checkpoints/epoch_K.pt
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--evaluating-hosted-pretrained-checkpoint-on-imagenet-zero-shot-prediction"></a>

##### Evaluating hosted pretrained checkpoint on ImageNet zero-shot prediction:

```bash
python -m training.main \
    --imagenet-val /path/to/imagenet/validation \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32
```


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--pretrained-model-details"></a>

#### Pretrained model details


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--laion-400m---httpslaionailaion-400-open-dataset"></a>

##### LAION-400M - https://laion.ai/laion-400-open-dataset

We are working on reproducing OpenAI's ViT results with the comparably sized (and open) LAION-400M dataset. Trained
weights may be found in release [v0.2](https://github.com/mlfoundations/open_clip/releases/tag/v0.2-weights).

The LAION400M weights have been trained on the JUWELS supercomputer (see acknowledgements section below).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b32-224x224"></a>

###### ViT-B/32 224x224

We replicate OpenAI's results on ViT-B/32, reaching a top-1 ImageNet-1k zero-shot accuracy of 62.96%.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot.png" width="700">

__Zero-shot comparison (courtesy of Andreas Fürst)__
<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_openai_compare_b32.jpg" width="700">

ViT-B/32 was trained with 128 A100 (40 GB) GPUs for ~36 hours, 4600 GPU-hours. The per-GPU batch size was 256 for a global batch size of 32768. 256 is much lower than it could have been (~320-384) due to being sized initially before moving to 'local' contrastive loss.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b16-224x224"></a>

###### ViT-B/16 224x224

The B/16 LAION400M training reached a top-1 ImageNet-1k zero-shot validation score of 67.07.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_b16.png" width="700">

This was the first major train session using the updated webdataset 0.2.x code. A bug was found that prevented shards from being shuffled properly between nodes/workers each epoch. This was fixed part way through training (epoch 26) but likely had an impact.

ViT-B/16 was trained with 176 A100 (40 GB) GPUS for ~61 hours, 10700 GPU-hours. Batch size per GPU was 192 for a global batch size of 33792.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b16-240x240"></a>

###### ViT-B/16+ 240x240

The B/16+ 240x240 LAION400M training reached a top-1 ImageNet-1k zero-shot validation score of 69.21.

This model is the same depth as the B/16, but increases the
  * vision width from 768 -> 896
  * text width from 512 -> 640
  * the resolution 224x224 -> 240x240 (196 -> 225 tokens)

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_b16_plus_240.png" width="700">

Unlike the B/16 run above, this model was a clean run with no dataset shuffling issues.

ViT-B/16+ was trained with 224 A100 (40 GB) GPUS for ~61 hours, 13620 GPU-hours. Batch size per GPU was 160 for a global batch size of 35840.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-l14-224x224"></a>

###### ViT-L/14 224x224

The L/14 LAION-400M training reached a top-1 ImageNet-1k zero-shot validation score of 72.77.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_l14.png" width="700">

ViT-L/14 was trained with 400 A100 (40 GB) GPUS for ~127 hours, 50800 GPU-hours. Batch size per GPU was 96 for a global batch size of 38400. Grad checkpointing was enabled.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--laion-2b-en---httpslaionailaion-5b-a-new-era-of-open-large-scale-multi-modal-datasets"></a>

##### LAION-2B (en) - https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/

A ~2B sample subset of LAION-5B with english captions (https://huggingface.co/datasets/laion/laion2B-en)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b32-224x224-1"></a>

###### ViT-B/32 224x224
A ViT-B/32 trained on LAION-2B, reaching a top-1 ImageNet-1k zero-shot accuracy of 65.62%.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion2b_clip_zeroshot_b32.png" width="700">

ViT-B/32 was trained with 112 A100 (40 GB) GPUs. The per-GPU batch size was 416 for a global batch size of 46592. Compute generously provided by [stability.ai](https://stability.ai/).

A second iteration of B/32 was trained on stability.ai cluster with a larger global batch size and learning rate, hitting 66.6% top-1. See https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-l14-224x224-1"></a>

###### ViT-L/14 224x224

A ViT-L/14 with a 75.3% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K

These weights use a different dataset mean and std than others. Instead of using the OpenAI mean & std, inception style normalization `[-1, 1]` is used via a mean and std of `[0.5, 0.5, 0.5]`. This is handled automatically if using `open_clip.create_model_and_transforms` from pretrained weights.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-h14-224x224"></a>

###### ViT-H/14 224x224

A ViT-H/14 with a 78.0% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-g14-224x224"></a>

###### ViT-g/14 224x224

A ViT-g/14 with a 76.6% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K

This model was trained with a shorted schedule than other LAION-2B models with 12B samples seen instead of 32+B. It matches LAION-400M training in samples seen. Many zero-shot results are lower as a result, but despite this it performs very well in some OOD zero-shot and retrieval tasks.



<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b32-roberta-base"></a>

###### ViT-B/32 roberta base

A ViT-B/32 with roberta base encoder with a 61.7% top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k
This is the first openclip model using a HF text tower. It has better performance on a range of tasks compared to the standard text encoder, see [metrics](https://huggingface.co/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k/blob/main/unknown.png)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-b32-xlm-roberta-base"></a>

###### ViT-B/32 xlm roberta base

A ViT-B/32 with xlm roberta base encoder with a 62.33% top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k
This is the first openclip model trained on the full laion5B dataset; hence the first multilingual clip trained with openclip. It has better performance on a range of tasks compared to the standard text encoder, see [metrics](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/blob/main/metrics.png)
A preliminary multilingual evaluation was run: 43% on imagenet1k italian (vs 21% for english B/32), 37% for imagenet1k japanese (vs 1% for english B/32 and 50% for B/16 clip japanese). It shows the multilingual property is indeed there as expected. Larger models will get even better performance.


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--vit-h14-xlm-roberta-large"></a>

###### ViT-H/14 xlm roberta large

A ViT-H/14 with xlm roberta large encoder with a 77.0% (vs 78% for the english equivalent) top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k

This model was trained following the [LiT](https://arxiv.org/abs/2111.07991) methodology: the image tower was frozen (initialized from english openclip ViT-H/14), the text tower was initialized from [xlm roberta large](https://huggingface.co/xlm-roberta-large) and unfrozen. This reduced training cost by a 3x factor.

See full english [metrics](https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/resolve/main/results_xlm_roberta_large.png)

On zero shot classification on imagenet with translated prompts this model reaches:

* 56% in italian (vs 21% for https://github.com/clip-italian/clip-italian)
* 53% in japanese (vs 54.6% for https://github.com/rinnakk/japanese-clip)
* 55.7% in chinese (to be compared with https://github.com/OFA-Sys/Chinese-CLIP)



<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--yfcc-15m"></a>

###### YFCC-15M

Below are checkpoints of models trained on YFCC-15M, along with their zero-shot top-1 accuracies on ImageNet and ImageNetV2. These models were trained using 8 GPUs and the same hyperparameters described in the "Sample running code" section, with the exception of `lr=5e-4` and `epochs=32`.

* [ResNet-50](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt) (32.7% / 27.9%)
* [ResNet-101](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt) (34.8% / 30.0%)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--cc12m---httpsgithubcomgoogle-research-datasetsconceptual-12m"></a>

###### CC12M - https://github.com/google-research-datasets/conceptual-12m

* [ResNet-50](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt) (36.45%)


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--pretrained-model-interface"></a>

##### Pretrained Model Interface

We offer a simple model interface to instantiate both pre-trained and untrained models.

NOTE: Many existing checkpoints use the QuickGELU activation from the original OpenAI models. This activation is actually less efficient than native torch.nn.GELU in recent versions of PyTorch. The model defaults are now nn.GELU, so one should use model definitions with `-quickgelu` postfix for the OpenCLIP pretrained weights. All OpenAI pretrained weights will always default to QuickGELU. One can also use the non `-quickgelu` model definitions with pretrained weights using QuickGELU but there will be an accuracy drop, for fine-tune that will likely vanish for longer runs.

Future trained models will use nn.GELU.

```python
>>> import open_clip
>>> open_clip.list_pretrained()
[('RN50', 'openai'),
 ('RN50', 'yfcc15m'),
 ('RN50', 'cc12m'),
 ('RN50-quickgelu', 'openai'),
 ('RN50-quickgelu', 'yfcc15m'),
 ('RN50-quickgelu', 'cc12m'),
 ('RN101', 'openai'),
 ('RN101', 'yfcc15m'),
 ('RN101-quickgelu', 'openai'),
 ('RN101-quickgelu', 'yfcc15m'),
 ('RN50x4', 'openai'),
 ('RN50x16', 'openai'),
 ('RN50x64', 'openai'),
 ('ViT-B-32', 'openai'),
 ('ViT-B-32', 'laion400m_e31'),
 ('ViT-B-32', 'laion400m_e32'),
 ('ViT-B-32', 'laion2b_e16'),
 ('ViT-B-32', 'laion2b_s34b_b79k'),
 ('ViT-B-32-quickgelu', 'openai'),
 ('ViT-B-32-quickgelu', 'laion400m_e31'),
 ('ViT-B-32-quickgelu', 'laion400m_e32'),
 ('ViT-B-16', 'openai'),
 ('ViT-B-16', 'laion400m_e31'),
 ('ViT-B-16', 'laion400m_e32'),
 ('ViT-B-16-plus-240', 'laion400m_e31'),
 ('ViT-B-16-plus-240', 'laion400m_e32'),
 ('ViT-L-14', 'openai'),
 ('ViT-L-14', 'laion400m_e31'),
 ('ViT-L-14', 'laion400m_e32'),
 ('ViT-L-14', 'laion2b_s32b_b82k'),
 ('ViT-L-14-336', 'openai'),
 ('ViT-H-14', 'laion2b_s32b_b79k'),
 ('ViT-g-14', 'laion2b_s12b_b42k'),
 ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
 ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
 ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),]

>>> model, train_transform, eval_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
```

<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--gradient-accumulation"></a>

##### Gradient accumulation

To simulate larger batches use `--accum-freq k`. If per gpu batch size, `--batch-size`, is `m`, then the effective batch size will be `k * m * num_gpus`.

When increasing `--accum-freq` from its default of 1, samples/s will remain approximately constant (batch size will double, as will time-per-batch). It is recommended to use other features to reduce batch size such as `--grad-checkpointing --local-loss --gather-with-grad` before increasing `--accum-freq`. `--accum-freq` can be used in addition to these features.

Instead of 1 forward pass per example, there are now 2 forward passes per-example. However, the first is done with `torch.no_grad`.

There is some additional GPU memory required --- the features and data from all `m` batches are stored in memory.

There are also `m` loss computations instead of the usual 1.

For more information see Cui et al. (https://arxiv.org/abs/2112.09331) or Pham et al. (https://arxiv.org/abs/2111.10050).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--scaling-trends"></a>

#### Scaling trends

The plot below shows how zero-shot performance of CLIP models varies as we scale the number of samples used for training. Zero-shot performance increases steadily for both ImageNet and [ImageNetV2](https://arxiv.org/abs/1902.10811), and is far from saturated at ~15M samples.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/scaling.png" width="700">


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--why-are-low-accuracy-clip-models-interesting"></a>

#### Why are low-accuracy CLIP models interesting?

**TL;DR:** CLIP models have high effective robustness, even at small scales.

CLIP models are particularly intriguing because they are more robust to natural distribution shifts (see Section 3.3 in the [CLIP paper](https://arxiv.org/abs/2103.00020)).
This phenomena is illustrated by the figure below, with ImageNet accuracy on the x-axis
and [ImageNetV2](https://arxiv.org/abs/1902.10811) (a reproduction of the ImageNet validation set with distribution shift) accuracy on the y-axis.
Standard training denotes training on the ImageNet train set and the CLIP zero-shot models
are shown as stars.

![CLIP scatter plot](https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/effective_robustness.png)

As observed by [Taori et al., 2020](https://arxiv.org/abs/2007.00644) and [Miller et al., 2021](https://arxiv.org/abs/2107.04649), the in-distribution
and out-of-distribution accuracies of models trained on ImageNet follow a predictable linear trend (the red line in the above plot). *Effective robustness*
quantifies robustness as accuracy beyond this baseline, i.e., how far a model lies above the red line. Ideally a model would not suffer from distribution shift and fall on the y = x line ([trained human labelers are within a percentage point of the y = x line](http://proceedings.mlr.press/v119/shankar20c.html)).

Even though the CLIP models trained with
this codebase achieve much lower accuracy than those trained by OpenAI, our models still lie on the same
trend of improved effective robustness (the purple line). Therefore, we can study what makes
CLIP robust without requiring industrial-scale compute.

For more information on effective robustness, please see:

- [Recht et al., 2019](https://arxiv.org/abs/1902.10811).
- [Taori et al., 2020](https://arxiv.org/abs/2007.00644).
- [Miller et al., 2021](https://arxiv.org/abs/2107.04649).

To know more about the factors that contribute to CLIP's robustness refer to [Fang et al., 2022](https://arxiv.org/abs/2205.01397).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--acknowledgments"></a>

#### Acknowledgments

We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--the-team"></a>

#### The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

[Gabriel Ilharco*](http://gabrielilharco.com/), [Mitchell Wortsman*](https://mitchellnw.github.io/), [Nicholas Carlini](https://nicholas.carlini.com/), [Rohan Taori](https://www.rohantaori.com/), [Achal Dave](http://www.achaldave.com/), [Vaishaal Shankar](http://vaishaal.com/), [John Miller](https://people.eecs.berkeley.edu/~miller_john/), [Hongseok Namkoong](https://hsnamkoong.github.io/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

Special thanks to [Jong Wook Kim](https://jongwook.kim/) and [Alec Radford](https://github.com/Newmu) for help with reproducing CLIP!


<a id="manual-ml-synthesis-pipeline-rag-module-chessclip-chessclip--citing"></a>

#### Citing

If you found this repository useful, please consider citing:
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)


<a id="manual-ml-synthesis-pipeline-scripts-extra-features"></a>

### ml/synthesis_pipeline/scripts/extra_features

Рабочий каталог для команд этого руководства — `ml/synthesis_pipeline/scripts/extra_features`.

<a id="manual-ml-synthesis-pipeline-scripts-extra-features--модуль-extra-features"></a>

### Модуль Extra Features

Пакет вычисления структурированных признаков позиции для генерации черновых описаний (Qwen) и последующей стандартизации (Claude).


<a id="manual-ml-synthesis-pipeline-scripts-extra-features--структура-модуля"></a>

#### Структура модуля

* `stockfish_eval.py` - расчет MultiPV, дельты сантипешек (`centipawn_loss`) и фиксация упущенного мата.
* `make_nag.py` - классификация хода по стандарту Lichess/Chess.com (`brilliant`, `great`, `best`, `blunder`, etc.).
* `board_flags.py` - расчет предикатов `python-chess` (связки, висячие фигуры, тип движения, баланс центра, детекция жертвы).
* `run_extra_features.py` - оркестратор пакета. Читает `data/01_raw/raw_dataset.jsonl` и сохраняет результат в `data/02_engine_analyzed/master_analyzed.jsonl`.


<a id="manual-ml-synthesis-pipeline-scripts-extra-features--запуск-из-корня-репозитория"></a>

#### Запуск из корня репозитория

```bash
STOCKFISH_PATH="/home/user/SFEDUCASTLING/Qwen3-chess-coach/stockfish/stockfish-ubuntu-x86-64-avx2" python scripts/extra_features/run_extra_features.py
```


<a id="manual-ml-synthesis-pipeline-scripts-extra-features--указание-пути-к-stockfish"></a>

#### Указание пути к Stockfish

По умолчанию путь определяется автоматически:
1. Значение переменной `STOCKFISH_PATH`
2. Поиск через `which stockfish`
3. Фоллбэк `/usr/games/stockfish`

Для ручного переопределения:
```bash
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish
python scripts/extra_features/run_extra_features.py
```

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
