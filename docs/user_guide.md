# Руководство пользователя SFEDUCASTLING

Проект — шахматный помощник: тактические паззлы, учебный курс по правилам и дебютам,
анализ партий, тренерские объяснения ходов (ассистент в реальном времени).

---

## Роли и права доступа

| Роль      | Гостевая активность | Доступ |
|-----------|---------------------|--------|
| `guest`   | Паззлы, анализ, чат с ассистентом, level-test | Без входа в аккаунт |
| `learner` | Всё, что у гостя + учебные модули | Регистрируется на вкладке «Обучение» |
| `admin`   | Всё + админ-эндпоинты (просмотр пользователей, статистика) | Назначается в БД |

Правила:
- Единственный источник прав — колонка `users.role` (`guest`/`learner`/`admin`).
  Флаг `is_admin` выводится из неё и в БД не хранится (колонка удалена миграцией `a1b2c3d4e5f6`).
- Учебные модули (`/api/training/*`) доступны **только зарегистрированным** (learner/admin).
  Гостю показывается плашка «Обучение доступно после регистрации».
- Сессия живёт **24 часа**: access-токен и refresh живут по суткам, после чего
  пользователь автоматически выходит из аккаунта.
- Логин: только латиница, цифры и символы `_` / `-`. Кириллица не принимается.
- Пароль: минимум 8 символов, нужны **и буквы, и цифры**. Хранится только bcrypt-хеш (cost 12).

---

## Запуск

### 1. Backend (Windows / 127.0.0.1:8005)

```bash
# окружение (пример venv)
python -m venv ml-env
ml-env\Scripts\activate

pip install -r requirements.txt

# конфигурация: .env в корне проекта (DATABASE_URL, JWT_SECRET, MAIA3_*, GIGACHESS_BASE_URL …)
# см. backend/config/settings.py

# миграции БД
python -m alembic upgrade head

# наполнение учебного курса (идемпотентно: повторные запуски обновляют данные)
python -m backend.services.training_seed

# запуск API (раздаёт фронт из frontend/dist)
python backend/app.py
```

### 2. Frontend (dev-режим; продакшен-сборка раздаётся самим backend)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, /api проксируется на :8005
npm run build    # сборка в dist/ (обновляет то, что отдаёт backend)
```

### 3. LLM-сервис (Qwen3-8B через vLLM, порт 8000)

```bash
bash start_llm.sh
```

---

## Тесты

```bash
# через актуальный venv (в системном Python 3.14 могут отсутствовать зависимости)
ml-env\Scripts\python.exe -m pytest tests/ -q
```

Прогон требует доступ к локальной БД (интеграционные тесты).

---

## Ветки и стейджинг (dev / feature-*)

За основу работы берётся ветка `dev` (стейдж). Новые фичи ведём в ветках `feature-*`,
которые вливаются в `dev`, а `dev` — в `main` только после проверки.

```bash
# создать/обновить стейдж-ветку
git checkout -b dev
git push -u origin dev

# фича-ветка от dev
git checkout -b feature/debuty-tasks dev

# после проверки — слияние
git checkout dev
git merge feature/debuty-tasks
git push origin dev
```

> Git-операции выполняет человек (см. процесс в репозитории), ИИ-ассистенту `git` не запускается.

CI:
- GitHub Actions × CodeQL (`codeql.yml`) — анализ безопасности для Python и JS/TS
  на `push` в `main`/`dev`/`feature-*` и на `pull_request` в `main`.