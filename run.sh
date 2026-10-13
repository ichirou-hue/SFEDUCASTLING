#!/usr/bin/env bash
# SFEDUCASTLING — установка и запуск (Linux / macOS).
# Использование:  ./run.sh
set -e
cd "$(dirname "$0")"

echo "=== SFEDUCASTLING — установка и запуск ==="

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

# 1. Виртуальное окружение
if [ ! -d ".venv" ]; then
    echo "Создаю виртуальное окружение..."
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Зависимости (устанавливаем только если чего-то не хватает)
# Maia3 лежит git-сабмодулем: инициализируем до pip install, иначе
# установка -e ./maia3 из requirements.txt упадёт на пустом каталоге.
# Нефатально: без сети/гита движок просто останется недоступен.
if [ -d ".git" ] && [ ! -f "maia3/pyproject.toml" ]; then
    echo "Инициализирую сабмодуль maia3..."
    git submodule update --init maia3 2>/dev/null || true
fi
if ! python -c "import fastapi, alembic, sqlalchemy, aiosqlite, dotenv" 2>/dev/null; then
    echo "Устанавливаю зависимости..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi
# Maia3 ставится из сабмодуля (-e ./maia3): доустанавливаем отдельно,
# если основной набор зависимостей уже был, а движка в venv нет.
if [ -f "maia3/pyproject.toml" ] && ! python -c "import maia3.uci" 2>/dev/null; then
    echo "Устанавливаю движок maia3 из сабмодуля..."
    python -m pip install -e ./maia3
fi

# 3. Конфигурация
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Создан .env из .env.example"
fi

# 4. База данных (миграции)
echo "Обновляю базу данных..."
python -m scripts.init_db

# 4b. Наполнение вкладки «Обучение» (идемпотентно: не создаёт дубликатов)
echo "Наполняю курс обучения..."
python -m backend.services.training_seed

# 5. Сервер: http://127.0.0.1:8005
# Запускаем через -m: так корень проекта попадает в sys.path,
# и пакет `backend` импортируется корректно (при запуске
# `python backend/app.py` sys.path содержал бы только backend/).
echo "Запускаю сервер: http://127.0.0.1:8005"
exec python -m backend.app
