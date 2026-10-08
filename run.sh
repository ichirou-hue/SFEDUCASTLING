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
if ! python -c "import fastapi, alembic, sqlalchemy, aiosqlite, dotenv" 2>/dev/null; then
    echo "Устанавливаю зависимости..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

# 3. Конфигурация
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Создан .env из .env.example"
fi

# 4. База данных (миграции)
echo "Обновляю базу данных..."
python -m scripts.init_db

# 5. Сервер: http://127.0.0.1:8005
# Запускаем через -m: так корень проекта попадает в sys.path,
# и пакет `backend` импортируется корректно (при запуске
# `python backend/app.py` sys.path содержал бы только backend/).
echo "Запускаю сервер: http://127.0.0.1:8005"
exec python -m backend.app
