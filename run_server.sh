#!/bin/bash
set -e

echo "=== SFEDUCASTLING — установка и запуск ==="

cd "$(dirname "$0")"

# Виртуальное окружение
if [ ! -d ".venv" ]; then
    echo "Создаю виртуальное окружение..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Зависимости
echo "Устанавливаю зависимости..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Запуск
echo "Запускаю сервер на порту 8000..."
cd backend
python app.py
