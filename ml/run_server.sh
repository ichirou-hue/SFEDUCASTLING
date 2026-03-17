#!/bin/bash
# Полная инструкция: запуск всего пайплайна на сервере
# Копируй этот файл на сервер и запускай: bash run_server.sh

echo "=========================================="
echo "SFEDUCASTLING — Установка на сервере"
echo "=========================================="

# 1. Создать виртуальное окружение
echo "[1/6] Создание venv..."
python3 -m venv venv
source venv/bin/activate

# 2. Установить зависимости
echo "[2/6] Установка Python-зависимостей..."
pip install --upgrade pip
pip install -r ML/requirements.txt
pip install -r backend/requirements.txt

# 3. Скачать Stockfish (если нет)
if [ ! -f "./stockfish" ]; then
    echo "[3/6] Скачивание Stockfish..."
    wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
    tar xf stockfish-ubuntu-x86-64-avx2.tar
    cp stockfish/stockfish-ubuntu-x86-64-avx2 ./stockfish
    chmod +x ./stockfish
    rm -rf stockfish-ubuntu-x86-64-avx2.tar stockfish/
    echo "       Stockfish установлен."
else
    echo "[3/6] Stockfish уже есть."
fi

# 4. Скачать PGN с Lichess (если нет)
if [ ! -f "./lichess_games.pgn" ]; then
    echo "[4/6] Скачивание партий с Lichess (январь 2024, ~2GB)..."
    wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
    zstd -d lichess_db_standard_rated_2024-01.pgn.zst -o lichess_games.pgn
    rm lichess_db_standard_rated_2024-01.pgn.zst
    echo "       Партии скачаны."
else
    echo "[4/6] PGN файл уже есть."
fi

# 5. Сгенерировать датасет
if [ ! -f "./dataset.jsonl" ]; then
    echo "[5/6] Генерация датасета (10000 позиций)..."
    python ML/prepare_dataset.py \
        --pgn lichess_games.pgn \
        --stockfish ./stockfish \
        --output dataset.jsonl \
        --limit 10000 \
        --depth 18
    echo "       Датасет готов."
else
    echo "[5/6] Датасет уже существует."
fi

# 6. Обучение модели
echo "[6/6] Запуск обучения QLoRA..."
python ML/train_qlora.py \
    --dataset dataset.jsonl \
    --output ./chess-llm-lora \
    --epochs 3

echo ""
echo "=========================================="
echo "ГОТОВО! Теперь запусти бэкенд:"
echo "  python backend/app.py"
echo "=========================================="
