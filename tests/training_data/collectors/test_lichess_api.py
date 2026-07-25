"""Тест скачивания партий с Lichess API."""

import sys
import json
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training_data.collectors.lichess_api import LichessClient


def main():
    client = LichessClient()

    # Скачиваем 5 blitz-партий Alireza
    games = client.fetch_games("alireza2003", max_games=5, perf_type="blitz")

    print(f"Скачано игр: {len(games)}\n")

    for i, game in enumerate(games, 1):
        print(f"=== Игра {i} ===")
        print(json.dumps(game, indent=2, ensure_ascii=False))
        print()

    # Сохраняем
    path = client.save_jsonl(games, "alireza_blitz.jsonl")
    print(f"Сохранено в {path}")


if __name__ == "__main__":
    main()
