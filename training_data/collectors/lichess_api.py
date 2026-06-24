"""
Клиент для скачивания шахматных партий с Lichess API.

Используются эндпоинты:
  GET /api/account               — проверка токена (требует токен)
  GET /api/games/user/{username} — скачивание партий (публичный)
  GET /api/player/top/{perf}     — топ игроков по рейтингу (публичный)

Формат сохранения: JSONL (одна JSON-строка = одна игра).

Пример использования:
  from training_data.collectors.lichess_api import LichessClient

  client = LichessClient()
  games = client.fetch_games("alireza2003", max_games=100, perf_type="blitz")
  client.save_jsonl(games, "data/alireza_blitz.jsonl")

CLI:
  python -m training_data.collectors.lichess_api --username alireza2003 --max 10
"""

import os
import json
import time
import logging
import http.client
import ssl
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "lichess.org"


class LichessClient:
    """Клиент для скачивания шахматных партий с Lichess API."""

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: Персональный токен Lichess (необязателен для публичных запросов).
                   Если не указан, читается из переменной окружения LICHESS_API_TOKEN.
        """
        self.token = token or os.getenv("LICHESS_API_TOKEN")
        # Отключаем проверку SSL-сертификата (нужно на Windows с корпоративным антивирусом)
        self.ctx = ssl._create_unverified_context()

    def _request(self, path: str, params: Optional[dict] = None) -> str:
        """Выполняет HTTPS GET запрос и возвращает тело ответа.

        Args:
            path: Путь к эндпоинту (начинается с /).
            params: Query-параметры.

        Returns:
            Тело ответа в виде строки.

        Raises:
            RuntimeError: если статус ответа не 200.
        """
        conn = http.client.HTTPSConnection(BASE_URL, context=self.ctx, timeout=30)
        headers = {"Accept": "application/x-ndjson"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{path}?{qs}"
        else:
            url = path

        conn.request("GET", url, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

        return body

    # ── 020: Проверка соединения ──

    def test_connection(self) -> dict:
        """Проверяет, что токен работает, через GET /api/account.

        Returns:
            Информация об аккаунте: username, id, createdAt, profile и т.д.

        Raises:
            RuntimeError: если токен невалидный (401).
        """
        body = self._request("/api/account")
        account = json.loads(body)
        logger.info("Подключение к Lichess API — авторизован как %s", account["username"])
        return account

    # ── 021: Скачивание партий ──

    def fetch_games(
        self,
        username: str,
        max_games: int = 100,
        rated: Optional[bool] = None,
        perf_type: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
    ) -> list[dict]:
        """Скачивает последние партии игрока с Lichess.

        GET /api/games/user/{username} → NDJSON → list[dict].

        Args:
            username: Имя пользователя на Lichess (например "alireza2003").
            max_games: Максимальное количество партий для скачивания.
            rated: True — только рейтинговые, False — только тренировочные,
                   None — все подряд.
            perf_type: Тип/скорость игры: "bullet", "blitz", "rapid",
                       "classical", "ultraBullet" и т.д.
            since: Unix timestamp в миллисекундах — скачать партии ПОСЛЕ этой даты.
            until: Unix timestamp в миллисекундах — скачать партии ДО этой даты.

        Returns:
            Список словарей с данными игр. Каждая игра содержит:
            id, rated, perf, speed, players (с рейтингом), moves, status,
            winner, createdAt и т.д.
        """
        params = {"max": max_games}
        if rated is not None:
            params["rated"] = str(rated).lower()
        if perf_type:
            params["perfType"] = perf_type
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        path = f"/api/games/user/{username}"
        logger.info("Скачиваю до %d партий с %s%s ...", max_games, BASE_URL, path)

        body = self._request(path, params)

        games = []
        for line in body.strip().split("\n"):
            if line:
                games.append(json.loads(line))
                if len(games) >= max_games:
                    break

        logger.info("Скачано %d партий для '%s'", len(games), username)
        return games

    def fetch_by_rating(
        self,
        min_rating: int = 1500,
        max_rating: int = 2500,
        max_games: int = 1000,
        perf_type: str = "blitz",
        max_players: int = 50,
    ) -> list[dict]:
        """Скачивает партии игроков в заданном диапазоне рейтинга.

        Стратегия:
          1. Получаем топ-N игроков с лидерборда Lichess.
          2. Фильтруем по рейтингу (min_rating .. max_rating).
          3. С каждого отфильтрованного игрока скачиваем его последние партии.
          4. Небольшая задержка между игроками, чтобы не получить бан по rate limit.

        Lichess API не умеет фильтровать партии напрямую по рейтингу,
        поэтому приходится действовать в обход.

        Args:
            min_rating: Минимальный рейтинг игрока (включительно).
            max_rating: Максимальный рейтинг игрока (включительно).
            max_games: Сколько всего партий собрать (суммарно по всем игрокам).
            perf_type: Тип игры для лидерборда (blitz, rapid, classical).
            max_players: Сколько топ-игроков запросить с лидерборда.

        Returns:
            Список словарей с данными игр (тот же формат, что и fetch_games).
        """
        # Шаг 1: получаем топ игроков
        body = self._request(f"/api/player/top/{perf_type}", {"nb": max_players})
        top = json.loads(body)

        # Шаг 2: фильтруем по рейтингу
        target = []
        for p in top.get("users", []):
            rating = p["perfs"][perf_type]["rating"]
            if min_rating <= rating <= max_rating:
                target.append((p["username"], rating))

        if not target:
            logger.warning(
                "Не найдено игроков с рейтингом %d-%d для %s",
                min_rating, max_rating, perf_type,
            )
            return []

        logger.info(
            "Найдено %d игроков с рейтингом %d-%d (например %s с рейтингом %d)",
            len(target), min_rating, max_rating,
            target[0][0], target[0][1],
        )

        # Шаг 3: скачиваем партии с каждого игрока
        per_player = max(1, max_games // len(target))
        all_games = []

        for username, rating in target:
            if len(all_games) >= max_games:
                break
            try:
                games = self.fetch_games(username, max_games=per_player,
                                         perf_type=perf_type)
                all_games.extend(games)
                logger.info("Скачано %d партий с %s (рейтинг %d)",
                            len(games), username, rating)
            except RuntimeError as e:
                logger.warning("Пропускаю %s: %s", username, e)
                continue
            time.sleep(0.5)  # чтобы не заблокировали за частые запросы

        logger.info("Всего собрано: %d партий", len(all_games))
        return all_games

    # ── Сохранение / Загрузка JSONL ──

    def save_jsonl(self, games: list[dict], output_path: str | Path) -> Path:
        """Сохраняет список игр в JSONL-файл.

        JSONL = одна JSON-строка на одну игру.
        Преимущества: можно читать построчно, удобно для pandas и HuggingFace Datasets.

        Args:
            games: Список игр (словарей).
            output_path: Путь к выходному .jsonl файлу.

        Returns:
            Путь к сохранённому файлу.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for game in games:
                f.write(json.dumps(game, ensure_ascii=False) + "\n")

        logger.info("Сохранено %d игр в %s", len(games), output_path)
        return output_path

    def load_jsonl(self, input_path: str | Path) -> list[dict]:
        """Загружает игры из JSONL-файла.

        Args:
            input_path: Путь к .jsonl файлу.

        Returns:
            Список словарей с данными игр.
        """
        games = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        logger.info("Загружено %d игр из %s", len(games), input_path)
        return games


# ── Точка входа для командной строки ──

def main():
    """CLI для быстрого тестирования и сбора данных.

    Примеры:
      # Проверить токен
      python lichess_api.py --token lip_xxx --test

      # Скачать 10 партий пользователя
      python lichess_api.py --username alireza2003

      # Скачать 100 рейтинговых blitz-партий
      python lichess_api.py --username alireza2003 --max 100 --rated --perf-type blitz

      # Скачать партии игроков с рейтингом 2000-2500
      python lichess_api.py --by-rating 2000 2500 --max 500 --perf-type rapid
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(
        description="Скачивание шахматных партий с Lichess API",
        epilog="Получить токен: https://lichess.org/account/oauth/token",
    )
    parser.add_argument("--token", help="API токен Lichess")
    parser.add_argument("--username", help="Чьи партии скачать")
    parser.add_argument("--test", action="store_true",
                        help="Проверить соединение с API")
    parser.add_argument("--output", "-o", default="games.jsonl",
                        help="Выходной файл (.jsonl)")
    parser.add_argument("--max", type=int, default=10,
                        help="Максимум партий")
    parser.add_argument("--perf-type", default=None,
                        help="Тип игры (blitz, rapid, classical, ...)")
    parser.add_argument("--rated", action=argparse.BooleanOptionalAction,
                        help="Только рейтинговые партии")
    parser.add_argument("--by-rating", nargs=2, type=int,
                        metavar=("MIN", "MAX"),
                        help="Диапазон рейтинга, например --by-rating 2000 2500")

    args = parser.parse_args()
    client = LichessClient(token=args.token)

    if args.test:
        info = client.test_connection()
        print(f"OK — {info['username']} (id={info['id']})")
        return

    if args.by_rating:
        games = client.fetch_by_rating(
            args.by_rating[0], args.by_rating[1],
            max_games=args.max,
            perf_type=args.perf_type or "blitz",
        )
    elif args.username:
        games = client.fetch_games(
            args.username, max_games=args.max,
            rated=args.rated, perf_type=args.perf_type,
        )
    else:
        parser.print_help()
        return

    path = client.save_jsonl(games, args.output)
    print(f"Сохранено {len(games)} игр в {path}")


if __name__ == "__main__":
    main()
