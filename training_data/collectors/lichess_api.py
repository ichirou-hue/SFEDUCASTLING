"""
Клиент для скачивания шахматных партий с Lichess API.

Использует http.client с IPv4 (на Windows IPv6/SNI падает с корпоративным
антивирусом — DECRYPTION_FAILED_OR_BAD_RECORD_MAC).

Формат сохранения: JSONL (одна JSON-строка = одна игра).
"""

import os
import json
import time
import logging
import socket
import ssl
import http.client
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "lichess.org"


def _resolve_ipv4() -> str:
    """Резолвит IPv4 адрес lichess.org."""
    return socket.getaddrinfo(BASE_URL, 443, socket.AF_INET)[0][4][0]


class LichessClient:
    """Клиент для скачивания шахматных партий с Lichess API."""

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: Персональный токен Lichess (необязателен для публичных запросов).
                   Если не указан, читается из переменной окружения LICHESS_API_TOKEN.
        """
        self.token = token or os.getenv("LICHESS_API_TOKEN")
        self._ip = _resolve_ipv4()

    def _request(self, path: str, params: Optional[dict] = None) -> str:
        """Выполняет HTTPS GET запрос через IPv4 + Host header."""
        headers = {"Accept": "application/x-ndjson", "Host": BASE_URL}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url = f"{path}?{qs}"
        else:
            url = path

        # Свежий SSL контекст на каждый запрос (антивирус режет кэшированные)
        ctx = ssl._create_unverified_context()
        conn = http.client.HTTPSConnection(self._ip, context=ctx, timeout=60)
        conn.request("GET", url, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

        return body

    # ── 020: Проверка соединения ──

    def test_connection(self) -> dict:
        """Проверяет, что токен работает, через GET /api/account."""
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
        """Скачивает последние партии игрока с Lichess."""
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

    def fetch_all_games(
        self,
        username: str,
        total_games: int = 10000,
        perf_type: Optional[str] = None,
        max_per_req: int = 500,
    ) -> list[dict]:
        """Скачивает все партии пагинацией через `until`."""
        all_games = []
        seen = set()
        until = None

        logger.info("Скачиваю %d игр для %s (пагинация)...", total_games, username)

        while len(all_games) < total_games:
            need = min(max_per_req, total_games - len(all_games))
            batch = self.fetch_games(username, max_games=need, perf_type=perf_type, until=until)

            if not batch:
                break

            new = [g for g in batch if g.get("id") and g["id"] not in seen]
            for g in new:
                seen.add(g["id"])
            all_games.extend(new)
            logger.info("Накоплено %d игр для %s", len(all_games), username)

            if not new:
                break

            until = new[-1].get("lastMoveAt") or new[-1].get("createdAt")
            if not until or until <= 0:
                break

            time.sleep(1.0)

        return all_games[:total_games]

    def fetch_by_rating(
        self,
        min_rating: int = 1500,
        max_rating: int = 2500,
        max_games: int = 1000,
        perf_type: str = "blitz",
        max_players: int = 50,
    ) -> list[dict]:
        """Скачивает партии игроков в заданном диапазоне рейтинга."""
        body = self._request(f"/api/player/top/{perf_type}", {"nb": max_players})
        top = json.loads(body)

        target = []
        for p in top.get("users", []):
            rating = p["perfs"][perf_type]["rating"]
            if min_rating <= rating <= max_rating:
                target.append((p["username"], rating))

        if not target:
            logger.warning("Не найдено игроков с рейтингом %d-%d для %s", min_rating, max_rating, perf_type)
            return []

        logger.info("Найдено %d игроков с рейтингом %d-%d (например %s %d)",
                    len(target), min_rating, max_rating, target[0][0], target[0][1])

        per_player = max(1, max_games // len(target))
        all_games = []

        for username, rating in target:
            if len(all_games) >= max_games:
                break
            try:
                games = self.fetch_games(username, max_games=per_player, perf_type=perf_type)
                all_games.extend(games)
                logger.info("Скачано %d партий с %s (рейтинг %d)", len(games), username, rating)
            except RuntimeError as e:
                logger.warning("Пропускаю %s: %s", username, e)
                continue
            time.sleep(0.5)

        logger.info("Всего собрано: %d партий", len(all_games))
        return all_games

    # ── Сохранение / Загрузка JSONL ──

    def save_jsonl(self, games: list[dict], output_path: str | Path) -> Path:
        """Сохраняет список игр в JSONL-файл."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for game in games:
                f.write(json.dumps(game, ensure_ascii=False) + "\n")

        logger.info("Сохранено %d игр в %s", len(games), output_path)
        return output_path

    def load_jsonl(self, input_path: str | Path) -> list[dict]:
        """Загружает игры из JSONL-файла."""
        games = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        logger.info("Загружено %d игр из %s", len(games), input_path)
        return games


# ── CLI ──

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(
        description="Скачивание шахматных партий с Lichess API",
        epilog="Получить токен: https://lichess.org/account/oauth/token",
    )
    parser.add_argument("--token", help="API токен Lichess")
    parser.add_argument("--username", help="Чьи партии скачать")
    parser.add_argument("--test", action="store_true", help="Проверить соединение с API")
    parser.add_argument("--output", "-o", default="games.jsonl", help="Выходной файл (.jsonl)")
    parser.add_argument("--max", type=int, default=10, help="Максимум партий")
    parser.add_argument("--perf-type", default=None, help="Тип игры (blitz, rapid, classical, ...)")
    parser.add_argument("--rated", action=argparse.BooleanOptionalAction, help="Только рейтинговые партии")
    parser.add_argument("--by-rating", nargs=2, type=int, metavar=("MIN", "MAX"), help="Диапазон рейтинга")
    parser.add_argument("--all", action="store_true", help="Скачать все партии (пагинация)")

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
        if args.all:
            games = client.fetch_all_games(args.username, total_games=args.max, perf_type=args.perf_type)
        else:
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