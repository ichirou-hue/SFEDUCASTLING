"""Endpoint для подтягивания профиля игрока с Lichess / Chess.com.

Используем официальные публичные API (без ключей):
  - Lichess:  GET https://lichess.org/api/user/{username}
  - Chess.com: GET https://api.chess.com/pub/player/{username}
               GET https://api.chess.com/pub/player/{username}/stats
Нормализуем ответ в единую структуру для фронта.
"""

import requests
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter(tags=["chess-profile"])

LICHESS_USER_URL = "https://lichess.org/api/user/{username}"
CHESSCOM_PLAYER_URL = "https://api.chess.com/pub/player/{username}"
CHESSCOM_STATS_URL = "https://api.chess.com/pub/player/{username}/stats"

USER_AGENT = "SfeduCastling/0.1 (registration-profile-preview)"
HTTP_TIMEOUT = 8.0


def _lichess_profile(username: str):
    try:
        resp = requests.get(
            LICHESS_USER_URL.format(username=username),
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return {"error": "Не удалось связаться с Lichess"}, 502

    if resp.status_code == 404:
        return {"error": "Пользователь не найден на Lichess"}, 404
    if resp.status_code != 200:
        return {"error": f"Lichess ответил {resp.status_code}"}, resp.status_code

    data = resp.json()
    perfs = data.get("perfs", {})
    profile = data.get("profile", {}) or {}

    def _perf(key: str):
        p = perfs.get(key, {})
        games = p.get("games", 0)
        return {"rating": p.get("rating") if games else None, "games": games}

    counts = data.get("count", {})
    return {
        "platform": "lichess",
        "username": data.get("username", username),
        "title": data.get("title"),
        "name": profile.get("firstName") or None,
        "country": profile.get("flag") or None,
        "avatar": data.get("avatar") or None,
        "perfs": {
            "bullet": _perf("bullet"),
            "blitz": _perf("blitz"),
            "rapid": _perf("rapid"),
            "classical": _perf("classical"),
        },
        "counts": {
            "all": counts.get("all", 0),
            "wins": counts.get("win", 0),
            "losses": counts.get("loss", 0),
            "draws": counts.get("draw", 0),
        },
    }, 200


def _chesscom_profile(username: str):
    headers = {"User-Agent": USER_AGENT}
    try:
        player_resp = requests.get(
            CHESSCOM_PLAYER_URL.format(username=username),
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return {"error": "Не удалось связаться с Chess.com"}, 502

    if player_resp.status_code == 404:
        return {"error": "Пользователь не найден на Chess.com"}, 404
    if player_resp.status_code != 200:
        return {"error": f"Chess.com ответил {player_resp.status_code}"}, player_resp.status_code

    data = player_resp.json()

    try:
        stats_resp = requests.get(
            CHESSCOM_STATS_URL.format(username=username),
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        stats = stats_resp.json() if stats_resp.status_code == 200 else {}
    except requests.RequestException:
        stats = {}

    ratings = {}
    counts = {"all": 0, "wins": 0, "losses": 0, "draws": 0}
    for key in ("bullet", "blitz", "rapid", "daily"):
        block = stats.get("chess_" + key, {})
        last = block.get("last", {})
        rec = block.get("record", {}) or {}
        games = rec.get("win", 0) + rec.get("loss", 0) + rec.get("draw", 0)
        ratings[key] = {"rating": last.get("rating"), "games": games}
        counts["all"] += games
        counts["wins"] += rec.get("win", 0)
        counts["losses"] += rec.get("loss", 0)
        counts["draws"] += rec.get("draw", 0)

    return {
        "platform": "chess.com",
        "username": data.get("username", username),
        "title": data.get("title"),
        "name": data.get("name"),
        "country": (data.get("country") or "").rsplit("/", 1)[-1] or None,
        "avatar": data.get("avatar") or None,
        "perfs": ratings,
        "counts": counts,
    }, 200


@router.get("/api/chess-profile")
def chess_profile(
    username: str = Query(..., min_length=1, max_length=64),
    platform: str = Query("lichess"),
):
    platform = platform.lower().strip()
    if platform in ("chess.com", "chesscom", "chess_com", "chess"):
        payload, status = _chesscom_profile(username)
    else:
        payload, status = _lichess_profile(username)
    return JSONResponse(content=payload, status_code=status)
