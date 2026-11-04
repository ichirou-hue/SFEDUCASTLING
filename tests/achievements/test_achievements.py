"""Достижения за первую игру, первый мат и первое поражение."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, time, timedelta

import pytest

from backend.db.session import async_session_factory
from backend.models.gigacoin_event import GigacoinEvent
from backend.services.achievements import (
    FIRST_GAME,
    FIRST_LOSS,
    FIRST_MATE,
    evaluate_game_achievements,
    replay_terminal_game,
)
from backend.services.gigacoin_service import MOSCOW_TZ, moscow_today

SCHOLAR_MATE = [
    "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
]
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]
STALEMATE = [
    "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5", "a5c7",
    "a6h6", "h2h4", "f7f6", "c7d7", "e8f7", "d7b7", "d8d3",
    "b7b8", "d3h7", "b8c8", "f7g6", "c8e6",
]


@pytest.fixture(autouse=True)
def no_background_recording(monkeypatch):
    """Не запускаем Stockfish-разбор партии в тестах достижений."""

    async def _noop(**_kwargs):
        return None

    monkeypatch.setattr("backend.api_gateway.routes.game.record_game", _noop)


def _finish(client, headers, moves, result, status="checkmate", player_color="w"):
    return client.post(
        "/api/game/finish",
        json={
            "moves": moves,
            "elo": 1100,
            "engine": "maia3",
            "result": result,
            "status": status,
            "player_color": player_color,
        },
        headers=headers,
    )


def _codes(response):
    return {item["code"] for item in response.json()["achievements"]}


class TestReplay:
    def test_scholar_mate_is_checkmate(self):
        verified = replay_terminal_game(SCHOLAR_MATE)
        assert verified is not None
        assert verified.status == "checkmate"
        assert verified.result == "1-0"

    def test_stalemate_is_draw(self):
        verified = replay_terminal_game(STALEMATE)
        assert verified is not None
        assert verified.status == "stalemate"
        assert verified.result == "1/2-1/2"

    def test_unfinished_and_illegal_games_are_rejected(self):
        assert replay_terminal_game(["e2e4", "e7e5"]) is None
        assert replay_terminal_game(["e2e5"]) is None


class TestAchievementRules:
    def test_checkmate_win_unlocks_first_game_and_first_mate(self):
        assert evaluate_game_achievements(SCHOLAR_MATE, "1-0", "w") == [
            FIRST_GAME, FIRST_MATE,
        ]

    def test_loss_unlocks_first_game_and_first_loss(self):
        assert evaluate_game_achievements(FOOLS_MATE, "0-1", "w") == [
            FIRST_GAME, FIRST_LOSS,
        ]

    def test_draw_unlocks_only_first_game(self):
        assert evaluate_game_achievements(STALEMATE, "1/2-1/2", "w") == [FIRST_GAME]

    def test_short_or_mismatched_games_unlock_nothing(self):
        assert evaluate_game_achievements(["e2e4", "e7e5"], "1-0", "w") == []
        assert evaluate_game_achievements(SCHOLAR_MATE, "0-1", "w") == []


class TestAchievementsApi:
    def test_requires_auth(self, client):
        assert client.get("/api/game/achievements").status_code == 401

    def test_first_win_awards_coins_and_listing_shows_unlocked(self, client, user_factory):
        player = user_factory("mate")
        response = _finish(client, player["headers"], SCHOLAR_MATE, "1-0")
        assert response.status_code == 200, response.text
        body = response.json()

        assert _codes(response) == {FIRST_GAME, FIRST_MATE}
        assert body["gigacoins_earned"] == 35
        assert body["gigacoins_balance"] == 35

        listing = client.get(
            "/api/game/achievements", headers=player["headers"]
        ).json()["achievements"]
        by_code = {item["code"]: item for item in listing}
        assert set(by_code) == {FIRST_GAME, FIRST_MATE, FIRST_LOSS}
        assert by_code[FIRST_GAME]["unlocked"] is True
        assert by_code[FIRST_GAME]["unlocked_at"].endswith("+00:00")
        assert by_code[FIRST_MATE]["unlocked"] is True
        assert by_code[FIRST_LOSS]["unlocked"] is False

    def test_first_loss_awards_coins(self, client, user_factory):
        player = user_factory("loss")
        response = _finish(client, player["headers"], FOOLS_MATE, "0-1")
        assert response.status_code == 200, response.text
        assert _codes(response) == {FIRST_GAME, FIRST_LOSS}
        assert response.json()["gigacoins_earned"] == 15
        assert response.json()["gigacoins_balance"] == 15

    def test_streak_bonus_does_not_inflate_achievement_reward(self, client, user_factory):
        player = user_factory("streak")
        today = moscow_today()

        async def _seed_streak() -> None:
            async with async_session_factory() as db:
                for offset in range(1, 7):
                    day = today - timedelta(days=offset)
                    created_at = datetime.combine(
                        day, time(12), tzinfo=MOSCOW_TZ
                    ).astimezone(UTC)
                    db.add(
                        GigacoinEvent(
                            user_id=player["user"]["id"],
                            reason="activity_day",
                            ref=day.isoformat(),
                            amount=0,
                            created_at=created_at,
                        )
                    )
                await db.commit()

        asyncio.run(_seed_streak())

        response = _finish(client, player["headers"], SCHOLAR_MATE, "1-0")
        assert response.status_code == 200, response.text
        body = response.json()
        assert _codes(response) == {FIRST_GAME, FIRST_MATE}
        # 10 + 25 за достижения и ещё 10 за седьмой активный день.
        assert body["gigacoins_earned"] == 45
        assert body["gigacoins_balance"] == 45
        assert {item["code"]: item["reward"] for item in body["achievements"]} == {
            FIRST_GAME: 10,
            FIRST_MATE: 25,
        }

        listing = client.get(
            "/api/game/achievements", headers=player["headers"]
        ).json()["achievements"]
        by_code = {item["code"]: item for item in listing}
        assert by_code[FIRST_GAME]["awarded"] == 10
        assert by_code[FIRST_MATE]["awarded"] == 25

    def test_draw_awards_only_first_game(self, client, user_factory):
        player = user_factory("draw")
        response = _finish(
            client, player["headers"], STALEMATE, "1/2-1/2", status="stalemate"
        )
        assert response.status_code == 200, response.text
        assert _codes(response) == {FIRST_GAME}
        # 10 за достижение + 1 за ничью на уровне Maia (партия длиннее 8 ходов).
        assert response.json()["gigacoins_earned"] == 11
        assert response.json()["gigacoins_balance"] == 11

    def test_rewards_are_idempotent(self, client, user_factory):
        player = user_factory("repeat")
        first = _finish(client, player["headers"], SCHOLAR_MATE, "1-0")
        assert first.status_code == 200

        second = _finish(client, player["headers"], SCHOLAR_MATE, "1-0")
        assert second.status_code == 200
        assert second.json()["achievements"] == []
        assert second.json()["gigacoins_earned"] == 0
        assert second.json()["gigacoins_balance"] == 35

    def test_forged_result_does_not_award(self, client, user_factory):
        player = user_factory("forge")
        response = _finish(client, player["headers"], SCHOLAR_MATE, "0-1")
        assert response.status_code == 200, response.text
        assert response.json()["achievements"] == []
        assert response.json()["gigacoins_earned"] == 0
        assert response.json()["gigacoins_balance"] == 0
