"""Personal bests and recent activity summaries for a user's profile."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import Iterable


# Moscow has observed fixed UTC+3 since 2014. A fixed offset avoids requiring
# the external tzdata package on Windows installations of Python.
MOSCOW = timezone(timedelta(hours=3), name="MSK")
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def _local_date(value: datetime) -> date:
    """Convert DB timestamps to the user's calendar date (naive DB values are UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(MOSCOW).date()


def _player_outcome(game) -> str | None:
    if game.result not in VALID_RESULTS:
        return None

    white = str(game.white or "player").casefold()
    black = str(game.black or "ai").casefold()
    player_is_black = black in {"player", "user"} and white not in {"player", "user"}
    player_is_white = white in {"player", "user"} or not player_is_black

    if game.result == "1/2-1/2":
        return "draw"
    player_won = game.result == ("0-1" if player_is_black else "1-0")
    return "win" if player_won else "loss"


def build_user_records(
    games: Iterable,
    puzzle_attempts: Iterable,
    *,
    now: datetime | None = None,
) -> dict:
    """Build all-time records and a 14-day daily series without storing duplicates."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    today = now.astimezone(MOSCOW).date()
    week_start = today - timedelta(days=today.weekday())
    chart_start = today - timedelta(days=13)

    ordered_games = sorted(games, key=lambda row: (row.created_at, row.id))
    ordered_attempts = sorted(puzzle_attempts, key=lambda row: (row.created_at, row.id))

    best_weekly_elo = None
    longest_win_streak = current_win_streak = 0
    daily_games = {
        (chart_start + timedelta(days=offset)).isoformat(): {
            "wins": 0, "draws": 0, "losses": 0,
        }
        for offset in range(14)
    }

    for game in ordered_games:
        outcome = _player_outcome(game)
        if outcome is None:
            continue

        if outcome == "win":
            current_win_streak += 1
            longest_win_streak = max(longest_win_streak, current_win_streak)
            if week_start <= _local_date(game.created_at) <= today and game.elo is not None:
                best_weekly_elo = max(best_weekly_elo or int(game.elo), int(game.elo))
        else:
            current_win_streak = 0

        game_date = _local_date(game.created_at).isoformat()
        if game_date in daily_games:
            outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
            daily_games[game_date][outcome_key] += 1

    longest_puzzle_streak = current_puzzle_streak = 0
    daily_puzzles = {
        (chart_start + timedelta(days=offset)).isoformat(): {
            "correct": 0, "incorrect": 0,
        }
        for offset in range(14)
    }

    for attempt in ordered_attempts:
        if attempt.correct:
            current_puzzle_streak += 1
            longest_puzzle_streak = max(longest_puzzle_streak, current_puzzle_streak)
        else:
            current_puzzle_streak = 0

        attempt_date = _local_date(attempt.created_at).isoformat()
        if attempt_date in daily_puzzles:
            key = "correct" if attempt.correct else "incorrect"
            daily_puzzles[attempt_date][key] += 1

    return {
        "records": {
            "weekly_best_opponent_elo": best_weekly_elo,
            "weekly_starts_on": week_start.isoformat(),
            "longest_win_streak": longest_win_streak,
            "best_puzzle_streak": longest_puzzle_streak,
        },
        "activity": {
            "starts_on": chart_start.isoformat(),
            "ends_on": today.isoformat(),
            "games": [
                {"date": day, **values} for day, values in daily_games.items()
            ],
            "puzzles": [
                {"date": day, **values} for day, values in daily_puzzles.items()
            ],
        },
    }
