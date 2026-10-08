"""Чистая логика B3: упрощённый SM-2 для учебных заданий."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

DEFAULT_EASE = 2.2
ERROR_INTERVAL_DAYS = 1.0


def next_review_state(
    *,
    reps: int,
    ease: float = DEFAULT_EASE,
    correct: bool,
    now: datetime | None = None,
) -> dict:
    """Возвращает новое состояние интервального повторения.

    Lite-SM-2:
    - ошибка сбрасывает серию и назначает повтор через 1 день;
    - первое верное решение после старта/ошибки -> 1 день;
    - каждое следующее подряд верное решение умножает интервал на ease (2.2).

    Интервал можно восстановить только из reps, поэтому отдельное поле interval
    в БД не требуется: 1, 2.2, 4.84, 10.648 ... дней.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)

    safe_reps = max(0, int(reps or 0))
    safe_ease = max(1.0, float(ease or DEFAULT_EASE))

    if not correct:
        new_reps = 0
        interval_days = ERROR_INTERVAL_DAYS
    else:
        new_reps = safe_reps + 1
        interval_days = 1.0 if new_reps == 1 else safe_ease ** (new_reps - 1)

    next_review_at = current + timedelta(days=interval_days)
    return {
        "reps": new_reps,
        "ease": safe_ease,
        "interval_days": round(interval_days, 3),
        "next_review_at": next_review_at,
    }
