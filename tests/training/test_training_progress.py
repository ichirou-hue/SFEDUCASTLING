from datetime import date, timedelta

from backend.services.training_service import _current_streak


def test_current_streak_counts_through_today():
    today = date(2026, 9, 14)
    days = {today, today - timedelta(days=1), today - timedelta(days=2)}
    assert _current_streak(days, today=today) == 3


def test_current_streak_keeps_yesterday_alive():
    today = date(2026, 9, 14)
    days = {today - timedelta(days=1), today - timedelta(days=2)}
    assert _current_streak(days, today=today) == 2


def test_current_streak_breaks_after_gap():
    today = date(2026, 9, 14)
    days = {today - timedelta(days=2), today - timedelta(days=3)}
    assert _current_streak(days, today=today) == 0
