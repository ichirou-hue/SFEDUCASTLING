"""Правила смены никнейма: валидация и экспоненциальная цена."""

import pytest

from backend.services.nickname import (
    nickname_change_price,
    normalize_nickname,
    validate_nickname,
)


def test_price_doubles_after_each_change():
    assert [nickname_change_price(changes) for changes in range(5)] == [
        10,
        20,
        40,
        80,
        160,
    ]


def test_price_handles_negative_and_large_counters():
    assert nickname_change_price(None) == 10
    assert nickname_change_price(-3) == 10
    # Верхняя граница защищает от выхода за пределы int32.
    assert nickname_change_price(1000) == nickname_change_price(20)


def test_empty_nickname_rejected():
    with pytest.raises(ValueError):
        validate_nickname("   ")


def test_nickname_is_normalized_and_allows_cyrillic():
    assert normalize_nickname("  Иван   Грозный  ") == "Иван Грозный"
    assert validate_nickname("  Иван   Грозный  ") == "Иван Грозный"
    assert validate_nickname("giga_chess-2026") == "giga_chess-2026"


def test_unsafe_nickname_rejected():
    for bad in ("<script>", "nick!", "a" * 33, "---"):
        with pytest.raises(ValueError):
            validate_nickname(bad)
