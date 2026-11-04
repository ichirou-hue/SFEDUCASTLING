"""Вероятность победы по оценке Stockfish и точность хода (модель Lichess).

Модуль держит общую математику для всех мест, где оценивается качество
хода: разбор партии (``services/game_review``), фоновая запись партий
(``services/game_recorder``) и точность по этапам
(``services/phase_accuracy``). Раньше каждый считал по-своему: разбор -
по падению вероятности победы, а запись партий - по сантипешкам, из-за
чего в профиле появлялись «хорошие» ходы, которые на деле теряли
позицию.

Оценка позиции переводится в вероятность победы логистической кривой,
как в Lichess. Это важно: потеря 100 cp в равной позиции и потеря
100 cp в безнадёжной - разные события. Мат сводится к предельной
оценке ``MATE_CP``: точное число ходов на коротком поиске недостоверно.
"""

from __future__ import annotations

import math

import chess

# Оценка мата в сантипешках. На коротком поиске число ходов до мата
# недостоверно, поэтому мат - это просто «позиция решена».
MATE_CP = 1000.0

# Предел оценки для расчётов: +10 пешек и мат для вероятности победы
# - одно и то же.
CP_LIMIT = 1000.0

# Пороги классификации по падению вероятности победы, в процентных
# пунктах. Основа - разметка Lichess: 10/20/30 пунктов на неточность,
# ошибку и зевок; более мягкие ступени отделяют просто хорошие ходы
# от лучшего.
DROP_BEST = 2.0
DROP_EXCELLENT = 5.0
DROP_GOOD = 10.0
DROP_INACCURACY = 20.0
DROP_MISTAKE = 30.0

# Вероятность победы, за пределами которой позиция уже решена. Там, где
# шансы упираются в предел, ход почти ничего не меняет: любой ход в
# безнадёжной позиции выглядит безупречным, а в выигранной - тоже.
# Поэтому такие ходы не идут ни в точность, ни в качество.
DECIDED_WIN_PERCENT = 15.0

CLASSES = ("best", "excellent", "good", "inaccuracy", "mistake", "blunder")


def clamp_cp(cp: float) -> float:
    """Ограничивает оценку пределом, при котором вероятность насыщается."""
    return max(-CP_LIMIT, min(CP_LIMIT, cp))


def is_decided(win_value: float | None, threshold: float = DECIDED_WIN_PERCENT) -> bool:
    """Позиция решена: шансы у предела, качество хода в ней ни о чём не говорит."""
    if win_value is None:
        return False

    return win_value <= threshold or win_value >= 100.0 - threshold


def win_percent(cp: float) -> float:
    """Вероятность победы белых в процентах по оценке в сантипешках."""
    return 50.0 + 50.0 * (
        2.0 / (1.0 + math.exp(-0.00368208 * clamp_cp(cp))) - 1.0
    )


def player_win_percent(cp: float, player_turn: chess.Color) -> float:
    """Вероятность победы игрока по оценке от лица белых.

    ``win_percent`` считает от белых, а ход игрока надо мерить от его
    стороны: рост перевеса белых для чёрных - это потеря.
    """
    value = win_percent(cp)
    return value if player_turn == chess.WHITE else 100.0 - value


def move_accuracy(win_before: float, win_after: float) -> float:
    """Точность одного хода (0..100) по падению вероятности победы.

    Формула Lichess: небольшая потеря почти не снижает точность, а зевок
    ферзя обнуляет её. Если вероятность выросла, ход не хуже лучшего -
    точность 100. Прибавка в единицу - та же, что в Lichess: оценка
    движка на коротком поиске сама по себе неточна.
    """
    if win_after >= win_before:
        return 100.0

    drop = win_before - win_after
    value = (
        103.1668100711649 * math.exp(-0.04354415386753951 * drop)
        - 3.166924740191411
        + 1.0
    )
    return max(0.0, min(100.0, value))


def classify(win_before: float, win_after: float) -> str:
    """Класс хода по падению вероятности победы.

    Args:
        win_before: Вероятность победы до хода, проценты.
        win_after: Вероятность победы после хода, проценты.
    """
    drop = max(0.0, win_before - win_after)

    if drop < DROP_BEST:
        return "best"
    if drop < DROP_EXCELLENT:
        return "excellent"
    if drop < DROP_GOOD:
        return "good"
    if drop < DROP_INACCURACY:
        return "inaccuracy"
    if drop < DROP_MISTAKE:
        return "mistake"
    return "blunder"


def classify_move(
    cp_before: float,
    cp_after: float,
    turn: chess.Color,
) -> str:
    """Класс хода по оценкам позиции до и после (cp от лица белых)."""
    win_before = player_win_percent(cp_before, turn)
    win_after = player_win_percent(cp_after, turn)
    return classify(win_before, win_after)
