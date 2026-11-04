"""Классификация качества ходов по падению вероятности победы.

Раньше здесь были пороги в сантипешках (20/50/100/300/500 cp): потеря
100 cp в равной позиции и та же потеря в безнадёжной считались
одинаково. Фоновая запись партий пользовалась этими порогами, а разбор
партии - моделью вероятности победы, из-за чего в профиле появлялись
«хорошие» ходы, которые на деле теряли позицию.

Теперь и разбор, и запись партий классифицируют ход одинаково.
Математика живёт в ``backend.analysis.win_probability``.
"""

from backend.analysis.win_probability import (
    CLASSES,
    classify,
    classify_move,
    move_accuracy,
    player_win_percent,
    win_percent,
)

__all__ = [
    "CLASSES",
    "classify",
    "classify_move",
    "move_accuracy",
    "player_win_percent",
    "win_percent",
]
