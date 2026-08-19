"""Классификация качества ходов (задача 82).

Оценка нормализуется в центепешки (cp) с точки зрения стороны,
делающей ход. Затем разница ``ev_after - ev_before`` сопоставляется
с порогами из плана: 20/50/100/300/500 cp.

Классы:
    - best (0..20 cp): лучший ход (или в пределах погрешности);
    - excellent (20..50 cp);
    - good (50..100 cp);
    - inaccuracy (100..300 cp);
    - mistake (300..500 cp);
    - blunder (>500 cp).
"""

from typing import Any

# Пороги в центепешках (как у Lichess, только в cp).
BEST_CP = 20
EXCELLENT_CP = 50
GOOD_CP = 100
INACCURACY_CP = 300
MISTAKE_CP = 500

# Большая оценка для мата: мат за N ходов ≈ ±(10000 - N) cp.
_MATE_SCALE = 10000


def normalize_eval(eval_dict: dict[str, Any] | None) -> float:
    """Переводит оценку Stockfish в центепешки с точки зрения ходящей стороны.

    Args:
        eval_dict: Словарь вида ``{"type": "cp", "value": 35}`` или
            ``{"type": "mate", "value": 2}`` (знак — от лица ходящей стороны).

    Returns:
        Оценка в cp: положительная — преимущество ходящего.
    """
    if not eval_dict:
        return 0.0

    eval_type = eval_dict.get("type")
    value = eval_dict.get("value", 0)

    if eval_type == "mate":
        if value > 0:
            return float(_MATE_SCALE - value)
        return float(-_MATE_SCALE - value)
    if eval_type == "cp":
        return float(value)
    return 0.0


def classify(diff_cp: float) -> str:
    """Классифицирует ход по потере оценки (в cp) относительно лучшего.

    Args:
        diff_cp: ``ev_best - ev_played`` в центепешках (положительное = хуже лучшего).

    Returns:
        Один из классов: best / excellent / good / inaccuracy / mistake / blunder.
    """
    if diff_cp <= BEST_CP:
        return "best"
    if diff_cp <= EXCELLENT_CP:
        return "excellent"
    if diff_cp <= GOOD_CP:
        return "good"
    if diff_cp <= INACCURACY_CP:
        return "inaccuracy"
    if diff_cp <= MISTAKE_CP:
        return "mistake"
    return "blunder"
