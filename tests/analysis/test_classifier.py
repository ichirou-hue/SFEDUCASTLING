"""Модульные тесты классификатора ходов (backend.analysis.classifier)."""

import chess
import pytest

from backend.analysis.classifier import classify, classify_move
from backend.analysis.win_probability import (
    DROP_BEST,
    DROP_EXCELLENT,
    DROP_GOOD,
    DROP_INACCURACY,
    DROP_MISTAKE,
    move_accuracy,
    player_win_percent,
    win_percent,
)


class TestWinPercent:
    def test_equal_position(self):
        assert win_percent(0) == pytest.approx(50.0)

    def test_positive_cp_helps_white(self):
        assert win_percent(100) > 50.0
        assert win_percent(500) > win_percent(100)

    def test_symmetric(self):
        assert win_percent(150) + win_percent(-150) == pytest.approx(100.0)

    def test_clamped(self):
        assert 0.0 <= win_percent(-10000) <= 100.0
        assert 0.0 <= win_percent(10000) <= 100.0

    def test_player_win_percent_for_black(self):
        assert player_win_percent(200, chess.BLACK) == pytest.approx(
            100.0 - win_percent(200)
        )
        assert player_win_percent(200, chess.WHITE) == pytest.approx(win_percent(200))


class TestMoveAccuracy:
    def test_no_loss_is_full_accuracy(self):
        assert move_accuracy(50.0, 50.0) == pytest.approx(100.0, abs=1e-3)
        assert move_accuracy(40.0, 60.0) == pytest.approx(100.0, abs=1e-3)

    def test_huge_loss_is_near_zero(self):
        assert move_accuracy(90.0, 10.0) < 10.0
        assert move_accuracy(100.0, 0.0) >= 0.0


class TestClassify:
    @pytest.mark.parametrize(
        "drop,expected",
        [
            (0.0, "best"),
            (DROP_BEST - 0.1, "best"),
            (DROP_BEST, "excellent"),
            (DROP_EXCELLENT - 0.1, "excellent"),
            (DROP_EXCELLENT, "good"),
            (DROP_GOOD - 0.1, "good"),
            (DROP_GOOD, "inaccuracy"),
            (DROP_INACCURACY - 0.1, "inaccuracy"),
            (DROP_INACCURACY, "mistake"),
            (DROP_MISTAKE - 0.1, "mistake"),
            (DROP_MISTAKE, "blunder"),
            (80.0, "blunder"),
        ],
    )
    def test_thresholds_by_win_percent_drop(self, drop, expected):
        assert classify(60.0, 60.0 - drop) == expected

    def test_negative_drop_is_best(self):
        assert classify(50.0, 55.0) == "best"

    def test_classify_move_uses_mover_perspective(self):
        # Белые теряют 6 пешек в равной позиции - это их грубая ошибка,
        # а для чёрных та же оценка означает отличный ход.
        assert classify_move(0.0, -600.0, chess.WHITE) == "blunder"
        assert classify_move(0.0, -600.0, chess.BLACK) == "best"

    def test_classify_move_matches_classify_for_black(self):
        assert classify_move(
            100.0, -100.0, chess.BLACK
        ) == classify(
            player_win_percent(100.0, chess.BLACK),
            player_win_percent(-100.0, chess.BLACK),
        )
