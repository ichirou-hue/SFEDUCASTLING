"""Модульные тесты классификатора ходов (backend.analysis.classifier)."""

import pytest

from backend.analysis.classifier import (
    BEST_CP,
    EXCELLENT_CP,
    GOOD_CP,
    INACCURACY_CP,
    MISTAKE_CP,
    classify,
    normalize_eval,
)


class TestNormalizeEval:
    def test_cp_positive(self):
        assert normalize_eval({"type": "cp", "value": 35}) == 35.0

    def test_cp_negative(self):
        assert normalize_eval({"type": "cp", "value": -120}) == -120.0

    def test_mate_positive(self):
        # Мат за 2 хода — большое преимущество.
        assert normalize_eval({"type": "mate", "value": 2}) > 9000

    def test_mate_negative(self):
        assert normalize_eval({"type": "mate", "value": -3}) < -9000

    def test_none(self):
        assert normalize_eval(None) == 0.0

    def test_unknown_type(self):
        assert normalize_eval({"type": "foo", "value": 1}) == 0.0


class TestClassify:
    @pytest.mark.parametrize(
        "diff,expected",
        [
            (0, "best"),
            (BEST_CP, "best"),
            (BEST_CP + 1, "excellent"),
            (EXCELLENT_CP, "excellent"),
            (EXCELLENT_CP + 1, "good"),
            (GOOD_CP, "good"),
            (GOOD_CP + 1, "inaccuracy"),
            (INACCURACY_CP, "inaccuracy"),
            (INACCURACY_CP + 1, "mistake"),
            (MISTAKE_CP, "mistake"),
            (MISTAKE_CP + 1, "blunder"),
            (100000, "blunder"),
        ],
    )
    def test_porogi(self, diff, expected):
        assert classify(float(diff)) == expected

    def test_negative_diff_is_best(self):
        # Ход лучше лучшего — тоже best.
        assert classify(-50.0) == "best"
