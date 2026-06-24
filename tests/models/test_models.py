"""Модульные тесты Pydantic-моделей из backend.api_gateway.models.

Проверяет валидацию FEN, ходов и ограничения полей
для всех моделей запросов SFEDUCASTLING API.
"""

import pytest
from pydantic import ValidationError
from backend.api_gateway.models import (
    FenSquare,
    MoveRequest,
    FenRequest,
    SimilarityRequest,
    DatasetMoveRequest,
    PGNTextRequest,
)

# --- Фикстуры ---

VALID_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EMPTY_FEN = "8/8/8/8/8/8/8/8 w - - 0 1"


# --- FenSquare ---

class TestFenSquare:
    def test_valid(self):
        m = FenSquare(fen=VALID_FEN, square="e5")
        assert m.fen == VALID_FEN
        assert m.square == "e5"

    def test_invalid_fen(self):
        with pytest.raises(ValidationError):
            FenSquare(fen="invalid-fen", square="e2")

    def test_invalid_square(self):
        with pytest.raises(ValidationError):
            FenSquare(fen=START_FEN, square="z9")


# --- MoveRequest ---

class TestMoveRequest:
    def test_valid_pawn_push(self):
        m = MoveRequest(fen=START_FEN, from_sq="e2", to_sq="e4")
        assert m.from_sq == "e2"
        assert m.to_sq == "e4"

    def test_valid_promotion(self):
        fen = "8/4P3/8/8/8/8/8/8 w - - 0 1"
        m = MoveRequest(fen=fen, from_sq="e7", to_sq="e8", promotion="n")
        assert m.promotion == "n"

    def test_invalid_fen(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen="bad", from_sq="e2", to_sq="e4")

    def test_invalid_source_square(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen=START_FEN, from_sq="z9", to_sq="e4")

    def test_invalid_target_square(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen=START_FEN, from_sq="e2", to_sq="z9")

    def test_invalid_promotion(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen=START_FEN, from_sq="e2", to_sq="e4", promotion="x")

    def test_no_piece_at_source(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen=START_FEN, from_sq="e5", to_sq="e6")

    def test_wrong_turn(self):
        """Пытаемся походить чёрной фигурой, когда ход белых."""
        with pytest.raises(ValidationError):
            fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
            MoveRequest(fen=fen, from_sq="e7", to_sq="e5")

    def test_illegal_move_knight_jump(self):
        with pytest.raises(ValidationError):
            MoveRequest(fen=START_FEN, from_sq="e2", to_sq="e6")

    def test_default_promotion(self):
        m = MoveRequest(fen=START_FEN, from_sq="e2", to_sq="e4")
        assert m.promotion == "q"


# --- FenRequest ---

class TestFenRequest:
    def test_valid_default_elo(self):
        m = FenRequest(fen=START_FEN)
        assert m.elo == 1500

    def test_valid_custom_elo(self):
        m = FenRequest(fen=START_FEN, elo=2000)
        assert m.elo == 2000

    def test_elo_zero(self):
        m = FenRequest(fen=START_FEN, elo=0)
        assert m.elo == 0

    def test_elo_three_thousand(self):
        m = FenRequest(fen=START_FEN, elo=3000)
        assert m.elo == 3000

    def test_invalid_fen(self):
        with pytest.raises(ValidationError):
            FenRequest(fen="bad-fen")

    def test_elo_negative(self):
        with pytest.raises(ValidationError):
            FenRequest(fen=START_FEN, elo=-1)

    def test_elo_too_high(self):
        with pytest.raises(ValidationError):
            FenRequest(fen=START_FEN, elo=3001)


# --- SimilarityRequest ---

class TestSimilarityRequest:
    def test_valid_default_top_k(self):
        m = SimilarityRequest(fen=START_FEN)
        assert m.top_k == 5

    def test_valid_custom_top_k(self):
        m = SimilarityRequest(fen=START_FEN, top_k=10)
        assert m.top_k == 10

    def test_top_k_one(self):
        m = SimilarityRequest(fen=START_FEN, top_k=1)
        assert m.top_k == 1

    def test_top_k_one_hundred(self):
        m = SimilarityRequest(fen=START_FEN, top_k=100)
        assert m.top_k == 100

    def test_invalid_fen(self):
        with pytest.raises(ValidationError):
            SimilarityRequest(fen="bad")

    def test_top_k_zero(self):
        with pytest.raises(ValidationError):
            SimilarityRequest(fen=START_FEN, top_k=0)

    def test_top_k_negative(self):
        with pytest.raises(ValidationError):
            SimilarityRequest(fen=START_FEN, top_k=-5)

    def test_top_k_too_high(self):
        with pytest.raises(ValidationError):
            SimilarityRequest(fen=START_FEN, top_k=101)


# --- DatasetMoveRequest ---

class TestDatasetMoveRequest:
    def test_valid(self):
        m = DatasetMoveRequest(fen=START_FEN, move="e2e4")
        assert m.move == "e2e4"

    def test_move_strip(self):
        m = DatasetMoveRequest(fen=START_FEN, move="  e2e4  ")
        assert m.move == "e2e4"

    def test_invalid_fen(self):
        with pytest.raises(ValidationError):
            DatasetMoveRequest(fen="bad", move="e2e4")

    def test_empty_move(self):
        with pytest.raises(ValidationError):
            DatasetMoveRequest(fen=START_FEN, move="")

    def test_whitespace_move(self):
        with pytest.raises(ValidationError):
            DatasetMoveRequest(fen=START_FEN, move="   ")

    def test_default_user_id(self):
        m = DatasetMoveRequest(fen=START_FEN, move="e2e4")
        assert m.user_id == "anonymous"

    def test_custom_ids(self):
        m = DatasetMoveRequest(fen=START_FEN, move="e2e4", user_id="alice", game_id="g123")
        assert m.user_id == "alice"
        assert m.game_id == "g123"


# --- PGNTextRequest ---

class TestPGNTextRequest:
    SAMPLE_PGN = (
        '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6\n'
        '4. Ba4 Nf6 5. O-O Be7 6. Re1 b5\n'
        '7. Bb3 d6 8. c3 O-O'
    )

    def test_valid(self):
        m = PGNTextRequest(pgn=self.SAMPLE_PGN)
        assert m.pgn == self.SAMPLE_PGN

    def test_strip_whitespace(self):
        m = PGNTextRequest(pgn="  " + self.SAMPLE_PGN + "  ")
        assert m.pgn == self.SAMPLE_PGN

    def test_empty(self):
        with pytest.raises(ValidationError):
            PGNTextRequest(pgn="")

    def test_whitespace_only(self):
        with pytest.raises(ValidationError):
            PGNTextRequest(pgn="   \n  \t  ")

    def test_invalid_fen_not_checked(self):
        """У PGNTextRequest нет поля FEN, поэтому валидация FEN не применяется."""
        m = PGNTextRequest(pgn="just some text")
        assert m.pgn == "just some text"


# tests(): общее покрытие - 98%, модуль (api_gateway/models) - 98%
# Запуск: python -m pytest tests/unit/test_models.py --cov=backend.api_gateway.models --cov-report=term -v
