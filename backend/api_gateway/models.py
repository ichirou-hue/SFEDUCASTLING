"""Pydantic-модели для SFEDUCASTLING API.

Все шахматные модели содержат валидацию FEN, чтобы в обработку
попадали только корректные позиции на доске.
"""

import chess
from pydantic import BaseModel, Field, field_validator, model_validator


def validate_fen(fen: str) -> str:
    """Проверяет FEN-строку через python-chess.

    Args:
        fen: FEN-строка (например 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')

    Returns:
        Проверенная FEN-строка.

    Raises:
        ValueError: Если FEN некорректен.
    """
    try:
        chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Некорректный FEN: {e}")
    return fen


class FenSquare(BaseModel):
    """Запрос: легальные ходы для фигуры на заданном поле.

    Attributes:
        fen: Текущая позиция в нотации FEN.
        square: Целевое поле (например 'e2', 'g1').
    """
    fen: str
    square: str

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("square")
    @classmethod
    def square_must_be_valid(cls, v: str) -> str:
        try:
            chess.parse_square(v)
        except ValueError:
            raise ValueError(f"Некорректное поле '{v}'. Используйте например 'e2', 'g8'.")
        return v


class MoveRequest(BaseModel):
    """Запрос: выполнить шахматный ход.

    Attributes:
        fen: Текущая позиция в нотации FEN.
        from_sq: Поле, откуда ходим (например 'e2').
        to_sq: Поле, куда ходим (например 'e4').
        promotion: Фигура превращения ('q', 'r', 'b', 'n'). По умолчанию 'q'.
    """
    fen: str
    from_sq: str
    to_sq: str
    promotion: str = "q"

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("from_sq", "to_sq")
    @classmethod
    def square_must_be_valid(cls, v: str) -> str:
        try:
            chess.parse_square(v)
        except ValueError:
            raise ValueError(f"Некорректное поле '{v}'. Используйте например 'e2', 'g8'.")
        return v

    @field_validator("promotion")
    @classmethod
    def promotion_must_be_valid(cls, v: str) -> str:
        if v not in ("q", "r", "b", "n"):
            raise ValueError(f"Некорректное превращение '{v}'. Используйте: q, r, b, n.")
        return v

    @model_validator(mode="after")
    def move_must_be_legal_on_board(self):
        try:
            board = chess.Board(self.fen)
            from_sq = chess.parse_square(self.from_sq)
            to_sq = chess.parse_square(self.to_sq)
            piece = board.piece_at(from_sq)
            if piece is None:
                raise ValueError(f"Нет фигуры на поле '{self.from_sq}'")
            if piece.color != board.turn:
                raise ValueError(f"Сейчас ход {'белых' if board.turn else 'чёрных'}, но фигура на '{self.from_sq}' — {'чёрная' if piece.color else 'белая'}")
            promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
            promo = None
            if piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
                promo = promo_map.get(self.promotion, chess.QUEEN)
            move = chess.Move(from_sq, to_sq, promotion=promo)
            if move not in board.legal_moves:
                raise ValueError(f"Недопустимый ход: {self.from_sq} -> {self.to_sq}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Ошибка валидации хода: {e}")
        return self


class FenRequest(BaseModel):
    """Запрос: анализ позиции на доске.

    Attributes:
        fen: Текущая позиция в нотации FEN.
        elo: Целевой рейтинг Elo (0–3000). По умолчанию 1500.
        moves: Список ходов в UCI (например ["e2e4", "e7e5"]), от стартовой
            позиции до текущей. Нужен Maia3 для истории партии.
        engine: Движок для хода: "maia3" (по умолчанию) или "stockfish".
    """
    fen: str
    elo: int = 1500
    moves: list[str] = []
    engine: str = "maia3"

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("elo")
    @classmethod
    def elo_must_be_in_range(cls, v: int) -> int:
        if v < 0 or v > 3000:
            raise ValueError(f"ELO должен быть от 0 до 3000, получено {v}")
        return v

    @field_validator("engine")
    @classmethod
    def engine_must_be_valid(cls, v: str) -> str:
        if v not in ("maia3", "stockfish"):
            raise ValueError(f"Движок должен быть 'maia3' или 'stockfish', получено {v}")
        return v


class CompareMovesRequest(BaseModel):
    """Запрос для сравнения Stockfish и Maia3."""

    fen: str
    elo: int = 1500
    moves: list[str] = []

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("elo")
    @classmethod
    def elo_must_be_in_range(cls, v: int) -> int:
        if v < 0 or v > 3000:
            raise ValueError(f"ELO должен быть от 0 до 3000, получено {v}")
        return v


class FinishGameRequest(BaseModel):
    """Запрос: сохранить завершённую партию с классификацией ходов.

    Attributes:
        moves: Полный список ходов партии в нотации UCI.
        user_id: Идентификатор пользователя (None для анонима).
        elo: Уровень сложности AI (Elo).
        engine: Движок AI ("maia3" или "stockfish").
        result: Результат партии ("1-0", "0-1", "1/2-1/2", "*").
        status: Статус окончания ("checkmate", "stalemate", "draw", ...).
    """

    moves: list[str] = []
    user_id: int | None = None
    elo: int | None = None
    engine: str = "maia3"
    result: str = "*"
    status: str = "playing"

    @field_validator("engine")
    @classmethod
    def engine_must_be_valid(cls, v: str) -> str:
        if v not in ("maia3", "stockfish"):
            raise ValueError(f"Движок должен быть 'maia3' или 'stockfish', получено {v}")
        return v


class SimilarityRequest(BaseModel):
    """Запрос: поиск похожих позиций в векторной базе.

    Attributes:
        fen: Позиция для поиска в нотации FEN.
        top_k: Количество результатов (1–100). По умолчанию 5.
    """
    fen: str
    top_k: int = 5

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_in_range(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"top_k должен быть от 1 до 100, получено {v}")
        return v


class DatasetMoveRequest(BaseModel):
    """Запрос: сохранить ход в тренировочный датасет.

    Attributes:
        fen: Позиция до хода в нотации FEN.
        move: Ход пользователя в нотации UCI или SAN.
        user_id: id залогиненного пользователя (NULL для анонима).
        game_id: Идентификатор игровой сессии. По умолчанию ''.
    """
    fen: str
    move: str
    user_id: int | None = None
    game_id: str = ""

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("move")
    @classmethod
    def move_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ход не может быть пустым")
        return v.strip()


class PGNTextRequest(BaseModel):
    """Запрос: разобрать PGN-текст.

    Attributes:
        pgn: PGN-текст партии.
    """
    pgn: str

    @field_validator("pgn")
    @classmethod
    def pgn_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("PGN-текст не может быть пустым")
        return v.strip()


class ExplainMoveRequest(BaseModel):
    """Запрос на объяснение хода игрока."""

    fen: str
    move: str
    elo: int = 1500
    moves: list[str] = []

    @field_validator("fen")
    @classmethod
    def fen_must_be_valid(cls, v: str) -> str:
        return validate_fen(v)

    @field_validator("move")
    @classmethod
    def move_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ход не может быть пустым")
        return v.strip()

    @field_validator("elo")
    @classmethod
    def elo_must_be_in_range(cls, v: int) -> int:
        if v < 0 or v > 3000:
            raise ValueError(
                f"ELO должен быть от 0 до 3000, получено {v}"
            )
        return v

    @field_validator("moves")
    @classmethod
    def moves_must_be_valid(cls, v: list[str]) -> list[str]:
        return [move.strip() for move in v if move.strip()]


class ChatAskRequest(BaseModel):
    """Запрос на чат с AI-ассистентом."""
    message: str = Field(..., min_length=1, max_length=4000)
    fen: str = ""
    moves: list[str] = []
    elo: int = 1500
    is_greeting: bool = False