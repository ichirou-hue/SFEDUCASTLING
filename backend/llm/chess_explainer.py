import re
import time

import chess

from backend.api_gateway.models import ExplainMoveRequest
from backend.api_gateway.routes.game import _maia3_move_strict
from backend.api_gateway.state import (
    ensure_stockfish,
    reset_stockfish,
    stockfish_lock,
    get_gigachess,
)
from backend.llm.gigachess import GigachessError


MAX_GIGACHESS_ATTEMPTS = 4
GIGACHESS_TRANSPORT_ATTEMPTS = 3
GIGACHESS_TRANSPORT_BACKOFF_SECONDS = (0.0, 0.6, 1.2)
GIGACHESS_INPUT_TYPE = "fen_content+attachment+compact_grounding_v37"


# ============================================================
# НАЗВАНИЯ
# ============================================================

PIECE_NAMES = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король",
}

COLOR_NAMES = {
    chess.WHITE: "белых",
    chess.BLACK: "чёрных",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _piece_name(piece: chess.Piece | None) -> str:
    if piece is None:
        return "нет фигуры"

    return (
        f"{PIECE_NAMES[piece.piece_type]} "
        f"{COLOR_NAMES[piece.color]}"
    )


def _piece_name_capitalized(piece: chess.Piece | None) -> str:
    text = _piece_name(piece)

    if not text:
        return text

    return text[0].upper() + text[1:]


def _piece_word_accusative(piece_word: str) -> str:
    forms = {
        "пешка": "пешку",
        "конь": "коня",
        "слон": "слона",
        "ладья": "ладью",
        "ферзь": "ферзя",
        "король": "короля",
    }
    return forms.get(piece_word, piece_word)


def _piece_name_accusative(piece_text: str) -> str:
    """
    "ферзь чёрных" -> "ферзя чёрных"
    "пешка белых" -> "пешку белых"
    """
    lowered = piece_text.lower().strip()
    for canonical in PIECE_NAMES.values():
        if lowered.startswith(canonical):
            suffix = piece_text[len(canonical):]
            return _piece_word_accusative(canonical) + suffix
    return piece_text


def _capture_object_case_patterns(piece_word: str) -> tuple[str, str]:
    """
    Возвращает:
      1) допустимую форму названия фигуры как прямого объекта взятия;
      2) ошибочную именительную форму.

    Цвет может стоять перед фигурой:
      "берёт чёрного ферзя" — допустимо,
      "берёт чёрный ферзь" — ошибка.
    """
    accusative_forms = {
        "пешка": r"пешку",
        "конь": r"коня",
        "слон": r"слона",
        "ладья": r"ладью",
        "ферзь": r"ферзя",
        "король": r"короля",
    }
    nominative_forms = {
        "пешка": r"пешка",
        "конь": r"конь",
        "слон": r"слон",
        "ладья": r"ладья",
        "ферзь": r"ферзь",
        "король": r"король",
    }
    return (
        accusative_forms.get(piece_word, re.escape(piece_word)),
        nominative_forms.get(piece_word, re.escape(piece_word)),
    )


# ============================================================
# ИНФОРМАЦИЯ О ХОДЕ
# ============================================================

def _move_info(
    board: chess.Board,
    move: chess.Move | None,
) -> dict | None:

    if move is None:
        return None

    if move not in board.legal_moves:
        return None

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    return {
        "uci": move.uci(),
        "san": board.san(move),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "piece": _piece_name(piece),
    }


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _join_squares(squares: list[str]) -> str:

    if not squares:
        return ""

    if len(squares) == 1:
        return squares[0]

    if len(squares) == 2:
        return f"{squares[0]} и {squares[1]}"

    return ", ".join(squares[:-1]) + " и " + squares[-1]


def _piece_movement_type(
    board: chess.Board,
    move: chess.Move,
) -> str | None:

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    piece_type = piece.piece_type

    if piece_type == chess.PAWN:
        return "pawn"

    if piece_type == chess.KNIGHT:
        return "knight_jump"

    if piece_type == chess.BISHOP:
        return "diagonal"

    if piece_type == chess.ROOK:

        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        if from_file == to_file:
            return "vertical"

        return "horizontal"

    if piece_type == chess.QUEEN:

        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)

        if from_file == to_file:
            return "vertical"

        if from_rank == to_rank:
            return "horizontal"

        return "diagonal"

    if piece_type == chess.KING:

        if board.is_castling(move):
            return "castling"

        return "king_step"

    return None


# ============================================================
# ФАКТЫ ХОДА
# ============================================================

def _move_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:

    piece = board_before.piece_at(move.from_square)

    if piece is None:
        raise ValueError(
            f"На поле {chess.square_name(move.from_square)} "
            f"нет фигуры."
        )

    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    center = {
        chess.D4,
        chess.E4,
        chess.D5,
        chess.E5,
    }

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------

    captured_piece = None
    captured_square = None

    if board_before.is_en_passant(move):

        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square),
        )

        captured_piece = board_before.piece_at(
            captured_square
        )

    else:

        captured_piece = board_before.piece_at(
            move.to_square
        )

        if captured_piece is not None:
            captured_square = move.to_square

    # --------------------------------------------------------
    # КОНТРОЛЬ ПОЛЕЙ
    # --------------------------------------------------------

    controls_before_set = set(
        board_before.attacks(move.from_square)
    )

    controls_after_set = set(
        board_after.attacks(move.to_square)
    )

    controls_before = sorted(
        chess.square_name(square)
        for square in controls_before_set
    )

    controls_after = sorted(
        chess.square_name(square)
        for square in controls_after_set
    )

    new_controls = sorted(
        chess.square_name(square)
        for square in (
            controls_after_set - controls_before_set
        )
    )

    lost_controls = sorted(
        chess.square_name(square)
        for square in (
            controls_before_set - controls_after_set
        )
    )

    controls_center_before = sorted(
        chess.square_name(square)
        for square in (
            controls_before_set & center
        )
    )

    controls_center_after = sorted(
        chess.square_name(square)
        for square in (
            controls_after_set & center
        )
    )

    new_center_controls = sorted(
        chess.square_name(square)
        for square in (
            (controls_after_set - controls_before_set) & center
        )
    )

    # --------------------------------------------------------
    # БАЗОВЫЕ ФАКТЫ
    # --------------------------------------------------------

    facts = {
        "piece": _piece_name(piece),

        "piece_type": piece.piece_type,

        "piece_symbol": piece.symbol().lower(),

        "piece_color": (
            "white"
            if piece.color == chess.WHITE
            else "black"
        ),

        "movement_type": _piece_movement_type(
            board_before,
            move,
        ),

        "from": from_square,

        "to": to_square,

        "captured_piece": (
            _piece_name(captured_piece)
            if captured_piece
            else None
        ),

        "captured_square": (
            chess.square_name(captured_square)
            if captured_square is not None
            else None
        ),

        "is_capture": board_before.is_capture(move),

        "is_en_passant": board_before.is_en_passant(move),

        "is_check": board_after.is_check(),

        "is_checkmate": board_after.is_checkmate(),

        "is_castling": board_before.is_castling(move),

        "is_promotion": move.promotion is not None,

        "promotion_piece": (
            PIECE_NAMES.get(
                move.promotion,
                chess.piece_name(move.promotion),
            )
            if move.promotion
            else None
        ),

        "turn": (
            "white"
            if board_before.turn == chess.WHITE
            else "black"
        ),

        "from_file": chess.square_file(move.from_square),

        "from_rank": chess.square_rank(move.from_square),

        "to_file": chess.square_file(move.to_square),

        "to_rank": chess.square_rank(move.to_square),

        "moves_to_center": move.to_square in center,

        "from_center": move.from_square in center,

        "occupies_center": move.to_square in center,

        "controls_before": controls_before,

        "controls_after": controls_after,

        "new_controls": new_controls,

        "lost_controls": lost_controls,

        "controls_center_before": controls_center_before,

        "controls_center_after": controls_center_after,

        "new_center_controls": new_center_controls,

        "gives_check": board_after.is_check(),
    }

    # --------------------------------------------------------
    # ПЕШКА
    # --------------------------------------------------------

    if piece.piece_type == chess.PAWN:

        rank_diff = abs(
            chess.square_rank(move.to_square)
            - chess.square_rank(move.from_square)
        )

        facts.update({
            "pawn_double_step": rank_diff == 2,

            "pawn_controls": controls_after,

            "pawn_controls_before": controls_before,

            "pawn_controls_after": controls_after,

            "pawn_new_controls": new_controls,

            "pawn_lost_controls": lost_controls,

            "pawn_controls_center": controls_center_after,

            "pawn_new_center_controls": new_center_controls,

            "pawn_reaches_center": move.to_square in center,
        })

    # --------------------------------------------------------
    # КОНЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.KNIGHT:

        starting_squares = {
            chess.G1,
            chess.B1,
            chess.G8,
            chess.B8,
        }

        facts.update({
            "knight_controls_before": controls_before,

            "knight_controls_after": controls_after,

            "knight_new_controls": new_controls,

            "knight_lost_controls": lost_controls,

            "knight_controls_center": controls_center_after,

            "knight_new_center_controls": new_center_controls,

            "knight_develops": (
                move.from_square in starting_squares
                and move.to_square not in starting_squares
            ),
        })

    # --------------------------------------------------------
    # СЛОН
    # --------------------------------------------------------

    elif piece.piece_type == chess.BISHOP:

        facts.update({
            "bishop_controls_before": controls_before,

            "bishop_controls_after": controls_after,

            "bishop_new_controls": new_controls,

            "bishop_lost_controls": lost_controls,

            "bishop_controls_center": controls_center_after,

            "bishop_new_center_controls": new_center_controls,

            "is_diagonal_move": True,
        })

    # --------------------------------------------------------
    # ЛАДЬЯ
    # --------------------------------------------------------

    elif piece.piece_type == chess.ROOK:

        facts.update({
            "rook_controls_before": controls_before,

            "rook_controls_after": controls_after,

            "rook_new_controls": new_controls,

            "rook_lost_controls": lost_controls,

            "rook_controls_center": controls_center_after,

            "rook_new_center_controls": new_center_controls,

            "is_file_move": (
                chess.square_file(move.from_square)
                == chess.square_file(move.to_square)
            ),

            "is_rank_move": (
                chess.square_rank(move.from_square)
                == chess.square_rank(move.to_square)
            ),
        })

    # --------------------------------------------------------
    # ФЕРЗЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.QUEEN:

        facts.update({
            "queen_controls_before": controls_before,

            "queen_controls_after": controls_after,

            "queen_new_controls": new_controls,

            "queen_lost_controls": lost_controls,

            "queen_controls_center": controls_center_after,

            "queen_new_center_controls": new_center_controls,
        })

    # --------------------------------------------------------
    # КОРОЛЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.KING:

        facts.update({
            "king_controls_before": controls_before,

            "king_controls_after": controls_after,

            "king_new_controls": new_controls,

            "king_lost_controls": lost_controls,
        })

    return facts


# ============================================================
# ФАКТЫ ИЗМЕНЕНИЯ ПОЗИЦИИ
# ============================================================

def _position_change_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:

    mover = board_before.turn
    opponent = not mover

    attacked_before = set()
    attacked_after = set()

    for square, piece in board_before.piece_map().items():

        if piece.color != mover:
            continue

        attacked_before.update(
            board_before.attacks(square)
        )

    for square, piece in board_after.piece_map().items():

        if piece.color != mover:
            continue

        attacked_after.update(
            board_after.attacks(square)
        )

    newly_attacked_pieces = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if (
            square in attacked_after
            and square not in attacked_before
        ):

            newly_attacked_pieces.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),

                "piece_type": piece.piece_type,

                "value": PIECE_VALUES.get(
                    piece.piece_type,
                    0,
                ),
            })

    no_longer_attacked = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if (
            square in attacked_before
            and square not in attacked_after
        ):

            no_longer_attacked.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    newly_defended = []

    for square, piece in board_after.piece_map().items():

        if piece.color != mover:
            continue

        defenders_before = len(
            board_before.attackers(
                mover,
                square,
            )
        )

        defenders_after = len(
            board_after.attackers(
                mover,
                square,
            )
        )

        if defenders_after > defenders_before:

            newly_defended.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    opponent_king = board_after.king(opponent)

    king_attackers = []

    if opponent_king is not None:

        king_attackers = [
            chess.square_name(square)
            for square in board_after.attackers(
                mover,
                opponent_king,
            )
        ]

    pinned_pieces = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if board_after.is_pinned(
            opponent,
            square,
        ):

            pinned_pieces.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    legal_replies = list(
        board_after.legal_moves
    )

    return {
        "newly_attacked_pieces": newly_attacked_pieces,

        "no_longer_attacked": no_longer_attacked,

        "newly_defended": newly_defended,

        "opponent_king_attackers": king_attackers,

        "pinned_pieces": pinned_pieces,

        "opponent_legal_moves": len(
            legal_replies
        ),

        "opponent_in_check": (
            board_after.is_check()
        ),
    }


# ============================================================
# STOCKFISH
# ============================================================

def _normalise_stockfish_result(
    board: chess.Board,
    infos: list,
) -> dict:

    if not infos:

        return {
            "available": True,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],
        }

    top_moves = []

    for info in infos:

        if not isinstance(info, dict):
            continue

        move_uci = info.get("Move")

        if not move_uci:
            continue

        try:

            move = chess.Move.from_uci(
                move_uci
            )

        except ValueError:

            continue

        if move not in board.legal_moves:
            continue

        move_info = _move_info(
            board,
            move,
        )

        if move_info is None:
            continue

        evaluation = info.get(
            "Centipawn"
        )

        mate = info.get(
            "Mate"
        )

        depth = (
            info.get("Depth")
            or info.get("SelectiveDepth")
        )

        top_moves.append({
            "move": move_info,

            "evaluation": evaluation,

            "mate": mate,

            "depth": depth,
        })

    if not top_moves:

        return {
            "available": True,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],
        }

    best = top_moves[0]

    if best.get("mate") is not None:

        evaluation = {
            "type": "mate",

            "value": best["mate"],
        }

    elif best.get("evaluation") is not None:

        evaluation = {
            "type": "cp",

            "value": best["evaluation"],
        }

    else:

        evaluation = None

    return {
        "available": True,

        "best_move": best["move"],

        "evaluation": evaluation,

        "top_moves": top_moves,
    }


def _stockfish_analysis(
    board: chess.Board,
) -> dict:

    stockfish = ensure_stockfish()

    if stockfish is None:

        return {
            "available": False,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],

            "error": "Stockfish недоступен.",
        }

    try:

        with stockfish_lock:

            stockfish.set_fen_position(
                board.fen()
            )

            infos = stockfish.get_top_moves(
                5
            )

        result = _normalise_stockfish_result(
            board,
            infos,
        )

        if result.get("best_move"):

            return result

        print(
            "[ChessExplainer] "
            "Stockfish MultiPV не вернул "
            "лучший ход. Переходим к get_best_move()."
        )

    except Exception as e:

        print(
            "[ChessExplainer] "
            f"Stockfish MultiPV error: {e}"
        )

    try:

        with stockfish_lock:

            stockfish.set_fen_position(
                board.fen()
            )

            move_uci = (
                stockfish.get_best_move()
            )

        if not move_uci:

            raise RuntimeError(
                "Stockfish не вернул bestmove."
            )

        move = chess.Move.from_uci(
            move_uci
        )

        if move not in board.legal_moves:

            raise RuntimeError(
                "Stockfish вернул "
                f"нелегальный ход: {move_uci}"
            )

        move_info = _move_info(
            board,
            move,
        )

        if move_info is None:

            raise RuntimeError(
                "Не удалось получить информацию "
                "о bestmove."
            )

        return {
            "available": True,

            "best_move": move_info,

            "evaluation": None,

            "top_moves": [
                {
                    "move": move_info,

                    "evaluation": None,

                    "mate": None,

                    "depth": None,
                }
            ],
        }

    except Exception as e:

        print(
            "[ChessExplainer] "
            f"Stockfish process error: {e}"
        )

        try:

            reset_stockfish()

        except Exception as reset_error:

            print(
                "[ChessExplainer] "
                f"Stockfish reset error: {reset_error}"
            )

        return {
            "available": False,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],

            "error": str(e),
        }


# ============================================================
# РАЗРЕШЁННЫЕ КООРДИНАТЫ
# ============================================================

def _line_opening_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> list[dict]:
    """
    Ищет безопасные педагогические факты вида
    «ход освободил линию для слона/ладьи/ферзя».

    Условие намеренно строгое:
    - рассматривается своя дальнобойная фигура;
    - поле FROM перемещённой фигуры до хода лежало в её атаке;
    - после освобождения FROM у дальнобойной фигуры появились
      новые контролируемые поля за бывшим блокером.

    Это не стратегическая интерпретация, а геометрический факт доски.
    """
    mover = board_before.turn
    result: list[dict] = []

    for square, piece in board_before.piece_map().items():
        if piece.color != mover:
            continue

        if piece.piece_type not in (
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ):
            continue

        # Перемещённую фигуру не анализируем как «открывшуюся».
        if square == move.from_square:
            continue

        same_piece_after = board_after.piece_at(square)
        if same_piece_after != piece:
            continue

        before_attacks = set(board_before.attacks(square))
        after_attacks = set(board_after.attacks(square))

        if move.from_square not in before_attacks:
            continue

        new_squares = sorted(
            after_attacks - before_attacks,
            key=lambda sq: chess.square_name(sq),
        )

        if not new_squares:
            continue

        result.append({
            "piece": _piece_name(piece),
            "square": chess.square_name(square),
            "new_controls": [
                chess.square_name(sq)
                for sq in new_squares
            ],
        })

    return result


def _build_derived_explanation_facts(
    *,
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:
    is_game_over = board_after.is_game_over(claim_draw=True)
    outcome = (
        board_after.outcome(claim_draw=True)
        if is_game_over
        else None
    )

    if outcome is None:
        result = "*"
        winner = None
        termination = None
    else:
        result = board_after.result(claim_draw=True)
        winner = (
            "white"
            if outcome.winner is chess.WHITE
            else "black"
            if outcome.winner is chess.BLACK
            else None
        )
        termination = getattr(
            outcome.termination,
            "name",
            str(outcome.termination),
        )

    opponent_king = board_after.king(board_after.turn)
    opponent_king_square = (
        chess.square_name(opponent_king)
        if opponent_king is not None
        else None
    )

    return {
        "line_openings": _line_opening_facts(
            board_before,
            board_after,
            move,
        ),
        "terminal": {
            "is_game_over": is_game_over,
            "is_checkmate": board_after.is_checkmate(),
            "is_stalemate": board_after.is_stalemate(),
            "is_insufficient_material": board_after.is_insufficient_material(),
            "result": result,
            "winner": winner,
            "termination": termination,
            "opponent_king_square": opponent_king_square,
        },
    }


def _build_gigachess_grounding(
    *,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
) -> dict:
    """
    Semantic fact compressor.

    Python может знать десятки точных клеток и отношений, но GigaChess
    получает только небольшой набор наиболее полезных фактов.

    ВАЖНО:
    - подробные move_facts/position_facts остаются внутри Python;
    - в prompt попадает максимум 5 коротких фактов;
    - validator разрешает конкретные клетки/фигуры только из этого
      сжатого контекста, а не из всего внутреннего анализа.
    """
    facts: list[str] = []
    allowed_squares: set[str] = set()
    allowed_piece_words: set[str] = set()
    moved_piece_controls: set[str] = set()
    attack_squares: set[str] = set()
    defended_squares: set[str] = set()

    from_square = str(explained_move["from"]).lower()
    to_square = str(explained_move["to"]).lower()
    uci = str(explained_move["uci"]).lower()
    piece_text = str(explained_move["piece"])
    piece_lower = piece_text.lower()

    allowed_squares.update({from_square, to_square})

    castling_rook_from = None
    castling_rook_to = None

    if move_facts.get("is_castling"):
        castling_map = {
            ("e1", "g1"): ("h1", "f1"),
            ("e1", "c1"): ("a1", "d1"),
            ("e8", "g8"): ("h8", "f8"),
            ("e8", "c8"): ("a8", "d8"),
        }
        rook_move = castling_map.get((from_square, to_square))
        if rook_move:
            castling_rook_from, castling_rook_to = rook_move
            allowed_squares.update(
                {castling_rook_from, castling_rook_to}
            )
            allowed_piece_words.add("ладья")

    moved_piece_word = next(
        (
            name
            for name in PIECE_NAMES.values()
            if name in piece_lower
        ),
        None,
    )
    if moved_piece_word:
        allowed_piece_words.add(moved_piece_word)

    mover_is_white = move_facts.get("piece_color") == "white"
    mover_side = "белых" if mover_is_white else "чёрных"
    terminal = derived_facts.get("terminal") or {}

    promotion_piece_word = None
    promotion_piece_acc = None

    if move_facts.get("is_promotion"):
        raw_promotion_piece = str(
            move_facts.get("promotion_piece") or ""
        ).lower()

        promotion_piece_word = next(
            (
                name
                for name in PIECE_NAMES.values()
                if name in raw_promotion_piece
            ),
            raw_promotion_piece or "фигура",
        )

        promotion_piece_acc = _piece_word_accusative(
            promotion_piece_word
        )
        allowed_piece_words.add(promotion_piece_word)

    # После promotion шах/мат даёт уже новая фигура, а не пешка.
    post_move_piece_word = (
        promotion_piece_word
        if move_facts.get("is_promotion")
        else moved_piece_word
    )

    movement_type = str(
        move_facts.get("movement_type") or ""
    ).lower()

    movement_geometry_phrase = ""

    if (
        movement_type == "diagonal"
        and moved_piece_word in {"слон", "ферзь"}
    ):
        movement_geometry_phrase = " по диагонали"

    elif (
        movement_type == "horizontal"
        and moved_piece_word in {"ладья", "ферзь"}
    ):
        movement_geometry_phrase = " по горизонтали"

    elif (
        movement_type == "vertical"
        and moved_piece_word in {"ладья", "ферзь"}
    ):
        movement_geometry_phrase = " по вертикали"

    elif (
        movement_type == "knight_jump"
        and moved_piece_word == "конь"
    ):
        movement_geometry_phrase = " ходом буквой «Г»"

    elif (
        movement_type == "king_step"
        and moved_piece_word == "король"
    ):
        movement_geometry_phrase = " на соседнюю клетку"

    elif (
        movement_type == "pawn"
        and moved_piece_word == "пешка"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        movement_geometry_phrase = " по диагонали вперёд на одну клетку"

    elif (
        movement_type == "pawn"
        and moved_piece_word == "пешка"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        movement_geometry_phrase = " прямо вперёд на одну клетку"

    def add_fact(value: str) -> None:
        value = value.strip()
        if value and value not in facts and len(facts) < 5:
            facts.append(value)

    # 1. Сам ход. Для взятия сразу формулируем завершённое событие,
    # чтобы GigaChess не превращал его в "может взять".
    if move_facts.get("is_capture"):
        captured_piece = str(
            move_facts.get("captured_piece") or "фигура соперника"
        )
        captured_square = str(
            move_facts.get("captured_square") or to_square
        ).lower()

        allowed_squares.add(captured_square)

        for name in PIECE_NAMES.values():
            if name in captured_piece.lower():
                allowed_piece_words.add(name)

        captured_acc = _piece_name_accusative(captured_piece)

        if move_facts.get("is_en_passant"):
            add_fact(
                f"{piece_text.capitalize()} идёт с {from_square} на {to_square} "
                f"взятием на проходе и этим ходом снимает {captured_acc} "
                f"с поля {captured_square}."
            )
        else:
            move_fact = (
                f"{piece_text.capitalize()} идёт с {from_square} на {to_square}"
                f"{movement_geometry_phrase} и этим ходом берёт "
                f"{captured_acc} на {captured_square}"
            )
            if move_facts.get("is_promotion") and promotion_piece_acc:
                move_fact += (
                    f", после чего на {to_square} превращается "
                    f"в {promotion_piece_acc}"
                )
            move_fact += "."
            add_fact(move_fact)
    else:
        move_fact = (
            f"{piece_text.capitalize()} идёт с {from_square} на {to_square}"
            f"{movement_geometry_phrase}"
        )
        if move_facts.get("pawn_double_step"):
            move_fact += " двойным ходом"
        if move_facts.get("is_promotion") and promotion_piece_acc:
            move_fact += (
                f" и на {to_square} превращается "
                f"в {promotion_piece_acc}"
            )
        move_fact += "."
        add_fact(move_fact)

    # 2. Остальные критические события хода.

    opponent_king_square = terminal.get("opponent_king_square")

    if move_facts.get("is_checkmate"):
        allowed_piece_words.add("король")
        if opponent_king_square:
            allowed_squares.add(opponent_king_square)
            mate_result_text = ""
            if terminal.get("result") == "1-0":
                mate_result_text = " Партия сразу заканчивается победой белых."
            elif terminal.get("result") == "0-1":
                mate_result_text = " Партия сразу заканчивается победой чёрных."

            add_fact(
                f"После этого хода {post_move_piece_word or 'фигура'} на {to_square} "
                f"ставит мат королю соперника на {opponent_king_square}; "
                f"у соперника нет легальных ходов."
                f"{mate_result_text}"
            )
        else:
            add_fact(
                "После этого хода королю соперника поставлен мат, "
                "и у соперника нет легальных ходов."
            )
    elif move_facts.get("is_check"):
        allowed_piece_words.add("король")
        if opponent_king_square:
            allowed_squares.add(opponent_king_square)
            add_fact(
                f"После этого хода {post_move_piece_word or 'фигура'} на {to_square} "
                f"объявляет шах королю соперника на {opponent_king_square}."
            )
        else:
            add_fact("После хода королю соперника объявлен шах.")

    if move_facts.get("is_castling"):
        if to_square in {"g1", "g8"}:
            castling_name = "короткая рокировка"
        elif to_square in {"c1", "c8"}:
            castling_name = "длинная рокировка"
        else:
            castling_name = "рокировка"

        if castling_rook_from and castling_rook_to:
            add_fact(
                f"Этим ходом выполняется {castling_name}: "
                f"король переходит с {from_square} на {to_square}, "
                f"а ладья одновременно переходит с "
                f"{castling_rook_from} на {castling_rook_to}."
            )
        else:
            add_fact(
                f"Этим ходом выполняется {castling_name}."
            )

    # Терминальный результат партии добавляем только если он реально
    # наступает сразу после этого хода.
    if (
        not move_facts.get("is_checkmate")
        and terminal.get("is_game_over")
    ):
        if terminal.get("is_stalemate"):
            add_fact(
                "После этого хода возникает пат, поэтому партия "
                "сразу заканчивается вничью."
            )
        elif terminal.get("result") == "1/2-1/2":
            add_fact(
                "После этого хода партия сразу заканчивается вничью."
            )
        elif terminal.get("result") == "1-0":
            add_fact(
                "После этого хода партия сразу заканчивается победой белых."
            )
        elif terminal.get("result") == "0-1":
            add_fact(
                "После этого хода партия сразу заканчивается победой чёрных."
            )

    # 3. Конкретный контроль перемещённой фигуры.
    # Не передаём модели длинные списки. Максимум 3 клетки.
    candidate_controls = [
        str(s).lower()
        for s in (
            move_facts.get("new_controls")
            or move_facts.get("controls_after")
            or []
        )
        if isinstance(s, str)
    ]

    # Приоритет центральным полям, затем остальным.
    center_names = {"d4", "e4", "d5", "e5"}
    ordered_controls = sorted(
        dict.fromkeys(candidate_controls),
        key=lambda s: (s not in center_names, s),
    )
    selected_controls = ordered_controls[:3]

    has_critical_event = bool(
        move_facts.get("is_capture")
        or move_facts.get("is_check")
        or move_facts.get("is_checkmate")
        or move_facts.get("is_castling")
        or move_facts.get("is_promotion")
    )

    if (
        selected_controls
        and not has_critical_event
        and len(facts) < 5
    ):
        moved_piece_controls.update(selected_controls)
        allowed_squares.update(selected_controls)
        add_fact(
            "После хода перемещённая фигура контролирует "
            f"{_join_squares(selected_controls)}."
        )

    # 4. Педагогический вывод о центре без новых координат.
    center_supported = bool(
        move_facts.get("occupies_center")
        or move_facts.get("new_center_controls")
    )
    if center_supported and len(facts) < 5 and not has_critical_event:
        add_fact(f"Ход усиливает влияние {mover_side} на центр.")

    # 5. Педагогический вывод о развитии без перечисления открывшихся клеток.
    line_openings = derived_facts.get("line_openings", [])
    development_supported = bool(
        not move_facts.get("is_castling")
        and (
            move_facts.get("knight_develops")
            or line_openings
        )
    )
    if development_supported and len(facts) < 5:
        add_fact(f"Ход помогает дальнейшему развитию фигур {mover_side}.")

    # Если ещё есть место — одна конкретная новая атака ИЛИ защита.
    # Для critical-event ходов этот слой подавляем: он часто дублирует
    # шах/мат и добавляет лишние сущности.
    if len(facts) < 5 and not has_critical_event:
        newly_attacked = [
            item
            for item in position_facts.get("newly_attacked_pieces", [])
            if (
                isinstance(item, dict)
                and item.get("piece")
                and item.get("square")
            )
        ]
        if newly_attacked:
            item = newly_attacked[0]
            square = str(item["square"]).lower()
            piece = str(item["piece"])
            allowed_squares.add(square)
            attack_squares.add(square)
            for name in PIECE_NAMES.values():
                if name in piece.lower():
                    allowed_piece_words.add(name)
            add_fact(
                f"После хода под атакой оказывается {piece} на {square}."
            )

    if len(facts) < 5 and not has_critical_event:
        newly_defended = [
            item
            for item in position_facts.get("newly_defended", [])
            if (
                isinstance(item, dict)
                and item.get("piece")
                and item.get("square")
            )
        ]
        if newly_defended:
            item = newly_defended[0]
            square = str(item["square"]).lower()
            piece = str(item["piece"])
            allowed_squares.add(square)
            defended_squares.add(square)
            for name in PIECE_NAMES.values():
                if name in piece.lower():
                    allowed_piece_words.add(name)
            add_fact(
                f"Ход усиливает защиту {piece} на {square}."
            )

    # Для тихих ходов полезно явно зафиксировать отсутствие критических
    # событий, но только если это не вытесняет более полезные позитивные факты.
    if (
        len(facts) < 5
        and not move_facts.get("is_capture")
        and not move_facts.get("is_check")
        and not move_facts.get("is_checkmate")
    ):
        add_fact("В этом ходе нет взятия, шаха или мата.")

    return {
        "facts": facts,
        "text": "\n".join(f"- {fact}" for fact in facts),
        "uci": uci,
        "from": from_square,
        "to": to_square,
        "moved_piece_word": moved_piece_word,
        "movement_type": movement_type,
        "pawn_capture_geometry_required": bool(
            movement_type == "pawn"
            and moved_piece_word == "пешка"
            and move_facts.get("is_capture")
            and not move_facts.get("is_en_passant")
            and not move_facts.get("is_promotion")
        ),
        "pawn_single_step_geometry_required": bool(
            movement_type == "pawn"
            and moved_piece_word == "пешка"
            and not move_facts.get("is_capture")
            and not move_facts.get("is_en_passant")
            and not move_facts.get("is_promotion")
            and not move_facts.get("pawn_double_step")
        ),
        "movement_geometry_required": bool(
            (
                movement_type == "diagonal"
                and moved_piece_word in {"слон", "ферзь"}
            )
            or (
                movement_type in {"horizontal", "vertical"}
                and moved_piece_word in {"ладья", "ферзь"}
            )
            or (
                movement_type == "knight_jump"
                and moved_piece_word == "конь"
            )
            or (
                movement_type == "king_step"
                and moved_piece_word == "король"
            )
            or (
                movement_type == "pawn"
                and moved_piece_word == "пешка"
                and move_facts.get("is_capture")
                and not move_facts.get("is_en_passant")
                and not move_facts.get("is_promotion")
            )
            or (
                movement_type == "pawn"
                and moved_piece_word == "пешка"
                and not move_facts.get("is_capture")
                and not move_facts.get("is_en_passant")
                and not move_facts.get("is_promotion")
                and not move_facts.get("pawn_double_step")
            )
        ),
        "allowed_squares": allowed_squares,
        "allowed_piece_words": allowed_piece_words,
        "moved_piece_controls": moved_piece_controls,
        "attack_squares": attack_squares,
        "defended_squares": defended_squares,
        "center_supported": center_supported,
        "development_supported": development_supported,
        "line_opening_supported": bool(line_openings),
        "terminal": terminal,
        "opponent_king_square": opponent_king_square,
        "opponent_legal_moves": int(
            position_facts.get("opponent_legal_moves") or 0
        ),
        "castling_rook_from": castling_rook_from,
        "castling_rook_to": castling_rook_to,
        "king_safety_supported": False,
        "generic_defense_supported": bool(defended_squares),
        "is_capture": bool(move_facts.get("is_capture")),
        "is_check": bool(move_facts.get("is_check")),
        "is_checkmate": bool(move_facts.get("is_checkmate")),
        "is_castling": bool(move_facts.get("is_castling")),
        "is_promotion": bool(move_facts.get("is_promotion")),
    }


def _is_negated_claim(lowered: str, keyword: str) -> bool:
    patterns = [
        rf"не\s+(?:да[её]т|ставит|является|делает|созда[её]т)\s+[^.!?]{{0,20}}{keyword}",
        rf"{keyword}[^.!?]{{0,8}}нет",
        rf"без\s+{keyword}",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _piece_word_present(
    text: str,
    canonical_piece_word: str,
) -> bool:
    """
    Проверяет русское название фигуры с учётом простых падежных форм:
    ферзь/ферзя, пешка/пешкой, король/короля и т.п.
    """
    patterns = {
        "пешка": r"\bпешк(?:а|и|е|у|ой|ою|ам|ами|ах)\b",
        "конь": r"\bкон(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
        "слон": r"\bслон(?:а|у|ом|е|ы|ов|ам|ами|ах)?\b",
        "ладья": r"\bладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)\b",
        "ферзь": r"\bферз(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
        "король": r"\bкорол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)\b",
    }
    pattern = patterns.get(canonical_piece_word)
    if pattern is None:
        return canonical_piece_word in text
    return bool(re.search(pattern, text, re.IGNORECASE))


def _validate_llm_explanation(
    text: str | None,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
) -> tuple[bool, list[str]]:
    """
    Factual validator для компактного grounding.

    В отличие от старой версии:
    - модель не обязана повторять все факты/координаты;
    - UCI проверяется отдельным regex (ловит d2e4/d2d4);
    - разрешены только те конкретные клетки/фигуры, которые реально
      присутствовали в СЖАТОМ prompt, а не во всём внутреннем Python-анализе;
    - одно хорошее предложение не бракуется только из-за формата;
    - фактические ошибки остаются hard constraints.
    """
    errors: list[str] = []

    if not isinstance(text, str):
        return False, ["Gigachess вернул не строку."]

    text = text.strip()
    if not text:
        return False, ["Gigachess вернул пустой ответ."]

    lowered = text.lower()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]

    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    expected_uci = grounding["uci"]
    from_square = grounding["from"]
    to_square = grounding["to"]
    moved_piece_word = grounding["moved_piece_word"]

    # --------------------------------------------------------
    # UCI — отдельная обязательная проверка.
    # \b между "2" и "e" не существует, поэтому square-regex сам по себе
    # никогда надёжно не поймает ошибку вида d2e4.
    # --------------------------------------------------------
    mentioned_uci = [
        value.lower()
        for value in re.findall(
            r"(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])",
            lowered,
        )
    ]
    for claimed_uci in mentioned_uci:
        if claimed_uci != expected_uci:
            errors.append(
                f"Модель назвала ход {claimed_uci}, но объясняется "
                f"проверенный ход {expected_uci}."
            )

    # --------------------------------------------------------
    # ФИГУРЫ — только из compact grounding.
    # --------------------------------------------------------
    mentioned_piece_words = {
        piece_name
        for piece_name in PIECE_NAMES.values()
        if _piece_word_present(lowered, piece_name)
    }
    unexpected_piece_words = (
        mentioned_piece_words - grounding["allowed_piece_words"]
    )
    if unexpected_piece_words:
        errors.append(
            "Названы фигуры, которых нет в переданном компактном контексте: "
            f"{sorted(unexpected_piece_words)}."
        )

    bad_subject_agreement = bool(
        re.search(
            r"\b(?:белые|ч[её]рные)\s+"
            r"(?:пешка|конь|слон|ладья|ферзь|король)\b",
            lowered,
        )
    )
    if bad_subject_agreement:
        errors.append(
            "Нарушено согласование цвета и названия фигуры "
            "(например, нужно «белая ладья», а не «белые ладья»)."
        )

    # --------------------------------------------------------
    # КООРДИНАТЫ — только те, которые реально были показаны модели.
    # --------------------------------------------------------
    used_squares = set(
        re.findall(r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])", lowered)
    )
    unexpected_squares = used_squares - grounding["allowed_squares"]
    if unexpected_squares:
        errors.append(
            "Использованы клетки, которых не было среди переданных "
            f"проверенных фактов: {sorted(unexpected_squares)}."
        )

    dangling_square_reference = bool(
        re.search(
            r"\b(?:на|в)\s+"
            r"(?:поле|клетку|клетке|клетке|клетки)"
            r"\s*(?:[.!?,;:]|$)",
            lowered,
        )
    )

    if dangling_square_reference:
        errors.append(
            "Ответ содержит незавершённую ссылку на клетку "
            "(например, «на поле» без координаты)."
        )

    # Явное движение "с X на Y".
    movement_pairs = re.findall(
        r"(?:с|из)\s+([a-h][1-8])[^.!?]{0,45}?(?:на|в)\s+([a-h][1-8])",
        lowered,
    )

    allowed_movement_pairs = {(from_square, to_square)}

    if move_facts.get("is_castling"):
        rook_from_for_move = grounding.get("castling_rook_from")
        rook_to_for_move = grounding.get("castling_rook_to")
        if rook_from_for_move and rook_to_for_move:
            allowed_movement_pairs.add(
                (rook_from_for_move, rook_to_for_move)
            )

    for claimed_from, claimed_to in movement_pairs:
        if (claimed_from, claimed_to) not in allowed_movement_pairs:
            if move_facts.get("is_castling"):
                expected_text = ", ".join(
                    f"{a}->{b}"
                    for a, b in sorted(allowed_movement_pairs)
                )
                errors.append(
                    "Неверно описано перемещение при рокировке: "
                    f"{claimed_from}->{claimed_to}; допустимы только "
                    f"проверенные пары {expected_text}."
                )
            else:
                errors.append(
                    "Неверно описано перемещение: "
                    f"{claimed_from}->{claimed_to}; проверенный ход "
                    f"{from_square}->{to_square}."
                )

    # Типичные конструкции с полем назначения.
    destination_patterns = [
        r"(?:ид[её]т|переход\w*|перемеща\w*|продвига\w*|"
        r"оказыва\w*)[^.!?]{0,24}?(?:на|в)\s+(?:поле\s+)?"
        r"([a-h][1-8])",
        r"занима\w*[^.!?]{0,20}?(?:поле\s+)?([a-h][1-8])",
    ]
    allowed_destination_squares = {to_square}
    if move_facts.get("is_castling"):
        rook_to_for_destination = grounding.get("castling_rook_to")
        if rook_to_for_destination:
            allowed_destination_squares.add(rook_to_for_destination)

    for pattern in destination_patterns:
        for claimed_to in re.findall(pattern, lowered):
            if claimed_to not in allowed_destination_squares:
                errors.append(
                    f"Неверно названо поле назначения {claimed_to}; "
                    "проверенные поля назначения — "
                    f"{sorted(allowed_destination_squares)}."
                )

    # --------------------------------------------------------
    # ГЕОМЕТРИЯ ХОДА — relation-specific validation.
    # --------------------------------------------------------
    expected_movement_type = str(
        grounding.get("movement_type") or ""
    ).lower()

    claimed_geometry: set[str] = set()

    if re.search(r"\bдиагонал\w*\b", lowered):
        claimed_geometry.add("diagonal")

    if re.search(r"\bвертикал\w*\b", lowered):
        claimed_geometry.add("vertical")

    if re.search(r"\bгоризонтал\w*\b", lowered):
        claimed_geometry.add("horizontal")

    knight_geometry_claim = bool(
        re.search(
            r"букв\w*\s*[«„\"']?г[»“\"']?",
            lowered,
        )
        or re.search(
            r"\bг[-\s]?образ\w*\b",
            lowered,
        )
    )
    if knight_geometry_claim:
        claimed_geometry.add("knight_jump")

    king_step_claim = bool(
        re.search(
            r"\bсоседн\w*\s+(?:поле|клетк\w*)\b",
            lowered,
        )
        or re.search(
            r"\bна\s+(?:один|одну|одно|одной|одного|одн\w*)\s+"
            r"(?:поле|клетк\w*|шаг\w*)\b",
            lowered,
        )
        or re.search(
            r"\b(?:один|одну|одно|одной|одного|одн\w*)\s+шаг\w*\b",
            lowered,
        )
    )
    if king_step_claim:
        claimed_geometry.add("king_step")

    known_geometries = {
        "diagonal",
        "vertical",
        "horizontal",
        "knight_jump",
        "king_step",
    }

    if (
        expected_movement_type in known_geometries
        and claimed_geometry
    ):
        wrong_geometry = (
            claimed_geometry - {expected_movement_type}
        )
        if wrong_geometry:
            geometry_names = {
                "diagonal": "диагональ",
                "vertical": "вертикаль",
                "horizontal": "горизонталь",
                "knight_jump": "ход буквой «Г»",
                "king_step": "ход на соседнюю клетку",
            }
            errors.append(
                "Неверно описана геометрия хода: указано "
                f"{[geometry_names[g] for g in sorted(wrong_geometry)]}; "
                "проверенный тип движения — "
                f"{geometry_names[expected_movement_type]}."
            )

    if grounding.get("pawn_single_step_geometry_required"):
        pawn_word_present = bool(
            re.search(
                r"\bпешк(?:а|и|у|ой|е|ам|ами|ах)\b",
                lowered,
            )
        )
        pawn_forward_claim = bool(
            re.search(r"\bвпер[её]д\b", lowered)
        )
        one_square_claim = bool(
            re.search(
                r"\b(?:на\s+)?(?:один|одну|одно|одной|одного|одн\w*)\s+"
                r"(?:поле|клетк\w*|шаг\w*)\b",
                lowered,
            )
            or re.search(
                r"\b(?:один|одну|одно|одной|одного|одн\w*)\s+шаг\w*\b",
                lowered,
            )
        )
        straight_claim = bool(
            re.search(
                r"\b(?:прямо|впер[её]д)\b",
                lowered,
            )
        )

        if not pawn_word_present:
            errors.append(
                "Для обычного хода пешки нужно явно назвать пешку."
            )

        if not pawn_forward_claim or not one_square_claim:
            errors.append(
                "Нужно явно указать правило обычного хода пешки: "
                "она идёт прямо вперёд на одну клетку."
            )

        if re.search(r"\bдиагонал\w*\b", lowered):
            errors.append(
                "Обычный ход пешки без взятия ошибочно описан как "
                "диагональный; пешка идёт прямо вперёд."
            )

        if re.search(r"\b(?:двойн\w*|на\s+две\s+клетк\w*)\b", lowered):
            errors.append(
                "Обычный ход пешки на одну клетку ошибочно описан "
                "как двойной ход."
            )

        if re.search(r"\b(?:бер[её]т|берут|взял\w*|забира\w*|снима\w*|бь[её]т|бьют)\b", lowered):
            errors.append(
                "Ответ выдумывает взятие, хотя проверенный ход пешки "
                "выполняется без взятия."
            )

    if grounding.get("pawn_capture_geometry_required"):
        pawn_word_present = bool(
            re.search(
                r"\bпешк(?:а|и|у|ой|е|ам|ами|ах)\b",
                lowered,
            )
        )
        pawn_diagonal_claim = bool(
            re.search(r"\bдиагонал\w*\b", lowered)
        )
        pawn_forward_claim = bool(
            re.search(r"\bвпер[её]д\b", lowered)
        )

        if not pawn_word_present:
            errors.append(
                "Для пешечного взятия нужно явно назвать пешку, "
                "которая выполняет ход."
            )

        if not pawn_diagonal_claim or not pawn_forward_claim:
            errors.append(
                "Нужно явно указать правило пешечного взятия: "
                "пешка берёт по диагонали вперёд."
            )

        if re.search(r"\b(?:горизонтал\w*|вертикал\w*)\b", lowered):
            errors.append(
                "Пешечное взятие ошибочно описано как движение "
                "по горизонтали или вертикали; пешка берёт "
                "по диагонали вперёд."
            )

        if re.search(r"\b(?:букв\w*\s*[«„\"']?г[»“\"']?|г[-\s]?образ\w*)\b", lowered):
            errors.append(
                "Пешечное взятие ошибочно описано как ход коня."
            )

    if expected_movement_type == "king_step":
        if re.search(r"\bрокиров\w*\b", lowered):
            errors.append(
                "Обычный ход короля ошибочно назван рокировкой."
            )

        if re.search(r"\b(?:прыга\w*|перепрыг\w*)\b", lowered):
            errors.append(
                "Обычный ход короля не является прыжком: "
                "король переходит на соседнюю клетку."
            )

    if grounding.get("movement_geometry_required"):
        expected_geometry_name = {
            "diagonal": "по диагонали",
            "horizontal": "по горизонтали",
            "vertical": "по вертикали",
            "knight_jump": "ходом буквой «Г»",
            "king_step": "на соседнюю клетку",
        }.get(expected_movement_type)

        if (
            expected_geometry_name
            and expected_movement_type not in claimed_geometry
            and not grounding.get("pawn_capture_geometry_required")
            and not grounding.get("pawn_single_step_geometry_required")
        ):
            errors.append(
                "Тип траектории должен быть назван явно: "
                f"проверенный ход выполняется {expected_geometry_name}."
            )

    # --------------------------------------------------------
    # КОРОЛЬ ПРИ РОКИРОВКЕ — relation-specific validation.
    # --------------------------------------------------------
    if move_facts.get("is_castling"):
        king_pair_pattern = (
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,45}?(?:с|из)\s+([a-h][1-8])"
            r"[^.!?]{0,35}?(?:на|в)\s+(?:поле\s+|клет(?:ку|ки|ке)\s+)?"
            r"([a-h][1-8])"
        )
        king_destination_pattern = (
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,30}?(?:переход\w*|перемеща\w*|ид[её]т|переш[её]л\w*)"
            r"[^.!?]{0,22}?(?:на|в)\s+(?:поле\s+|клет(?:ку|ки|ке)\s+)?"
            r"([a-h][1-8])"
        )

        for claimed_from, claimed_to in re.findall(
            king_pair_pattern,
            lowered,
        ):
            if claimed_from != from_square or claimed_to != to_square:
                errors.append(
                    "Неверно описано перемещение короля при рокировке: "
                    f"{claimed_from}->{claimed_to}; проверенное перемещение "
                    f"{from_square}->{to_square}."
                )

        for claimed_to in re.findall(
            king_destination_pattern,
            lowered,
        ):
            if claimed_to != to_square:
                errors.append(
                    "Неверно названо поле назначения короля при рокировке: "
                    f"{claimed_to}; проверенное поле — {to_square}."
                )

    # --------------------------------------------------------
    # ЛАДЬЯ ПРИ РОКИРОВКЕ — relation-specific validation.
    # --------------------------------------------------------
    castling_rook_from = grounding.get("castling_rook_from")
    castling_rook_to = grounding.get("castling_rook_to")

    if move_facts.get("is_castling") and castling_rook_from and castling_rook_to:
        rook_pair_pattern = (
            r"ладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)"
            r"[^.!?]{0,45}?(?:с|из)\s+([a-h][1-8])"
            r"[^.!?]{0,35}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
        )
        rook_destination_pattern = (
            r"ладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)"
            r"[^.!?]{0,25}?(?:переход\w*|перемеща\w*|ид[её]т)"
            r"[^.!?]{0,20}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
        )

        for claimed_from, claimed_to in re.findall(
            rook_pair_pattern,
            lowered,
        ):
            if (
                claimed_from != castling_rook_from
                or claimed_to != castling_rook_to
            ):
                errors.append(
                    "Неверно описано перемещение ладьи при рокировке: "
                    f"{claimed_from}->{claimed_to}; проверенное перемещение "
                    f"{castling_rook_from}->{castling_rook_to}."
                )

        for claimed_to in re.findall(
            rook_destination_pattern,
            lowered,
        ):
            if claimed_to != castling_rook_to:
                errors.append(
                    "Неверно названо поле назначения ладьи при рокировке: "
                    f"{claimed_to}; проверенное поле — {castling_rook_to}."
                )

    # --------------------------------------------------------
    # ПОЛЕ КОРОЛЯ СОПЕРНИКА — relation-specific validation.
    # --------------------------------------------------------
    opponent_king_square = grounding.get("opponent_king_square")

    if opponent_king_square:
        king_square_claims: set[str] = set()

        king_patterns = [
            # "король находится/стоит/остаётся на поле h8"
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"[^.!?]{0,24}?"
            r"(?:находи\w*|стои\w*|оста[её]т\w*|располож\w*)"
            r"[^.!?]{0,18}?(?:на|в)\s+(?:поле\s+)?([a-h][1-8])",

            # "королю на h8 поставлен мат"
            # Здесь координата непосредственно относится к слову "король".
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)"
            r"\s+(?:соперника\s+|ч[её]рных\s+|белых\s+)?"
            r"(?:на|в)\s+(?:поле\s+)?([a-h][1-8])",

            # "на поле h8 находится/стоит чёрный король"
            r"(?:на|в)\s+(?:поле\s+)?([a-h][1-8])"
            r"[^.!?]{0,18}?"
            r"(?:находи\w*|стои\w*|оста[её]т\w*|располож\w*)"
            r"[^.!?]{0,18}?"
            r"корол(?:ь|я|ю|ём|ем|е|и|ей|ям|ями|ях)",
        ]

        for pattern in king_patterns:
            king_square_claims.update(
                value.lower()
                for value in re.findall(pattern, lowered)
            )

        invalid_king_squares = (
            king_square_claims - {opponent_king_square}
        )
        if invalid_king_squares:
            errors.append(
                "Неверно указано поле короля соперника: "
                f"{sorted(invalid_king_squares)}; проверенное поле — "
                f"{opponent_king_square}."
            )

    # --------------------------------------------------------
    # КОНТРОЛЬ / АТАКА / ЗАЩИТА — проверяем именно ОТНОШЕНИЕ.
    # --------------------------------------------------------
    for sentence in sentences:
        sentence_lower = sentence.lower()

        control_match = re.search(r"контрол", sentence_lower)
        if control_match:
            tail = sentence_lower[control_match.start():]
            control_squares = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    tail,
                )
            )
            invalid = control_squares - grounding["moved_piece_controls"]
            if invalid:
                errors.append(
                    "Неверно описан контроль перемещённой фигуры: "
                    f"{sorted(invalid)} не входят в переданный проверенный "
                    f"набор {sorted(grounding['moved_piece_controls'])}."
                )

        if re.search(r"атаку|атакует|под атак", sentence_lower):
            attack_squares_in_sentence = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    sentence_lower,
                )
            )
            # FROM/TO сами по себе не являются целями атаки.
            attack_squares_in_sentence -= {from_square, to_square}
            invalid = (
                attack_squares_in_sentence - grounding["attack_squares"]
            )
            if invalid:
                errors.append(
                    "Неверно описана конкретная атака: "
                    f"{sorted(invalid)} не подтверждены compact grounding."
                )

        if "защищ" in sentence_lower or "защит" in sentence_lower:
            defended_in_sentence = set(
                re.findall(
                    r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])",
                    sentence_lower,
                )
            )
            defended_in_sentence -= {from_square, to_square}
            invalid = (
                defended_in_sentence - grounding["defended_squares"]
            )
            if invalid:
                errors.append(
                    "Неверно описана конкретная защита: "
                    f"{sorted(invalid)} не подтверждены compact grounding."
                )

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------
    capture_completed_patterns = [
        r"\bбер(?:[её]т|ут|[её]м|[её]те)\b",
        r"\bвзял(?:а|и|о)?\b",
        r"\bвзяв\b",
        r"\bзабира(?:ет|ют|ем|ете)\b",
        r"\bзабрал(?:а|и|о)?\b",
        r"\bзахватыва(?:ет|ют|ем|ете|я)\b",
        r"\bзахватил(?:а|и|о)?\b",
        r"\bснима(?:ет|ют|ем|ете)\b",
        r"\bснял(?:а|и|о)?\b",
        r"\bбь(?:[её]т|ют|[её]м|[её]те)\b",
        r"\bпобил(?:а|и|о)?\b",
        r"\bвзят(?:ие|ия|ием|ую|ой)\b",
        r"\b(?:фигура|пешка|конь|слон|ладья|ферзь)\s+снят\w*\b",
    ]

    capture_future_patterns = [
        r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,35}\bвзять\b",
        r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,35}\bзабрать\b",
        r"\bугрожа\w*[^.!?]{0,35}\bвзять\b",
        r"\bготов\w*[^.!?]{0,35}\bвзят",
    ]

    capture_completed = any(
        re.search(pattern, lowered)
        for pattern in capture_completed_patterns
    )
    capture_future_claim = any(
        re.search(pattern, lowered)
        for pattern in capture_future_patterns
    )

    capture_negated = any(
        phrase in lowered
        for phrase in (
            "взятия нет",
            "без взятия",
            "не является взятием",
        )
    )
    positive_capture = capture_completed and not capture_negated

    if move_facts.get("is_capture"):
        if capture_future_claim:
            errors.append(
                "Ответ описывает взятие как будущую возможность, "
                "но проверенный ход уже совершает взятие."
            )

        if not positive_capture:
            errors.append(
                "Ход является уже совершившимся взятием. Нужно прямо сказать, "
                "что фигура этим ходом берёт/забирает/снимает указанную фигуру."
            )

        captured_piece = str(
            move_facts.get("captured_piece") or ""
        ).lower()
        captured_piece_word = next(
            (
                name
                for name in PIECE_NAMES.values()
                if name in captured_piece
            ),
            None,
        )
        if (
            captured_piece_word
            and not _piece_word_present(
                lowered,
                captured_piece_word,
            )
        ):
            errors.append(
                "Не названа взятая фигура "
                f"«{captured_piece_word}»."
            )

        if captured_piece_word:
            (
                captured_acc_pattern,
                captured_nom_pattern,
            ) = _capture_object_case_patterns(
                captured_piece_word
            )

            capture_verb_stem = (
                r"(?:бер(?:[её]т|ут|[её]м|[её]те)|"
                r"взял(?:а|и|о)?|взяв|"
                r"забира(?:ет|ют|ем|ете)|"
                r"забрал(?:а|и|о)?|"
                r"снима(?:ет|ют|ем|ете)|"
                r"снял(?:а|и|о)?|"
                r"бь(?:[её]т|ют|[её]м|[её]те)|"
                r"захватыва(?:ет|ют|ем|ете|я)|"
                r"захватил(?:а|и|о)?)"
            )

            bad_capture_object = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b"
                    rf"[^.!?]{{0,18}}?"
                    rf"\b(?:белый|ч[её]рный)?\s*"
                    rf"{captured_nom_pattern}\b",
                    lowered,
                )
            )

            good_capture_object = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b"
                    rf"[^.!?]{{0,18}}?"
                    rf"\b(?:белого|ч[её]рного|белую|ч[её]рную)?\s*"
                    rf"{captured_acc_pattern}\b",
                    lowered,
                )
            )

            # Для существительного "взятие" допускается родительный:
            # "взятие ферзя", "взятие ладьи" и т.п.
            nominal_capture_ok = bool(
                re.search(
                    rf"\bвзят\w*\b[^.!?]{{0,16}}?"
                    rf"\b(?:белого|ч[её]рного|белой|ч[её]рной)?\s*"
                    rf"(?:{captured_acc_pattern}|"
                    rf"{'ладьи' if captured_piece_word == 'ладья' else 'пешки' if captured_piece_word == 'пешка' else captured_acc_pattern})\b",
                    lowered,
                )
            )

            if bad_capture_object and not good_capture_object:
                errors.append(
                    "Взятая фигура названа в неверном падеже: "
                    f"после глагола взятия нужно использовать форму "
                    f"«{_piece_word_accusative(captured_piece_word)}»."
                )

            # Если используется именно глагольная конструкция взятия,
            # объект должен быть грамматически выражен корректно.
            has_capture_verb = bool(
                re.search(
                    rf"\b{capture_verb_stem}\b",
                    lowered,
                )
            )
            if (
                has_capture_verb
                and not good_capture_object
                and not nominal_capture_ok
                and not bad_capture_object
            ):
                errors.append(
                    "После глагола взятия не удалось однозначно найти "
                    "взятую фигуру в корректной форме."
                )

        if move_facts.get("is_en_passant"):
            en_passant_completed = bool(
                re.search(
                    r"\bвзят\w*\s+на\s+проходе\b"
                    r"|\bбер\w*\s+на\s+проходе\b"
                    r"|\ben\s*passant\b",
                    lowered,
                )
            )

            en_passant_future = bool(
                re.search(
                    r"\bмож(?:ет|но|гут|ем|ете)\b[^.!?]{0,40}"
                    r"\b(?:взять|бить)\b[^.!?]{0,20}\bна\s+проходе\b"
                    r"|\bготов\w*[^.!?]{0,35}\bвзят\w*\s+на\s+проходе\b",
                    lowered,
                )
            )

            if en_passant_future:
                errors.append(
                    "Ответ описывает взятие на проходе как будущую "
                    "возможность, но оно уже выполняется этим ходом."
                )

            if not en_passant_completed:
                errors.append(
                    "Ход является взятием на проходе, но ответ не называет "
                    "этот специальный механизм."
                )

            captured_square = str(
                move_facts.get("captured_square") or ""
            ).lower()
            if captured_square and captured_square not in used_squares:
                errors.append(
                    "Для взятия на проходе нужно назвать поле снятой пешки "
                    f"{captured_square}."
                )
    elif positive_capture or capture_future_claim:
        errors.append(
            "Ответ выдумывает взятие или возможность взятия, "
            "которых нет в проверенном ходе."
        )

    # --------------------------------------------------------
    # ШАХ / МАТ / РОКИРОВКА / ПРЕВРАЩЕНИЕ
    # --------------------------------------------------------
    has_check_word = bool(
        re.search(r"\bшах(?:а|у|ом|е)?\b", lowered)
    )

    check_completed_patterns = [
        # Активные / деепричастные формы.
        r"\bобъяв(?:ля\w*|ил(?:а|и|о)?|ив)\s+шах\b",
        r"\bда[её]т\w*\s+шах\b",
        r"\bдал(?:а|и|о)?\s+шах\b",
        r"\bпоставил(?:а|и|о)?\s+шах\b",
        r"\bставит\s+шах\b",

        # Пассивные завершённые формы.
        r"\b(?:был\s+)?объявлен\s+шах\b",
        r"\bшах\s+(?:был\s+)?объявлен\b",
        r"\b(?:был\s+)?дан\s+шах\b",
        r"\bшах\s+(?:был\s+)?дан\b",

        # Краткие устойчивые формы.
        r"\bс\s+шахом\b",
        r"\bпод\s+шахом\b",
        r"\bшахует\b",
    ]
    check_completed = any(
        re.search(pattern, lowered)
        for pattern in check_completed_patterns
    )

    check_future_patterns = [
        r"\bугрожа\w*[^.!?]{0,40}\bшах(?:ом|а)?\b",
        r"\bмож(?:ет|но|гут)\b[^.!?]{0,45}"
        r"(?:объявить|дать|поставить)\s+шах\b",
        r"\bготовит\w*[^.!?]{0,35}\bшах\b",
        r"\bпозволя\w*[^.!?]{0,45}"
        r"(?:объявить|дать|поставить)\s+шах\b",
    ]
    check_future_claim = any(
        re.search(pattern, lowered)
        for pattern in check_future_patterns
    )

    check_positive = (
        check_completed
        and not _is_negated_claim(
            lowered,
            r"шах(?:а|у|ом|е)?",
        )
    )

    has_mate_word = bool(
        re.search(r"\bмат(?:а|у|ом|е)?\b", lowered)
    )

    # "Есть слово мат" недостаточно. Для реального checkmate нужен
    # завершённый факт: "ставит мат", "это мат", "матует" и т.п.
    mate_completed_patterns = [
        r"\bстав(?:ит|ят|я)\s+(?:шах\s+и\s+)?мат\b",
        r"\bпоставил(?:а|и|о)?\s+(?:шах\s+и\s+)?мат\b",
        r"\bпоставлен\w*\s+(?:шах\s+и\s+)?мат\b",
        r"\bобъявля(?:ет|ют)\s+мат\b",
        r"\bматует\b",
        r"\bзаматовал(?:а|и|о)?\b",
        r"\bэто\s+(?:и\s+есть\s+)?мат\b",
        r"\bшах\s+и\s+мат\b",
        r"\bполучается\s+мат\b",
    ]
    mate_completed = any(
        re.search(pattern, lowered)
        for pattern in mate_completed_patterns
    )

    # Формулировки будущей возможности/угрозы НЕ описывают ход,
    # который уже является матом.
    mate_future_patterns = [
        r"\bмож(?:но|ет|ем|ете|гут)\b[^.!?]{0,45}\bпоставить\s+мат\b",
        r"\bпозволя\w*\b[^.!?]{0,45}\bпоставить\s+мат\b",
        r"\bготовит\w*\b[^.!?]{0,35}\bмат\b",
        r"\bугрожа\w*\b[^.!?]{0,35}\bмат(?:ом|а)?\b",
        r"\bсозда[её]т\w*\b[^.!?]{0,35}\bугроз\w*\b[^.!?]{0,25}\bмат",
        r"\bмат\s+в\s+один\s+ход\b",
    ]
    mate_future_claim = any(
        re.search(pattern, lowered)
        for pattern in mate_future_patterns
    )

    mate_positive = (
        mate_completed
        and not _is_negated_claim(
            lowered,
            r"мат(?:а|у|ом|е)?",
        )
    )

    has_castling = "рокиров" in lowered

    castling_completed_patterns = [
        r"\bвыполня\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bвыполненн\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bдела\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bсоверша\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bрокиру\w*",
        r"\bэто\s+(?:уже\s+)?(?:и\s+есть\s+)?"
        r"(?:выполненн\w*\s+)?"
        r"(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bпроисход\w*\s+рокиров",
        r"\bознача\w*\s+(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
        r"\bявля\w*\s+(?:собой\s+)?(?:коротк\w*\s+|длинн\w*\s+)?рокиров",
    ]
    castling_completed = any(
        re.search(pattern, lowered)
        for pattern in castling_completed_patterns
    )

    castling_future_patterns = [
        r"\bготов\w*\s+(?:к\s+)?рокиров",
        r"\bподготавлива\w*\s+рокиров",
        r"\bпозволя\w*[^.!?]{0,35}\bрокиров",
        r"\bмож(?:но|ет|ем|ете|гут)\b[^.!?]{0,35}\bрокиров",
        r"\bсозда[её]т\w*[^.!?]{0,35}\bвозможност\w*[^.!?]{0,20}\bрокиров",
    ]
    castling_future_claim = any(
        re.search(pattern, lowered)
        for pattern in castling_future_patterns
    )

    castling_positive = (
        castling_completed
        and not any(
            phrase in lowered
            for phrase in (
                "рокировки нет",
                "без рокировки",
                "не является рокировкой",
            )
        )
    )

    has_promotion = "превращ" in lowered

    promotion_completed_patterns = [
        r"\bпревраща\w*\s+в\b",
        r"\bпревратил(?:ась|ся|и|ось)?\s+в\b",
        r"\bпревративш\w*\s+в\b",
        r"\bстановит(?:ся|ься)\b[^.!?]{0,20}"
        r"(?:ферз|ладь|слон|кон)",
    ]
    promotion_completed = any(
        re.search(pattern, lowered)
        for pattern in promotion_completed_patterns
    )

    promotion_future_patterns = [
        r"\bмож(?:ет|но|гут)\b[^.!?]{0,40}\bпреврат",
        r"\bготов\w*[^.!?]{0,35}\bпревращ",
        r"\bприближа\w*[^.!?]{0,35}\bпол[юя]\s+превращ",
        r"\bпродвига\w*[^.!?]{0,35}\b(?:к|до)\s+пол[яю]\s+превращ",
        r"\bид[её]т\w*[^.!?]{0,30}\bк\s+пол[юя]\s+превращ",
    ]
    promotion_future_claim = any(
        re.search(pattern, lowered)
        for pattern in promotion_future_patterns
    )

    promotion_positive = (
        promotion_completed
        and not any(
            phrase in lowered
            for phrase in (
                "превращения нет",
                "без превращения",
                "не является превращением",
            )
        )
    )

    if move_facts.get("is_check"):
        if check_future_claim:
            errors.append(
                "Ответ описывает шах как угрозу или будущую возможность, "
                "но проверенный ход уже объявляет шах."
            )
        if not check_positive and not mate_positive:
            errors.append(
                "Ход уже даёт шах, но ответ не описывает шах "
                "как совершившееся событие."
            )
    elif check_positive or check_future_claim:
        errors.append("Ответ выдумывает шах или угрозу шаха.")

    if move_facts.get("is_checkmate"):
        if mate_future_claim:
            errors.append(
                "Ответ описывает мат как будущую возможность или угрозу, "
                "но проверенный ход уже сам ставит мат."
            )
        if not mate_positive:
            errors.append(
                "Ход уже ставит мат. Ответ должен прямо описать завершённое "
                "событие: «ставит мат», «это мат» или эквивалентную формулировку."
            )
    elif mate_positive or mate_future_claim:
        errors.append("Ответ выдумывает мат или угрозу мата.")

    if move_facts.get("is_castling"):
        if castling_future_claim:
            errors.append(
                "Ответ описывает рокировку как будущую возможность или "
                "подготовку, но проверенный ход уже сам является рокировкой."
            )
        if not castling_positive:
            errors.append(
                "Ход уже является рокировкой. Ответ должен прямо сказать, "
                "что рокировка выполняется этим ходом."
            )
        if not re.search(r"\bладь(?:я|и|е|ю|ёй|ей|ям|ями|ях)\b", lowered):
            errors.append(
                "При рокировке одновременно перемещается ладья; "
                "объяснение должно упомянуть ладью."
            )

        castling_required_squares = {
            str(explained_move.get("from") or "").lower(),
            str(explained_move.get("to") or "").lower(),
            str(grounding.get("castling_rook_from") or "").lower(),
            str(grounding.get("castling_rook_to") or "").lower(),
        }
        castling_required_squares.discard("")

        missing_castling_squares = {
            square
            for square in castling_required_squares
            if not re.search(
                rf"(?<![a-z0-9]){re.escape(square)}(?![a-z0-9])",
                lowered,
            )
        }

        if missing_castling_squares:
            errors.append(
                "Для объяснения рокировки должны быть сохранены точные "
                "координаты обоих перемещений. Не названы клетки: "
                f"{sorted(missing_castling_squares)}."
            )
    elif castling_positive or castling_future_claim:
        errors.append("Ответ выдумывает рокировку или подготовку к ней.")

    if move_facts.get("is_promotion"):
        promotion_piece = str(
            move_facts.get("promotion_piece") or ""
        ).lower()

        if promotion_future_claim:
            errors.append(
                "Ответ описывает превращение как будущую возможность или "
                "приближение к полю превращения, но оно уже происходит "
                "этим ходом."
            )

        if not promotion_positive:
            errors.append(
                "Ход уже превращает пешку в новую фигуру, но ответ "
                "не описывает завершённое превращение."
            )

        if (
            promotion_piece
            and not _piece_word_present(
                lowered,
                promotion_piece,
            )
        ):
            errors.append(
                "Не названа фигура превращения "
                f"«{promotion_piece}»."
            )

        promotion_square_mentions: set[str] = set()

        # "на поле a7 превращается...", "на b8 превращается..."
        for match in re.finditer(
            r"\bна\s+(?:поле\s+)?([a-h][1-8])"
            r"\s+превращ\w*",
            lowered,
        ):
            promotion_square_mentions.add(match.group(1))

        # "превращается ... на поле b8"
        for match in re.finditer(
            r"\bпревращ\w*[^.!?]{0,30}"
            r"\bна\s+(?:поле\s+)?([a-h][1-8])\b",
            lowered,
        ):
            promotion_square_mentions.add(match.group(1))

        expected_promotion_square = str(
            move_facts.get("to") or explained_move.get("to") or ""
        ).lower()

        wrong_promotion_squares = sorted(
            square
            for square in promotion_square_mentions
            if expected_promotion_square
            and square != expected_promotion_square
        )

        if wrong_promotion_squares:
            errors.append(
                "Неверно указано поле превращения: "
                f"{wrong_promotion_squares}; проверенное поле — "
                f"{expected_promotion_square}."
            )

        pawn_continues_after_promotion = bool(
            re.search(
                r"\bпешк\w*[^.!?]{0,45}\bпродолж\w*\s+движ",
                lowered,
            )
            or re.search(
                r"\bпозволя\w*\s+(?:ей|пешк\w*)"
                r"[^.!?]{0,35}\bпродолж\w*\s+движ",
                lowered,
            )
            or re.search(
                r"\bпосле\s+превращ\w*[^.!?]{0,35}"
                r"\bпешк\w*\b",
                lowered,
            )
        )
        if pawn_continues_after_promotion:
            errors.append(
                "После превращения пешка как пешка больше не существует; "
                "ответ ошибочно приписывает ей дальнейшее движение "
                "или действие после превращения."
            )
    elif promotion_positive or promotion_future_claim:
        errors.append("Ответ выдумывает превращение или подготовку к нему.")

    # --------------------------------------------------------
    # АТАКА НА КОРОЛЯ / УГРОЗЫ
    # --------------------------------------------------------
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if (
            "корол" in sentence_lower
            and re.search(
                r"атаку|атакует|под атак|угрож",
                sentence_lower,
            )
            and not move_facts.get("is_check")
        ):
            errors.append(
                "Ответ утверждает непосредственную атаку/угрозу королю, "
                "но после проверенного хода шаха нет."
            )

    # --------------------------------------------------------
    # УТВЕРЖДЕНИЯ О СЛЕДУЮЩЕМ ХОДЕ СОПЕРНИКА.
    # --------------------------------------------------------
    opponent_legal_moves = int(
        grounding.get("opponent_legal_moves") or 0
    )
    terminal_state = grounding.get("terminal") or {}

    opponent_move_claim = bool(
        re.search(
            r"вынужден\w*[^.!?]{0,55}?"
            r"(?:сделать\s+ход|пойти|перейти|ходить)",
            lowered,
        )
        or re.search(
            r"(?:должен|должны|прид[её]тся)[^.!?]{0,45}?"
            r"(?:сделать\s+ход|пойти|перейти|ходить)",
            lowered,
        )
        or re.search(
            r"сделать\s+ход\s+корол[её]?м",
            lowered,
        )
        or re.search(
            r"ход\s+корол[её]?м\s+(?:на|в)\s+(?:поле\s+)?[a-h][1-8]",
            lowered,
        )
    )

    if (
        opponent_move_claim
        and (
            terminal_state.get("is_game_over")
            or opponent_legal_moves == 0
        )
    ):
        errors.append(
            "Ответ утверждает, что соперник должен сделать следующий ход, "
            "но после проверенного хода у соперника нет легальных ходов."
        )

    no_opponent_moves_claim = bool(
        re.search(
            r"\b(?:больше\s+)?не\s+мог\w*[^.!?]{0,24}"
            r"(?:сделать\s+)?ход\w*\b",
            lowered,
        )
        or re.search(
            r"\bнет\s+(?:ни\s+одного\s+)?"
            r"(?:(?:легальн|доступн|возможн)\w*\s+)?"
            r"ход\w*\b",
            lowered,
        )
        or re.search(
            r"\bход(?:ов|а)?\s+(?:больше\s+)?не\s+"
            r"остал\w*\b",
            lowered,
        )
        or re.search(
            r"\bне\s+остал\w*[^.!?]{0,24}\bход\w*\b",
            lowered,
        )
        or re.search(
            r"\bлиш[её]н\w*[^.!?]{0,20}\bход\w*\b",
            lowered,
        )
    )

    if no_opponent_moves_claim and opponent_legal_moves > 0:
        errors.append(
            "Ответ утверждает, что у соперника не осталось доступных ходов, "
            f"но Python нашёл {opponent_legal_moves} легальн"
            + (
                "ый ответ."
                if opponent_legal_moves == 1
                else "ых ответа."
                if 2 <= opponent_legal_moves <= 4
                else "ых ответов."
            )
        )

    # --------------------------------------------------------
    # ИТОГ ПАРТИИ / НИЧЬЯ / ПОБЕДА
    # --------------------------------------------------------
    terminal = grounding.get("terminal") or {}
    actual_result = terminal.get("result") or "*"
    is_game_over = bool(terminal.get("is_game_over"))

    draw_claim = bool(
        re.search(
            r"\bничь(?:я|ей|ю|е)\b|\bвничью\b",
            lowered,
        )
    )
    white_win_claim = bool(
        re.search(
            r"(?:партия|игра)\s+(?:сразу\s+)?"
            r"(?:заканчива\w*|заверша\w*)\s+"
            r"(?:победой\s+)?бел",
            lowered,
        )
        or re.search(
            r"\bпобед(?:а|е|ой|у)\s+бел\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:привод\w*|привел\w*|привёл\w*|"
            r"вед[её]т|гарантир\w*)\s+к\s+"
            r"побед\w*\s+бел\w*\b",
            lowered,
        )
        or re.search(
            r"\bбел\w*\s+"
            r"(?:выигр\w*|побед(?:ил\w*|ят\w*|ил|или|ила|ило|а|ают|ит))\b",
            lowered,
        )
    )
    black_win_claim = bool(
        re.search(
            r"(?:партия|игра)\s+(?:сразу\s+)?"
            r"(?:заканчива\w*|заверша\w*)\s+"
            r"(?:победой\s+)?ч[её]рн",
            lowered,
        )
        or re.search(
            r"\bпобед(?:а|е|ой|у)\s+ч[её]рн\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:привод\w*|привел\w*|привёл\w*|"
            r"вед[её]т|гарантир\w*)\s+к\s+"
            r"побед\w*\s+ч[её]рн\w*\b",
            lowered,
        )
        or re.search(
            r"\bч[её]рн\w*\s+"
            r"(?:выигр\w*|побед(?:ил\w*|ят\w*|ил|или|ила|ило|а|ают|ит))\b",
            lowered,
        )
    )
    unsupported_future_claim = bool(
        re.search(
            r"\bмат\s+(?:неизбеж\w*|гарантир\w*|неотвратим\w*)\b",
            lowered,
        )
        or re.search(
            r"\b(?:неизбеж\w*|гарантир\w*|неотвратим\w*)\s+мат\b",
            lowered,
        )
        or re.search(
            r"\bпешк\w*\s+(?:нельзя|невозможно)\s+остановить\b",
            lowered,
        )
        or re.search(
            r"\b(?:неудержим\w*|неостановим\w*)\s+пешк\w*\b",
            lowered,
        )
        or re.search(
            r"\bкорол\w*\s+не\s+мож\w*\s+догнать\s+пешк\w*\b",
            lowered,
        )
        or re.search(
            r"\bпешк\w*\s+гарантир\w*\s+превращ\w*\b",
            lowered,
        )
    )

    unique_move_claim = bool(
        re.search(
            r"\bединственн\w*\s+"
            r"(?:(?:возможн\w*|допустим\w*)\s+)?"
            r"(?:ход|вариант|решени\w*)\b",
            lowered,
        )
        or re.search(
            r"\bтолько\s+(?:этот|данный|такой)\s+ход\b",
            lowered,
        )
        or re.search(
            r"\b(?:другого|иных|альтернативн\w*)\s+"
            r"(?:хода|ходов|варианта|вариантов)\s+нет\b",
            lowered,
        )
        or re.search(
            r"\bнет\s+(?:другого|иных|альтернативн\w*)\s+"
            r"(?:хода|ходов|варианта|вариантов)\b",
            lowered,
        )
    )

    generic_win_claim = bool(
        re.search(
            r"\b(?:выигрывая|побеждая)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)\b",
            lowered,
        )
        or re.search(
            r"\b(?:обеспечивая|принося|гарантируя)\s+"
            r"(?:немедленную\s+|сразу\s+)?побед\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:обеспечива\w*|принос\w*|гарантир\w*)\s+"
            r"(?:немедленную\s+|сразу\s+)?побед\w*\b",
            lowered,
        )
        or re.search(
            r"\b(?:вед[её]т|привод\w*|привел\w*|привёл\w*)\s+"
            r"к\s+(?:немедленн\w*\s+)?(?:выигрыш\w*|побед\w*)\b",
            lowered,
        )
    )

    resignation_claim = bool(
        re.search(
            r"\b(?:"
            r"сдаться|"
            r"сда[её]тся|сдаются|"
            r"сдал(?:ся|ась|ись)|"
            r"капитулиру\w*|"
            r"капитуляц\w*"
            r")\b",
            lowered,
        )
    )

    if resignation_claim:
        errors.append(
            "Ответ выдумывает сдачу или капитуляцию. "
            "Проверенные факты описывают позицию после хода, "
            "но не содержат события сдачи."
        )

    generic_terminal_claim = bool(
        re.search(
            r"(?:партия|игра)\s+"
            r"(?:заканчива\w*|заверша\w*|оканчива\w*|окончен\w*)",
            lowered,
        )
        or re.search(
            r"(?:заканчива\w*|заверша\w*|оканчива\w*)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)",
            lowered,
        )
        or re.search(
            r"(?:завершая|заканчивая|оканчивая)\s+"
            r"(?:эту\s+|данную\s+)?(?:партию|игру)",
            lowered,
        )
        or re.search(
            r"(?:партия|игра)\s+(?:уже\s+)?"
            r"(?:закончена|завершена|окончена)",
            lowered,
        )
    )

    if unsupported_future_claim:
        errors.append(
            "Ответ делает неподтверждённый прогноз о будущем позиции "
            "(например, неизбежный мат, неудержимая пешка или невозможность "
            "догнать её). Такие выводы не были подтверждены Python."
        )

    if unique_move_claim:
        errors.append(
            "Ответ утверждает, что ход является единственным, "
            "но уникальность хода не была подтверждена Python."
        )

    if generic_win_claim and not is_game_over:
        errors.append(
            "Ответ утверждает, что этим ходом партия уже выигрывается, "
            "но после проверенного хода позиция не является терминальной."
        )

    if draw_claim and actual_result != "1/2-1/2":
        errors.append(
            "Ответ утверждает ничью, но после проверенного хода "
            "партия не заканчивается вничью."
        )

    if white_win_claim and actual_result != "1-0":
        errors.append(
            "Ответ утверждает немедленную победу белых, но такой "
            "результат после этого хода не наступает."
        )

    if black_win_claim and actual_result != "0-1":
        errors.append(
            "Ответ утверждает немедленную победу чёрных, но такой "
            "результат после этого хода не наступает."
        )

    # Любое "партия заканчивается ..." запрещаем в нетерминальной позиции,
    # даже если модель не уточнила результат.
    if generic_terminal_claim and not is_game_over:
        errors.append(
            "Ответ утверждает, что партия заканчивается после этого хода, "
            "но позиция после хода не является терминальной."
        )

    decisive_finish_claim = bool(
        re.search(
            r"(?:реша\w*\s+(?:исход|партию|игру)|"
            r"заверша\w*\s+(?:партию|игру)|"
            r"заканчива\w*\s+(?:партию|игру))",
            lowered,
        )
    )
    if decisive_finish_claim and not is_game_over:
        errors.append(
            "Ответ приписывает ходу немедленное завершение партии, "
            "хотя после хода игра продолжается."
        )

    # --------------------------------------------------------
    # НЕПОДТВЕРЖДЁННЫЕ ЯРЛЫКИ И СИЛЬНЫЕ ВЫВОДЫ
    # --------------------------------------------------------
    if re.search(r"\bгамбит\w*\b", lowered):
        errors.append(
            "Название гамбита не подтверждено программным контекстом."
        )

    if re.search(
        r"\b(?:дебют|защита|система)\s+[а-яёa-z-]+",
        lowered,
    ):
        errors.append(
            "Конкретное название дебюта/системы не подтверждено."
        )

    strong_patterns = [
        r"выигрыва\w*\s+(?:ферз|ладь|слон|кон|пеш|фигур|материал)",
        r"теря\w*\s+(?:ферз|ладь|слон|кон|пеш|фигур|материал)",
        r"жертв\w*",
        r"вынужда\w*\s+соперник",
        r"форсир\w*\s+(?:выигрыш|мат)",
        r"решающ\w*\s+преимуществ",
        r"победн\w*\s+позици",
        r"вед[её]т\s+к\s+(?:ничь|побед)",
        r"гарантир\w*\s+(?:ничь|побед)",
    ]
    for pattern in strong_patterns:
        if re.search(pattern, lowered):
            errors.append(
                "Ответ содержит сильный тактический/материальный вывод, "
                "которого нет в compact grounding."
            )
            break

    # --------------------------------------------------------
    # НЕПОДТВЕРЖДЁННЫЕ ОЦЕНОЧНЫЕ ЦЕЛИ / БЕЗОПАСНОСТЬ.
    # --------------------------------------------------------
    king_safety_supported = bool(
        grounding.get("king_safety_supported")
    )
    generic_defense_supported = bool(
        grounding.get("generic_defense_supported")
    )

    safety_claim = bool(
        re.search(
            r"\bбезопасн\w*\b"
            r"|\bукрыва\w*\s+корол"
            r"|\bобезопас\w*\s+корол"
            r"|\bзащища\w*\s+корол",
            lowered,
        )
    )
    if safety_claim and not king_safety_supported:
        errors.append(
            "Ответ делает вывод о безопасности/укрытии короля, "
            "которого нет в compact grounding."
        )

    generic_defense_claim = bool(
        re.search(
            r"\bдля\s+защит\w*\b"
            r"|\bс\s+целью\s+защит\w*\b"
            r"|\bдля\s+обороны\b"
            r"|\bмож(?:ет|гут|но)\b[^.!?]{0,30}\bзащища\w*\b"
            r"|\bзащища\w*\s+корол",
            lowered,
        )
    )
    if generic_defense_claim and not generic_defense_supported:
        errors.append(
            "Ответ приписывает ходу общую защитную цель, "
            "которая не подтверждена compact grounding."
        )

    if move_facts.get("is_castling"):
        unsupported_castling_strategy_claim = bool(
            re.search(
                r"\bсоединя\w*\b"
                r"|\bактивизир\w*\b"
                r"|\bвводи\w*\s+ладь"
                r"|\bразвива\w*\s+ладь"
                r"|\bразвити\w*\s+ладь"
                r"|\bулучша\w*\s+(?:положени|позици)\w*\s+ладь",
                lowered,
            )
        )
        if unsupported_castling_strategy_claim:
            errors.append(
                "Ответ добавляет стратегический эффект рокировки, "
                "которого нет среди переданных проверенных фактов."
            )

    # --------------------------------------------------------
    # ПЕДАГОГИЧЕСКИЕ ВЫВОДЫ
    # --------------------------------------------------------
    if (
        re.search(
            r"контрол\w*\s+центр|влияни\w*\s+на\s+центр|"
            r"занима\w*\s+центр|борьб\w*\s+за\s+центр",
            lowered,
        )
        and not grounding["center_supported"]
    ):
        errors.append(
            "Вывод о центре не подтверждён compact grounding."
        )

    if (
        "развит" in lowered
        and not grounding["development_supported"]
    ):
        errors.append(
            "Вывод о развитии фигур не подтверждён compact grounding."
        )

    if (
        re.search(
            r"открыва\w*\s+(?:лини|диагон)|"
            r"освобожда\w*\s+(?:лини|диагон)",
            lowered,
        )
        and not grounding["line_opening_supported"]
    ):
        errors.append(
            "Утверждение об открывшейся линии/диагонали не подтверждено."
        )

    # --------------------------------------------------------
    # НЕСТАНДАРТНАЯ ОПИСАТЕЛЬНАЯ НОТАЦИЯ.
    # GigaChess иногда генерирует QB2-KR2 / KB1 и похожие обозначения.
    # Для нашего API допустимы только обычные клетки a1-h8 и UCI/SAN.
    # --------------------------------------------------------
    descriptive_notation = re.findall(
        r"(?<![A-Za-zА-Яа-я0-9])"
        r"[KQRBN]{1,3}[1-8]"
        r"(?![A-Za-zА-Яа-я0-9])",
        text,
    )

    if descriptive_notation:
        errors.append(
            "Ответ использует неподдерживаемую описательную шахматную "
            f"нотацию {sorted(set(descriptive_notation))}. "
            "Используй только обычные координаты a1-h8 или естественный текст."
        )

    # --------------------------------------------------------
    # ФОРМАТ: мягче, чем раньше.
    # Одно содержательное предложение допустимо; факты важнее формы.
    # --------------------------------------------------------
    if len(text) > 1400:
        errors.append(
            f"Ответ слишком длинный ({len(text)} символов)."
        )

    if len(sentences) > 5:
        errors.append(
            f"Ответ слишком раздроблен: {len(sentences)} предложений."
        )

    if any(token in text for token in ("```", "{", "}")):
        errors.append(
            "Нужен обычный текст без Markdown/JSON."
        )

    errors = list(dict.fromkeys(errors))
    return not errors, errors


# ============================================================
# ДЕТЕРМИНИРОВАННОЕ ОБЪЯСНЕНИЕ
# ============================================================

def _deterministic_explanation(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    facts: dict,
) -> str:

    piece = board_before.piece_at(
        move.from_square
    )

    if piece is None:

        return (
            "Stockfish рекомендует этот ход. "
            "Фигура перемещается на указанное поле. "
            "Точные данные хода получены из позиции."
        )

    san = board_before.san(
        move
    )

    quality_text = (
        f"Stockfish рекомендует ход {san}. "
    )

    from_square = chess.square_name(
        move.from_square
    )

    to_square = chess.square_name(
        move.to_square
    )

    piece_text = _piece_name_capitalized(
        piece
    )

    # --------------------------------------------------------
    # ПРЕВРАЩЕНИЕ — обрабатываем раньше шаха/взятия, потому что
    # превращение может одновременно быть взятием, шахом или матом.
    # --------------------------------------------------------

    if facts.get("is_promotion"):

        promotion_piece = str(
            facts.get("promotion_piece") or "фигура"
        ).lower()
        promotion_piece_acc = _piece_word_accusative(
            promotion_piece
        )

        movement_text = (
            f"Пешка переходит с {from_square} на {to_square}"
        )

        if facts.get("is_capture"):
            captured = str(
                facts.get("captured_piece") or "фигура соперника"
            )
            captured_acc = _piece_name_accusative(captured)
            movement_text += (
                f", берёт {captured_acc} на {to_square}"
            )

        movement_text += (
            f" и превращается в {promotion_piece_acc}. "
        )

        opponent_king = board_after.king(board_after.turn)
        opponent_king_square = (
            chess.square_name(opponent_king)
            if opponent_king is not None
            else None
        )

        if facts.get("is_checkmate"):
            if opponent_king_square:
                movement_text += (
                    f"После превращения {promotion_piece} на {to_square} "
                    f"ставит мат королю соперника на "
                    f"{opponent_king_square}."
                )
            else:
                movement_text += (
                    f"После превращения {promotion_piece} ставит мат."
                )
        elif facts.get("is_check"):
            if opponent_king_square:
                movement_text += (
                    f"После превращения {promotion_piece} на {to_square} "
                    f"объявляет шах королю соперника на "
                    f"{opponent_king_square}."
                )
            else:
                movement_text += (
                    f"После превращения {promotion_piece} объявляет шах."
                )

        return quality_text + movement_text

    # --------------------------------------------------------
    # МАТ
    # --------------------------------------------------------

    if facts.get(
        "is_checkmate"
    ):

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на {to_square} "
            f"и ставит мат. "
            f"После этого у соперника нет "
            f"легального ответа."
        )

    # --------------------------------------------------------
    # ВЗЯТИЕ НА ПРОХОДЕ
    # --------------------------------------------------------

    if facts.get("is_en_passant"):

        captured_square = str(
            facts.get("captured_square") or ""
        )
        mover_color = str(
            facts.get("piece_color") or ""
        ).lower()
        mover_side = "белая" if mover_color == "white" else "чёрная"
        captured_side = "чёрную" if mover_color == "white" else "белую"

        return (
            quality_text
            + f"{mover_side.capitalize()} пешка выполняет взятие на проходе: "
            f"переходит с {from_square} на {to_square} и снимает "
            f"{captured_side} пешку с {captured_square}."
        )

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------

    if facts.get(
        "is_capture"
    ):

        captured = facts.get(
            "captured_piece"
        )

        if captured:

            captured_square = (
                facts.get(
                    "captured_square"
                )
                or to_square
            )

            captured_acc = _piece_name_accusative(
                str(captured)
            )

            capture_geometry = ""

            if (
                facts.get("movement_type") == "diagonal"
                and piece.piece_type in {
                    chess.BISHOP,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по диагонали"

            elif (
                facts.get("movement_type") == "horizontal"
                and piece.piece_type in {
                    chess.ROOK,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по горизонтали"

            elif (
                facts.get("movement_type") == "vertical"
                and piece.piece_type in {
                    chess.ROOK,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по вертикали"

            elif (
                facts.get("movement_type") == "knight_jump"
                and piece.piece_type == chess.KNIGHT
            ):
                capture_geometry = " ходом буквой «Г»"

            elif (
                facts.get("movement_type") == "king_step"
                and piece.piece_type == chess.KING
            ):
                capture_geometry = " на соседнюю клетку"

            elif (
                facts.get("movement_type") == "pawn"
                and piece.piece_type == chess.PAWN
                and facts.get("is_capture")
                and not facts.get("is_en_passant")
                and not facts.get("is_promotion")
            ):
                capture_geometry = " по диагонали вперёд на одну клетку"

            elif (
                facts.get("movement_type") == "pawn"
                and piece.piece_type == chess.PAWN
                and not facts.get("is_capture")
                and not facts.get("is_en_passant")
                and not facts.get("is_promotion")
                and not facts.get("pawn_double_step")
            ):
                capture_geometry = " прямо вперёд на одну клетку"

            return (
                quality_text
                + f"{piece_text} переходит с "
                f"{from_square} на {to_square}"
                f"{capture_geometry} и берёт "
                f"{captured_acc} на "
                f"{captured_square}. "
                f"После хода взятая фигура "
                f"удалена с доски."
            )

    # --------------------------------------------------------
    # ШАХ
    # --------------------------------------------------------

    if facts.get(
        "is_check"
    ):

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на {to_square} "
            f"и даёт шах. "
            f"Король соперника находится "
            f"под атакой."
        )

    # --------------------------------------------------------
    # РОКИРОВКА
    # --------------------------------------------------------

    if facts.get(
        "is_castling"
    ):

        is_kingside = (
            chess.square_file(move.to_square)
            > chess.square_file(move.from_square)
        )
        side = "короткая" if is_kingside else "длинная"

        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(
            7 if is_kingside else 0,
            rank,
        )
        rook_to = chess.square(
            5 if is_kingside else 3,
            rank,
        )

        rook_from_name = chess.square_name(rook_from)
        rook_to_name = chess.square_name(rook_to)

        return (
            quality_text
            + f"Этим ходом выполняется {side} рокировка: "
            f"король переходит с {from_square} на {to_square}, "
            f"а ладья одновременно переходит с "
            f"{rook_from_name} на {rook_to_name}."
        )

    # --------------------------------------------------------
    # ПЕШКА
    # --------------------------------------------------------

    if piece.piece_type == chess.PAWN:

        if facts.get(
            "pawn_double_step"
        ):

            movement_text = (
                f"Пешка переходит с "
                f"{from_square} на "
                f"{to_square} двойным шагом."
            )

        else:

            movement_text = (
                f"Пешка переходит с "
                f"{from_square} на "
                f"{to_square} прямо вперёд "
                "на одну клетку."
            )

        if facts.get(
            "pawn_reaches_center"
        ):

            movement_text += (
                f" Она занимает "
                f"центральное поле "
                f"{to_square}."
            )

        new_controls = facts.get(
            "pawn_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + movement_text
                + " После хода пешка "
                "контролирует "
                + _join_squares(
                    new_controls
                )
                + "."
            )

        return (
            quality_text
            + movement_text
            + " После хода меняется "
            "набор полей, которые "
            "контролирует пешка."
        )

    # --------------------------------------------------------
    # КОНЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.KNIGHT:

        new_controls = facts.get(
            "knight_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} прыгает "
                f"с {from_square} на "
                f"{to_square}. "
                f"После хода конь получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} прыгает "
            f"с {from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых конём."
        )

    # --------------------------------------------------------
    # СЛОН
    # --------------------------------------------------------

    if piece.piece_type == chess.BISHOP:

        new_controls = facts.get(
            "bishop_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"по диагонали с "
                f"{from_square} на "
                f"{to_square}. "
                f"После хода слон получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"по диагонали с "
            f"{from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых слоном."
        )

    # --------------------------------------------------------
    # ЛАДЬЯ
    # --------------------------------------------------------

    if piece.piece_type == chess.ROOK:

        direction = (
            "по вертикали"
            if facts.get(
                "movement_type"
            ) == "vertical"
            else "по горизонтали"
        )

        new_controls = facts.get(
            "rook_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"с {from_square} на "
                f"{to_square} {direction}. "
                f"После хода ладья получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"с {from_square} на "
            f"{to_square} {direction}. "
            f"После хода меняется набор "
            f"полей, контролируемых ладьёй."
        )

    # --------------------------------------------------------
    # ФЕРЗЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.QUEEN:

        movement_type = facts.get(
            "movement_type"
        )

        if movement_type == "diagonal":

            direction = "по диагонали"

        elif movement_type == "vertical":

            direction = "по вертикали"

        else:

            direction = "по горизонтали"

        new_controls = facts.get(
            "queen_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"с {from_square} на "
                f"{to_square} {direction}. "
                f"После хода ферзь получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"с {from_square} на "
            f"{to_square} {direction}. "
            f"После хода меняется набор "
            f"полей, контролируемых ферзём."
        )

    # --------------------------------------------------------
    # КОРОЛЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.KING:

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых королём."
        )

    return (
        quality_text
        + f"{piece_text} переходит с "
        f"{from_square} на "
        f"{to_square}. "
        f"Это конкретное перемещение "
        f"Stockfish рекомендует "
        f"в данной позиции."
    )


# ============================================================
# ПРОВЕРЕННЫЙ КОНТЕКСТ ДЛЯ GIGACHESS
# ============================================================

def _build_verified_explanation_context(
    *,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
) -> str:
    """
    Возвращает ТОЛЬКО compact grounding, реально отправляемый модели.
    Подробные Python-факты остаются в API/debug, но не засоряют prompt.
    """
    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )
    return grounding["text"]


def _render_compact_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
    retry_focus: list[str] | None = None,
) -> list[dict]:
    """
    Реальный production-формат по успешному эксперименту E:

      FEN в content
      + тот же FEN в attachments
      + максимум 5 коротких фактов
      + одно user message.

    Retry не получает предыдущий ответ и не получает его ошибочные клетки.
    Он генерирует текст с нуля из того же чистого grounding.
    """
    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    focus_block = ""
    if retry_focus:
        focus_block = (
            "\n\nПосле автоматической проверки предыдущая генерация "
            "была отклонена. Сгенерируй новый ответ с нуля.\n"
            + "\n".join(f"- {item}" for item in retry_focus[:3])
        )

    user = f"""
Позиция FEN:
{fen}

Объясни ход Stockfish {explained_move['uci']} как шахматный тренер для игрока рейтинга около {elo} Elo.

Проверенные факты:
{grounding['text']}

Используй только эти шахматные факты.
Если среди фактов явно указан тип траектории «по диагонали», «по горизонтали», «по вертикали» или «ходом буквой „Г“», сохрани именно этот тип движения.
Для коня формулировка «ходом буквой „Г“» означает его особый прыжок; не называй такой ход диагональю, горизонталью или вертикалью.
Для обычного хода короля формулировка «на соседнюю клетку» означает переход ровно на одно соседнее поле; не называй такой ход прыжком или рокировкой.
Если проверенный ход — обычное взятие пешкой, обязательно скажи, что пешка берёт по диагонали вперёд. Не заменяй это общим «белые/чёрные берут фигуру»: назови пешку явно.
Если проверенный ход — обычный ход пешки без взятия и без двойного шага, обязательно скажи, что пешка идёт прямо вперёд на одну клетку.
Важно: для линейных фигур тип траектории описывает движение ОТ исходного поля ДО конечного поля. Не переопределяй его по букве или цифре конечной клетки: например, поле h1 само по себе не означает, что ход был «по вертикали h».
Если среди фактов есть взятие, шах, мат, рокировка или превращение, обязательно опиши это как событие, уже происходящее в данном ходе, а не как будущую возможность.
При рокировке обязательно упомяни оба перемещения — короля и ладьи — если они указаны среди фактов.
Не добавляй цели вроде «безопасное поле», «укрывает короля», «для защиты» или другие стратегические оценки, если их нет среди проверенных фактов.
Не утверждай, что у соперника нет ходов или что он не может сделать ход, если это прямо не указано среди проверенных фактов.
Не утверждай и косвенно, что ход уже выигрывает партию («выигрывая партию», «принося победу»), если немедленная победа не указана среди проверенных фактов.
Не называй ход «единственным», «единственно возможным» или «единственным вариантом»: backend не передаёт подтверждённый факт уникальности хода.
Не прогнозируй «неизбежный мат», «неудержимую пешку», «король не может догнать пешку» или гарантированное превращение, если такого проверенного факта нет.
Пиши грамматически корректно по-русски: согласовывай род и падеж («белая ладья», «берёт чёрного ферзя»).
Не оставляй незавершённые конструкции вроде «на поле» без координаты; если координата не нужна, просто не упоминай поле.
Используй только обычные координаты вида e1, g1, h1, f1; не используй описательную нотацию вроде QB2 или KR2.
Не называй другие конкретные клетки, фигуры, взятия, угрозы, дебюты или тактические события.
Свяжи факты в естественное и понятное объяснение.
Не перечисляй факты механически.
Напиши 1-4 коротких предложения.
{focus_block}
Верни только объяснение обычным русским текстом.
""".strip()

    return [{
        "role": "user",
        "content": user,
        "attachments": [fen],
    }]


def _build_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
) -> list[dict]:
    return _render_compact_prompt(
        fen=fen,
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
        elo=elo,
        retry_focus=None,
    )


def _compact_retry_focus(
    validation_errors: list[str],
) -> list[str]:
    """
    Превращает подробные ошибки validator в короткие БЕЗОПАСНЫЕ подсказки.
    Неверные координаты/предыдущий ответ обратно в prompt не попадают.
    """
    lowered = " ".join(validation_errors).lower()
    focus: list[str] = []

    if any(
        token in lowered
        for token in (
            "ход ",
            "перемещ",
            "координ",
            "клет",
            "поле назначения",
            "uci",
        )
    ):
        focus.append(
            "Особенно внимательно сохрани точный ход и клетки из проверенных фактов."
        )

    if any(
        token in lowered
        for token in (
            "фигур",
            "пешк",
            "конь",
            "слон",
            "ладь",
            "ферз",
            "корол",
        )
    ):
        focus.append(
            "Не меняй тип объясняемой фигуры и не добавляй другие фигуры."
        )

    if (
        "совершившимся взятием" in lowered
        or "формулировка «может взять»" in lowered
        or "не названа взятая фигура" in lowered
    ):
        focus.append(
            "Обязательно опиши взятие как уже совершившееся этим ходом "
            "и не опускай тип взятой фигуры."
        )

    if "ход даёт шах" in lowered or "шах, но ответ" in lowered:
        focus.append(
            "Обязательно скажи, что именно этим ходом сопернику объявлен шах."
        )

    if (
        "ход ставит мат" in lowered
        or "ход уже ставит мат" in lowered
        or "будущую возможность" in lowered
        or "мат, но ответ" in lowered
        or "нет легальных ходов" in lowered
        or "поле короля соперника" in lowered
    ):
        focus.append(
            "Это уже мат данным ходом, а не угроза мата и не возможность "
            "поставить мат следующим ходом. Скажи прямо, что фигура приходит "
            "на проверенное поле назначения и этим ходом ставит мат."
        )

    if (
        "ход является рокировкой" in lowered
        or "ход уже является рокировкой" in lowered
        or "будущую возможность" in lowered
        or "подготовку" in lowered
    ):
        focus.append(
            "Это уже выполненная рокировка данным ходом, а не подготовка "
            "к ней. Скажи прямо, что король и ладья одновременно переходят "
            "на проверенные поля."
        )

    if (
        "безопасности/укрытии короля" in lowered
        or "общую защитную цель" in lowered
        or "стратегический эффект рокировки" in lowered
        or "должно упомянуть ладью" in lowered
    ):
        focus.append(
            "Для этой рокировки не добавляй слова про безопасность, "
            "защиту, соединение или активизацию фигур. Опиши только "
            "проверенную механику: это уже рокировка; король и ладья "
            "одновременно переходят на указанные клетки."
        )

    if (
        "взятия на проходе" in lowered
        or "поле снятой пешки" in lowered
        or "специальный механизм" in lowered
    ):
        focus.append(
            "Это уже выполненное взятие на проходе. Обязательно назови сам "
            "механизм и отдельно укажи поле назначения движущейся пешки "
            "и поле, с которого снимается чужая пешка."
        )

    if (
        "ход является превращением" in lowered
        or "ход уже превращает пешку" in lowered
        or "будущую возможность" in lowered
        or "поля превращения" in lowered
        or "после превращения пешка" in lowered
    ):
        focus.append(
            "Превращение уже завершается этим ходом: пешка становится "
            "проверенной новой фигурой. Если ход даёт шах/мат, опиши его "
            "как уже возникший (например, «объявлен шах» или "
            "«новая фигура объявляет шах»). Координаты можно вообще "
            "не называть."
        )

    if any(
        token in lowered
        for token in (
            "не осталось доступных ходов",
            "не может сделать ход",
            "легальный ответ",
            "легальных ответа",
            "легальных ответов",
        )
    ):
        focus.append(
            "Не утверждай, что у соперника нет ходов. "
            "После проверенного хода у него остаётся как минимум один "
            "легальный ответ."
        )

    if any(
        token in lowered
        for token in (
            "ничью",
            "вничью",
            "партия заканчивается",
            "игра продолжается",
            "позиция после хода не является терминальной",
            "немедленную победу",
            "победу белых",
            "победу чёрных",
            "победе белых",
            "победе чёрных",
            "немедленное завершение партии",
        )
    ):
        focus.append(
            "Не объявляй ничью, победу или окончание партии, если этого "
            "нет среди проверенных фактов."
        )

    if any(
        token in lowered
        for token in (
            "выдумывает",
            "не подтвержден",
            "гамбит",
            "дебют",
            "угроз",
        )
    ):
        focus.append(
            "Не добавляй события или тактические выводы, которых нет в фактах."
        )

    if any(
        token in lowered
        for token in (
            "неподтверждённый прогноз",
            "неизбежный мат",
            "неудержимая пешка",
            "невозможность догнать",
        )
    ):
        focus.append(
            "Не делай прогнозов о будущем: не называй мат неизбежным, "
            "пешку неудержимой и не утверждай, что король не сможет её догнать, "
            "если это не дано среди проверенных фактов."
        )

    if any(
        token in lowered
        for token in (
            "является единственным",
            "уникальность хода",
        )
    ):
        focus.append(
            "Не утверждай, что ход единственный или что других вариантов нет. "
            "Уникальность хода Python не подтверждала."
        )

    if any(
        token in lowered
        for token in (
            "партия уже выигрывается",
            "немедленную победу",
            "результат после этого хода не наступает",
        )
    ):
        focus.append(
            "Не утверждай победу, выигрыш партии или немедленное завершение, "
            "если такого терминального результата нет среди проверенных фактов."
        )

    if any(
        token in lowered
        for token in (
            "обычного хода пешки",
            "прямо вперёд на одну клетку",
            "ошибочно описан как двойной ход",
            "выдумывает взятие",
        )
    ):
        focus.append(
            "Назови пешку явно и укажи правило: она идёт прямо вперёд "
            "на одну клетку. Не добавляй взятие, диагональ, двойной ход, "
            "короля, мат, победу или прогноз о дальнейшем превращении."
        )

    if any(
        token in lowered
        for token in (
            "пешечного взятия",
            "назвать пешку",
            "по диагонали вперёд",
        )
    ):
        focus.append(
            "Назови пешку явно и укажи правило этого хода: "
            "она берёт по диагонали вперёд. Не заменяй это фразой "
            "«белые/чёрные берут»."
        )

    if any(
        token in lowered
        for token in (
            "неверном падеже",
            "согласование цвета",
            "незавершённую ссылку на клетку",
            "корректной форме",
        )
    ):
        focus.append(
            "Исправь русскую грамматику и не оставляй оборванных ссылок "
            "на поле. Для взятия используй корректный объект: например, "
            "«белая ладья берёт чёрного ферзя»."
        )

    if any(
        token in lowered
        for token in (
            "геометрия хода",
            "диагональный ход",
            "тип движения",
            "диагональ",
            "вертикаль",
            "горизонталь",
        )
    ):
        focus.append(
            "Сохрани проверенный тип движения и назови его явно: "
            "по диагонали, по горизонтали, по вертикали, ходом буквой «Г» "
            "для коня или на соседнюю клетку для обычного хода короля — "
            "строго согласно проверенному факту."
        )

    if any(
        token in lowered
        for token in (
            "контрол",
            "защит",
            "отношен",
        )
    ):
        focus.append(
            "Не меняй связь между фигурой и полями контроля/защиты."
        )

    if any(
        token in lowered
        for token in (
            "слишком корот",
            "60 символ",
        )
    ):
        focus.append(
            "Сделай объяснение чуть подробнее, но не добавляй новых шахматных фактов."
        )

    if not focus:
        focus.append(
            "Строго используй только перечисленные проверенные факты."
        )

    return list(dict.fromkeys(focus))[:3]




def _render_protected_castling_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Repair рокировки с защищёнными слотами.

    Модель отвечает только за естественную русскую формулировку.
    Реальные координаты ей не передаются и подставляются Python после ответа.
    """
    side = "короткая" if move_facts.get("to") in {"g1", "g8"} else "длинная"

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй естественное короткое объяснение уже выполненной {side} рокировки.

Проверенная механика:
- Король переходит с <KING_FROM> на <KING_TO>.
- Ладья одновременно переходит с <ROOK_FROM> на <ROOK_TO>.
- Это уже выполненная {side} рокировка.
- В этом ходе нет взятия, шаха или мата.

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Сохрани маркеры <KING_FROM>, <KING_TO>, <ROOK_FROM>, <ROOK_TO> ТОЧНО в таком виде.
- Каждый из четырёх маркеров используй ровно один раз.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши вообще никаких других координат вида a1-h8.
- Не добавляй безопасность, защиту, развитие, активизацию, угрозы или стратегическую пользу.
- Не используй QB2, KR2, KB1 и подобную описательную нотацию.
- Прямо скажи, что это рокировка, уже выполняемая данным ходом.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    # В strict repair FEN намеренно отсутствует:
    # позицию уже разобрал Python, повторный анализ модели не нужен.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_castling_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    grounding: dict,
) -> tuple[str, list[str]]:
    """
    Проверяет защищённые слоты и только после этого подставляет координаты.
    """
    errors: list[str] = []

    slot_values = {
        "<KING_FROM>": str(explained_move.get("from") or "").lower(),
        "<KING_TO>": str(explained_move.get("to") or "").lower(),
        "<ROOK_FROM>": str(grounding.get("castling_rook_from") or "").lower(),
        "<ROOK_TO>": str(grounding.get("castling_rook_to") or "").lower(),
    }

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)
        if count != 1:
            errors.append(
                f"Защищённый маркер {slot} должен встречаться ровно один раз; "
                f"получено: {count}."
            )
        if not value:
            errors.append(
                f"Для маркера {slot} отсутствует проверенное значение."
            )

    # В raw-ответе вообще не должно быть шахматных координат:
    # только защищённые слоты.
    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected repair модель самостоятельно назвала координаты "
            f"{raw_squares}; координаты должны приходить только из Python-слотов."
        )

    descriptive = re.findall(
        r"(?<![A-Za-zА-Яа-я0-9])[KQRBN]{1,3}[1-8]"
        r"(?![A-Za-zА-Яа-я0-9])",
        raw_answer,
    )
    if descriptive:
        errors.append(
            "В protected repair использована неподдерживаемая описательная "
            f"нотация: {sorted(set(descriptive))}."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        materialized = materialized.replace(slot, value)

    return materialized, []




def _render_protected_capture_promotion_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Жёсткий repair для комбинации:
        capture + promotion [+ check | mate]

    Вместо множества мелких координатных slots модель получает
    несколько атомарных смысловых маркеров. Python позже заменяет
    каждый маркер целиком проверенным событием.

    GigaChess отвечает только за связность, порядок и естественный
    русский текст между событиями.
    """
    required_events = [
        "<CAPTURE_EVENT>",
        "<PROMOTION_EVENT>",
    ]

    if move_facts.get("is_checkmate"):
        required_events.append("<MATE_EVENT>")
    elif move_facts.get("is_check"):
        required_events.append("<CHECK_EVENT>")

    events_line = " → ".join(required_events)

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй короткое естественное объяснение одного шахматного хода.

Этот ход состоит из уже совершившихся событий.
Используй protected-маркеры строго в таком порядке:

{events_line}

Предыдущая генерация была отклонена:
{focus_text}

Что означают маркеры:
- <CAPTURE_EVENT> — уже выполненное взятие проверенной фигуры.
- <PROMOTION_EVENT> — уже выполненное превращение пешки.
- <CHECK_EVENT> — уже объявленный этим ходом шах.
- <MATE_EVENT> — уже поставленный этим ходом мат.

Жёсткие правила:
- Используй каждый маркер из указанной последовательности РОВНО один раз.
- Не пропускай ни один обязательный маркер.
- Не добавляй другие маркеры.
- Не заменяй маркеры своими названиями фигур или координатами.
- Не пиши собственные шахматные координаты вида a1-h8.
- Сохрани порядок событий: взятие → превращение → шах/мат.
- Можно добавлять только естественные связующие слова и пунктуацию.
- Не добавляй другие фигуры, взятия, угрозы, сдачу, ничью,
  победу, дебют или стратегические оценки.
- Верни 1-2 коротких предложения.
- Верни только объяснение.

Пример допустимой структуры без раскрытия фактов:
«Пешка <CAPTURE_EVENT> и <PROMOTION_EVENT>. После этого <CHECK_EVENT>.»
Используй только те маркеры, которые перечислены в обязательной
последовательности выше.
""".strip()

    # Никакого FEN и никаких реальных фигур/координат.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_capture_promotion_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
    derived_facts: dict,
) -> tuple[str, list[str]]:
    """
    Материализует атомарные события capture+promotion.
    """
    errors: list[str] = []

    from_square = str(
        explained_move.get("from") or ""
    ).lower()
    to_square = str(
        explained_move.get("to") or ""
    ).lower()

    captured_piece = str(
        move_facts.get("captured_piece") or ""
    )
    captured_acc = _piece_name_accusative(
        captured_piece
    )
    captured_square = str(
        move_facts.get("captured_square")
        or explained_move.get("to")
        or ""
    ).lower()

    promotion_piece = str(
        move_facts.get("promotion_piece") or ""
    ).lower()
    promotion_acc = _piece_word_accusative(
        promotion_piece
    )

    terminal = derived_facts.get("terminal") or {}
    king_square = str(
        terminal.get("opponent_king_square") or ""
    ).lower()

    event_values = {
        "<CAPTURE_EVENT>": (
            f"переходит с {from_square} на {to_square} и берёт "
            f"{captured_acc} на {captured_square}"
        ),
        "<PROMOTION_EVENT>": (
            f"на {to_square} превращается в {promotion_acc}"
        ),
    }

    required_events = [
        "<CAPTURE_EVENT>",
        "<PROMOTION_EVENT>",
    ]

    if move_facts.get("is_checkmate"):
        event_values["<MATE_EVENT>"] = (
            f"после превращения {promotion_piece} на {to_square} "
            f"ставит мат королю соперника на {king_square}"
        )
        required_events.append("<MATE_EVENT>")

    elif move_facts.get("is_check"):
        event_values["<CHECK_EVENT>"] = (
            f"после превращения {promotion_piece} на {to_square} "
            f"объявляет шах королю соперника на {king_square}"
        )
        required_events.append("<CHECK_EVENT>")

    # Каждый обязательный атомарный event — ровно один раз.
    for event in required_events:
        count = raw_answer.count(event)
        if count != 1:
            errors.append(
                f"Protected event {event} должен встречаться "
                f"ровно один раз; получено: {count}."
            )

    known_events = set(event_values)
    unknown_events = sorted(
        set(re.findall(r"<[A-Z_]+>", raw_answer))
        - known_events
    )
    if unknown_events:
        errors.append(
            "В capture-promotion repair использованы неизвестные "
            f"маркеры: {unknown_events}."
        )

    # Проверяем порядок маркеров.
    present_positions = [
        raw_answer.find(event)
        for event in required_events
        if raw_answer.find(event) >= 0
    ]
    if (
        len(present_positions) == len(required_events)
        and present_positions != sorted(present_positions)
    ):
        errors.append(
            "Нарушен порядок событий: сначала взятие, затем "
            "превращение, затем шах/мат."
        )

    # В raw-ответе модель не имеет права самостоятельно писать клетки.
    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected capture-promotion repair модель самостоятельно "
            f"назвала координаты {raw_squares}; координаты должен "
            "подставлять только Python."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for event, value in event_values.items():
        materialized = materialized.replace(
            event,
            value,
        )

    return materialized, []


def _render_protected_promotion_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
    derived_facts: dict,
) -> list[dict]:
    """
    Repair превращения с защищёнными слотами.

    GigaChess отвечает только за русский текст. Координаты и тип новой
    фигуры материализуются Python после генерации.
    """
    promotion_piece = str(
        move_facts.get("promotion_piece") or "фигура"
    ).lower()

    gives_mate = bool(move_facts.get("is_checkmate"))
    gives_check = bool(move_facts.get("is_check"))
    is_capture = bool(move_facts.get("is_capture"))

    event_lines = []

    if is_capture:
        event_lines.append(
            "- Пешка переходит с <PAWN_FROM> на <PROMOTION_SQUARE> "
            "и этим же ходом берёт <CAPTURED_PIECE_ACC> "
            "на <CAPTURE_SQUARE>."
        )
    else:
        event_lines.append(
            "- Пешка переходит с <PAWN_FROM> на <PROMOTION_SQUARE>."
        )

    event_lines.append(
        "- На <PROMOTION_SQUARE_EVENT> она превращается в "
        "<PROMOTED_PIECE_ACC>."
    )

    if gives_mate:
        event_lines.append(
            "- После превращения <PROMOTED_PIECE_NOM> на "
            "<CHECKER_SQUARE> ставит мат королю соперника на "
            "<KING_SQUARE>."
        )
    elif gives_check:
        event_lines.append(
            "- После превращения <PROMOTED_PIECE_NOM> на "
            "<CHECKER_SQUARE> объявляет шах королю соперника на "
            "<KING_SQUARE>."
        )

    focus_text = "\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    events_text = "\n".join(event_lines)

    capture_rules = ""
    if is_capture:
        capture_rules = """
- В этом ходе ОБЯЗАТЕЛЬНО опиши уже совершившееся взятие.
- Сохрани <CAPTURED_PIECE_ACC> и <CAPTURE_SQUARE> ТОЧНО.
- Каждый из этих двух маркеров используй ровно один раз.
- Не опускай взятую фигуру: превращение произошло через взятие.
""".strip()

    user = f"""
Сформулируй естественное короткое объяснение уже выполненного превращения пешки.

Проверенная механика:
{events_text}

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Координаты в итоговой фразе НЕ обязательны, кроме обязательных protected-маркеров,
  если они перечислены ниже.
- Если используешь какой-либо маркер в угловых скобках, сохрани его ТОЧНО.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши собственные координаты вида a1-h8.
{capture_rules}
- Можно кратко сказать, что пешка превратилась в проверенную новую фигуру
  и этим же ходом дала шах/мат, если это указано в механике.
- Превращение уже происходит этим ходом, а не готовится на будущее.
- После превращения пешка больше не является пешкой; дальнейшие действия
  совершает новая фигура.
- Если указан шах или мат, он уже возникает этим ходом, а не является угрозой.
- Не говори о сдаче, капитуляции, ничьей, победе или окончании партии,
  если этого нет в проверенной механике.
- Не добавляй стратегическую пользу, центр, развитие, защиту или угрозы.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_promotion_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
    derived_facts: dict,
) -> tuple[str, list[str]]:
    """
    Проверяет protected slots и подставляет только проверенные Python-факты.
    """
    errors: list[str] = []

    promotion_piece = str(
        move_facts.get("promotion_piece") or ""
    ).lower()
    promotion_piece_acc = _piece_word_accusative(
        promotion_piece
    )

    terminal = derived_facts.get("terminal") or {}
    king_square = str(
        terminal.get("opponent_king_square") or ""
    ).lower()

    slot_values = {
        "<PAWN_FROM>": str(explained_move.get("from") or "").lower(),
        "<PROMOTION_SQUARE>": str(explained_move.get("to") or "").lower(),
        "<PROMOTION_SQUARE_EVENT>": str(explained_move.get("to") or "").lower(),
        "<PROMOTED_PIECE_ACC>": promotion_piece_acc,
    }

    mandatory_slots: set[str] = set()

    if move_facts.get("is_capture"):
        captured_piece = str(
            move_facts.get("captured_piece") or ""
        )
        captured_square = str(
            move_facts.get("captured_square")
            or explained_move.get("to")
            or ""
        ).lower()

        slot_values.update({
            "<CAPTURED_PIECE_ACC>": _piece_name_accusative(
                captured_piece
            ),
            "<CAPTURE_SQUARE>": captured_square,
        })
        mandatory_slots.update({
            "<CAPTURED_PIECE_ACC>",
            "<CAPTURE_SQUARE>",
        })

    if move_facts.get("is_check") or move_facts.get("is_checkmate"):
        slot_values.update({
            "<PROMOTED_PIECE_NOM>": promotion_piece,
            "<CHECKER_SQUARE>": str(explained_move.get("to") or "").lower(),
            "<KING_SQUARE>": king_square,
        })

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)

        if count > 1:
            errors.append(
                f"Защищённый маркер {slot} нельзя дублировать; "
                f"получено вхождений: {count}."
            )

        if slot in mandatory_slots and count != 1:
            errors.append(
                f"Для превращения со взятием маркер {slot} "
                f"обязателен ровно один раз; получено: {count}."
            )

        if count == 1 and not value:
            errors.append(
                f"Для использованного маркера {slot} "
                "отсутствует проверенное значение."
            )

    known_slots = set(slot_values)
    unknown_slots = sorted(set(
        re.findall(r"<[A-Z_]+>", raw_answer)
    ) - known_slots)

    if unknown_slots:
        errors.append(
            "В protected promotion repair использованы неизвестные "
            f"маркеры: {unknown_slots}."
        )

    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected promotion repair модель самостоятельно назвала "
            f"координаты {raw_squares}; они должны приходить только из Python."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        if slot in materialized:
            materialized = materialized.replace(slot, value)

    return materialized, []



def _render_protected_en_passant_repair_prompt(
    *,
    move_facts: dict,
    retry_focus: list[str],
) -> list[dict]:
    """
    Repair en passant с защищёнными координатами.

    Модель формулирует только естественный русский текст.
    Три критические клетки материализуются Python:
      - откуда идёт пешка;
      - куда она приходит;
      - откуда снимается чужая пешка.
    """
    mover_side = (
        "белая" if move_facts.get("piece_color") == "white"
        else "чёрная"
    )
    captured_side = (
        "чёрную" if move_facts.get("piece_color") == "white"
        else "белую"
    )

    focus_text = "\\n".join(
        f"- {item}" for item in retry_focus[:3]
    ) or "- Предыдущий ответ не прошёл автоматическую проверку."

    user = f"""
Сформулируй короткое естественное объяснение уже выполненного взятия на проходе.

Проверенная механика:
- {mover_side.capitalize()} пешка выполняет взятие на проходе.
- Она переходит с <PAWN_FROM> на <PAWN_TO>.
- Этим же ходом она снимает {captured_side} пешку с <CAPTURED_PAWN_SQUARE>.
- Снятая пешка находится НЕ на поле назначения движущейся пешки.
- Партия после этого хода продолжается.

Предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Обязательно используй выражение «взятие на проходе».
- Сохрани <PAWN_FROM>, <PAWN_TO>, <CAPTURED_PAWN_SQUARE> ТОЧНО.
- Каждый из этих трёх маркеров используй ровно один раз.
- Не заменяй маркеры шахматными координатами самостоятельно.
- Не пиши никаких других координат вида a1-h8.
- Не говори, что партия заканчивается, заканчивается вничью или победой.
- Не описывай взятие как будущую возможность.
- Не добавляй шах, мат, угрозы, дебют или стратегическую пользу.
- Верни 1-2 коротких предложения.
- Верни только объяснение.
""".strip()

    # FEN здесь намеренно не нужен: Python уже полностью разобрал механику.
    return [{
        "role": "user",
        "content": user,
    }]


def _materialize_protected_en_passant_answer(
    raw_answer: str,
    *,
    explained_move: dict,
    move_facts: dict,
) -> tuple[str, list[str]]:
    errors: list[str] = []

    slot_values = {
        "<PAWN_FROM>": str(
            explained_move.get("from") or ""
        ).lower(),
        "<PAWN_TO>": str(
            explained_move.get("to") or ""
        ).lower(),
        "<CAPTURED_PAWN_SQUARE>": str(
            move_facts.get("captured_square") or ""
        ).lower(),
    }

    for slot, value in slot_values.items():
        count = raw_answer.count(slot)

        if count != 1:
            errors.append(
                f"Защищённый маркер {slot} должен встречаться ровно один раз; "
                f"получено: {count}."
            )

        if not value:
            errors.append(
                f"Для маркера {slot} отсутствует проверенное значение."
            )

    unknown_slots = sorted(
        set(re.findall(r"<[A-Z_]+>", raw_answer))
        - set(slot_values)
    )
    if unknown_slots:
        errors.append(
            "В protected en passant repair использованы неизвестные "
            f"маркеры: {unknown_slots}."
        )

    raw_squares = sorted(set(
        re.findall(
            r"(?<![a-zA-Z0-9])([a-h][1-8])(?![a-zA-Z0-9])",
            raw_answer.lower(),
        )
    ))
    if raw_squares:
        errors.append(
            "В protected en passant repair модель самостоятельно назвала "
            f"координаты {raw_squares}; они должны приходить только из Python."
        )

    if not re.search(
        r"\bвзят\w*\s+на\s+проходе\b",
        raw_answer.lower(),
    ):
        errors.append(
            "Protected en passant repair должен прямо назвать "
            "«взятие на проходе»."
        )

    if errors:
        return raw_answer, errors

    materialized = raw_answer
    for slot, value in slot_values.items():
        materialized = materialized.replace(slot, value)

    return materialized, []


def _render_strict_special_repair_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
    retry_focus: list[str],
) -> list[dict]:
    """
    Строгий режим repair для специальных ходов.

    Python уже определил шахматную механику. Модель не анализирует позицию
    заново, а только превращает проверенные отношения в естественный текст.
    """
    grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    special_name = "специальный ход"
    relation_rules: list[str] = []

    movement_type = str(
        grounding.get("movement_type")
        or move_facts.get("movement_type")
        or ""
    ).lower()

    movement_geometry_phrase = {
        "diagonal": "по диагонали",
        "horizontal": "по горизонтали",
        "vertical": "по вертикали",
        "knight_jump": "ходом буквой «Г»",
        "king_step": "на соседнюю клетку",
    }.get(movement_type)

    if (
        movement_type == "pawn"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        relation_rules.extend([
            "Ход выполняет именно пешка.",
            "В этом ходе нет взятия.",
            "Пешка идёт прямо вперёд на одну клетку.",
            (
                "Обязательно назови пешку и сохрани формулировку "
                "«прямо вперёд на одну клетку»."
            ),
            (
                "Не упоминай короля соперника, мат, победу, "
                "неизбежность превращения, неудержимость пешки "
                "или возможность/невозможность её догнать, "
                "если этого нет среди обязательных отношений."
            ),
        ])

    if (
        movement_type == "pawn"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        movement_geometry_phrase = "по диагонали вперёд"

    if (
        movement_type == "pawn"
        and not move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
        and not move_facts.get("pawn_double_step")
    ):
        movement_geometry_phrase = "прямо вперёд на одну клетку"

    if (
        movement_type == "pawn"
        and move_facts.get("is_capture")
        and not move_facts.get("is_en_passant")
        and not move_facts.get("is_promotion")
    ):
        relation_rules.extend([
            "Ход выполняет именно пешка.",
            "Это уже совершившееся взятие.",
            "Пешка берёт фигуру на одну клетку по диагонали вперёд.",
            (
                "Обязательно назови пешку и используй формулировку "
                "«по диагонали вперёд»."
            ),
        ])

    if (
        grounding.get("movement_geometry_required")
        and movement_geometry_phrase
    ):
        relation_rules.extend([
            f"Проверенная траектория хода: {movement_geometry_phrase}.",
            (
                f"Обязательно используй формулировку «{movement_geometry_phrase}» "
                "и не заменяй её другим направлением."
            ),
            (
                "Не определяй траекторию по букве или цифре конечного поля; "
                "она уже вычислена Python по исходной и конечной клетке."
            ),
            (
                "Если проверенный тип — ход коня буквой «Г», не заменяй его "
                "диагональю, горизонталью или вертикалью."
            ),
            (
                "Если проверенный тип — обычный ход короля на соседнюю клетку, "
                "не называй его прыжком или рокировкой."
            ),
        ])

    if move_facts.get("is_castling"):
        special_name = "рокировка"
        king_from = str(explained_move["from"]).lower()
        king_to = str(explained_move["to"]).lower()
        rook_from = grounding.get("castling_rook_from")
        rook_to = grounding.get("castling_rook_to")

        relation_rules.extend([
            f"Король: {king_from} -> {king_to}.",
            f"Ладья: {rook_from} -> {rook_to}.",
            "Это уже выполненная рокировка данным ходом.",
            "Обязательно назови обе фигуры и все четыре координаты.",
            "Не объясняй цель, пользу или безопасность рокировки.",
        ])

    elif move_facts.get("is_checkmate"):
        special_name = "мат"
        relation_rules.extend([
            "Мат уже поставлен именно этим ходом.",
            "Не описывай мат как будущую возможность или угрозу.",
        ])

    elif move_facts.get("is_capture"):
        special_name = "взятие"
        relation_rules.extend([
            "Взятие уже совершено именно этим ходом.",
            "Не пиши «может взять» или другие будущие возможности.",
        ])

    elif move_facts.get("is_check"):
        special_name = "шах"
        relation_rules.extend([
            "Шах уже объявлен именно этим ходом.",
        ])

    elif move_facts.get("is_promotion"):
        special_name = "превращение"
        relation_rules.extend([
            "Превращение уже происходит именно этим ходом.",
        ])

    focus_text = "\n".join(f"- {item}" for item in retry_focus[:3])
    relations_text = "\n".join(f"- {item}" for item in relation_rules)

    user = f"""
Нужно заново сформулировать объяснение хода {explained_move['uci']} для игрока около {elo} Elo.

Режим: СТРОГОЕ ПЕРЕФРАЗИРОВАНИЕ проверенных шахматных отношений.
Позиция FEN намеренно не передаётся: не восстанавливай доску и не анализируй её самостоятельно.
Не анализируй позицию заново и не объясняй, зачем ход полезен.

Проверенные факты:
{grounding['text']}

Обязательные отношения для события «{special_name}»:
{relations_text}

После автоматической проверки предыдущая генерация была отклонена:
{focus_text}

Жёсткие правила:
- Сохраняй все названные координаты без изменений.
- Не называй ни одной другой клетки.
- Если среди обязательных отношений указан тип траектории, назови его буквально и не заменяй другим.
- Не называй «вертикаль h», «горизонталь 1» или другую линию доски от себя, если такой формулировки нет в проверенных фактах.
- Не добавляй новые фигуры, угрозы, защиту, безопасность, дебют или стратегическую пользу.
- Не объявляй победу, ничью, сдачу или окончание партии, если этого нет в проверенных фактах.
- Не утверждай, что у соперника нет ходов или что он не может сделать ход, если такого факта нет среди обязательных отношений.
- Не добавляй «выигрывая партию», «принося победу» и похожие выводы, если немедленная победа не дана среди обязательных отношений.
- Не называй ход единственным или единственно возможным: такого проверенного отношения нет.
- Не добавляй прогнозы вроде «мат неизбежен», «пешку нельзя остановить», «король не может догнать пешку» или «превращение гарантировано», если их нет среди обязательных отношений.
- Пиши грамматически корректно: «белая ладья», «белый ферзь», «берёт чёрного ферзя», «берёт чёрную ладью».
- Не оставляй фразу «на поле» или «в клетку» без конкретной координаты; если координата не нужна, убери эту конструкцию целиком.
- Не используй описательную нотацию QB2, KR2, KB1 и подобную.
- Специальное событие происходит уже этим ходом, а не готовится на будущее.
- Для рокировки не заменяй конкретные клетки словами «безопасное поле» или «своя позиция».
- Верни 1-2 коротких естественных предложения.
- Верни только объяснение, без списков и комментариев.
""".strip()

    return [{
        "role": "user",
        "content": user,
    }]


def _build_repair_prompt(
    *,
    fen: str,
    explained_move: dict,
    move_facts: dict,
    position_facts: dict,
    derived_facts: dict,
    elo: int,
    previous_answer: str | None,
    validation_errors: list[str],
    attempt_number: int,
) -> list[dict]:
    """
    Clean regeneration.

    previous_answer намеренно НЕ используется: неправильный текст модели
    не возвращается ей обратно и не загрязняет следующую генерацию.
    """
    _ = previous_answer

    retry_focus = _compact_retry_focus(validation_errors)

    repair_grounding = _build_gigachess_grounding(
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    has_special_event = any([
        move_facts.get("is_capture"),
        move_facts.get("is_check"),
        move_facts.get("is_checkmate"),
        move_facts.get("is_castling"),
        move_facts.get("is_promotion"),
        move_facts.get("is_en_passant"),
    ])

    needs_strict_repair = bool(
        has_special_event
        or repair_grounding.get("movement_geometry_required")
        or repair_grounding.get("pawn_capture_geometry_required")
        or repair_grounding.get("pawn_single_step_geometry_required")
    )

    if move_facts.get("is_castling"):
        return _render_protected_castling_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if move_facts.get("is_en_passant"):
        return _render_protected_en_passant_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if (
        move_facts.get("is_promotion")
        and move_facts.get("is_capture")
    ):
        return _render_protected_capture_promotion_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
        )

    if move_facts.get("is_promotion"):
        return _render_protected_promotion_repair_prompt(
            move_facts=move_facts,
            retry_focus=retry_focus,
            derived_facts=derived_facts,
        )

    if needs_strict_repair:
        return _render_strict_special_repair_prompt(
            fen=fen,
            explained_move=explained_move,
            move_facts=move_facts,
            position_facts=position_facts,
            derived_facts=derived_facts,
            elo=elo,
            retry_focus=retry_focus,
        )

    # Для обычных ходов без обязательной геометрии остаётся
    # прежняя clean regeneration.
    if attempt_number == 1:
        retry_focus.append(
            "Начни с корректной фигуры и опиши только смысл проверенного хода."
        )
    elif attempt_number == 2:
        retry_focus.append(
            "Не пиши UCI-запись в итоговом тексте; объясни ход обычными словами."
        )
    else:
        retry_focus.append(
            "Сделай максимально простое объяснение в 1-2 предложениях "
            "и не вводи новых шахматных сущностей."
        )

    return _render_compact_prompt(
        fen=fen,
        explained_move=explained_move,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
        elo=elo,
        retry_focus=retry_focus[-3:],
    )


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def explain_move(
    req: ExplainMoveRequest,
    played_move: str | None = None,
) -> dict:
    # ========================================================
    # FEN
    # ========================================================
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        return {
            "ok": False,
            "error": f"Некорректный FEN: {e}",
        }

    fen_before = board.fen()

    # ========================================================
    # ХОД ПОЛЬЗОВАТЕЛЯ — только для API-ответа
    # ========================================================
    actual_played_move = played_move or getattr(req, "move", None)
    played_move_info = None

    if actual_played_move:
        try:
            played_chess_move = chess.Move.from_uci(actual_played_move)
        except ValueError:
            return {
                "ok": False,
                "error": f"Некорректный UCI ход: {actual_played_move}",
            }

        if played_chess_move not in board.legal_moves:
            return {
                "ok": False,
                "error": (
                    f"Ход {actual_played_move} нелегален для переданного FEN."
                ),
            }

        played_move_info = _move_info(board, played_chess_move)

    # ========================================================
    # STOCKFISH — единственный источник объясняемого хода
    # ========================================================
    stockfish = _stockfish_analysis(board)
    if not stockfish.get("available"):
        return {
            "ok": False,
            "error": "Stockfish недоступен.",
            "played_move": played_move_info,
            "stockfish": stockfish,
        }

    stockfish_move_info = stockfish.get("best_move")
    if not stockfish_move_info:
        return {
            "ok": False,
            "error": "Stockfish не вернул лучший ход.",
            "played_move": played_move_info,
            "stockfish": stockfish,
        }

    try:
        stockfish_move = chess.Move.from_uci(stockfish_move_info["uci"])
    except (ValueError, KeyError):
        return {
            "ok": False,
            "error": "Stockfish вернул некорректный ход.",
        }

    if stockfish_move not in board.legal_moves:
        return {
            "ok": False,
            "error": "Ход Stockfish отсутствует среди легальных ходов.",
        }

    stockfish_move_info = _move_info(board, stockfish_move)
    if stockfish_move_info is None:
        return {
            "ok": False,
            "error": "Не удалось получить информацию о ходе Stockfish.",
        }

    # ========================================================
    # ПОЗИЦИЯ ПОСЛЕ ХОДА + ПРОВЕРЕННЫЕ ФАКТЫ
    # ========================================================
    board_after = board.copy()
    board_after.push(stockfish_move)
    fen_after = board_after.fen()

    move_facts = _move_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )
    position_facts = _position_change_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )
    derived_facts = _build_derived_explanation_facts(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
    )

    print("[ChessExplainer] === STOCKFISH MOVE ===")
    print(f"[ChessExplainer] played_move={played_move_info}")
    print(f"[ChessExplainer] stockfish_move={stockfish_move_info}")
    print(f"[ChessExplainer] FEN before={fen_before}")
    print(f"[ChessExplainer] FEN after={fen_after}")

    # ========================================================
    # MAIA3 — не влияет на объясняемый ход
    # ========================================================
    try:
        maia_move = _maia3_move_strict(req, board.copy())
    except Exception as e:
        print(f"[ChessExplainer] Maia3 error: {e}")
        maia_move = None

    maia3 = {
        "available": maia_move is not None,
        "move": _move_info(board, maia_move),
        "elo": req.elo,
    }
    same_as_maia3 = (
        maia_move is not None
        and maia_move == stockfish_move
    )

    deterministic_explanation = _deterministic_explanation(
        board_before=board,
        board_after=board_after,
        move=stockfish_move,
        facts=move_facts,
    )

    verified_context = _build_verified_explanation_context(
        explained_move=stockfish_move_info,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )
    gigachess_grounding = _build_gigachess_grounding(
        explained_move=stockfish_move_info,
        move_facts=move_facts,
        position_facts=position_facts,
        derived_facts=derived_facts,
    )

    print("[ChessExplainer] === GIGACHESS INPUT FORMAT ===")
    print("[ChessExplainer] FEN is duplicated: content + attachments")
    print(f"[ChessExplainer] FEN in content={fen_before}")
    print("[ChessExplainer] === COMPACT GROUNDING SENT TO GIGACHESS ===")
    print(verified_context)

    base_result = {
        "ok": True,
        "played_move": played_move_info,
        "stockfish": stockfish,
        "maia3": maia3,
        "same_as_maia3": same_as_maia3,
        "fen_before": fen_before,
        "fen_after": fen_after,
        "move_facts": move_facts,
        "position_facts": position_facts,
        "derived_facts": derived_facts,
        "explained_move": stockfish_move_info,
        "gigachess_grounding": {
            "facts": gigachess_grounding["facts"],
            "allowed_squares": sorted(gigachess_grounding["allowed_squares"]),
            "allowed_piece_words": sorted(
                gigachess_grounding["allowed_piece_words"]
            ),
            "moved_piece_controls": sorted(
                gigachess_grounding["moved_piece_controls"]
            ),
            "attack_squares": sorted(gigachess_grounding["attack_squares"]),
            "defended_squares": sorted(
                gigachess_grounding["defended_squares"]
            ),
            "terminal": gigachess_grounding["terminal"],
            "opponent_king_square": gigachess_grounding[
                "opponent_king_square"
            ],
            "opponent_legal_moves": gigachess_grounding[
                "opponent_legal_moves"
            ],
            "castling_rook_from": gigachess_grounding[
                "castling_rook_from"
            ],
            "castling_rook_to": gigachess_grounding[
                "castling_rook_to"
            ],
            "king_safety_supported": gigachess_grounding[
                "king_safety_supported"
            ],
            "generic_defense_supported": gigachess_grounding[
                "generic_defense_supported"
            ],
        },
    }

    # ========================================================
    # GIGACHESS + ITERATIVE REPAIR LOOP
    # ========================================================
    client = get_gigachess()
    if client is None:
        base_result["gigachess"] = {
            "available": False,
            "used": False,
            "retry": False,
            "attempt_count": 0,
            "input_type": GIGACHESS_INPUT_TYPE,
        }
        base_result["explanation_source"] = "deterministic"
        base_result["explanation"] = deterministic_explanation
        return base_result

    from backend.config.settings import settings

    attempts_debug: list[dict] = []
    previous_answer: str | None = None
    validation_errors: list[str] = []

    for attempt in range(1, MAX_GIGACHESS_ATTEMPTS + 1):
        if attempt == 1:
            generation_mode = "initial_compact"
            messages = _build_prompt(
                fen=fen_before,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
                elo=req.elo,
            )
            temperature = 0.0
            top_p = 1.0
            max_tokens = min(
                int(settings.gigachess.max_tokens),
                280,
            )
        else:
            if move_facts.get("is_castling"):
                generation_mode = "protected_castling_rewrite"
            elif move_facts.get("is_en_passant"):
                generation_mode = "protected_en_passant_rewrite"
            elif (
                move_facts.get("is_promotion")
                and move_facts.get("is_capture")
            ):
                generation_mode = "protected_capture_promotion_rewrite"
            elif move_facts.get("is_promotion"):
                generation_mode = "protected_promotion_rewrite"
            elif any([
                move_facts.get("is_capture"),
                move_facts.get("is_check"),
                move_facts.get("is_checkmate"),
                move_facts.get("is_en_passant"),
                gigachess_grounding.get("movement_geometry_required"),
                gigachess_grounding.get("pawn_capture_geometry_required"),
                gigachess_grounding.get("pawn_single_step_geometry_required"),
            ]):
                generation_mode = "strict_special_rewrite"
            else:
                generation_mode = "compact_repair"
            messages = _build_repair_prompt(
                fen=fen_before,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
                elo=req.elo,
                previous_answer=previous_answer,
                validation_errors=validation_errors,
                attempt_number=attempt - 1,
            )
            repair_temperatures = {
                2: 0.12,
                3: 0.18,
                4: 0.22,
            }
            temperature = repair_temperatures.get(attempt, 0.18)
            top_p = 1.0
            max_tokens = 280

        print(
            "[ChessExplainer] === GIGACHESS SEMANTIC ATTEMPT "
            f"{attempt}/{MAX_GIGACHESS_ATTEMPTS} ==="
        )

        answer = None
        transport_debug: list[dict] = []
        transport_errors: list[str] = []

        for transport_attempt in range(
            1,
            GIGACHESS_TRANSPORT_ATTEMPTS + 1,
        ):
            print(
                "[ChessExplainer] --- transport attempt "
                f"{transport_attempt}/"
                f"{GIGACHESS_TRANSPORT_ATTEMPTS} "
                f"for semantic attempt {attempt} ---"
            )

            if transport_attempt > 1:
                delay_index = min(
                    transport_attempt - 1,
                    len(GIGACHESS_TRANSPORT_BACKOFF_SECONDS) - 1,
                )
                delay = GIGACHESS_TRANSPORT_BACKOFF_SECONDS[
                    delay_index
                ]
                if delay > 0:
                    print(
                        "[ChessExplainer] transport backoff: "
                        f"{delay:.1f}s"
                    )
                    time.sleep(delay)

            try:
                answer = client.chat(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                transport_debug.append({
                    "transport_attempt": transport_attempt,
                    "ok": True,
                    "error": None,
                })
                break

            except GigachessError as e:
                error_message = (
                    f"Gigachess request error: {e}"
                )
                print(
                    "[ChessExplainer] "
                    f"{error_message}"
                )
                transport_errors.append(error_message)
                transport_debug.append({
                    "transport_attempt": transport_attempt,
                    "ok": False,
                    "error": error_message,
                })

        if answer is None:
            validation_errors = transport_errors or [
                "Gigachess transport failed without response."
            ]
            previous_answer = None

            attempts_debug.append({
                "attempt": attempt,
                "mode": generation_mode,
                "temperature": temperature,
                "answer": None,
                "valid": False,
                "errors": validation_errors,
                "transport_attempt_count": len(transport_debug),
                "transport_attempts": transport_debug,
            })

            print(
                "[ChessExplainer] Transport retries exhausted "
                "for current semantic attempt. "
                "Semantic prompt was not evaluated."
            )

            # Не переходим к следующей semantic attempt:
            # без ответа модели нет причины менять repair prompt
            # или температуру. Сразу используем deterministic fallback.
            break

        print("[ChessExplainer] GIGACHESS RESPONSE:")
        print(repr(answer))

        raw_answer = answer
        protected_errors: list[str] = []

        if generation_mode == "protected_castling_rewrite":
            current_grounding = _build_gigachess_grounding(
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                position_facts=position_facts,
                derived_facts=derived_facts,
            )
            answer, protected_errors = _materialize_protected_castling_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                grounding=current_grounding,
            )

            if not protected_errors:
                print("[ChessExplainer] PROTECTED CASTLING MATERIALIZED:")
                print(repr(answer))

        elif generation_mode == "protected_en_passant_rewrite":
            answer, protected_errors = _materialize_protected_en_passant_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
            )

            if not protected_errors:
                print("[ChessExplainer] PROTECTED EN PASSANT MATERIALIZED:")
                print(repr(answer))

        elif generation_mode == "protected_capture_promotion_rewrite":
            answer, protected_errors = (
                _materialize_protected_capture_promotion_answer(
                    raw_answer,
                    explained_move=stockfish_move_info,
                    move_facts=move_facts,
                    derived_facts=derived_facts,
                )
            )

            if not protected_errors:
                print(
                    "[ChessExplainer] "
                    "PROTECTED CAPTURE PROMOTION MATERIALIZED:"
                )
                print(repr(answer))

        elif generation_mode == "protected_promotion_rewrite":
            answer, protected_errors = _materialize_protected_promotion_answer(
                raw_answer,
                explained_move=stockfish_move_info,
                move_facts=move_facts,
                derived_facts=derived_facts,
            )

            if not protected_errors:
                print(
                    "[ChessExplainer] "
                    "PROTECTED PROMOTION MATERIALIZED:"
                )
                print(repr(answer))

        if protected_errors:
            valid = False
            validation_errors = protected_errors
        else:
            valid, validation_errors = _validate_llm_explanation(
                answer,
                stockfish_move_info,
                move_facts,
                position_facts,
                derived_facts,
            )

        attempt_debug = {
            "attempt": attempt,
            "mode": generation_mode,
            "temperature": temperature,
            "answer": answer,
            "valid": valid,
            "errors": validation_errors,
            "transport_attempt_count": len(transport_debug),
            "transport_attempts": transport_debug,
        }

        if generation_mode in {
            "protected_castling_rewrite",
            "protected_en_passant_rewrite",
            "protected_promotion_rewrite",
            "protected_capture_promotion_rewrite",
        }:
            attempt_debug["raw_answer"] = raw_answer

        attempts_debug.append(attempt_debug)

        if valid:
            base_result["gigachess"] = {
                "available": True,
                "used": True,
                "retry": (
                    attempt > 1
                    or len(transport_debug) > 1
                ),
                "attempt_count": attempt,
                "input_type": GIGACHESS_INPUT_TYPE,
                "attempts": attempts_debug,
            }
            base_result["explanation_source"] = (
                "gigachess"
                if attempt == 1
                else f"gigachess_repair_{attempt - 1}"
            )
            base_result["explanation"] = answer.strip()
            return base_result

        print("[ChessExplainer] GIGACHESS VALIDATION FAILED:")
        for index, error in enumerate(validation_errors, start=1):
            print(f"[ChessExplainer]   {index}. {error}")

        previous_answer = answer

    # ========================================================
    # FALLBACK ПОСЛЕ ВСЕХ НЕУДАЧНЫХ ПОПЫТОК
    # ========================================================
    print(
        "[ChessExplainer] Gigachess generation unavailable or all "
        "semantic attempts rejected. Using deterministic explanation."
    )

    base_result["gigachess"] = {
        "available": True,
        "used": False,
        "retry": True,
        "attempt_count": len(attempts_debug),
        "input_type": GIGACHESS_INPUT_TYPE,
        "attempts": attempts_debug,
        "validation_errors": validation_errors,
    }
    base_result["explanation_source"] = "deterministic"
    base_result["explanation"] = deterministic_explanation
    return base_result

