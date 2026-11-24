"""Детерминированные факты python-chess: позиция ДО, дельта хода, позиция ПОСЛЕ.

Все клетки — строки a1..h8, фигуры — FEN-символы (P N B R Q K / p n b r q k),
чтобы JSON был читаемым и проверяемым без python-chess.
"""
from __future__ import annotations

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def fen(board: chess.Board) -> str:
    # Контракт 02 требует именно такую сериализацию.
    return board.fen(en_passant="fen")


def _ps(board: chess.Board, sq: int) -> dict:
    return {"square": chess.square_name(sq), "piece": board.piece_at(sq).symbol()}


def material(board: chess.Board) -> int:
    """Материал с точки зрения белых, в пешках."""
    s = 0
    for p in board.piece_map().values():
        s += VALUE[p.piece_type] if p.color == chess.WHITE else -VALUE[p.piece_type]
    return s


def hanging(board: chess.Board, color: bool) -> list[dict]:
    """Фигуры цвета color под боем соперника и без своей защиты (король не считается)."""
    out = []
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type == chess.KING:
            continue
        if board.attackers(not color, sq) and not board.attackers(color, sq):
            out.append(_ps(board, sq))
    return sorted(out, key=lambda d: -VALUE[chess.Piece.from_symbol(d["piece"]).piece_type])


def pins(board: chess.Board, color: bool) -> list[dict]:
    """Абсолютные связки фигур цвета color: {square, piece, pinner_square, pinner_piece}."""
    out = []
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type == chess.KING or not board.is_pinned(color, sq):
            continue
        for s in board.pin(color, sq):
            q = board.piece_at(s)
            if q and q.color != color and q.piece_type in SLIDERS:
                out.append({**_ps(board, sq), "pinner_square": chess.square_name(s), "pinner_piece": q.symbol()})
                break
    return out


def position_facts(board: chess.Board) -> dict:
    return {
        "fen": fen(board),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "in_check": board.is_check(),
        "castling_rights": board.castling_xfen(),
        "material_white_minus_black": material(board),
        "pins": {"w": pins(board, chess.WHITE), "b": pins(board, chess.BLACK)},
        "hanging": {"w": hanging(board, chess.WHITE), "b": hanging(board, chess.BLACK)},
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "legal_moves": board.legal_moves.count(),
    }


def _key(items: list[dict]) -> set:
    return {(d["square"], d["piece"]) for d in items}


def move_delta(before: chess.Board, move: chess.Move) -> dict:
    """Дельта хода (блок 2 пайплайна). before не меняется."""
    mover = before.turn
    piece = before.piece_at(move.from_square)
    after = before.copy(stack=False)
    after.push(move)
    to, frm = move.to_square, move.from_square
    moved_now = after.piece_at(to)  # с учётом превращения

    captured = None
    if before.is_en_passant(move):
        ep_sq = chess.square(chess.square_file(to), chess.square_rank(frm))
        captured = {"square": chess.square_name(ep_sq), "piece": before.piece_at(ep_sq).symbol()}
    elif before.is_capture(move):
        captured = _ps(before, to)

    castling = None
    if before.is_castling(move):
        castling = "O-O" if before.is_kingside_castling(move) else "O-O-O"

    # 4–5. Новые нападения и защиты самой сходившей фигуры.
    att_before = before.attacks(frm)
    att_after = after.attacks(to)
    new_attacks, new_defenses = [], []
    for t in att_after:
        q = after.piece_at(t)
        if not q or t in att_before:
            continue
        if q.color != mover:
            new_attacks.append(_ps(after, t))
        elif q.piece_type != chess.KING:
            new_defenses.append(_ps(after, t))
    new_attacks.sort(key=lambda d: -VALUE[chess.Piece.from_symbol(d["piece"]).piece_type] if d["piece"].lower() != "k" else -100)

    # 6. Вскрытия (линии через освобождённое поле) и перекрытия (линии через занятое поле).
    discovered, blocked = [], []
    for sq, p in after.piece_map().items():
        if p.piece_type not in SLIDERS or sq == to or before.piece_at(sq) != p:
            continue
        a0, a1 = before.attacks(sq), after.attacks(sq)
        for t in a1 - a0:
            q = after.piece_at(t)
            if q and q.color != p.color and frm in chess.SquareSet(chess.between(sq, t)):
                discovered.append({"slider_square": chess.square_name(sq), "slider_piece": p.symbol(), **_ps(after, t)})
        lost_targets = [t for t in a0 - a1 if to in chess.SquareSet(chess.between(sq, t)) or t == to]
        if lost_targets and p.color != mover:
            tgt = next((t for t in lost_targets if before.piece_at(t) and t != to), None)
            item = {"slider_square": chess.square_name(sq), "slider_piece": p.symbol()}
            if tgt is not None:
                item.update({"was_target_square": chess.square_name(tgt), "was_target_piece": before.piece_at(tgt).symbol()})
            blocked.append(item)

    # Контроль поля назначения после хода.
    enemy_control = [_ps(after, s) for s in after.attackers(not mover, to)]
    own_support = [_ps(after, s) for s in after.attackers(mover, to) if s != to]

    # 7–8. Связки и висячие фигуры: что появилось/исчезло в результате хода.
    me, opp = ("w", "b") if mover == chess.WHITE else ("b", "w")
    pins_b = {"w": pins(before, chess.WHITE), "b": pins(before, chess.BLACK)}
    pins_a = {"w": pins(after, chess.WHITE), "b": pins(after, chess.BLACK)}
    hang_b = {"w": hanging(before, chess.WHITE), "b": hanging(before, chess.BLACK)}
    hang_a = {"w": hanging(after, chess.WHITE), "b": hanging(after, chess.BLACK)}

    def added(a, b):
        kb = _key(b)
        return [d for d in a if (d["square"], d["piece"]) not in kb]

    return {
        "uci": move.uci(),
        "san": before.san(move),
        "color": "w" if mover == chess.WHITE else "b",
        "piece": piece.symbol(),
        "from": chess.square_name(frm),
        "to": chess.square_name(to),
        "vacated_square": chess.square_name(frm),
        "occupied_square": chess.square_name(to),
        "is_capture": captured is not None,
        "captured": captured,
        "is_en_passant": before.is_en_passant(move),
        "promotion": moved_now.symbol() if move.promotion else None,
        "castling": castling,
        "gives_check": after.is_check(),
        "new_attacks": new_attacks,
        "new_defenses": new_defenses,
        "discovered_attacks": [d for d in discovered if d["slider_piece"].isupper() == (mover == chess.WHITE)],
        "opened_for_opponent": [d for d in discovered if d["slider_piece"].isupper() != (mover == chess.WHITE)],
        "interpositions": blocked,
        "square_control": {"enemy_attackers": enemy_control, "own_defenders": own_support},
        "new_pins_on_opponent": added(pins_a[opp], pins_b[opp]),
        "new_pins_on_self": added(pins_a[me], pins_b[me]),
        "released_pins": added(pins_b[me], pins_a[me]),
        "new_hanging_self": added(hang_a[me], hang_b[me]),
        "new_hanging_opponent": added(hang_a[opp], hang_b[opp]),
        # Фигура, ушедшая с висячего поля на другое висячее, спасённой не считается.
        "saved_hanging_self": [
            d for d in added(hang_b[me], hang_a[me])
            if not (d["square"] == chess.square_name(frm) and chess.square_name(to) in {h["square"] for h in hang_a[me]})
        ],
        "castling_rights_before": before.castling_xfen(),
        "castling_rights_after": after.castling_xfen(),
    }


def all_facts(fen_before: str, move_uci: str) -> dict:
    before = chess.Board(fen_before)
    move = chess.Move.from_uci(move_uci)
    if move not in before.legal_moves:
        raise ValueError(f"нелегальный ход {move_uci} в позиции {fen_before}")
    after = before.copy(stack=False)
    after.push(move)
    return {"before": position_facts(before), "delta": move_delta(before, move), "after": position_facts(after)}
