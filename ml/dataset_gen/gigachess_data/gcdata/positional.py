"""Позиционные изменения, вызванные ходом: пешечная структура, активность фигур, безопасность королей.

Сообщаем только то, что ход изменил (сравнение до/после), для обеих сторон.
Всё — подсчёты python-chess, без оценок движка, чтобы LLM было нечего додумывать.
"""
from __future__ import annotations

import chess

CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]


def _pawn_files(board: chess.Board, color: bool) -> list[int]:
    return [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]


def doubled_files(board: chess.Board, color: bool) -> set:
    fs = _pawn_files(board, color)
    return {f for f in set(fs) if fs.count(f) > 1}


def isolated_pawns(board: chess.Board, color: bool) -> set:
    fs = set(_pawn_files(board, color))
    return {s for s in board.pieces(chess.PAWN, color)
            if chess.square_file(s) - 1 not in fs and chess.square_file(s) + 1 not in fs}


def passed_pawns(board: chess.Board, color: bool) -> set:
    from .motifs import is_passed
    return {s for s in board.pieces(chess.PAWN, color) if is_passed(board, s)}


def is_outpost(board: chess.Board, sq: int) -> bool:
    """Конь/слон в лагере соперника, защищён своей пешкой и не может быть атакован пешкой соперника."""
    p = board.piece_at(sq)
    if not p or p.piece_type not in (chess.KNIGHT, chess.BISHOP):
        return False
    rank = chess.square_rank(sq) if p.color == chess.WHITE else 7 - chess.square_rank(sq)
    if not 3 <= rank <= 5:
        return False
    if not any(board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN
               for s in board.attackers(p.color, sq)):
        return False
    f = chess.square_file(sq)
    ahead = range(chess.square_rank(sq) + 1, 8) if p.color == chess.WHITE else range(chess.square_rank(sq) - 1, -1, -1)
    for ff in (f - 1, f + 1):
        if 0 <= ff < 8:
            for r in ahead:
                q = board.piece_at(chess.square(ff, r))
                if q and q.piece_type == chess.PAWN and q.color != p.color:
                    return False
    return True


def center_control(board: chess.Board, color: bool) -> int:
    return sum(1 for s in CENTER if board.attackers(color, s))


def bishop_pair(board: chess.Board, color: bool) -> bool:
    return len(board.pieces(chess.BISHOP, color)) >= 2


def king_zone(board: chess.Board, color: bool) -> chess.SquareSet:
    k = board.king(color)
    return chess.SquareSet(chess.BB_KING_ATTACKS[k]) | chess.SquareSet.from_square(k) if k is not None else chess.SquareSet()


def king_attackers(board: chess.Board, color: bool) -> int:
    """Сколько фигур соперника бьют поля вокруг короля color."""
    zone = king_zone(board, color)
    return len({a for s in zone for a in board.attackers(not color, s)})


def pawn_shield(board: chess.Board, color: bool) -> int:
    """Свои пешки на трёх вертикалях у короля, на 1–2 ряда впереди."""
    k = board.king(color)
    if k is None:
        return 0
    kf, kr = chess.square_file(k), chess.square_rank(k)
    step = 1 if color == chess.WHITE else -1
    n = 0
    for f in (kf - 1, kf, kf + 1):
        for d in (1, 2):
            r = kr + step * d
            if 0 <= f < 8 and 0 <= r < 8:
                q = board.piece_at(chess.square(f, r))
                n += bool(q and q.piece_type == chess.PAWN and q.color == color)
    return n


def rooks_connected(board: chess.Board, color: bool) -> bool:
    rooks = list(board.pieces(chess.ROOK, color))
    return len(rooks) >= 2 and rooks[1] in board.attacks(rooks[0])


def mobility(board: chess.Board, sq: int) -> int:
    """Поля, которые бьёт фигура, кроме занятых своими."""
    p = board.piece_at(sq)
    return len([t for t in board.attacks(sq) if not (board.piece_at(t) and board.piece_at(t).color == p.color)])


def positional_delta(before: chess.Board, move: chess.Move) -> list[dict]:
    """Список изменений {kind, side, ...} в порядке важности."""
    after = before.copy(stack=False)
    after.push(move)
    me = before.turn
    out = []
    sides = (me, not me)

    # Безопасность королей.
    for c in sides:
        sh0, sh1 = pawn_shield(before, c), pawn_shield(after, c)
        if sh1 < sh0:
            out.append({"kind": "king_shield", "side": c, "before": sh0, "after": sh1})
        a0, a1 = king_attackers(before, c), king_attackers(after, c)
        if a1 - a0 >= 1 and a1 >= 2:
            out.append({"kind": "king_pressure", "side": c, "before": a0, "after": a1})
        elif a0 - a1 >= 1 and a0 >= 2:
            out.append({"kind": "king_relief", "side": c, "before": a0, "after": a1})

    # Форпост сходившей фигуры.
    if is_outpost(after, move.to_square) and not is_outpost(before, move.from_square):
        out.append({"kind": "outpost", "side": me, "piece": after.piece_at(move.to_square).symbol(),
                    "square": chess.square_name(move.to_square)})

    # Пешечная структура (обе стороны: взятие меняет и чужую структуру).
    for c in sides:
        for f in sorted(doubled_files(after, c) - doubled_files(before, c)):
            out.append({"kind": "doubled", "side": c, "file": chess.FILE_NAMES[f]})
        for s in sorted(isolated_pawns(after, c) - isolated_pawns(before, c)):
            out.append({"kind": "isolated", "side": c, "square": chess.square_name(s)})
        moved = {move.from_square: move.to_square} if c == me else {}
        old_passed = {moved.get(s, s) for s in passed_pawns(before, c)}
        for s in sorted(passed_pawns(after, c) - old_passed):
            out.append({"kind": "passed", "side": c, "square": chess.square_name(s)})
        if bishop_pair(before, c) and not bishop_pair(after, c):
            out.append({"kind": "bishop_pair_lost", "side": c})

    # Активность.
    p = after.piece_at(move.to_square)
    if p.piece_type != chess.KING and not before.is_castling(move):
        m0, m1 = mobility(before, move.from_square), mobility(after, move.to_square)
        if abs(m1 - m0) >= 3:
            out.append({"kind": "mobility", "side": me, "piece": p.symbol(),
                        "square": chess.square_name(move.to_square), "before": m0, "after": m1})
    for c in sides:
        c0, c1 = center_control(before, c), center_control(after, c)
        if c0 != c1:
            out.append({"kind": "center", "side": c, "before": c0, "after": c1})
    if p.piece_type == chess.ROOK:
        seventh = 6 if me == chess.WHITE else 1
        if chess.square_rank(move.to_square) == seventh and chess.square_rank(move.from_square) != seventh:
            out.append({"kind": "rook_seventh", "side": me, "square": chess.square_name(move.to_square)})
    if rooks_connected(after, me) and not rooks_connected(before, me):
        out.append({"kind": "rooks_connected", "side": me})
    return out
