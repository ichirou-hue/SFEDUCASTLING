"""Шахматный смысл хода, вычисленный python-chess: тактические мотивы и описание линий.

Это слой «зачем и что из этого следует» поверх сырой геометрии из facts.py.
Всё детерминировано; LLM потом только связывает эти формулировки.
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def _sym(board: chess.Board, sq: int) -> str:
    return board.piece_at(sq).symbol()


def move_info(board: chess.Board, move: chess.Move) -> dict:
    """Структурное описание хода в позиции board (до хода)."""
    p = board.piece_at(move.from_square)
    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    else:
        cap_sq = move.to_square
    cap = board.piece_at(cap_sq) if board.is_capture(move) else None
    after = board.copy(stack=False)
    after.push(move)
    castling = None
    if board.is_castling(move):
        castling = "O-O" if board.is_kingside_castling(move) else "O-O-O"
    return {
        "uci": move.uci(), "san": board.san(move), "piece": p.symbol(),
        "from": chess.square_name(move.from_square), "to": chess.square_name(move.to_square),
        "captured": {"piece": cap.symbol(), "square": chess.square_name(cap_sq)} if cap else None,
        "promotion": chess.Piece(move.promotion, p.color).symbol() if move.promotion else None,
        "castling": castling,
        "check": after.is_check(), "mate": after.is_checkmate(),
    }


def line_info(board: chess.Board, pv: list, n: int) -> tuple[list[dict], dict]:
    """Первые n полуходов линии + итог по взятым фигурам с каждой стороны."""
    b = board.copy(stack=False)
    plies, lost = [], {"w": [], "b": []}
    # Не обрываем размен на середине: «белые теряют коня» без ответного взятия — неправда.
    k = n
    while k < min(len(pv), n + 2) and board_capture_chain(board, pv, k):
        k += 1
    for mv in pv[:k]:
        if mv not in b.legal_moves:
            break
        info = move_info(b, mv)
        if info["captured"]:
            lost["w" if info["captured"]["piece"].isupper() else "b"].append(info["captured"]["piece"])
        plies.append(info)
        b.push(mv)
    net_w = sum(VAL[chess.Piece.from_symbol(s).piece_type] for s in lost["b"]) - \
        sum(VAL[chess.Piece.from_symbol(s).piece_type] for s in lost["w"])
    return plies, {"white_lost": lost["w"], "black_lost": lost["b"], "net_for_white": net_w}


def board_capture_chain(board: chess.Board, pv: list, k: int) -> bool:
    """pv[k] — взятие на поле, где взятием закончился pv[k-1]."""
    b = board.copy(stack=False)
    for mv in pv[:k - 1]:
        if mv not in b.legal_moves:
            return False
        b.push(mv)
    prev, nxt = pv[k - 1], pv[k]
    if prev not in b.legal_moves or not b.is_capture(prev):
        return False
    b.push(prev)
    return nxt in b.legal_moves and b.is_capture(nxt) and nxt.to_square == prev.to_square


def en_prise(board: chess.Board, sq: int) -> bool:
    """Фигуру можно выгодно взять: не защищена или бьётся более дешёвой фигурой."""
    p = board.piece_at(sq)
    if not p or p.piece_type == chess.KING:
        return False
    att = board.attackers(not p.color, sq)
    if not att:
        return False
    if not board.attackers(p.color, sq):
        return True
    return min(VAL[board.piece_at(a).piece_type] for a in att) < VAL[p.piece_type]


def capture_kind(before: chess.Board, move: chess.Move) -> str | None:
    if not before.is_capture(move):
        return None
    after = before.copy(stack=False)
    after.push(move)
    attacker = VAL[before.piece_at(move.from_square).piece_type]
    victim = 1 if before.is_en_passant(move) else VAL[before.piece_at(move.to_square).piece_type]
    recapturable = bool(after.attackers(after.turn, move.to_square))
    if not recapturable or victim > attacker:
        return "выигрыш материала"
    if victim == attacker:
        return "размен"
    return "жертва" if en_prise(after, move.to_square) else "взятие"


def fork_targets(after: chess.Board, sq: int) -> list[dict]:
    """Цели двойного удара сходившей фигурой: король, более ценные или незащищённые фигуры (не пешки)."""
    p = after.piece_at(sq)
    out = []
    for t in after.attacks(sq):
        q = after.piece_at(t)
        if not q or q.color == p.color or q.piece_type == chess.PAWN:
            continue
        if q.piece_type == chess.KING or VAL[q.piece_type] > VAL[p.piece_type] or not after.attackers(q.color, t):
            out.append({"piece": q.symbol(), "square": chess.square_name(t)})
    return out if len(out) >= 2 else []


def is_passed(board: chess.Board, sq: int) -> bool:
    p = board.piece_at(sq)
    if not p or p.piece_type != chess.PAWN:
        return False
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if p.color == chess.WHITE else range(r - 1, -1, -1)
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            for rr in ahead:
                q = board.piece_at(chess.square(ff, rr))
                if q and q.piece_type == chess.PAWN and q.color != p.color:
                    return False
    return True


def file_status(board: chess.Board, sq: int) -> str | None:
    """Для ладьи/ферзя: открытая (без пешек) или полуоткрытая (без своих пешек) вертикаль."""
    p = board.piece_at(sq)
    f = chess.square_file(sq)
    pawns = [board.piece_at(chess.square(f, r)) for r in range(8)]
    pawns = [q for q in pawns if q and q.piece_type == chess.PAWN]
    if not pawns:
        return "открытая"
    if not any(q.color == p.color for q in pawns):
        return "полуоткрытая"
    return None


def static_motifs(before: chess.Board, move: chess.Move) -> dict:
    """Мотивы, видимые без движка."""
    after = before.copy(stack=False)
    after.push(move)
    to = move.to_square
    p = after.piece_at(to)
    res = {
        "capture_kind": capture_kind(before, move),
        "fork": fork_targets(after, to) if not after.is_checkmate() else [],
        "moved_piece_en_prise": en_prise(after, to),
        "passed_pawn": p.piece_type == chess.PAWN and is_passed(after, to) and not is_passed(before, move.from_square),
        "rook_file": None,
        "castling": before.is_castling(move),
        "lost_castling_rights": (before.castling_xfen() != after.castling_xfen()) and not before.is_castling(move)
        and before.has_castling_rights(before.turn) and not after.has_castling_rights(before.turn),
        "development": p.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(move.from_square) == (0 if p.color == chess.WHITE else 7),
    }
    if p.piece_type in (chess.ROOK, chess.QUEEN) and chess.square_file(move.from_square) != chess.square_file(to):
        res["rook_file"] = file_status(after, to)
    # Свои фигуры, которые после хода можно выгодно взять (кроме сходившей).
    from .positional import positional_delta
    res["positional"] = positional_delta(before, move)
    res["own_en_prise"] = [{"piece": _sym(after, s), "square": chess.square_name(s)}
                           for s, q in after.piece_map().items()
                           if q.color == p.color and s != to and en_prise(after, s) and not en_prise(before, s)]
    res["overloaded"] = overloaded(after, not p.color)
    return res


def overloaded(board: chess.Board, color: bool) -> list[dict]:
    """Фигуры цвета color, которые в одиночку защищают две и более атакованные фигуры."""
    sole = {}
    for s, q in board.piece_map().items():
        if q.color != color or q.piece_type == chess.KING or not board.attackers(not color, s):
            continue
        defs = list(board.attackers(color, s))
        if len(defs) == 1:
            sole.setdefault(defs[0], []).append({"piece": q.symbol(), "square": chess.square_name(s)})
    return [{"piece": _sym(board, d), "square": chess.square_name(d), "defends": ts}
            for d, ts in sole.items() if len(ts) >= 2]
