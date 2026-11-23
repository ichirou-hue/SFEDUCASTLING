"""Ключи терминов ядра v1: тактика, маты, атака на короля (docs/term_keys.md, Часть 0).

Правила повторяют по смыслу разметчик тем задач lichess (ornicar/lichess-puzzler, tagger/cook.py, util.py;
AGPL) — переписаны заново на python-chess, код не копировался. Цель первого шага — совпасть с темами базы
задач lichess (проверка: tools/validate_keys.py); наши дополнительные условия (не ошибка, выигрыш в линии)
накладываются потом в concepts.py.

Вариант задаётся как в lichess: mainline = [ход соперника перед вариантом, ход X, ответ O, ход X, ...].
Сторона X (pov) делает ходы с нечётными индексами. Для наших данных первого хода соперника может не быть —
тогда pre=None, и правила, которым он нужен, его не используют.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
from chess import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, SquareSet

VALUES = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9}
KING_VALUES = {**VALUES, KING: 99}
RAY = (QUEEN, ROOK, BISHOP)
# Сколько полуходов ответа видят ключи. Столько же идёт во вход модели (template.engine_block):
# всё, на чём держится термин, должно быть во входе, иначе модель учится его выдумывать.
KEY_PLIES = 6


# --- вспомогательные проверки ---------------------------------------------------------------

def is_defended(board: chess.Board, piece: chess.Piece, sq: int) -> bool:
    """Поле защищено своей фигурой — напрямую или сквозь атакующую дальнобойную фигуру соперника."""
    if board.attackers(piece.color, sq):
        return True
    for a in board.attackers(not piece.color, sq):
        ap = board.piece_at(a)
        if ap and ap.piece_type in RAY:
            b = board.copy(stack=False)
            b.remove_piece_at(a)
            if b.attackers(piece.color, sq):
                return True
    return False


def is_hanging(board: chess.Board, piece: chess.Piece, sq: int) -> bool:
    return not is_defended(board, piece, sq)


def lower_attacker(board: chess.Board, piece: chess.Piece, sq: int) -> bool:
    return any(board.piece_type_at(a) != KING and VALUES[board.piece_type_at(a)] < VALUES.get(piece.piece_type, 99)
               for a in board.attackers(not piece.color, sq))


def bad_spot(board: chess.Board, sq: int) -> bool:
    """Фигуру бьют, и она не защищена или её бьёт фигура дешевле."""
    piece = board.piece_at(sq)
    if piece is None:
        return False
    return bool(board.attackers(not piece.color, sq)) and (is_hanging(board, piece, sq) or lower_attacker(board, piece, sq))


def is_trapped(board: chess.Board, sq: int) -> bool:
    """Фигура в bad_spot, и любой её ход (кроме выгодного взятия) снова ведёт в bad_spot."""
    if board.is_check() or board.is_pinned(board.turn, sq):
        return False
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type in (PAWN, KING) or not bad_spot(board, sq):
        return False
    for m in board.legal_moves:
        if m.from_square != sq:
            continue
        cap = board.piece_at(m.to_square)
        if cap and VALUES.get(cap.piece_type, 99) >= VALUES[piece.piece_type]:
            return False
        board.push(m)
        escaped = not bad_spot(board, m.to_square)
        board.pop()
        if escaped:
            return False
    return True


def material_diff(board: chess.Board, color: bool) -> int:
    def count(c):
        return sum(len(board.pieces(pt, c)) * v for pt, v in VALUES.items())
    return count(color) - count(not color)


def advanced_pawn(board_before: chess.Board, move: chess.Move, very: bool = False) -> bool:
    """Пешка идёт на 6-ю горизонталь и дальше (very — на 7-ю и дальше) или превращается."""
    if move.promotion:
        return True
    if board_before.piece_type_at(move.from_square) != PAWN:
        return False
    r = chess.square_rank(move.to_square)
    if board_before.turn == chess.WHITE:
        return r >= (6 if very else 5)
    return r <= (1 if very else 2)


# --- вариант --------------------------------------------------------------------------------

@dataclass
class Line:
    """mainline[0] — ход соперника перед вариантом (или None), дальше чередуются X и O."""
    moves: list
    boards: list = field(default_factory=list)   # boards[k] — позиция перед moves[k]; boards[-1] — финальная
    pov: bool = chess.WHITE

    @classmethod
    def build(cls, board: chess.Board, moves: list[chess.Move], pre: chess.Move | None = None) -> "Line":
        """board — позиция перед pre (если он есть) или перед первым ходом X."""
        b = board.copy(stack=False)
        boards, mm = [], []
        if pre is not None:
            boards.append(b.copy(stack=False))
            b.push(pre)
            mm.append(pre)
        else:
            boards.append(None)
            mm.append(None)
        pov = b.turn
        for m in moves:
            boards.append(b.copy(stack=False))
            b.push(m)
            mm.append(m)
        boards.append(b.copy(stack=False))
        return cls(mm, boards, pov)

    def __len__(self):
        return len(self.moves)

    def before(self, k):
        return self.boards[k]

    def after(self, k):
        return self.boards[k + 1]

    def x_idx(self):
        return range(1, len(self.moves), 2)

    def moved_type(self, k):
        return self.after(k).piece_type_at(self.moves[k].to_square)

    def captured(self, k):
        b = self.before(k)
        return b.piece_at(self.moves[k].to_square) if b is not None else None

    def is_capture(self, k):
        b = self.before(k)
        return b is not None and b.is_capture(self.moves[k])

    def final(self):
        return self.boards[-1]


def _reply_pv(facts: dict, max_plies: int) -> tuple[list, bool]:
    """Полный вариант ответа (engine.best_reply.pv_uci), если сохранён, иначе — обрезанная линия плана."""
    reply = facts["engine"].get("best_reply") or {}
    pv = reply.get("pv_uci")
    if pv is None:
        return [m["uci"] for m in reply.get("line") or []], False
    return list(pv)[:max_plies], True


def line_from_facts(facts: dict, prev: dict | None = None, max_plies: int = KEY_PLIES):
    """Вариант стороны, сделавшей ход: [предыдущий ход партии] + сыгранный ход + ответ движка.

    prev — {"prev_move_uci", "fen_prev"} из записи (если есть). Возвращает (Line | None, info).
    """
    pv, full = _reply_pv(facts, max_plies)
    moves = [chess.Move.from_uci(u) for u in [facts["delta"]["uci"]] + pv]
    pre = (prev or {}).get("prev_move_uci")
    fen_prev = (prev or {}).get("fen_prev")
    try:
        if pre and fen_prev:
            L = Line.build(chess.Board(fen_prev), moves, pre=chess.Move.from_uci(pre))
        else:
            L = Line.build(chess.Board(facts["before"]["fen"]), moves)
    except (ValueError, AssertionError):
        return None, {"full_pv": full, "pre": False}
    return L, {"full_pv": full, "pre": bool(pre and fen_prev)}


def opponent_line(facts: dict, max_plies: int = KEY_PLIES):
    """Вариант с точки зрения соперника: сыгранный ход — «ход перед вариантом», дальше ответ движка.
    Нужен для причин ошибки: что соперник получает после сыгранного хода (вилку, сквозной удар…)."""
    pv, _ = _reply_pv(facts, max_plies)
    if not pv:
        return None
    try:
        return Line.build(chess.Board(facts["before"]["fen"]), [chess.Move.from_uci(u) for u in pv],
                          pre=chess.Move.from_uci(facts["delta"]["uci"]))
    except (ValueError, AssertionError):
        return None


def line_for_record(rec: dict, max_plies: int = KEY_PLIES):
    return line_from_facts(rec["facts"], rec, max_plies)


# --- тактика --------------------------------------------------------------------------------

def fork(L: Line) -> bool:
    for k in list(L.x_idx())[:-1]:
        b = L.after(k)
        to = L.moves[k].to_square
        if bad_spot(b, to):
            continue
        mover = L.moved_type(k)
        hits = []
        for sq in b.attacks(to):
            p = b.piece_at(sq)
            if not p or p.color == L.pov or p.piece_type == PAWN:
                continue
            if KING_VALUES[p.piece_type] > KING_VALUES[mover] or (
                    is_hanging(b, p, sq) and sq not in b.attackers(not L.pov, to)):
                hits.append(sq)
        if len(hits) > 1:
            return {"k": k, "sq": to, "targets": hits}
    return None


def discovered_check(L: Line) -> bool:
    for k in L.x_idx():
        b = L.after(k)
        ch = b.checkers()
        if ch and L.moves[k].to_square not in ch:
            return {"k": k, "checkers": list(ch)}
    return None


def double_check(L: Line):
    for k in L.x_idx():
        ch = L.after(k).checkers()
        if len(ch) > 1:
            return {"k": k, "checkers": list(ch)}
    return None


def discovered_attack(L: Line):
    dc = discovered_check(L)
    if dc:
        return dc
    for k in list(L.x_idx())[1:]:
        if not L.is_capture(k):
            continue
        m, prev_op, prev_x = L.moves[k], L.moves[k - 1], L.moves[k - 2]
        if prev_op.to_square == m.to_square:
            return None
        between = SquareSet.between(m.from_square, m.to_square)
        if (prev_x.from_square in between and prev_x.to_square != m.to_square
                and prev_x.to_square != m.from_square and not L.before(k - 2).is_castling(prev_x)):
            # k — взятие дальнобойной фигурой, k-2 — ход, который открыл ей линию.
            return {"k": k - 2, "slider": m.from_square, "target": m.to_square, "vacated": prev_x.from_square}
    return None


def sacrifice(L: Line) -> bool:
    b0 = L.after(0) if L.moves[0] is not None else L.before(1)
    initial = material_diff(b0, L.pov)
    for k in list(L.x_idx())[1:]:
        d = material_diff(L.after(k), L.pov) - initial
        if d <= -2:
            if any(L.moves[j].promotion for j in range(2, len(L), 2)):
                return None
            return {"k": k, "deficit": -d}
    return None


def quiet_move(L: Line) -> bool:
    for k in list(L.x_idx())[:-1]:
        b0, b1, m = L.before(k), L.after(k), L.moves[k]
        if b1.is_check() or b0.is_check():
            continue
        if b0.is_capture(m) or any((p := b1.piece_at(s)) and p.color != L.pov for s in b1.attacks(m.to_square)):
            continue
        if advanced_pawn(b0, m) or b0.piece_type_at(m.from_square) == KING:
            continue
        return {"k": k}
    return None


def deflection(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        m = L.moves[k]
        cap = L.captured(k)
        if not (cap or m.promotion):
            continue
        mover = L.moved_type(k)
        if cap and KING_VALUES[cap.piece_type] > KING_VALUES[mover]:
            continue
        sq = m.to_square
        prev_op, prev_x = L.moves[k - 1], L.moves[k - 2]
        prev_x_cap = L.before(k - 2).piece_at(prev_x.to_square)
        grandpa_board = L.after(k - 2)
        # В оригинале ценность взятой фигуры сравнивается с номером типа фигуры — повторяем как есть.
        if ((not prev_x_cap or VALUES[prev_x_cap.piece_type] < L.moved_type(k - 2))
                and sq != prev_op.to_square and sq != prev_x.to_square
                and (prev_op.to_square == prev_x.to_square or grandpa_board.is_check())
                and (sq in grandpa_board.attacks(prev_op.from_square)
                     or (m.promotion and chess.square_file(m.to_square) == chess.square_file(prev_op.from_square)
                         and m.from_square in grandpa_board.attacks(prev_op.from_square)))
                and sq not in L.after(k - 1).attacks(prev_op.to_square)):
            # Фигуру соперника увели с prev_op.from_square, и X берёт на sq, который она защищала.
            return {"k": k - 2, "defender_from": prev_op.from_square, "defender_to": prev_op.to_square, "sq": sq}
    return None


def attraction(L: Line) -> bool:
    for k in L.x_idx():
        if k + 2 >= len(L):
            break
        sq = L.moves[k].to_square
        op = L.moves[k + 1]
        if op.to_square != sq:
            continue
        op_piece = L.moved_type(k + 1)
        if op_piece not in (KING, QUEEN, ROOK):
            continue
        nxt = L.moves[k + 2]
        if sq not in L.after(k + 2).attacks(nxt.to_square):
            continue
        if op_piece == KING or (k + 4 < len(L) and L.moves[k + 4].to_square == sq):
            return {"k": k, "sq": sq, "lured": op.from_square}
    return None


def skewer(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        m = L.moves[k]
        cap = L.captured(k)
        if not cap or L.moved_type(k) not in RAY or L.after(k).is_checkmate():
            continue
        op = L.moves[k - 1]
        between = SquareSet.between(m.from_square, m.to_square)
        if op.to_square == m.to_square or op.from_square not in between:
            continue
        if KING_VALUES[L.moved_type(k - 1)] > KING_VALUES[cap.piece_type] and bad_spot(L.after(k - 1), m.to_square):
            return {"k": k - 2, "slider": m.from_square, "front": op.from_square, "back": m.to_square}
    return None


def x_ray(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        if not L.is_capture(k):
            continue
        m, op, px = L.moves[k], L.moves[k - 1], L.moves[k - 2]
        if op.to_square != m.to_square or L.moved_type(k - 1) == KING:
            continue
        if px.to_square != op.to_square:
            continue
        if op.from_square in SquareSet.between(m.from_square, m.to_square):
            return {"k": k, "slider": m.from_square, "sq": m.to_square}
    return None


def intermezzo(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        if not L.is_capture(k):
            continue
        m = L.moves[k]
        sq = m.to_square
        op = L.moves[k - 1]
        prev_x_board = L.after(k - 2)
        if op.from_square in prev_x_board.attackers(not L.pov, sq):
            continue
        if L.moves[k - 2].to_square == sq:
            continue
        if k - 3 < 0 or L.moves[k - 3] is None:
            return None
        prev_op = L.moves[k - 3]
        if prev_op.to_square == sq and L.is_capture(k - 3) and m in L.after(k - 3).legal_moves:
            # Ход k-2 — промежуточный: вместо немедленного ответного взятия на sq.
            return {"k": k - 2, "sq": sq}
        return None
    return None


def _interference(L: Line, by_opponent: bool) -> bool:
    """Фигура встаёт между дальнобойным защитником и защищаемой фигурой, которую X затем берёт.
    by_opponent=False — перекрывает X (перекрытие), True — сам соперник (самоперекрытие)."""
    for k in list(L.x_idx())[1:]:
        prev_board = L.before(k)
        sq = L.moves[k].to_square
        cap = prev_board.piece_at(sq)
        if not (cap and is_hanging(prev_board, cap, sq)):
            continue
        if not by_opponent and sq == L.moves[k - 1].to_square:
            continue
        j = k - 1 if by_opponent else k - 2      # ход, который перекрывает
        init = L.before(j)
        defenders = init.attackers(cap.color, sq)
        if not defenders:
            continue
        d = min(defenders)
        dp = init.piece_at(d)
        if dp and dp.piece_type in RAY and L.moves[j].to_square in SquareSet.between(sq, d):
            return {"k": j, "block": L.moves[j].to_square, "defender": d, "target": sq, "by_opponent": by_opponent}
    return None


def interference(L: Line):
    return _interference(L, by_opponent=True) or _interference(L, by_opponent=False)


def clearance(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        m = L.moves[k]
        b = L.after(k)
        if L.before(k).piece_at(m.to_square):
            continue
        piece = b.piece_at(m.to_square)
        if not piece or piece.piece_type not in RAY:
            continue
        pm = L.moves[k - 2]
        if (not pm.promotion and pm.to_square != m.from_square and pm.to_square != m.to_square
                and not L.before(k).is_check() and (not b.is_check() or L.moved_type(k - 1) != KING)):
            if pm.from_square == m.to_square or pm.from_square in SquareSet.between(m.from_square, m.to_square):
                if not L.before(k - 2).piece_at(pm.to_square) or bad_spot(L.after(k - 2), pm.to_square):
                    return {"k": k - 2, "vacated": pm.from_square, "user": m.from_square, "to": m.to_square}
    return None


def capturing_defender(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        m = L.moves[k]
        b = L.after(k)
        cap = L.captured(k)
        mover = L.moved_type(k)
        if b.is_checkmate() or (cap and mover != KING and VALUES[cap.piece_type] <= VALUES[mover]
                                and is_hanging(L.before(k), cap, m.to_square)
                                and L.moves[k - 1].to_square != m.to_square):
            pm = L.moves[k - 2]
            if L.after(k - 2).is_check() or pm.to_square == m.from_square:
                continue
            init = L.before(k - 2)
            defender = init.piece_at(pm.to_square)
            if (defender and defender.color != L.pov and m.to_square in init.attacks(pm.to_square)
                    and not init.is_check()):
                return {"k": k - 2, "defender": pm.to_square, "target": m.to_square}
    return None


def trapped_piece(L: Line) -> bool:
    for k in list(L.x_idx())[1:]:
        sq = L.moves[k].to_square
        cap = L.captured(k)
        if cap and cap.piece_type != PAWN:
            op = L.moves[k - 1]
            if op.to_square == sq:
                sq = op.from_square
            if is_trapped(L.before(k - 1), sq):
                return {"k": k, "sq": sq}
    return None


def hanging_piece(L: Line) -> bool:
    if len(L) < 2 or L.moves[0] is None:
        return False
    b = L.before(1)
    to = L.moves[1].to_square
    cap = b.piece_at(to)
    if b.is_check() and (not cap or cap.piece_type == PAWN):
        return False
    if cap and cap.piece_type != PAWN and is_hanging(b, cap, to):
        op = L.moves[0]
        op_cap = L.before(0).piece_at(op.to_square)
        if op_cap and VALUES.get(op_cap.piece_type, 99) >= VALUES[cap.piece_type] and op.to_square == to:
            return False
        if len(L) < 4 or material_diff(L.after(3), L.pov) >= material_diff(L.after(1), L.pov):
            return {"k": 1, "sq": to}
    return None


def pin(L: Line) -> bool:
    """Связка, которая мешает: связанная фигура не может напасть или уйти без потерь."""
    for k in L.x_idx():
        b = L.after(k)
        for sq, p in b.piece_map().items():
            if p.color == L.pov:
                continue
            ray = b.pin(p.color, sq)
            if ray == SquareSet(chess.BB_ALL):
                continue
            ray = SquareSet(ray)
            # Мешает напасть.
            for a in b.attacks(sq):
                t = b.piece_at(a)
                if t and t.color == L.pov and a not in ray and (
                        KING_VALUES[t.piece_type] > KING_VALUES[p.piece_type] or is_hanging(b, t, a)):
                    return {"k": k, "pinned": sq, "pinner": _pinner(b, sq, ray, L.pov)}
            # Мешает уйти.
            for att in b.attackers(L.pov, sq):
                if att not in ray:
                    continue
                ap = b.piece_at(att)
                if VALUES.get(p.piece_type, 99) > VALUES.get(ap.piece_type, 99) or (
                        is_hanging(b, p, sq) and sq not in b.attackers(not L.pov, att)
                        and any(m.from_square == sq and m.to_square not in ray for m in b.pseudo_legal_moves)):
                    return {"k": k, "pinned": sq, "pinner": att}
    return None


def _pinner(b: chess.Board, sq: int, ray: SquareSet, pov: bool):
    """Дальнобойная фигура X на линии связки, которая бьёт связанную фигуру напрямую."""
    for s in ray:
        p = b.piece_at(s)
        if p and p.color == pov and p.piece_type in RAY and sq in b.attacks(s):
            return s
    return None


# --- король ---------------------------------------------------------------------------------

def attacking_f2_f7(L: Line) -> bool:
    for k in L.x_idx():
        sq = L.moves[k].to_square
        cap = L.captured(k)
        if cap and sq in (chess.F2, chess.F7):
            king_sq = chess.E8 if sq == chess.F7 else chess.E1
            return L.before(k).piece_at(king_sq) == chess.Piece(KING, not L.pov)
    return False


def side_attack(L: Line, corner_file: int, king_files: list, nb_pieces: int) -> bool:
    back = 7 if L.pov == chess.WHITE else 0
    init = L.before(1)
    ksq = init.king(not L.pov)
    if (ksq is None or chess.square_rank(ksq) != back or chess.square_file(ksq) not in king_files
            or len(init.piece_map()) < nb_pieces or not any(L.after(k).is_check() for k in L.x_idx())):
        return False
    corner = chess.square(corner_file, back)
    score = 0
    for k in L.x_idx():
        dist = chess.square_distance(corner, L.moves[k].to_square)
        if L.after(k).is_check():
            score += 1
        if L.is_capture(k) and dist <= 3:
            score += 1
        elif dist >= 5:
            score -= 1
    return score >= 2


def kingside_attack(L: Line) -> bool:
    return side_attack(L, 7, [6, 7], 20)


def queenside_attack(L: Line) -> bool:
    return side_attack(L, 0, [0, 1, 2], 18)


def exposed_king(L: Line) -> bool:
    b = L.before(1)
    if L.pov == chess.WHITE:
        pov = L.pov
    else:
        pov = not L.pov
        b = b.mirror()
    ksq = b.king(not pov)
    if ksq is None or chess.square_rank(ksq) < 5:
        return False
    sqs = [ksq - 8]
    if chess.square_file(ksq) > 0:
        sqs += [ksq - 1, ksq - 9]
    if chess.square_file(ksq) < 7:
        sqs += [ksq + 1, ksq - 7]
    for s in sqs:
        if 0 <= s < 64 and b.piece_at(s) == chess.Piece(PAWN, not pov):
            return False
    return any(L.after(k).is_check() for k in list(L.x_idx())[1:-1])


# --- маты (финальная позиция) ---------------------------------------------------------------

def _around(sq):
    return [s for s in chess.SQUARES if chess.square_distance(s, sq) == 1]


def smothered_mate(L: Line) -> bool:
    b = L.final()
    ksq = b.king(not L.pov)
    for c in b.checkers():
        if b.piece_type_at(c) == KNIGHT:
            for s in _around(ksq):
                p = b.piece_at(s)
                if not p or p.color == L.pov:
                    return False
            return True
    return False


def back_rank_mate(L: Line) -> bool:
    b = L.final()
    ksq = b.king(not L.pov)
    back = 7 if L.pov == chess.WHITE else 0
    if not (b.is_checkmate() and chess.square_rank(ksq) == back):
        return False
    step = -8 if L.pov == chess.WHITE else 8
    sqs = [ksq + step]
    f = chess.square_file(ksq)
    if f < 7:
        sqs.append(ksq + step + 1)
    if f > 0:
        sqs.append(ksq + step - 1)
    for s in sqs:
        p = b.piece_at(s)
        if p is None or p.color == L.pov or b.attackers(L.pov, s):
            return False
    return any(chess.square_rank(c) == back for c in b.checkers())


def anastasia_mate(L: Line) -> bool:
    b = L.final().copy(stack=False)
    ksq = b.king(not L.pov)
    last = L.moves[-1]
    if not (chess.square_file(ksq) in (0, 7) and chess.square_rank(ksq) not in (0, 7)):
        return False
    if chess.square_file(last.to_square) != chess.square_file(ksq) or b.piece_type_at(last.to_square) not in (QUEEN, ROOK):
        return False
    if chess.square_file(ksq) != 0:
        b.apply_transform(chess.flip_horizontal)
    ksq = b.king(not L.pov)
    blocker = b.piece_at(ksq + 1)
    knight = b.piece_at(ksq + 3)
    return bool(blocker and blocker.color != L.pov and knight and knight.color == L.pov and knight.piece_type == KNIGHT)


def hook_mate(L: Line) -> bool:
    b = L.final()
    ksq = b.king(not L.pov)
    to = L.moves[-1].to_square
    if b.piece_type_at(to) != ROOK or chess.square_distance(to, ksq) != 1:
        return False
    for d in b.attackers(L.pov, to):
        if b.piece_type_at(d) == KNIGHT and chess.square_distance(d, ksq) == 1:
            if any(b.piece_type_at(p) == PAWN for p in b.attackers(L.pov, d)):
                return True
    return False


def arabian_mate(L: Line) -> bool:
    b = L.final()
    ksq = b.king(not L.pov)
    to = L.moves[-1].to_square
    if not (chess.square_file(ksq) in (0, 7) and chess.square_rank(ksq) in (0, 7)):
        return False
    if b.piece_type_at(to) != ROOK or chess.square_distance(to, ksq) != 1:
        return False
    for n in b.attackers(L.pov, to):
        if (b.piece_type_at(n) == KNIGHT and abs(chess.square_rank(n) - chess.square_rank(ksq)) == 2
                and abs(chess.square_file(n) - chess.square_file(ksq)) == 2):
            return True
    return False


def boden_or_double_bishop(L: Line) -> str | None:
    b = L.final()
    ksq = b.king(not L.pov)
    bishops = list(b.pieces(BISHOP, L.pov))
    if len(bishops) < 2:
        return None
    for s in [s for s in chess.SQUARES if chess.square_distance(s, ksq) < 2]:
        if not all(b.piece_type_at(a) == BISHOP for a in b.attackers(L.pov, s)):
            return None
    kf = chess.square_file(ksq)
    if (chess.square_file(bishops[0]) < kf) == (chess.square_file(bishops[1]) > kf):
        return "bodenMate"
    return "doubleBishopMate"


def dovetail_mate(L: Line) -> bool:
    b = L.final()
    ksq = b.king(not L.pov)
    if chess.square_file(ksq) in (0, 7) or chess.square_rank(ksq) in (0, 7):
        return False
    q = L.moves[-1].to_square
    if (b.piece_type_at(q) != QUEEN or chess.square_file(q) == chess.square_file(ksq)
            or chess.square_rank(q) == chess.square_rank(ksq) or chess.square_distance(q, ksq) > 1):
        return False
    for s in _around(ksq):
        if s == q:
            continue
        att = list(b.attackers(L.pov, s))
        if att == [q]:
            if b.piece_at(s):
                return False
        elif att:
            return False
    return True


# --- прочее ---------------------------------------------------------------------------------

def piece_endgame(L: Line, pt: int) -> bool:
    for b in (L.before(1), L.after(1)):
        types = {p.piece_type for p in b.piece_map().values()}
        if not types <= {KING, PAWN, pt} or (pt != PAWN and pt not in types):
            return False
    return True


def tags(L: Line) -> set:
    """Темы по правилам lichess-разметчика (названия — как в базе задач lichess)."""
    t = set()
    end = L.final()
    # В задачах lichess матует всегда решающий; в наших вариантах мат может поставить и соперник.
    if end.is_checkmate() and end.turn != L.pov:
        n = (len(L) // 2)
        t |= {"mate", f"mateIn{n}" if n <= 4 else "mateIn5"}
        if smothered_mate(L):
            t.add("smotheredMate")
        elif back_rank_mate(L):
            t.add("backRankMate")
        elif anastasia_mate(L):
            t.add("anastasiaMate")
        elif hook_mate(L):
            t.add("hookMate")
        elif arabian_mate(L):
            t.add("arabianMate")
        else:
            bd = boden_or_double_bishop(L)
            if bd:
                t.add(bd)
            elif dovetail_mate(L):
                t.add("dovetailMate")
    checks = {
        "attraction": attraction, "deflection": deflection, "doubleCheck": double_check,
        "quietMove": quiet_move, "sacrifice": sacrifice, "xRayAttack": x_ray, "fork": fork,
        "hangingPiece": hanging_piece, "trappedPiece": trapped_piece, "discoveredAttack": discovered_attack,
        "exposedKing": exposed_king, "skewer": skewer, "interference": interference, "intermezzo": intermezzo,
        "pin": pin, "attackingF2F7": attacking_f2_f7, "clearance": clearance,
        "capturingDefender": capturing_defender,
    }
    for name, fn in checks.items():
        try:
            if fn(L):
                t.add(name)
        except (IndexError, TypeError, AttributeError):
            # Короткий вариант или нет хода перед вариантом — правило неприменимо.
            pass
    for k in L.x_idx():
        b0, m = L.before(k), L.moves[k]
        if advanced_pawn(b0, m, very=True):
            t.add("advancedPawn")
        if b0.is_en_passant(m):
            t.add("enPassant")
        if m.promotion:
            t.add("promotion")
            if m.promotion != QUEEN:
                t.add("underPromotion")
    for pt, name in ((PAWN, "pawnEndgame"), (QUEEN, "queenEndgame"), (ROOK, "rookEndgame"),
                     (BISHOP, "bishopEndgame"), (KNIGHT, "knightEndgame")):
        if piece_endgame(L, pt):
            t.add(name)
            break
    if "backRankMate" not in t and "fork" not in t:
        if kingside_attack(L):
            t.add("kingsideAttack")
        elif queenside_attack(L):
            t.add("queensideAttack")
    return t
