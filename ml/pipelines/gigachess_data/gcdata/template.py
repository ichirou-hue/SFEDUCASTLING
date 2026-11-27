"""Вход модели GigaChess: вопрос о ходе и оценка длины последовательности.

Ограничения из паспорта модели и контракта 02:
- одна доска на пример: в fen кладём позицию ДО хода, сам ход описываем в вопросе;
- координаты в вопросе/ответе — только «e4» отдельным словом: prepare заменяет их на gigatoken,
  поэтому SAN вида Nxe5 и латинские буквы фигур в тексте недопустимы;
- весь пример (prompt + 69 шахматных слотов + ответ + стоп-токен) <= 512 позиций.
"""
from __future__ import annotations

import chess

from .keys import KEY_PLIES
from .ru import PIECE_RU, side_gen_ru

BOARD_SLOTS = 69
MAX_SEQ = 512
# Служебные токены шаблона: BOS, «[» «]» пустого списка функций, роли, маркер доски, стоп 128001.
TEMPLATE_OVERHEAD = 16
# Без tokenizer.json считаем грубо и с запасом: русский текст ≈ 2.6 символа на токен.
CHARS_PER_TOKEN = 2.6


def move_text(delta: dict) -> str:
    color = chess.WHITE if delta["color"] == "w" else chess.BLACK
    side = side_gen_ru(color)
    if delta["castling"]:
        kind = "короткая" if delta["castling"] == "O-O" else "длинная"
        return f"Ход {side}: {kind} рокировка, король с {delta['from']} на {delta['to']}"
    piece = PIECE_RU[chess.Piece.from_symbol(delta["piece"]).piece_type]
    s = f"Ход {side}: {piece} с {delta['from']} на {delta['to']}"
    if delta["is_capture"]:
        s += " со взятием"
    if delta["promotion"]:
        s += f", превращение в {_acc(chess.Piece.from_symbol(delta['promotion']).piece_type)}"
    return s


def _acc(pt: int) -> str:
    return {chess.QUEEN: "ферзя", chess.ROOK: "ладью", chess.BISHOP: "слона", chess.KNIGHT: "коня"}[pt]


def _ply(board: chess.Board, m: chess.Move) -> str:
    p = board.piece_at(m.from_square)
    s = f"{PIECE_RU[p.piece_type]} {chess.square_name(m.from_square)} на {chess.square_name(m.to_square)}"
    if board.is_capture(m):
        s += " со взятием"
    if m.promotion:
        s += f", превращение в {_acc(m.promotion)}"
    board.push(m)
    if board.is_checkmate():
        s += ", мат"
    elif board.is_check():
        s += ", шах"
    return s


def line_text(fen: str, ucis: list, first: list | None = None) -> str:
    """Вариант словами: «ферзь d5 на e5, шах; слон f1 на e2». Координаты — отдельными словами (gigatoken)."""
    b = chess.Board(fen)
    for u in first or []:
        b.push(chess.Move.from_uci(u))
    return "; ".join(_ply(b, chess.Move.from_uci(u)) for u in ucis)


def _score_text(s: dict) -> str:
    if s["mate"] is not None:
        if s["mate"] == 0:
            return "мат на доске"
        return f"мат в {abs(s['mate'])} за {'белых' if s['mate'] > 0 else 'чёрных'}"
    return f"{s['cp'] / 100:+.2f}"


def engine_block(rec: dict, reply_plies: int = KEY_PLIES, best_plies: int = 5) -> str:
    """Факты движка для входа модели — ровно то, на чём держится план ответа (gcdata/concepts, draft).
    Числа здесь допустимы: это вход; в ответе они передаются словами."""
    f = rec["facts"]
    eng, d = f["engine"], f["delta"]
    fen = f["before"]["fen"]
    lines = [f"Оценка до хода: {_score_text(eng['score_before'])}; после хода: {_score_text(eng['score_after'])} "
             "(плюс — в пользу белых)."]
    b = eng["best_move_before"]
    if eng["played_is_best"]:
        lines.append("Сыгран лучший ход.")
    else:
        lines.append(f"Лучший ход: {line_text(fen, (b.get('pv_uci') or [b['uci']])[:best_plies])}.")
    lines.append(f"Отрыв лучшего хода от второго по шансам на выигрыш: {eng['only_move_gap']:.2f}.")
    reply = (eng.get("best_reply") or {}).get("pv_uci") or [m["uci"] for m in (eng.get("best_reply") or {}).get("line") or []]
    if reply:
        lines.append(f"Ответ соперника и продолжение: {line_text(fen, reply[:reply_plies], first=[d['uci']])}.")
    if eng.get("own_threat"):
        t = eng["own_threat"]["move"]
        lines.append(f"Угроза после хода: {line_text(fen, [t['uci']], first=[d['uci'], '0000'])}.")
    if eng.get("opp_threat_before"):
        t = eng["opp_threat_before"]
        lines.append(f"Угроза соперника до хода: {line_text(fen, [t['move']['uci']], first=['0000'])} "
                     f"({'ход её снимает' if t.get('parried') else 'ход её не снимает'}).")
    if rec.get("prev_move_uci") and rec.get("fen_prev"):
        lines.append(f"Предыдущий ход соперника: {line_text(rec['fen_prev'], [rec['prev_move_uci']])}.")
    return "Факты движка:\n" + "\n".join(lines)


def build_question(delta: dict, intent: str | None, engine: str | None = None) -> str:
    q = move_text(delta) + "."
    if intent:
        q += f"\nЗамысел игрока: {intent.rstrip('.')}."
    if engine:
        q += "\n" + engine
    q += "\nОпиши этот ход."
    return q


def user_body(fen: str, question: str) -> str:
    # Тело пользовательского сообщения из контракта 02 (до кодирования координат).
    return f"Позиция: <|gigatoken_6|>\nПозиция FEN: {fen}\n{question}"


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None):
        self.tok = None
        if tokenizer_path:
            from tokenizers import Tokenizer  # tokenizers==0.21.4, как в контракте
            self.tok = Tokenizer.from_file(tokenizer_path)

    @property
    def exact(self) -> bool:
        return self.tok is not None

    def count(self, text: str) -> int:
        if self.tok:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        return int(len(text) / CHARS_PER_TOKEN) + 1

    def sequence_length(self, fen: str, question: str, answer: str) -> int:
        body = user_body(fen, question).replace("<|gigatoken_6|>", "")
        return BOARD_SLOTS + TEMPLATE_OVERHEAD + self.count(body) + self.count(answer)

    def answer_budget(self, fen: str, question: str, max_seq: int = MAX_SEQ) -> int:
        """Сколько токенов остаётся на ответ."""
        body = user_body(fen, question).replace("<|gigatoken_6|>", "")
        return max_seq - BOARD_SLOTS - TEMPLATE_OVERHEAD - self.count(body)
