"""Блок 4: план комментария из фактов.

Слой фактов (facts, motifs, positional, engine) -> словарь терминов (concepts) -> планировщик (здесь):
выбирает главную историю хода по весам терминов и связывает факты отношениями
(причина / в результате / однако / вместо этого). LLM потом только склоняет и связывает.
Текст намеренно «рваный»: факты через «;», фигуры в именительном падеже, термины в [скобках].
"""
from __future__ import annotations

import chess

from .concepts import TERM_PATTERNS, find_concepts
from .ru import PIECE_RU, eval_words, piece_sym_ru, side_gen_ru, side_ru

LIMITS = {"thesis": 2, "second_weight": 55}


def _ps(d: dict) -> str:
    return f"{piece_sym_ru(d['piece'])} {d['square']}"


def ply_ru(m: dict) -> str:
    if m.get("castling"):
        s = f"{'короткая' if m['castling'] == 'O-O' else 'длинная'} рокировка ({piece_sym_ru(m['piece'])} {m['from']} -> {m['to']})"
    else:
        s = f"{piece_sym_ru(m['piece'])} {m['from']} -> {m['to']}"
    if m.get("captured"):
        s += f" со взятием: {_ps(m['captured'])}"
    if m.get("promotion"):
        s += f" с превращением: {piece_sym_ru(m['promotion'])}"
    if m.get("mate"):
        s += ", мат"
    elif m.get("check"):
        s += ", шах"
    return s


def material_words(m: int) -> str:
    """Баланс материала (белые минус чёрные, в пешках) словами, без чисел."""
    if m == 0:
        return "материал равный"
    side = "белых" if m > 0 else side_gen_ru(chess.BLACK)
    a = abs(m)
    if a == 1:
        return f"у {side} лишняя пешка"
    if a == 2:
        return f"у {side} небольшой перевес в материале"
    if a <= 4:
        return f"у {side} лишняя фигура"
    if a <= 7:
        return f"у {side} лишняя ладья"
    return f"у {side} большой перевес в материале"


def summary_ru(summary: dict) -> str | None:
    def names(syms):
        return ", ".join(PIECE_RU[chess.Piece.from_symbol(s).piece_type] for s in syms)

    parts = []
    if summary["white_lost"]:
        parts.append(f"белые теряют: {names(summary['white_lost'])}")
    if summary["black_lost"]:
        parts.append(f"{side_ru(chess.BLACK)} теряют: {names(summary['black_lost'])}")
    return "; ".join(parts) if parts else None


def _c(c) -> str:
    # В [скобках] — только термины словаря: их LLM обязана назвать, остальные концепты — просто факты.
    ev = "; ".join(c.evidence)
    return f"[{c.term}] {ev}" if c.term in TERM_PATTERNS else ev


def build_draft(rec: dict, intent: str | None) -> str:
    """Планировщик описания хода: одна главная история, понятная человеку, без чисел.

    Пункты плана: 1) что делает ход (1–2 главных термина); 2) итог: вердикт, перевес словами,
    причина (при ошибке) или главное последствие; 3) что было сильнее — только при ошибке.
    Ответ LLM — один абзац. Варианты движка ход за ходом не пересказываем: ход соперника попадает
    в план, только если он и есть причина (вилка соперника и т. п. — в обосновании термина).
    Метки отношений («причина:», «в результате:», «однако:», «вместо этого:») LLM передаёт союзами.
    """
    f = rec["facts"]
    d, after, eng = f["delta"], f["after"], f["engine"]
    color = chess.WHITE if d["color"] == "w" else chess.BLACK
    concepts = find_concepts(f, rec)
    by_role = {r: sorted([c for c in concepts if c.role == r], key=lambda c: -c.weight)
               for r in ("move", "cons", "reason", "better")}
    bad = eng["nag"] in (2, 4, 6) or bool(eng.get("allows_mate_in"))
    mover_move = {"piece": d["piece"], "from": d["from"], "to": d["to"], "captured": d["captured"],
                  "promotion": d["promotion"], "castling": d["castling"],
                  "check": d["gives_check"], "mate": after["is_checkmate"]}

    # 1. Ход: тезис — 1–2 самых весомых термина «что делает ход».
    s1 = [side_ru(color), ply_ru(mover_move)]
    if intent:
        # Слова игрока из партии — не факт: LLM передаёт их с атрибуцией.
        # Квадратные скобки в плане — только для терминов: в цитате заменяем на круглые.
        quote = intent.rstrip(".").replace("[", "(").replace("]", ")").replace("«", "\"").replace("»", "\"")
        s1.append(f"замысел игрока (цитата): «{quote}»")
    # Второй факт — только если он сам весомый: иначе мелочь («открытая вертикаль») отвлекает от главного.
    thesis = by_role["move"][:1] + [c for c in by_role["move"][1:LIMITS["thesis"]] if c.weight >= LIMITS["second_weight"]]
    s1 += [_c(c) for c in thesis]

    # 2. Итог: вердикт и перевес словами; при ошибке — причина, иначе — главное последствие.
    sb, sa = eng["score_before"], eng["score_after"]
    s2 = [("однако: " if intent and bad else "") + f"вердикт: {eng['verdict']}"]
    w_before, w_after = eval_words(sb["cp"], sb["mate"]), eval_words(sa["cp"], sa["mate"])
    if after["is_checkmate"]:
        s2.append("партия окончена матом")
    elif after["is_stalemate"]:
        s2.append("пат, ничья")
    elif w_before == w_after:
        s2.append(f"оценка не меняется: {w_after}")
    else:
        s2.append(f"оценка до хода: {w_before}; после хода: {w_after}")
    # Материал словами, если он меняется в варианте: иначе «выиграл ладью, но позиция равная» непонятно
    # (обычно сторона просто отыгрывает отданное раньше).
    summ = (eng.get("best_reply") or {}).get("summary")
    # Только заметная смена (от фигуры) и без мата в оценке: пешка туда-сюда или материал при мате — шум.
    if summ and not after["is_checkmate"] and sa["mate"] is None and sb["mate"] is None:
        m_before = f["before"]["material_white_minus_black"]
        m_end = f["after"]["material_white_minus_black"] + summ["net_for_white"]
        cp_after = sa["cp"] or 0
        # Материал против оценки (лишняя фигура у одних, перевес у других) — вариант оборван посреди
        # комбинации: такой «итог по материалу» запутает читателя.
        contradicts = (m_end > 0 and cp_after <= -100) or (m_end < 0 and cp_after >= 100)
        if abs(m_end - m_before) >= 3 and not contradicts:
            s2.append(f"материал: до хода {material_words(m_before)}; после ответного варианта {material_words(m_end)}")
    if bad:
        reasons = by_role["reason"][:2] or by_role["cons"][:1]
        if reasons:
            s2 += ["причина: " + _c(c) for c in reasons]
        else:
            # Ни одно правило-причина не сработало: показываем, чем отвечает соперник, — без «потому что».
            r = eng.get("best_reply") or {}
            if r.get("line") and not eng.get("mate_involved"):
                s2.append("лучший ответ соперника: " + ply_ru(r["line"][0]))
    else:
        s2 += ["в результате: " + _c(c) for c in by_role["cons"][:1]]

    # 3. Что было сильнее — только при ошибке. Вместо варианта ход за ходом — что он давал.
    s3 = []
    if bad and not eng["played_is_best"]:
        b = eng["best_move_before"]
        s = "вместо этого: сильнее было " + ply_ru(b)
        better = by_role["better"][:1]
        if better and better[0].term == "упущенная возможность" and not b.get("mate_in"):
            gain = summary_ru(b.get("line_summary") or {"white_lost": [], "black_lost": []})
            s += "; " + f"[упущенная возможность] этот ход выигрывал материал" + (f" ({gain})" if gain else "")
        elif better:
            s += "; " + _c(better[0])
        s3.append(s)

    parts = ["Ход: " + "; ".join(s1), "Итог: " + "; ".join(s2)]
    if s3:
        parts.append("Лучше было: " + "; ".join(s3))
    return "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
