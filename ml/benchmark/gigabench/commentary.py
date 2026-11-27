#!/usr/bin/env python3
"""Frozen M4-M7-auto commentary suite and deterministic scoring helpers."""
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
from . import __version__  # noqa: E402
from .io import read_jsonl, sha, unique, write_json, write_jsonl  # noqa: E402

CRITERIA_VERSION = "m4-m7-auto-v1"
DEFAULT_BUDGET = 512

PIECE_NAMES = {"P": "пешка", "N": "конь", "B": "слон", "R": "ладья", "Q": "ферзь", "K": "король"}
PIECE_STEMS = {
    "пешк": "P", "кон": "N", "слон": "B", "лад": "R", "ферз": "Q", "корол": "K",
}
VERDICT_GROUPS = (
    ("грубая ошибка", r"\bгруб\w*\s+ошибк\w*|\bзевок\b"),
    ("неточность", r"\bнеточн\w*\b"),
    ("ошибка", r"\bошибк\w*|\bплох\w*\s+ход\w*|\bслаб\w*\s+ход\w*"),
    ("точный ход", r"\bточн\w*[\s—–-]+ход\w*|\bход\w*[\s—–-]+точн\w*|\bверн\w*[\s—–-]+ход\w*"),
    ("лучший ход", r"\bлучш\w*[\s—–-]+ход\w*|\bход\w*[\s—–-]+лучш\w*"),
    ("сильный ход", r"\bсильн\w*[\s—–-]+ход\w*|\bход\w*[\s—–-]+сильн\w*"),
    ("хороший ход", r"\bхорош\w*[\s—–-]+ход\w*|\bход\w*[\s—–-]+хорош\w*"),
    ("приемлемый ход", r"\bприемлем\w*[\s—–-]+ход\w*|\bход\w*[\s—–-]+приемлем\w*"),
)
POSITIVE_VERDICTS = {"точный ход", "лучший ход", "сильный ход", "хороший ход", "приемлемый ход"}
NEGATIVE_VERDICTS = {"грубая ошибка", "неточность", "ошибка"}
GOAL_RE = re.compile(
    r"\b(?:чтобы|с целью|позволяет|созда[её]т|угрожает|освобождает|занимает|"
    r"вста[её]т|оказывается|атакует|защищает|укрепляет|развивает|вскрывает|давит|"
    r"контролирует|препятствует|выигрывает|сохраняет|уводит|переводит|объявляет|"
    r"да[её]т|отражает|ставит|вынужден\w*|теряет\w*|попадает|бь[её]т|бер[её]т|"
    r"отвлекает|завлекает|связывает|выигрыш|ловля|профилактика|вилка|размен|"
    r"жертв\w*|открывает|усиливает|активизирует|снимает|мешает|готовит)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(?:точн\w*|лучш\w*|сильн\w*|хорош\w*|приемлем\w*|ошибк\w*|неточн\w*|зевок|"
    r"лучше|сильнее|следует|вместо|рекомендую|стоило|надо было|верн\w*|плох\w*|"
    r"отражает|выигрывает|вед[её]т|сохраняет)\b",
    re.I,
)
RECOMMEND_RE = re.compile(
    r"\b(?:лучше|лучш\w*|сильнее|следовало|надо было|стоило|рекоменд\w*|вместо|"
    r"можно\s+(?:сыграть|играть)|стоит\s+(?:сыграть|играть))\b",
    re.I,
)
CHECK_RE = re.compile(r"\bшах(?!мат|и)\w*", re.I)
MATE_RE = re.compile(
    r"(?:объявля\w*|ставит|поставлен\w*|получается|это|окончена|заканчивается|закончилась)"
    r"\s+(?:мат|матом)\b|\bс\s+матом\b|\bмат\b(?=\s*[.!;]|$)",
    re.I,
)
MATE_THREAT_RE = re.compile(
    r"(?:угрожа\w*\s+матом|мат\s+в\s+\d+|нацелен\w*\s+на\s+мат|"
    r"форсированн\w*\s+мат|сохраня\w*\s+(?:форсированн\w*\s+)?мат|"
    r"возможность\s+с\s+матом|упущенн\w*\s+возможность[^.!?\n]{0,24}мат)",
    re.I,
)
FUTURE_RE = re.compile(
    r"\b(?:до хода|после хода|до него|после него|был|была|было|мог|могла|"
    r"угрожа\w*|гроз(?:ит|ят|ил|ила|или)|отвеч\w*|ответ\w*|взамен|раньше|могли|"
    r"не\s+отразил\w*\s+угроз\w*|не\s+отражен\w*\s+угроз\w*|угроз\w*(?=\s*:)|"
    r"следующ\w*|затем|потом|дальше|в варианте|если|профилактик\w*|"
    r"препятств\w*|не даёт|не дает|перестаёт|перестает|отход\w*|"
    r"продолжение|нацелен\w*|ловушк\w*)\b",
    re.I,
)
PLAYED_RE = re.compile(
    r"\b(?:идет|идёт|сыгран\w*|сыграл\w*|сходил\w*|пошел|пошёл|пошл[аи]|"
    r"продвинул(?:ся|ась|ись)|переместил(?:ся|ась|ись)|объявляет|встает|"
    r"встаёт|занимает|берет|берёт|бьет|бьёт|ставит|дает|даёт|отражает|созда[её]т|"
    r"берут|бьют|вынужден\w*|превращается|уходит|отходит)\b",
    re.I,
)
CAPTURE_RE = re.compile(r"\b(?:бь[её]т|бер[её]т|взял\w*|взяти\w*|съел\w*|захват\w*)\b", re.I)
CAPTURE_CLAIM_RE = re.compile(
    r"\b(?:бер[её]т|берут|взял\w*|съел\w*|захват\w*)\b|"
    r"(?<!без )(?<!вместо )\bсо?\s+взятием\b",
    re.I,
)
CAPTURE_EXPLICIT = r"\b(?:бер[её]т|берут|взял\w*|съел\w*|захват\w*)\b"
CAPTURE_NA_RE = re.compile(
    CAPTURE_EXPLICIT + r"[^.!?\n]{0,30}?\bна\s+(?P<square>[a-h][1-8])\b", re.I
)
CAPTURE_TWO_SQUARES_RE = re.compile(
    CAPTURE_EXPLICIT + r"[^.!?\n]{0,20}?\b(?P<origin>[a-h][1-8])\b[^.!?\n]{0,20}?"
    r"(?P<piece>пешк\w*|кон\w*|слон\w*|лад\w*|ферз\w*|корол\w*)\s+"
    r"(?P<square>[a-h][1-8])\b",
    re.I,
)
CAPTURE_PIECE_SQUARE_RE = re.compile(
    CAPTURE_EXPLICIT + r"[^.!?\n]{0,12}?"
    r"(?P<piece>пешк\w*|кон\w*|слон\w*|лад\w*|ферз\w*|корол\w*)\s+"
    r"(?P<square>[a-h][1-8])\b",
    re.I,
)
CAPTURE_DIRECT_RE = re.compile(CAPTURE_EXPLICIT + r"\s+(?P<square>[a-h][1-8])\b", re.I)
CASTLING_RE = re.compile(r"\bрокировк\w*\b", re.I)
CASTLING_CLAIM_RE = re.compile(
    r"\bрокировал\w*|\bрокировк[аи]\b|"
    r"\b(?:сделал[аи]?|сделана|делает|делают)\s+(?:короткую|длинную)?\s*рокировк\w*",
    re.I,
)
PROMOTION_RE = re.compile(r"\bпревращ\w*\b", re.I)
PROMOTION_CLAIM_RE = re.compile(
    r"\bпревраща\w*|\bпреврати\w*|\bпревращени\w*\s+в\s+(?:ферз|лад|слон|кон)",
    re.I,
)
SQUARE_RE = re.compile(r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])", re.I)
UCI_RE = re.compile(r"(?<![a-z0-9])([a-h][1-8][a-h][1-8][qrbn]?)(?![a-z0-9])", re.I)
MOVE_RE = re.compile(
    r"(?:(?P<piece>пешк\w*|кон\w*|слон\w*|лад\w*|ферз\w*|корол\w*)\s+)?"
    r"(?:с\s+)?(?P<from>[a-h][1-8])\s*(?:на|в|->|→|-)\s*(?P<to>[a-h][1-8])",
    re.I,
)
PIECE_SUBJECT_RE = re.compile(
    r"\b(пешка|конь|слон|ладья|ферзь|король)\s+"
    r"(?:идет|пошел|пошла|продвинулся|продвинулась|переместился|переместилась|"
    r"уходит|отходит|встает|занимает)\s*$"
)
PLAYED_SAN_RE = re.compile(
    r"(?i:\b(?:сыграл[аи]?|сыгран(?:\s+ход)?)\s+)"
    r"(?P<san>(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
    r"O-O(?:-O)?)[+#]?)(?![\w+=#-])"
)
CURRENT_MOVE_RE = re.compile(r"^(?:а\s+)?(?:этот|сыгранный|сделанный)\s+ход\b")
LINE_CONTINUATION_RE = re.compile(
    r"^(?:а\s+)?(?:после\s+ответ\w*|после\s+ответного|затем|потом|в ответ|взамен|"
    r"следующ\w*\s+ход\w*)\b"
)
CAPTURE_WITH_SQUARE_RE = re.compile(
    r"\b(?:бь[её]т|взял\w*|съел\w*|захват\w*)\b[^.!?\n]{0,24}?"
    r"(?:(?P<piece>пешк\w*|кон\w*|слон\w*|лад\w*|ферз\w*|корол\w*)\s+)?"
    r"\b(?P<square>[a-h][1-8])\b",
    re.I,
)
CAPTURE_PIECE_RE = re.compile(
    r"\b(?:бь[её]т|взял\w*|съел\w*|захват\w*)\b[^.!?\n]{0,24}?"
    r"(?P<piece>пешк\w*|кон\w*|слон\w*|лад\w*|ферз\w*|корол\w*)",
    re.I,
)
EVAL_SIDE_RE = re.compile(
    r"(?:\bу\s+|\bза\s+)?(?P<side>бел\w*|черн\w*)\b[^.!?\n]{0,36}\b(?:перевес|ведут|имеют|получают)\b",
    re.I,
)
EVAL_SIDE_REV_RE = re.compile(
    r"\b(?P<side>бел\w*|черн\w*)\s+(?:сохраня\w*\s+)?"
    r"(?:решающ\w*|больш\w*|небольш\w*|заметн\w*|сильн\w*)?\s*перевес\b",
    re.I,
)
EVAL_SIDE_GEN_RE = re.compile(
    r"\bперевес\w*\s+(?P<side>бел\w*|черн\w*)",
    re.I,
)
EVAL_EQUAL_RE = re.compile(
    r"\bравн\w*\s+(?:игра|позиция|борьба|шансы)|"
    r"\b(?:игра|позиция|борьба)\s+(?:\w+\s+){0,2}равн\w*|"
    r"\bпримерно\s+равн\w*|\bоценка\s+равн\w*",
    re.I,
)
EVAL_UNCHANGED_RE = re.compile(
    r"\bоценк\w*\b[^.!?\n]{0,24}\b(?:не\s+меняется|почти\s+не\s+меняется|стабильн\w*)\b",
    re.I,
)
PLAYED_PASSIVE_RE = re.compile(r"\b(?:был[аи]?|быть)\s+(?:сыгран|сделан|выполнен)\w*\b", re.I)
NEGATED_PREDICATE_RE = re.compile(r"\b(?:не|без)\s+(?:\w+\s+){0,3}$", re.I)
NEG_CHECK_RE = re.compile(r"\bне\s+(?:\w+\s+){0,2}шах\w*|\bбез\s+шах\w*", re.I)
NEG_MATE_RE = re.compile(r"\bне\s+(?:\w+\s+){0,2}мат\w*|\bбез\s+мат\w*", re.I)
NEG_CAPTURE_RE = re.compile(
    r"\bбез\s+(?:\w+\s+){0,2}взяти\w*|"
    r"\bне\s+(?:\w+\s+){0,2}(?:бер[её]т|берут|взял\w*|съел\w*|захват\w*)",
    re.I,
)
NEG_CASTLING_RE = re.compile(r"\bбез\s+рокировк\w*|\bне\s+(?:\w+\s+){0,2}рокиров\w*", re.I)
NEG_PROMOTION_RE = re.compile(r"\bбез\s+превращени\w*|\bне\s+(?:\w+\s+){0,2}превраща\w*", re.I)


def _norm(text):
    return text.lower().replace("ё", "е")


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _piece_from_word(word):
    word = _norm(word)
    for stem, piece in PIECE_STEMS.items():
        if word.startswith(stem):
            return piece
    return None


def _verdict_group(text):
    """Return the first (longest-priority) explicit verdict in text."""
    normalized = _norm(text)
    for group, pattern in VERDICT_GROUPS:
        if re.search(pattern, normalized):
            return group
    return None


def _asserted_verdicts(text):
    """Return non-negated verdicts in text order; "не ошибка" asserts nothing."""
    normalized = _norm(text)
    found = []
    for group, pattern in VERDICT_GROUPS:
        for match in re.finditer(pattern, normalized):
            prefix = normalized[max(0, match.start() - 30):match.start()]
            if NEGATED_PREDICATE_RE.search(prefix):
                continue
            found.append((match.start(), group))
    return [group for _, group in sorted(found)]


HYPOTHETICAL_RE = re.compile(
    r"(?:угроз\w*|гроз\w*|не\s+отразил\w*|не\s+отражен\w*|взяти\w*|вместо|"
    r"мог\w*|ответ\w*|затем|потом|после|до хода|следующ\w*)\s*(?:[^.!?;:]{0,40})$",
    re.I,
)


def _hypothetical(text, position):
    return bool(HYPOTHETICAL_RE.search(text[max(0, position - 70):position]))


def _last(pattern, text):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _negated(text, position, window=40):
    return bool(NEGATED_PREDICATE_RE.search(text[max(0, position - window):position]))


def _line_context(sentence):
    """True when a sentence describes a past/future line rather than the played move."""
    normalized = _norm(sentence)
    if PLAYED_PASSIVE_RE.match(normalized):
        return False
    if not FUTURE_RE.search(normalized):
        return False
    alternative = re.search(
        r"\b(?:лучше|сильнее|стоило|следовало|надо было)\s+(?:было|был|была|были)\b",
        normalized,
    )
    if not alternative:
        return True
    return bool(re.search(
        r"\b(?:ответ|следующ|угрож|вариант|затем|потом|если|до хода|после хода)\b",
        normalized,
    ))


def _move_role(sentence, position):
    """Classify a move from its nearest preceding played/recommendation cue."""
    prefix = _norm(sentence[:position])
    prefix = re.split(r"[,;:]", prefix)[-1]
    played_matches = list(PLAYED_RE.finditer(prefix))
    recommendation_matches = list(RECOMMEND_RE.finditer(prefix))
    played_at = played_matches[-1].start() if played_matches else -1
    recommendation_at = recommendation_matches[-1].start() if recommendation_matches else -1
    recommendation = recommendation_at > played_at
    played = bool(played_matches) and not _line_context(sentence) and not recommendation
    return recommendation, played


def _actual_eval_side(cp):
    if cp is None:
        return None
    if cp > 50:
        return "белые"
    if cp < -50:
        return "чёрные"
    return "равно"


def _eval_bucket(cp):
    if cp is None:
        return None
    if cp >= 300:
        return "решающий перевес белых"
    if cp >= 100:
        return "большой перевес белых"
    if cp >= 50:
        return "небольшой перевес белых"
    if cp > -50:
        return "равная игра"
    if cp > -100:
        return "небольшой перевес чёрных"
    if cp > -300:
        return "большой перевес чёрных"
    return "решающий перевес чёрных"


def _eval_text(cp):
    side = _actual_eval_side(cp)
    if side == "равно":
        return "равная игра"
    if side == "белые":
        return "перевес белых"
    if side == "чёрные":
        return "перевес чёрных"
    return None


def _best_move(engine):
    best = engine.get("best_move_before") or {}
    if not best:
        return None
    return {
        "uci": best.get("uci"),
        "san": best.get("san"),
        "from": best.get("from"),
        "to": best.get("to"),
        "piece": best.get("piece"),
    }


def build_rubric(facts, target):
    """Create a frozen, transparent rubric from engine facts and the played move."""
    delta = facts["delta"]
    after = facts["after"]
    engine = facts["engine"]
    verdict = _verdict_group(_norm(target.get("verdict") or engine.get("verdict") or "")) or target.get("verdict") or engine.get("verdict")
    cpl = target.get("cpl", engine.get("cpl"))
    score_after = (engine.get("score_after") or {}).get("cp")
    score_before = (engine.get("score_before") or {}).get("cp")
    best = _best_move(engine)
    ideas = [{
        "id": "move", "kind": "move", "text": f"Ход: {delta['from']}→{delta['to']}",
        "from": delta["from"], "to": delta["to"], "piece": delta.get("piece"),
        "san": delta.get("san"),
    }]
    if after.get("is_checkmate"):
        ideas.append({"id": "effect", "kind": "effect", "effect": "mate", "text": "Мат"})
    elif delta.get("gives_check"):
        ideas.append({"id": "effect", "kind": "effect", "effect": "check", "text": "Шах"})
    elif delta.get("is_capture"):
        ideas.append({"id": "effect", "kind": "effect", "effect": "capture", "text": "Взятие"})
    elif delta.get("castling"):
        ideas.append({"id": "effect", "kind": "effect", "effect": "castling", "text": "Рокировка"})
    elif delta.get("promotion"):
        ideas.append({"id": "effect", "kind": "effect", "effect": "promotion", "text": "Превращение"})
    if verdict:
        ideas.append({"id": "verdict", "kind": "verdict", "verdict": verdict,
                      "text": f"Вердикт: {verdict}"})
    if len(ideas) < 2:
        evaluation = _eval_text(score_after)
        if evaluation:
            ideas.append({"id": "evaluation", "kind": "evaluation", "text": f"Оценка: {evaluation}",
                          "side": _actual_eval_side(score_after), "unchanged": False})
        elif score_after is not None and score_before is not None and abs(score_after - score_before) <= 50:
            ideas.append({"id": "evaluation", "kind": "evaluation", "text": "Оценка: не меняется",
                          "side": None, "unchanged": True})
    ideas = ideas[:5]
    if len(ideas) < 2:
        ideas.append({"id": "evaluation", "kind": "evaluation", "text": "Оценка позиции",
                      "side": _actual_eval_side(score_after), "unchanged": False})
    return {
        "version": CRITERIA_VERSION,
        "learner_level": "начинающий",
        "required_ideas": ideas,
        "allowed_facts": {
            "move_uci": delta.get("uci"),
            "san": delta.get("san"),
            "piece": delta.get("piece"),
            "from": delta.get("from"),
            "to": delta.get("to"),
            "is_capture": bool(delta.get("is_capture")),
            "captured": delta.get("captured"),
            "gives_check": bool(delta.get("gives_check")),
            "is_checkmate": bool(after.get("is_checkmate")),
            "castling": delta.get("castling"),
            "promotion": delta.get("promotion"),
            "verdict": verdict,
            "cpl": cpl,
            "score_before_cp": score_before,
            "score_after_cp": score_after,
            "best_move": best,
        },
        "critical_error_examples": _critical_examples(delta, after),
    }


def _critical_examples(delta, after):
    examples = ["неверная фигура или клетка сыгранного хода"]
    if delta.get("is_capture"):
        examples.append("ложное или пропущенное взятие")
    if delta.get("gives_check"):
        examples.append("ложный или пропущенный шах")
    if after.get("is_checkmate"):
        examples.append("ложный или пропущенный мат")
    if delta.get("castling"):
        examples.append("ложная рокировка")
    if delta.get("promotion"):
        examples.append("ложное превращение")
    examples.append("противоположный вердикт или сторона перевеса")
    examples.append("нелегальный рекомендуемый ход")
    return examples


def _move_matches(claim, move):
    same_piece = (not claim.get("piece") or not move.get("piece")
                  or claim["piece"].upper() == move["piece"].upper())
    same_promotion = (not claim.get("promotion") or not move.get("uci")
                      or claim["promotion"] == move["uci"][4:])
    return (claim.get("from") == move.get("from") and claim.get("to") == move.get("to")
            and same_piece and same_promotion)


def _claim(kind, text, **fields):
    if "from_" in fields:
        fields["from"] = fields.pop("from_")
    return {"kind": kind, "text": text, **fields}


def _capture_target(sentence):
    """Return (square, piece) for an explicit capture, preferring the target over the origin."""
    normalized = _norm(sentence)
    for pattern in (
        CAPTURE_NA_RE, CAPTURE_TWO_SQUARES_RE, CAPTURE_PIECE_SQUARE_RE, CAPTURE_DIRECT_RE,
    ):
        match = pattern.search(normalized)
        if not match:
            continue
        square = match["square"]
        prefix = normalized[:match.start("square")]
        if re.search(r"\bс\s*$", prefix):
            continue
        piece = _piece_from_word(match["piece"]) if "piece" in match.groupdict() and match["piece"] else None
        return square, piece
    return None, None


def _claim_segments(sentence):
    """Drop unchecked continuations but retain explicit assertions about the actual move."""
    in_line = False
    for clause in re.split(r"[,;]\s*", sentence):
        clause = clause.strip(" ,;:")
        normalized = _norm(clause)
        if CURRENT_MOVE_RE.match(normalized):
            in_line = False
        if not clause or in_line:
            continue
        if not _line_context(clause):
            yield clause
            continue
        first = FUTURE_RE.search(normalized)
        if first:
            prefix = clause[:first.start()].strip(" ,;:")
            if len(prefix.split()) >= 3 and not _line_context(prefix):
                yield prefix
            marker = normalized[first.start():]
            in_line = bool(LINE_CONTINUATION_RE.match(marker) or marker.startswith(
                ("угроз", "угрож", "гроз", "не отраз", "если", "в варианте")))


def extract_claims(text, facts):
    """Extract only assertions that can be checked against the played move and engine facts.

    Past/future lines ("до хода", "угрожает", "ответ соперника") are skipped instead
    of guessed at.  M4 then counts confirmed assertions over *checkable* assertions,
    matching the expert definition: an unverifiable sentence is not an error.
    """
    claims = []
    seen = set()
    delta = facts["delta"]
    captured = delta.get("captured") or {}
    captured_square = captured.get("square") if isinstance(captured, dict) else None
    capture_squares = {square for square in (delta.get("to"), captured_square) if square}

    def add(claim):
        key = (claim["kind"], claim.get("from"), claim.get("to"), claim.get("square"),
               claim.get("piece"), claim.get("san"), claim.get("verdict"), claim.get("side"),
               claim.get("unchanged"), claim.get("promotion"), claim.get("expected"),
               claim.get("text"))
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    for sentence in _sentences(text):
        for segment in _claim_segments(sentence):
            normalized = _norm(segment)
            recommendation = bool(RECOMMEND_RE.search(normalized))
            played = bool(PLAYED_RE.search(normalized))
            for match in MOVE_RE.finditer(normalized):
                recommendation, is_played = _move_role(normalized, match.start())
                if not recommendation and _hypothetical(normalized, match.start()):
                    continue
                subject = PIECE_SUBJECT_RE.search(normalized[:match.start()])
                add(_claim(
                    "move", segment, from_=match["from"], to=match["to"],
                    piece=_piece_from_word(match["piece"] or (subject[1] if subject else "")),
                    recommendation=recommendation, played=is_played,
                ))
            for match in PLAYED_SAN_RE.finditer(segment):
                is_alternative, is_played = _move_role(normalized, match.start("san"))
                if is_played and not is_alternative and not _negated(normalized, match.start("san")):
                    add(_claim("move", segment, san=match["san"], recommendation=False, played=True))
            for match in UCI_RE.finditer(normalized):
                recommendation, is_played = _move_role(normalized, match.start())
                if not recommendation and _hypothetical(normalized, match.start()):
                    continue
                uci = match[1].lower()
                add(_claim(
                    "move", segment, from_=uci[:2], to=uci[2:4], piece=None,
                    promotion=uci[4] if len(uci) > 4 else None,
                    recommendation=recommendation, played=is_played,
                ))
            if recommendation:
                # The move is the only checkable part of an alternative recommendation.
                continue
            castling_match = CASTLING_CLAIM_RE.search(normalized)
            if castling_match and not _negated(normalized, castling_match.start()):
                add(_claim("castling", segment, played=played, expected=True))
            elif NEG_CASTLING_RE.search(normalized):
                add(_claim("castling", segment, played=played, expected=False))
            promotion_match = PROMOTION_CLAIM_RE.search(normalized)
            if promotion_match and not _negated(normalized, promotion_match.start()):
                add(_claim("promotion", segment, played=played, expected=True))
            elif NEG_PROMOTION_RE.search(normalized):
                add(_claim("promotion", segment, played=played, expected=False))
            negative_capture = NEG_CAPTURE_RE.search(normalized)
            capture_match = CAPTURE_CLAIM_RE.search(normalized)
            if negative_capture and not capture_match:
                add(_claim("capture", segment, square=None, piece=None,
                           played=played, expected=False))
            elif capture_match and not _negated(normalized, capture_match.start()):
                square, piece = _capture_target(segment)
                add(_claim("capture", segment, square=square, piece=piece,
                           played=played, expected=True))
            elif not capture_match:
                square_matches = list(CAPTURE_WITH_SQUARE_RE.finditer(normalized))
                for match in square_matches:
                    square = match["square"]
                    if (square in capture_squares and not _hypothetical(normalized, match.start())
                            and not _negated(normalized, match.start())):
                        add(_claim("capture", segment, square=square,
                                   piece=_piece_from_word(match["piece"] or ""), played=played))
                piece_match = _last(CAPTURE_PIECE_RE, normalized)
                verb = re.search(r"\bбь[её]т\b", normalized)
                if not square_matches and piece_match and verb:
                    between = normalized[verb.end():piece_match.start("piece")]
                    actual_capture = delta.get("captured") or {}
                    actual_piece = actual_capture.get("piece") if isinstance(actual_capture, dict) else None
                    claimed_piece = _piece_from_word(piece_match["piece"])
                    if (not re.search(r"\bпо\b", between)
                            and delta.get("is_capture")
                            and actual_piece
                            and claimed_piece == actual_piece.upper()
                            and not _hypothetical(normalized, verb.start())):
                        add(_claim("capture", segment, square=None,
                                   piece=claimed_piece, played=played))
            negative_check = bool(NEG_CHECK_RE.search(normalized))
            check_match = CHECK_RE.search(normalized)
            mate_match = MATE_RE.search(normalized)
            if (check_match and not MATE_THREAT_RE.search(normalized)
                    and not negative_check and not _hypothetical(normalized, check_match.start())
                    and not _negated(normalized, check_match.start())):
                add(_claim("check", segment, expected=True, played=played))
            elif negative_check and not mate_match and not MATE_THREAT_RE.search(normalized):
                add(_claim("check", segment, expected=False, played=played))
            if (mate_match and not MATE_THREAT_RE.search(normalized)
                    and not _hypothetical(normalized, mate_match.start())):
                expected_mate = not bool(NEG_MATE_RE.search(normalized)) and not _negated(
                    normalized, mate_match.start())
                add(_claim("mate", segment, expected=expected_mate, played=played))
            for verdict in _asserted_verdicts(normalized):
                add(_claim("verdict", segment, verdict=verdict, played=played))
            if "материал" in normalized:
                continue
            if EVAL_UNCHANGED_RE.search(normalized):
                add(_claim("evaluation", segment, unchanged=True, side=None, played=played))
            equal_match = EVAL_EQUAL_RE.search(normalized)
            change_match = re.search(
                r"\b(?:сменя\w*|стал\w*|станов\w*|теперь|в результате|после)\b",
                normalized,
            )
            changed_to_equal = bool(
                equal_match and change_match and equal_match.start() > change_match.start()
            )
            if changed_to_equal:
                add(_claim("evaluation", segment, unchanged=False, side="равно", played=played))
                continue
            side_match = (_last(EVAL_SIDE_GEN_RE, normalized) or _last(EVAL_SIDE_REV_RE, normalized)
                          or _last(EVAL_SIDE_RE, normalized))
            if side_match:
                side = "белые" if side_match["side"].startswith("бел") else "чёрные"
                add(_claim("evaluation", segment, unchanged=False, side=side, played=played))
            elif EVAL_EQUAL_RE.search(normalized):
                add(_claim("evaluation", segment, unchanged=False, side="равно", played=played))
    return claims

def _verify_move(claim, facts):
    actual = facts["delta"]
    if claim["recommendation"]:
        board = chess.Board(facts["before"]["fen"])
        uci = claim["from"] + claim["to"] + (claim.get("promotion") or "")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return "incorrect", True
        if move not in board.legal_moves:
            if claim.get("promotion") is None and any(
                candidate.from_square == move.from_square and candidate.to_square == move.to_square
                and candidate.promotion for candidate in board.legal_moves
            ):
                return "unconfirmed", False
            return "incorrect", True
        best = facts["engine"].get("best_move_before") or {}
        return ("confirmed" if uci == best.get("uci")
                else "unconfirmed"), False
    if not claim.get("played"):
        return "unconfirmed", False
    if claim.get("san"):
        board = chess.Board(facts["before"]["fen"])
        try:
            move = board.parse_san(claim["san"])
        except chess.AmbiguousMoveError:
            return "unconfirmed", False
        except ValueError:
            return "incorrect", True
        if move.uci() != actual.get("uci"):
            return "incorrect", True
        board.push(move)
        if ((claim["san"].endswith("#") and not board.is_checkmate())
                or (claim["san"].endswith("+") and not board.is_check())):
            return "incorrect", True
        return "confirmed", False
    if _move_matches(claim, actual):
        return "confirmed", False
    return "incorrect", True


def _verify_claim(claim, facts, rubric):
    kind = claim["kind"]
    delta = facts["delta"]
    after = facts["after"]
    if kind == "move":
        return _verify_move(claim, facts)
    if kind == "capture":
        expected = claim.get("expected", True)
        actual_capture = bool(delta.get("is_capture"))
        if expected != actual_capture:
            return "incorrect", True
        if not expected:
            return "confirmed", False
        captured = delta.get("captured") or {}
        actual_square = captured.get("square") if isinstance(captured, dict) else None
        actual_piece = captured.get("piece") if isinstance(captured, dict) else None
        if claim.get("square") and claim["square"] != (actual_square or delta.get("to")):
            return "incorrect", True
        if claim.get("piece") and actual_piece and claim["piece"].upper() != actual_piece.upper():
            return "incorrect", True
        return "confirmed", False
    if kind == "check":
        actual = bool(delta.get("gives_check") or after.get("in_check"))
        return ("confirmed", False) if claim["expected"] == actual else ("incorrect", True)
    if kind == "mate":
        actual = bool(after.get("is_checkmate"))
        return ("confirmed", False) if claim["expected"] == actual else ("incorrect", True)
    if kind in ("castling", "promotion"):
        expected = claim.get("expected", True)
        actual = bool(delta.get(kind))
        return ("confirmed", False) if expected == actual else ("incorrect", True)
    if kind == "verdict":
        actual = _verdict_group(_norm(rubric["allowed_facts"].get("verdict") or ""))
        if actual is None:
            return "unconfirmed", False
        if claim["verdict"] == actual:
            return "confirmed", False
        if claim["verdict"] in POSITIVE_VERDICTS and actual in POSITIVE_VERDICTS:
            return "unconfirmed", False
        if claim["verdict"] in NEGATIVE_VERDICTS and actual in NEGATIVE_VERDICTS:
            return "unconfirmed", False
        return "incorrect", True
    if kind == "evaluation":
        before = (facts["engine"].get("score_before") or {}).get("cp")
        after_cp = (facts["engine"].get("score_after") or {}).get("cp")
        if claim.get("unchanged"):
            if before is None or after_cp is None:
                return "unconfirmed", False
            stable = abs(after_cp - before) <= 100 or _eval_bucket(before) == _eval_bucket(after_cp)
            return ("confirmed" if stable else "unconfirmed", False)
        if after_cp is None:
            return "unconfirmed", False
        side = claim["side"]
        if side == "равно":
            return ("confirmed", False) if abs(after_cp) <= 100 else ("incorrect", True)
        if side == "белые":
            if after_cp > 0:
                return "confirmed", False
            return ("unconfirmed", False) if after_cp > -50 else ("incorrect", True)
        if side == "чёрные":
            if after_cp < 0:
                return "confirmed", False
            return ("unconfirmed", False) if after_cp < 50 else ("incorrect", True)
        return "unconfirmed", False
    return "unconfirmed", False

def _mentions_squares(text, squares):
    found = set(SQUARE_RE.findall(text))
    return all(square in found for square in squares if square)


def _mentions_san(text, san):
    if not san:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(_norm(san).rstrip("+#")) + r"(?![a-z0-9])",
                     _norm(text)) is not None


def _idea_met(idea, claims, text, facts):
    kind = idea["kind"]
    current_sentences = [sentence for sentence in _sentences(text) if not _line_context(sentence)]
    if kind in ("move", "best_move"):
        recommendation = kind == "best_move"
        if any(
            claim["kind"] == "move" and claim["verdict"] != "incorrect"
            and (kind == "move" or claim.get("recommendation") == recommendation)
            and _move_matches(claim, idea)
            for claim in claims
        ):
            return True
        if any(_mentions_san(sentence, idea.get("san")) for sentence in current_sentences):
            return True
        if kind == "move":
            return any(
                claim["kind"] == "capture" and claim["verdict"] == "confirmed"
                and _mentions_squares(claim["text"], (idea.get("from"), idea.get("to")))
                for claim in claims
            )
        if any(RECOMMEND_RE.search(_norm(sentence)) for sentence in current_sentences):
            piece = idea.get("piece")
            piece_stem = next(
                (stem for stem, code in PIECE_STEMS.items() if code == (piece or "").upper()),
                None,
            )
            return bool(
                piece_stem
                and any(
                    re.search(r"\b" + piece_stem + r"\w*\b", _norm(sentence))
                    and _mentions_squares(sentence, (idea.get("to"),))
                    for sentence in current_sentences
                )
            )
    if kind == "effect":
        effect = idea["effect"]
        claim_kind = {"mate": "mate", "check": "check", "capture": "capture",
                      "castling": "castling", "promotion": "promotion"}.get(effect)
        return any(
            claim["kind"] == claim_kind and claim["verdict"] == "confirmed"
            and claim.get("expected", True)
            for claim in claims
        )
    if kind == "verdict":
        if any(
            claim["kind"] == "verdict" and claim["verdict"] == "confirmed"
            and claim.get("asserted_verdict") == idea["verdict"]
            for claim in claims
        ):
            return True
        pattern = dict(VERDICT_GROUPS).get(idea["verdict"])
        if pattern:
            return any(
                not re.search(r"\bне\s+$", _norm(sentence[max(0, match.start() - 12):match.start()]))
                for sentence in current_sentences
                for match in re.finditer(pattern, _norm(sentence))
            )
    if kind == "evaluation":
        if any(
            claim["kind"] == "evaluation" and claim["verdict"] == "confirmed"
            and claim.get("unchanged", False) == idea.get("unchanged", False)
            and claim.get("side") == idea.get("side")
            for claim in claims
        ):
            return True
        normalized = _norm(text)
        if idea.get("unchanged"):
            return bool(EVAL_UNCHANGED_RE.search(normalized))
        cp = (facts["engine"].get("score_after") or {}).get("cp")
        side = idea.get("side")
        if cp is None or side not in ("равно", "белые", "чёрные"):
            return False
        evaluation_matches = [
            match
            for pattern in (
                EVAL_EQUAL_RE, EVAL_SIDE_GEN_RE, EVAL_SIDE_REV_RE, EVAL_SIDE_RE,
            )
            for match in pattern.finditer(normalized)
        ]
        if not evaluation_matches:
            return False
        evaluation_match = max(evaluation_matches, key=lambda match: match.start())
        prefix = normalized[:evaluation_match.start()]
        past = list(re.finditer(r"\b(?:до хода|до него|перед ходом)\b", prefix))
        current = list(re.finditer(
            r"\b(?:после хода|после него|после этого|теперь|в результате|стала|сменил\w*)\b",
            prefix,
        ))
        if past and (not current or past[-1].start() > current[-1].start()):
            return False
        if EVAL_EQUAL_RE.fullmatch(evaluation_match.group()):
            claimed_side = "равно"
        else:
            claimed_side = "белые" if evaluation_match["side"].startswith("бел") else "чёрные"
        if claimed_side:
            if claimed_side == side and ((claimed_side == "белые" and cp > 0)
                                         or (claimed_side == "чёрные" and cp < 0)):
                return True
            if abs(cp) <= 50 and side == "равно" and (
                    (claimed_side == "белые" and cp > 0)
                    or (claimed_side == "чёрные" and cp < 0)):
                return True
        return side == "равно" and abs(cp) <= 100 and bool(EVAL_EQUAL_RE.search(normalized))
    return False


def _m7(text):
    normalized = _norm(text)
    sentences = _sentences(text)
    words = re.findall(r"[a-zа-я0-9]+", normalized)
    clarity = (
        1 <= len(sentences) <= 6
        and (not words or len(words) / len(sentences) <= 40)
        and "<|" not in text
        and "gigatoken" not in normalized
    )
    negative_check = bool(re.search(r"\bне\s+шах\b|\bбез\s+шах\w*", normalized))
    no_capture = bool(re.search(r"\bбез\s+(?:\w+\s+){0,2}взяти", normalized))
    effect = (
        (bool(CHECK_RE.search(normalized)) and not negative_check)
        or bool(MATE_RE.search(normalized) and not MATE_THREAT_RE.search(normalized))
        or (bool(CAPTURE_RE.search(normalized)) and not no_capture)
        or bool(CASTLING_RE.search(normalized))
        or bool(PROMOTION_RE.search(normalized))
    )
    return {
        "goal": bool(GOAL_RE.search(normalized) or effect),
        "position_link": bool(SQUARE_RE.search(normalized)),
        "clarity": clarity,
        "actionable": bool(ACTION_RE.search(normalized)),
    }


def score_response(text, case, truncated=False):
    """Score one free-text reply against a frozen automatic rubric."""
    gold = case["gold"]
    facts = gold["facts"]
    rubric = gold["rubric"]
    base = {
        "id": case["id"],
        "M4": 0.0, "M5": 0.0, "M6": 0, "M7": 0.0, "accepted": False,
        "claims": [], "idea_scores": [0] * len(rubric["required_ideas"]),
        "criteria": {"goal": False, "position_link": False, "clarity": False, "actionable": False},
        "empty": not text.strip(), "truncated": bool(truncated),
        "critical_error": False,
    }
    if not text.strip() or truncated:
        return base
    claims = extract_claims(text, facts)
    verified = []
    for claim in claims:
        verdict, critical = _verify_claim(claim, facts, rubric)
        verified.append({
            **claim,
            "asserted_verdict": claim.get("verdict") if claim["kind"] == "verdict" else None,
            "verdict": verdict,
            "critical": critical,
        })
    checked = [c for c in verified if c["verdict"] in ("confirmed", "incorrect")]
    m4 = (sum(c["verdict"] == "confirmed" for c in checked) / len(checked)) if checked else 0.0
    idea_scores = [1 if _idea_met(idea, verified, text, facts) else 0
                   for idea in rubric["required_ideas"]]
    m5 = sum(idea_scores) / len(idea_scores) if idea_scores else 0.0
    critical = any(c["critical"] for c in verified)
    m6 = int(critical)
    criteria = _m7(text)
    m7 = sum(criteria.values()) / 4
    accepted = bool(m4 == 1 and m5 >= 0.8 and m6 == 0 and m7 >= 0.75)
    base.update({
        "M4": m4, "M5": m5, "M6": m6, "M7": m7, "accepted": accepted,
        "claims": verified, "idea_scores": idea_scores, "criteria": criteria,
        "confirmed_claims": sum(c["verdict"] == "confirmed" for c in verified),
        "incorrect_claims": sum(c["verdict"] == "incorrect" for c in verified),
        "unconfirmed_claims": sum(c["verdict"] == "unconfirmed" for c in verified),
        "critical_error": critical,
    })
    return base


def _validate_case(row, facts_record):
    board = chess.Board(row["fen"])
    if not board.is_valid():
        raise ValueError("invalid FEN")
    move = chess.Move.from_uci(row["move_uci"])
    if move not in board.legal_moves:
        raise ValueError("illegal move")
    facts = facts_record["facts"]
    if facts["delta"]["uci"] != row["move_uci"]:
        raise ValueError("facts describe another move")
    if facts["before"]["fen"] != row["fen"]:
        raise ValueError("facts describe another FEN")
    after = board.copy()
    after.push(move)
    if row.get("target", {}).get("fen_after") and after.fen(en_passant="fen") != row["target"]["fen_after"]:
        raise ValueError("target FEN differs")
    return facts


def commentary_cases(commentary, facts_path, split, budget, limit=None):
    rows = read_jsonl(commentary)
    facts = unique(read_jsonl(facts_path))
    cases, excluded = [], []
    for row in rows:
        if row.get("split") != split or row.get("task") != "move_commentary":
            continue
        try:
            if row["id"] not in facts:
                raise ValueError("missing facts record")
            facts_bundle = _validate_case(row, facts[row["id"]])
        except (KeyError, ValueError) as exc:
            excluded.append({"id": row.get("id"), "reason": str(exc)})
            continue
        target = row.get("target") or {}
        rubric = build_rubric(facts_bundle, target)
        cases.append({
            "id": "commentary:" + row["id"],
            "suite": "commentary",
            "task": "move_commentary",
            "fen": row["fen"],
            "prompt": row["question"],
            "score_type": "explanation_auto",
            "gold": {
                "move_uci": row["move_uci"],
                "fen_after": target.get("fen_after"),
                "facts": facts_bundle,
                "rubric": rubric,
            },
            "group_id": row.get("group_id") or row["id"],
            "params": {"split": split, "rubric_version": CRITERIA_VERSION},
            "max_new_tokens": budget,
            "private": True,
            "provenance": {"source": "handoff-desc-v1", "record_id": row["id"]},
        })
    if limit:
        cases = cases[:limit]
    return cases, excluded


def load_frozen_suite(path=None):
    suite = Path(path) if path else ROOT / "data/commentary_m4_m7_auto_v1"
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("criteria_version") != CRITERIA_VERSION:
        raise ValueError("frozen commentary suite uses another criteria version")
    for filename, hash_key in (
        ("cases.jsonl", "cases_sha256"),
        ("queries.jsonl", "queries_sha256"),
    ):
        if manifest.get(hash_key) != sha(suite / filename):
            raise ValueError(f"frozen commentary suite {filename} hash mismatch")

    cases = read_jsonl(suite / "cases.jsonl")
    queries = read_jsonl(suite / "queries.jsonl")
    rubrics = read_jsonl(suite / "auto_rubrics.jsonl")
    audit = read_jsonl(suite / "audit.jsonl")
    if len(cases) != manifest.get("case_count") or not cases:
        raise ValueError("frozen commentary suite case count mismatch")
    if len(audit) != manifest.get("excluded_commentary_cases"):
        raise ValueError("frozen commentary suite audit count mismatch")
    unique(cases)
    if len({query["id"] for query in queries}) != len(queries):
        raise ValueError("frozen commentary suite has duplicate query IDs")
    expected_queries = [
        {"id": case["id"], "fen": case["fen"], "prompt": case["prompt"],
         "max_new_tokens": case["max_new_tokens"]}
        for case in cases
    ]
    if queries != expected_queries:
        raise ValueError("frozen commentary suite queries differ from its cases")
    expected_rubrics = [
        {"id": case["id"], **case["gold"]["rubric"]} for case in cases
    ]
    if rubrics != expected_rubrics:
        raise ValueError("frozen commentary suite rubrics differ from its cases")
    if any(
        case["score_type"] != "explanation_auto"
        or case["params"].get("rubric_version") != CRITERIA_VERSION
        or "reference_answer" in case["gold"]
        for case in cases
    ):
        raise ValueError("frozen commentary suite contains invalid or reference cases")
    return cases, manifest


def prepare_run(args):
    from .pipeline import source_hashes

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    budget = args.max_new_tokens or DEFAULT_BUDGET
    if budget < 1:
        raise ValueError("max_new_tokens must be positive")
    cases, excluded = commentary_cases(args.commentary, args.facts, args.split, budget, args.limit)
    if not cases:
        raise ValueError("no commentary cases selected")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    queries = [{"id": c["id"], "fen": c["fen"], "prompt": c["prompt"],
                "max_new_tokens": c["max_new_tokens"]} for c in cases]
    rubrics = [{"id": c["id"], **c["gold"]["rubric"]} for c in cases]
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "auto_rubrics.jsonl", rubrics)
    write_jsonl(out / "audit.jsonl", excluded)
    config = dict(config)
    config["answer_protocol"] = "legacy"
    config["generation"] = dict(config.get("generation", {}), max_new_tokens=budget)
    manifest = {
        "version": __version__,
        "profile": "commentary-auto",
        "model_input_profile": "contract_v1_fen",
        "model_input_template": "Позиция FEN: {fen}\n{prompt}",
        "config": config,
        "criteria_version": CRITERIA_VERSION,
        "suite_id": (
            f"commentary-{args.split}-{sha(out / 'cases.jsonl')[:8]}-"
            f"{sha(out / 'queries.jsonl')[:8]}"
        ),
        "commentary_split": args.split,
        "case_count": len(cases),
        "automatic_queries": len(queries),
        "explanation_cases": 0,
        "excluded_explanation_cases": 0,
        "excluded_commentary_cases": len(excluded),
        "tasks": dict(Counter(c["suite"] + "/" + c["task"] for c in cases)),
        "code_sha256": source_hashes(),
        "cases_sha256": sha(out / "cases.jsonl"),
        "queries_sha256": sha(out / "queries.jsonl"),
        "source_commentary_sha256": sha(args.commentary),
        "source_facts_sha256": sha(args.facts),
        "contains_local_source_comments": False,
        "contains_reference_answers": False,
        "contains_private_cases": True,
        "source_legacy_sha256": None,
        "external_sources": {},
        "chess_version": chess.__version__,
        "chess_module_sha256": sha(chess.__file__),
        "gold_sources": "facts_v2 engine facts + python-chess; automatic M4-M7 rubric, not expert-approved",
        "upstream_comparability": "Separate commentary block with automatic criteria; not the official GigaChess score",
    }
    write_json(out / "manifest.json", manifest)
    return manifest, excluded


def score_run(args):
    run = Path(args.run)
    suite_path = Path(args.suite) if getattr(args, "suite", None) else run
    cases_path = suite_path / "cases.jsonl"
    queries_path = suite_path / "queries.jsonl"
    suite_manifest = None
    if getattr(args, "suite", None):
        suite_manifest = json.loads((suite_path / "manifest.json").read_text(encoding="utf-8"))
        if suite_manifest.get("criteria_version") != CRITERIA_VERSION:
            raise ValueError("suite uses a different criteria version")
        if suite_manifest.get("cases_sha256") != sha(cases_path):
            raise ValueError("frozen suite cases hash does not match manifest")
        if suite_manifest.get("queries_sha256") != sha(queries_path):
            raise ValueError("frozen suite queries hash does not match manifest")
    if suite_manifest is None and (run / "manifest.json").is_file():
        run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if run_manifest.get("criteria_version") != CRITERIA_VERSION:
            raise ValueError("run uses a different criteria version")
        if run_manifest.get("cases_sha256") != sha(cases_path):
            raise ValueError("run cases hash does not match manifest")
        if (run / "queries.jsonl").is_file() and run_manifest.get("queries_sha256") != sha(
                run / "queries.jsonl"):
            raise ValueError("run queries hash does not match manifest")
    case_rows = read_jsonl(cases_path)
    cases = {c["id"]: c for c in case_rows}
    if len(cases) != len(case_rows):
        raise ValueError("suite contains duplicate case IDs")
    if getattr(args, "suite", None):
        queries = read_jsonl(queries_path)
        query_ids = [q["id"] for q in queries]
        if len(set(query_ids)) != len(query_ids) or set(query_ids) != set(cases):
            raise ValueError("suite queries and scoring cases have different IDs")
        run_queries_path = run / "queries.jsonl"
        if run_queries_path.exists():
            if read_jsonl(run_queries_path) != queries:
                raise ValueError("run queries do not match the frozen suite")
        else:
            write_jsonl(run_queries_path, queries)
    responses = unique(read_jsonl(run / "responses.jsonl"))
    missing = sorted(set(cases) - set(responses))
    unexpected = sorted(set(responses) - set(cases))
    if unexpected:
        raise ValueError("responses contain unknown IDs")
    if missing and not args.allow_partial:
        raise ValueError(f"missing {len(missing)} responses; use --allow-partial to score a subset")
    rows = []
    for identifier, case in cases.items():
        if identifier not in responses:
            continue
        response = responses[identifier]
        rows.append(score_response(response.get("text", ""), case,
                                   truncated=bool(response.get("truncated"))))
    if not rows:
        raise ValueError("no responses to score")
    n = len(rows)
    summary = {
        "criteria_version": CRITERIA_VERSION,
        "suite_id": suite_manifest.get("suite_id") if suite_manifest else None,
        "cases_sha256": sha(cases_path),
        "queries_sha256": sha(queries_path) if queries_path.is_file() else None,
        "n": n,
        "expected": len(cases),
        "complete": not missing,
        "M4": 100 * sum(r["M4"] for r in rows) / n,
        "M5": 100 * sum(r["M5"] for r in rows) / n,
        "M6": 100 * sum(r["M6"] for r in rows) / n,
        "M7": 100 * sum(r["M7"] for r in rows) / n,
        "accepted": sum(r["accepted"] for r in rows),
        "accepted_percent": 100 * sum(r["accepted"] for r in rows) / n,
        "empty": sum(r.get("empty") for r in rows),
        "truncated": sum(r.get("truncated") for r in rows),
        "claims_total": sum(len(r.get("claims", [])) for r in rows),
        "claims_confirmed": sum(r.get("confirmed_claims", 0) for r in rows),
        "claims_incorrect": sum(r.get("incorrect_claims", 0) for r in rows),
        "claims_unconfirmed": sum(r.get("unconfirmed_claims", 0) for r in rows),
        "critical_errors": sum(r.get("critical_error") for r in rows),
    }
    write_jsonl(run / "m4_m7.jsonl", rows)
    write_json(run / "m4_m7_summary.json", summary)
    (run / "m4_m7_report.html").write_text(
        render_report(summary, rows, title="Автоматические М4–М7"), encoding="utf-8")
    return summary


def _pct(value):
    return "—" if value is None else f"{value:.1f}%"


def _delta(after, before, lower_is_better=False):
    if after is None or before is None:
        return "—", "flat"
    diff = after - before
    if abs(diff) < 1e-9:
        return "0,0 п. п.", "flat"
    good = diff < 0 if lower_is_better else diff > 0
    return f"{diff:+.1f} п. п.".replace("-", "−"), "up" if good else "down"


def render_report(summary, rows, title):
    case_rows = []
    for row in rows:
        details = "; ".join(
            f"{html.escape(c['kind'])}: {html.escape(c['verdict'])}"
            for c in row.get("claims", [])
        )
        case_rows.append(
            "<tr>"
            f"<td>{html.escape(row['id'])}</td>"
            f"<td class='num'>{100*row['M4']:.0f}%</td>"
            f"<td class='num'>{100*row['M5']:.0f}%</td>"
            f"<td class='num'>{row['M6']}</td>"
            f"<td class='num'>{row['M7']*4:.0f}/4</td>"
            f"<td class='{'ok' if row['accepted'] else 'no'}'>{'да' if row['accepted'] else 'нет'}</td>"
            f"<td>{details or '—'}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f6f2;--surface:#fff;--ink:#1b2420;--muted:#5d6b65;--line:#dcd8cc;--good:#1f7a4d;--bad:#b4481a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}}
.big{{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}}.tag{{font-size:12px;color:var(--muted);text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:13px;color:var(--muted)}}td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.scroll{{overflow-x:auto}}.ok{{color:var(--good);font-weight:700}}.no{{color:var(--bad);font-weight:700}}
.note{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:12px 0}}
</style><main>
<h1>{html.escape(title)}</h1>
<p class="muted">Критерии {html.escape(summary['criteria_version'])}. Оценено {summary['n']} из {summary['expected']} ответов.</p>
<div class="cards">
<div class="card"><div class="tag">M4-auto факты</div><div class="big">{_pct(summary['M4'])}</div></div>
<div class="card"><div class="tag">M5-auto идеи</div><div class="big">{_pct(summary['M5'])}</div></div>
<div class="card"><div class="tag">M6-auto критич. ошибки</div><div class="big">{_pct(summary['M6'])}</div><div class="muted">меньше — лучше</div></div>
<div class="card"><div class="tag">M7-auto структура</div><div class="big">{_pct(summary['M7'])}</div></div>
<div class="card"><div class="tag">Принято</div><div class="big">{_pct(summary['accepted_percent'])}</div><div class="muted">{summary['accepted']} / {summary['n']}</div></div>
</div>
<p class="note">M4–M7-auto — детерминированные критерии по фактам движка и python-chess. Это не экспертная оценка педагогической пользы. M6=0 означает «скрипт не нашёл критической ошибки», а не доказанное её отсутствие. Пустой или обрезанный ответ получает 0.</p>
<p>Проверяемых утверждений: {summary['claims_total']}; подтверждено: {summary['claims_confirmed']}; неверно: {summary['claims_incorrect']}; не подтверждено: {summary['claims_unconfirmed']}. Критических ответов: {summary['critical_errors']}. Пустых: {summary['empty']}; обрезанных: {summary['truncated']}.</p>
<div class="scroll"><table>
<thead><tr><th>Ответ</th><th class="num">M4-auto</th><th class="num">M5-auto</th><th class="num">M6-auto</th><th class="num">M7-auto</th><th>Принят</th><th>Утверждения</th></tr></thead>
<tbody>{''.join(case_rows)}</tbody></table></div>
</main></html>
"""


def compare_runs(args):
    before_path, after_path = Path(args.before), Path(args.after)
    before = json.loads((before_path / "m4_m7_summary.json").read_text(encoding="utf-8"))
    after = json.loads((after_path / "m4_m7_summary.json").read_text(encoding="utf-8"))
    before_queries = unique(read_jsonl(before_path / "queries.jsonl"))
    after_queries = unique(read_jsonl(after_path / "queries.jsonl"))
    if before_queries != after_queries:
        raise ValueError("runs are not comparable: queries, FENs or generation budgets differ")
    if (before["criteria_version"] != after["criteria_version"]
            or before.get("suite_id") != after.get("suite_id")
            or before.get("cases_sha256") != after.get("cases_sha256")
            or not before.get("complete") or not after.get("complete")
            or before["n"] != before["expected"] or after["n"] != after["expected"]):
        raise ValueError("runs are not comparable: incomplete runs or different scoring cases")
    if before["n"] != after["n"] or before["n"] != len(before_queries):
        raise ValueError("runs are not comparable: different case counts")
    labels = (args.before_label or "до", args.after_label or "после")
    rows = []
    for key, title, lower in (
        ("M4", "M4-auto факты", False), ("M5", "M5-auto обязательные идеи", False),
        ("M6", "M6-auto критические ошибки", True), ("M7", "M7-auto структура", False),
        ("accepted_percent", "Принятые объяснения", False),
    ):
        delta, css = _delta(after[key], before[key], lower)
        rows.append(
            f"<tr><th>{html.escape(title)}</th><td class='num'>{_pct(before[key])}</td>"
            f"<td class='num'>{_pct(after[key])}</td><td class='{css}'>{delta}</td></tr>"
        )
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Сравнение М4–М7-auto</title>
<style>body{{margin:0;background:#f3f6f9;color:#172532;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:900px;margin:auto;padding:24px 16px}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dbe3eb}}
th,td{{padding:12px;border-bottom:1px solid #dbe3eb;text-align:left}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.up{{color:#126333;font-weight:700}}.down{{color:#a21b2d;font-weight:700}}.flat{{color:#46586a;font-weight:700}}</style>
<main><h1>Сравнение М4–М7-auto</h1>
<p>Критерии {html.escape(before['criteria_version'])}; по {before['n']} ответам у каждой модели.
M6 — доля ответов с критической ошибкой, меньше лучше.</p>
<table><thead><tr><th>Показатель</th><th>{html.escape(labels[0])}</th><th>{html.escape(labels[1])}</th><th>Δ</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Автоматические критерии не заменяют слепую экспертную разметку.</p></main></html>"""
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="build a commentary run with frozen automatic rubrics")
    prepare.add_argument("--commentary", type=Path, required=True)
    prepare.add_argument("--facts", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--max-new-tokens", type=int, default=DEFAULT_BUDGET)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    score = sub.add_parser("score", help="score responses with automatic M4-M7 criteria")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--suite", type=Path,
                       help="frozen suite directory; defaults to cases in --run")
    score.add_argument("--allow-partial", action="store_true")
    compare = sub.add_parser("compare", help="compare two M4-M7-auto summaries")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--before-label")
    compare.add_argument("--after-label")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest, excluded = prepare_run(args)
        print(json.dumps({"cases": manifest["case_count"], "excluded": excluded,  # noqa: T201
                          "out": str(args.out)}, ensure_ascii=False))
    elif args.command == "score":
        print(json.dumps(score_run(args), ensure_ascii=False, indent=2))  # noqa: T201
    else:
        print(compare_runs(args))  # noqa: T201


if __name__ == "__main__":
    main()
