"""Шаг 2. Замысел из входных партий: человеческий комментарий, очищенный скриптом. Без LLM.

Замысел — это входные данные (слова игрока/аннотатора), а не результат генерации.
Скрипт только чистит комментарий и оставляет предложения про намерение:
  - вычищает [%eval]/[%clk]/ссылки;
  - отбрасывает мусор целиком: приветствия, «поставьте лайк», автоаннотации lichess, результат партии;
  - из оставшегося берёт только предложения с маркерами намерения и без вердиктов, оценок и вариантов
    («9. Nd5 is worth looking», «this one is particularly bad» отбрасываются).

Результат: intent = {"status", "text", "reasons"}; status: none | dropped_rule | ok.
Какая доля примеров получит замысел во входе (пайплайн: 15% с ним, 85% без), решает шаг 4.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.io import read_jsonl, write_jsonl

MAX_INTENT_CHARS = 200

CMD_RE = re.compile(r"\[%[^\]]*\]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
JUNK_RE = re.compile(
    r"(click|heart|like (this|the) study|subscribe|follow me|thanks for (watching|reading)|welcome to|"
    r"hello|hi everyone|nothing to annotate|no comment|youtube|twitch|discord|patreon|chapter \d|in this study|lichess\.org|chess\.com|"
    r"лайк|подпис|всем привет|глава \d)", re.IGNORECASE)
AUTO_ANNOT_RE = re.compile(r"^(inaccuracy|mistake|blunder)\.|\bwas best\b|^\(?[+-]?\d+\.\d+\)?\s*(→|->)", re.IGNORECASE)
RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|½-½)\b|^(white|black) (wins|resigns)\.?$", re.IGNORECASE)
INTENT_MARKERS_RE = re.compile(
    r"(\bto [a-z]+|in order|so that|so as|intend|plan|idea|aim|prepar|threat|menac|hop(e|es|ing)\b|"
    r"\btry|tries|want|going to|target|purpose|looking (for|to)|heading|pressure|presur|attack|"
    r"fight|provok|protect|defen|prevent|stop|develop|hold|keep|control|support|activat|free|open|"
    r"sacrific|trap|\bpin|fork|castl|break|exploit|improv|prophyla|avoid|"
    r"\bum\b.*\bzu\b|droht|idee|angriff|verteidig|vorberei|"
    r"чтобы|идея|план|готов|угрож|собира|хочет|замысел|с целью|намерен|стрем|атак|давлен|"
    r"защищ|предотвр|развив|контрол|жертв|вскры)",
    re.IGNORECASE)
# Предложение — не замысел, если в нём вердикт, оценка или вариант с номерами ходов.
VERDICT_RE = re.compile(
    r"(better|worse|best|mistake|blunder|inaccura|\bbad\b|\bgood\b|excellent|brilliant|winning|losing|"
    r"\blost\b|\bwon\b|equal|advantage|decisive|\bnot sure|should|could have|would have|of course|"
    r"schlecht|besser|fehler|entscheidend|verkennung|durfte|niemals|aussichtslos|hätte|sollte|"
    r"лучше|хуже|ошибк|зевок|сильн|слаб|выигра|проигра|равен|преимуществ|[!?]{2}|\(\s*[!?]|\d+\s*\.\s*(\.\.)?\s*[KQRBNO]?[a-h]?x?[a-h][1-8]|"
    r"[+-]\d+[.,]\d+|\d+\s*\.(\s|$)|\([^)]*$)", re.IGNORECASE)
# Предложение — не замысел этого хода, если оно про прошлое: сожаление о прежних ходах, «давно надо было».
RETRO_RE = re.compile(
    r"(\bhad to\b|\bhad been\b|long (time )?ago|\bearlier\b|\bpreviously\b|\bmissed\b|\bforgot|"
    r"should('ve| have)|\bwas (a|an|the|already)\b|früher|hätte|давно|раньше|надо было|следовало|упустил)",
    re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\s+~\s+")


def clean(comment: str) -> str:
    c = CMD_RE.sub(" ", comment)
    c = URL_RE.sub(" ", c)
    return re.sub(r"\s+", " ", c).strip()


SIDE_RE = {
    "w": re.compile(r"\b(white|weiß|weiss)\b|бел(ые|ых|ым)", re.IGNORECASE),
    "b": re.compile(r"\b(black|schwarz)\b|ч[её]рн(ые|ых|ым)", re.IGNORECASE),
}


def about_other_side(text: str, fen: str) -> bool:
    """Комментарий в ChessGPT часто про ход соперника («Black adds a defender…» к ходу белых):
    замысел, где названа только другая сторона, к этому ходу не относится."""
    parts = fen.split()
    if len(parts) < 2:
        return False
    me, other = parts[1], "b" if parts[1] == "w" else "w"
    return bool(SIDE_RE[other].search(text)) and not SIDE_RE[me].search(text)


def extract_intent(comment: str) -> tuple[str | None, str | None]:
    """-> (текст замысла, причина отбраковки)."""
    c = clean(comment)
    if not c:
        return None, "empty"
    if RESULT_RE.search(c):
        return None, "game_result"
    if AUTO_ANNOT_RE.search(c):
        return None, "engine_auto_annotation"
    if JUNK_RE.search(c):
        return None, "junk_meta"
    kept = []
    for s in SENT_SPLIT_RE.split(c):
        s = s.strip()
        if len(s.split()) < 3 or not INTENT_MARKERS_RE.search(s) or VERDICT_RE.search(s) or RETRO_RE.search(s):
            continue
        if len(" ".join(kept + [s])) > MAX_INTENT_CHARS:
            break
        kept.append(s)
    if not kept:
        return None, "no_intent_sentence"
    return " ".join(kept), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out, stats, reasons = [], Counter(), Counter()
    for rec in read_jsonl(args.inp):
        # Явный замысел от разметчика идёт по тем же правилам, что и комментарий.
        raw = (rec.get("intent_given") or rec.get("comment_raw") or "").strip()
        if not raw:
            rec["intent"] = {"status": "none", "text": None, "reasons": []}
        else:
            text, why = extract_intent(raw)
            if text and about_other_side(text, rec.get("fen_before", "")):
                text, why = None, "other_side"
            rec["intent"] = ({"status": "ok", "text": text, "reasons": []} if text
                             else {"status": "dropped_rule", "text": None, "reasons": [why]})
            reasons.update(rec["intent"]["reasons"])
        stats[rec["intent"]["status"]] += 1
        out.append(rec)
    write_jsonl(args.out, out)
    print("замысел:", dict(stats), "| причины отбраковки:", dict(reasons))


if __name__ == "__main__":
    main()
