"""Шаг 4. Сборка несвязного фактологического черновика + вопрос для модели.

Dropout замысла (пайплайн, блок 1): только у ~15% примеров замысел человека попадает во вход
и в комментарий; остальные описывают мотив по геометрии хода. Иначе модель научится ждать
подсказку, которой на инференсе обычно нет. Решение детерминировано по id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.draft import build_draft
from gcdata.io import read_jsonl, stable_unit, write_jsonl
from gcdata.template import CHARS_PER_TOKEN, TokenCounter, build_question, engine_block


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--intent-rate", type=float, default=0.15,
                    help="целевая доля примеров с замыслом во входе среди ВСЕХ примеров (пайплайн: 0.15)")
    ap.add_argument("--max-answer-chars", type=int, default=600)
    ap.add_argument("--tokenizer", help="tokenizer.json модели для точного подсчёта длины")
    ap.add_argument("--seed", default="gigachess-v2")
    args = ap.parse_args()

    tc = TokenCounter(args.tokenizer)
    recs = list(read_jsonl(args.inp))
    with_intent = sum((r.get("intent") or {}).get("status") == "ok" for r in recs)
    # Вероятность взять замысел, чтобы в итоге получилось ~intent_rate от всего набора.
    p = min(1.0, args.intent_rate * len(recs) / with_intent) if with_intent else 0.0
    out, n_intent, n_tight = [], 0, 0
    for rec in recs:
        intent = rec.get("intent") or {}
        use_intent = intent.get("status") == "ok" and stable_unit(args.seed, "intent", rec["id"]) < p
        intent_text = intent["text"] if use_intent else None
        question = build_question(rec["facts"]["delta"], intent_text, engine_block(rec))
        draft = build_draft(rec, intent_text)
        # Бюджет ответа в символах: не больше того, что влезет в 512 позиций.
        budget_tokens = tc.answer_budget(rec["fen_before"], question)
        max_chars = min(args.max_answer_chars, int(budget_tokens * CHARS_PER_TOKEN) - 20)
        if max_chars < 300:
            n_tight += 1
            continue
        n_intent += bool(intent_text)
        eng = rec["facts"]["engine"]
        out.append({
            "id": rec["id"], "group_id": rec["group_id"], "source": rec["source"], "ply": rec.get("ply"),
            "fen_before": rec["fen_before"], "fen_after": rec["facts"]["after"]["fen"],
            "move_uci": rec["move_uci"], "move_san": rec["move_san"],
            "question": question, "draft": draft, "max_chars": max_chars,
            "intent_used": bool(intent_text), "intent": intent,
            "verdict": eng["verdict"], "nag": eng["nag"], "cpl": eng["cpl"],
            "facts": rec["facts"],
        })
    write_jsonl(args.out, out)
    print(f"черновиков: {len(out)}, с замыслом во входе: {n_intent}, отброшено по длине: {n_tight}")


if __name__ == "__main__":
    main()
