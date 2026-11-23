"""Сверка ключей gcdata/keys.py с темами базы задач lichess (CC0).

Темы в базе проставил разметчик lichess-puzzler (их версия кода могла отличаться от текущей),
поэтому это проверка того, насколько точно мы воспроизвели правила, а не независимая истина.
Выход: точность и полнота по каждой теме + примеры расхождений в <out>.jsonl.

  python tools/validate_keys.py --csv data/lichess_puzzles/head200k.csv --n 20000 --out work_keys/val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.keys import Line, tags  # noqa: E402

# Темы, которые проверяем (остальные темы базы — длина, фаза, оценка и т. п. — не наши ключи).
THEMES = [
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "deflection", "attraction", "hangingPiece",
    "trappedPiece", "xRayAttack", "capturingDefender", "interference", "clearance", "intermezzo", "sacrifice",
    "quietMove", "exposedKing", "attackingF2F7", "kingsideAttack", "queensideAttack", "advancedPawn",
    "enPassant", "promotion", "underPromotion", "mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
    "backRankMate", "smotheredMate", "anastasiaMate", "arabianMate", "hookMate", "bodenMate",
    "doubleBishopMate", "dovetailMate", "pawnEndgame", "rookEndgame", "bishopEndgame", "knightEndgame",
    "queenEndgame",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--skip", type=int, default=0, help="пропустить первые N задач (для отложенной выборки)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=15, help="сколько расхождений каждого вида сохранить")
    args = ap.parse_args()

    stat = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    examples = defaultdict(list)
    done = errors = 0
    with open(args.csv, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < args.skip:
                continue
            if done >= args.n:
                break
            moves = [chess.Move.from_uci(u) for u in row["Moves"].split()]
            try:
                L = Line.build(chess.Board(row["FEN"]), moves[1:], pre=moves[0])
                ours = tags(L)
            except Exception as e:  # битая строка — считаем и идём дальше
                errors += 1
                continue
            theirs = set(row["Themes"].split())
            for t in THEMES:
                a, b = t in ours, t in theirs
                if a and b:
                    stat[t][0] += 1
                elif a:
                    stat[t][1] += 1
                    if len(examples[(t, "fp")]) < args.examples:
                        examples[(t, "fp")].append(row["PuzzleId"])
                elif b:
                    stat[t][2] += 1
                    if len(examples[(t, "fn")]) < args.examples:
                        examples[(t, "fn")].append(row["PuzzleId"])
            done += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"задач: {done}, ошибок разбора: {errors}\n")
    print(f"{'тема':22} {'у lichess':>9} {'у нас':>7} {'точность':>9} {'полнота':>8}")
    for t in THEMES:
        tp, fp, fn = stat[t]
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        rows.append({"theme": t, "lichess": tp + fn, "ours": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec,
                     "fp_examples": examples[(t, "fp")], "fn_examples": examples[(t, "fn")]})
        f = lambda x: "—" if x is None else f"{x:.3f}"
        print(f"{t:22} {tp + fn:9d} {tp + fp:7d} {f(prec):>9} {f(rec):>8}")
    with open(args.out + ".jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
