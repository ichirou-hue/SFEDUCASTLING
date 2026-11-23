"""Шаг 6. Сборка итогового датасета.

Выход (в --out-dir):
  commentary_v2.jsonl   разборы ходов (задача move_commentary, черновик контракта v2)
  tqu_v1.jsonl          атомарные вопросы free_square / piece_on_square строго по контракту v1
  facts_v2.jsonl        все факты и черновик по id — для аудита и Валидатора №1/№2 у команды обучения
  dpo_pairs.jsonl       chosen (прошёл проверки) / rejected (срезан проверкой) — только train
  stats.json            распределения, длины, причины отказов, SHA256 файлов

Правила разделения из контракта 02: сплит по group_id (партия), одинаковая расстановка фигур
не пересекает сплиты (группы с общей расстановкой склеиваются), heldout-доски исключаются.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess

from gcdata.io import read_jsonl, sha256_file, stable_unit, write_jsonl
from gcdata.ru import piece_ru
from gcdata.template import MAX_SEQ, TokenCounter, user_body

V2 = "gigachess-dataset-v2-draft"
V1 = "gigachess-dataset-v1"


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def layout(fen: str) -> str:
    return fen.split()[0]


def load_heldout(path: str | None) -> set:
    """Расстановки из heldout_boards.json — формат файла не фиксируем, ищем FEN-подобные строки."""
    if not path:
        return set()
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.count("/") == 7:
            try:
                out.add(chess.Board(x if " " in x else x + " w - - 0 1").board_fen())
            except ValueError:
                pass

    walk(json.load(open(path, encoding="utf-8")))
    return out


def assign_splits(groups_layouts: dict, ratios: tuple, seed: str) -> dict:
    dsu = DSU()
    first_by_layout = {}
    for g, lays in groups_layouts.items():
        dsu.find(g)
        for lay in lays:
            if lay in first_by_layout:
                dsu.union(g, first_by_layout[lay])
            else:
                first_by_layout[lay] = g
    names = ("train", "validation", "test")
    out = {}
    for g in groups_layouts:
        u = stable_unit(seed, "split", dsu.find(g))
        acc = 0.0
        for name, r in zip(names, ratios):
            acc += r
            if u < acc:
                out[g] = name
                break
        else:
            out[g] = names[-1]
    return out


def tqu_records(d: dict, split: str, k: int, seed: str) -> list[dict]:
    """k вопросов по одной позиции, пустые и занятые поля поровну, задачи чередуются."""
    board = chess.Board(d["fen_before"])
    occ = sorted(chess.SquareSet(board.occupied), key=lambda s: stable_unit(seed, d["id"], "o", str(s)))
    emp = sorted(chess.SquareSet(~board.occupied & chess.BB_ALL), key=lambda s: stable_unit(seed, d["id"], "e", str(s)))
    # Сдвиг по позиции: при одном вопросе на позицию иначе всегда выпадал бы первый вариант
    # (занятое поле + «какая фигура») — модель учила бы, что на спрошенной клетке всегда фигура.
    off = int(stable_unit(seed, d["id"], "tqu-offset") * 4)
    picks = []
    for i in range(k):
        j = i + off
        pool = occ if j % 2 == 0 else emp
        if pool:
            picks.append((j, pool.pop(0)))
    out = []
    for j, sq in picks:
        name = chess.square_name(sq)
        p = board.piece_at(sq)
        task = "piece_on_square" if (j // 2) % 2 == 0 else "free_square"
        if task == "piece_on_square":
            q = f"Какая фигура находится на клетке {name}?"
            a = f"На поле {name} находится {piece_ru(p)}." if p else f"На поле {name} ничего нет."
        else:
            q = f"Поле {name} свободно?"
            a = f"Нет, на поле {name} находится {piece_ru(p)}." if p else f"Да, поле {name} свободно."
        out.append({
            "schema_version": V1, "id": f"{d['id']}_tqu_{name}", "split": split, "group_id": d["group_id"],
            "task": task, "fen": d["fen_before"], "square": name, "question": q, "answer": a,
            "target": {"piece": p.symbol() if p else ".", "is_empty": p is None}, "source": d["source"],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--polished", required=True)
    ap.add_argument("--rejected", help="по умолчанию <polished>.rejected.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heldout", help="evidence/heldout_boards.json из комплекта модели")
    ap.add_argument("--split", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    ap.add_argument("--tqu-share", type=float, default=0.30, help="доля TQU в итоговой смеси (пайплайн: 70/30)")
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--seed", default="gigachess-v2")
    args = ap.parse_args()
    if abs(sum(args.split) - 1) > 1e-6:
        sys.exit("--split должен суммироваться в 1")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tc = TokenCounter(args.tokenizer)
    heldout = load_heldout(args.heldout)
    answers = {r["id"]: r for r in read_jsonl(args.polished)}
    drafts = [d for d in read_jsonl(args.drafts) if (answers.get(d["id"]) or {}).get("answer")]

    drop = Counter()
    kept = []
    seen_pairs = set()
    for d in drafts:
        if layout(d["fen_before"]) in heldout:
            drop["heldout_board"] += 1
            continue
        pair = (layout(d["fen_before"]), d["move_uci"])
        if pair in seen_pairs:
            drop["duplicate_position_move"] += 1  # одна и та же партия у разных аннотаторов
            continue
        seen_pairs.add(pair)
        ans = answers[d["id"]]["answer"]
        n = tc.sequence_length(d["fen_before"], d["question"], ans)
        if n > args.max_seq:
            drop["too_long_for_512"] += 1
            continue
        d["_answer"], d["_seq"] = ans, n
        d["_generator"] = answers[d["id"]].get("model")
        kept.append(d)

    groups = {}
    for d in kept:
        groups.setdefault(d["group_id"], set()).add(layout(d["fen_before"]))
    split_of = assign_splits(groups, tuple(args.split), args.seed)

    comm, facts, tqu = [], [], []
    n_tqu_total = round(len(kept) * args.tqu_share / (1 - args.tqu_share)) if args.tqu_share < 1 else 0
    per_pos = max(1, -(-n_tqu_total // max(1, len(kept))))  # потолок деления
    for d in kept:
        sp = split_of[d["group_id"]]
        eng = d["facts"]["engine"]
        comm.append({
            "schema_version": V2, "id": d["id"], "split": sp, "group_id": d["group_id"],
            "task": "move_commentary", "fen": d["fen_before"], "move_uci": d["move_uci"],
            "question": d["question"], "answer": d["_answer"],
            "target": {"move_uci": d["move_uci"], "fen_after": d["fen_after"], "nag": eng["nag"],
                       "verdict": eng["verdict"], "cpl": eng["cpl"]},
            "source": d["source"],
            # Какая модель связала план в текст: можно учить только на части (например, только Claude).
            "generator": d["_generator"],
        })
        facts.append({"id": d["id"], "draft": d["draft"], "intent_used": d["intent_used"],
                      "intent": d["intent"], "seq_len": d["_seq"], "seq_len_exact": tc.exact, "facts": d["facts"]})
        if len(tqu) < n_tqu_total:
            tqu.extend(tqu_records(d, sp, min(per_pos, n_tqu_total - len(tqu)), args.seed))

    # DPO: только train, чтобы validation/test не протекли в выравнивание.
    rej_path = args.rejected or str(Path(args.polished).with_suffix("")) + ".rejected.jsonl"
    by_id = {c["id"]: c for c in comm if c["split"] == "train"}
    dpo, rej_reasons = [], Counter()
    if Path(rej_path).exists():
        for r in read_jsonl(rej_path):
            for reason in r["reasons"]:
                rej_reasons[reason.split(":")[0]] += 1
            c = by_id.get(r["id"])
            if c:
                dpo.append({"id": f"{r['id']}_dpo{r['attempt']}", "fen": c["fen"],
                            "prompt": user_body(c["fen"], c["question"]),
                            "chosen": c["answer"], "rejected": r["text"], "rejected_reasons": r["reasons"]})

    files = {
        "commentary_v2.jsonl": comm, "tqu_v1.jsonl": tqu, "facts_v2.jsonl": facts, "dpo_pairs.jsonl": dpo,
    }
    for name, rows in files.items():
        write_jsonl(out / name, rows)

    seqs = [f["seq_len"] for f in facts]
    stats = {
        "records": {k: len(v) for k, v in files.items()},
        "commentary_by_split": Counter(c["split"] for c in comm),
        "groups_by_split": Counter(split_of[g] for g in groups),
        "tqu_by_split": Counter(t["split"] for t in tqu),
        "tqu_tasks": Counter(t["task"] for t in tqu),
        "tqu_empty": Counter(t["target"]["is_empty"] for t in tqu),
        "nag": Counter(str(c["target"]["nag"]) for c in comm),
        "verdict": Counter(c["target"]["verdict"] for c in comm),
        "intent_in_input_share": round(sum(f["intent_used"] for f in facts) / max(1, len(facts)), 3),
        "answer_chars_mean": round(sum(len(c["answer"]) for c in comm) / max(1, len(comm)), 1),
        "seq_len": {"max": max(seqs, default=0), "mean": round(sum(seqs) / max(1, len(seqs)), 1),
                    "exact_tokenizer": tc.exact},
        "dropped": drop,
        "llm_rejection_reasons": rej_reasons.most_common(),
        "heldout_boards_loaded": len(heldout),
        "sha256": {name: sha256_file(out / name) for name in files},
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("records", "commentary_by_split", "nag", "dropped", "seq_len")},
                     ensure_ascii=False, indent=1))
    if not tc.exact:
        print("ВНИМАНИЕ: длина оценена по символам; для приёмки запустите с --tokenizer tokenizer.json модели")


if __name__ == "__main__":
    main()
