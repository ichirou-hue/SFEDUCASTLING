"""Делит ещё не сгенерированные планы на простые (их связывает локальная T-lite) и сложные
(тактика, несколько причин — их оставляем на Claude: на них T-lite путает смысл).

  python tools/split_for_local.py --drafts work_ds/desc3000_drafts.jsonl --done work_ds/desc3000_polished.jsonl \\
      --out-simple work_tlite/simple_drafts.jsonl --out-complex work_tlite/complex_drafts.jsonl
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.io import read_jsonl, write_jsonl  # noqa: E402

TACTIC = {
    "вилка", "связка", "сквозной удар", "вскрытое нападение", "вскрытый шах", "двойной шах", "отвлечение",
    "завлечение", "рентген", "уничтожение защиты", "перекрытие", "освобождение линии", "промежуточный ход",
    "ловля фигуры", "тихий ход", "жертва", "мат по последней горизонтали", "спёртый мат", "мат Анастасии",
    "арабский мат", "мат «крюк»", "мат Бодена", "мат двумя слонами", "мат «ласточкин хвост»",
}


def is_complex(draft: str) -> bool:
    terms = set(re.findall(r"\[([^\]]+)\]", draft))
    return bool(terms & TACTIC) or draft.count("причина:") >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--out-simple", required=True)
    ap.add_argument("--out-complex", required=True)
    args = ap.parse_args()
    done = {r["id"] for r in read_jsonl(args.done)}
    rest = [d for d in read_jsonl(args.drafts) if d["id"] not in done]
    simple = [d for d in rest if not is_complex(d["draft"])]
    complex_ = [d for d in rest if is_complex(d["draft"])]
    Path(args.out_simple).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_simple, simple)
    write_jsonl(args.out_complex, complex_)
    print(f"осталось {len(rest)}: простых {len(simple)} (T-lite), сложных {len(complex_)} (на Claude)")


if __name__ == "__main__":
    main()
