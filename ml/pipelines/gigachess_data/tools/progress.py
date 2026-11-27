"""Живой счётчик генерации: сколько примеров готово, темп, сколько осталось. Ctrl-C — выход.

  python tools/progress.py --drafts work_ds/desc3000_drafts.jsonl --polished work_ds/desc3000_polished.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path


def count(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, ""
    n, model = 0, ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("answer"):
                n += 1
                model = r.get("model", model)
    return n, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--polished", required=True)
    ap.add_argument("--every", type=float, default=10)
    args = ap.parse_args()
    total = sum(1 for _ in open(args.drafts, encoding="utf-8"))
    hist = deque(maxlen=30)  # окно для темпа: последние ~5 минут
    while True:
        n, model = count(Path(args.polished))
        now = time.time()
        hist.append((now, n))
        rate = 0.0
        if len(hist) > 1 and hist[-1][0] > hist[0][0]:
            rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0]) * 60
        left = total - n
        eta = f"~{left / rate:.0f} мин" if rate > 0 else "—"
        bar = "#" * int(30 * n / total) + "." * (30 - int(30 * n / total))
        print(f"{time.strftime('%H:%M:%S')}  [{bar}]  {n} / {total} ({n / total:.0%})  "
              f"темп {rate:.0f}/мин  осталось {eta}  модель: {model}", flush=True)
        if n >= total:
            print("Связывание закончено. Дальше идут перепроверка движком, судья и выгрузка.")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
