"""Шаг 3. Обогащение контекстом: python-chess (ДО / дельта / ПОСЛЕ) + Stockfish + NAG.

Каждый воркер держит свой процесс Stockfish. Прогон можно прерывать: уже готовые id пропускаются.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.engine import Analyzer
from gcdata.facts import all_facts
from gcdata.io import done_ids, read_jsonl, write_jsonl

_an = None


def _init(engine, depth, movetime, threads, hash_mb):
    global _an
    _an = Analyzer(engine, depth=depth, movetime=movetime, threads=threads, hash_mb=hash_mb)


def _work(rec):
    try:
        f = all_facts(rec["fen_before"], rec["move_uci"])
        f["engine"] = _an.analyse_move(rec["fen_before"], rec["move_uci"])
        rec["facts"] = f
        return rec, None
    except Exception:  # одна битая позиция не должна ронять прогон
        return rec, traceback.format_exc(limit=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True, help="путь к бинарнику Stockfish 16+")
    ap.add_argument("--depth", type=int, default=22)  # на 18 вердикт менялся в 14% позиций при перепроверке на 24
    ap.add_argument("--movetime", type=float, help="секунд на позицию вместо фиксированной глубины")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="потоков Stockfish на воркер")
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--skip-intent", nargs="*", default=[],
                    help="статусы замысла, ходы с которыми не анализировать (по умолчанию берём все)")
    args = ap.parse_args()

    seen = done_ids(args.out)
    todo = [r for r in read_jsonl(args.inp)
            if r["id"] not in seen and (r.get("intent") or {}).get("status") not in set(args.skip_intent)]
    print(f"к анализу: {len(todo)} (уже готово: {len(seen)})")
    # Проверяем движок до пула: упавший initializer Pool пересоздаёт бесконечно.
    try:
        Analyzer(args.engine, depth=1).close()
    except Exception as e:
        sys.exit(f"не удалось запустить движок {args.engine}: {e!r}")
    err_path = Path(args.out).with_suffix(".errors.log")
    ok = bad = 0
    with Pool(args.workers, initializer=_init,
              initargs=(args.engine, args.depth, args.movetime, args.threads, args.hash)) as pool:
        for rec, err in pool.imap_unordered(_work, todo, chunksize=4):
            if err:
                bad += 1
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(f"{rec['id']}\n{err}\n")
                continue
            write_jsonl(args.out, [rec], append=True)
            ok += 1
            if ok % 500 == 0:
                print(f"  {ok} готово")
    print(f"готово: {ok}, ошибок: {bad}" + (f" (см. {err_path})" if bad else ""))


if __name__ == "__main__":
    main()
