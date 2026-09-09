"""Пересборка backend/knowledge/puzzles.json.

Источник: официальная база Lichess (lichess_db_puzzle.csv.zst).

ВАЖНО. Формат официальной базы (см. database.lichess.org/#puzzles):
  - FEN — это позиция ПЕРЕД ходом соперника;
  - позиция, которую показывают игроку, получается применением ПЕРВОГО хода
    из Moves;
  - настоящая задача для игрока начинается со ВТОРОГО хода (он играет цве-
    том, ПРОТИВОПОЛОЖНЫМ стороне хода в FEN).

Все прежние варианты генератора считали FEN «позицией игрока» и брали
Moves[0] за первый ответ игрока — из-за этого каждая задача оказывалась
«за проигравшего» (игрок должен был повторять ход соперника, закрывавший
выигрыш). Здесь это исправлено:

  - presented_fen = FEN после Moves[0] (ход соперника), ход у игрока;
  - moves         = Moves[1:] (полное решение игрока);
  - игрок в presented_fen — это цвет, ПРОТИВОПОЛОЖНЫЙ стороне хода в FEN.

Отбираем ТОЛЬКО выигрывающие тактические задачи с чётким «уроком».

СЕМАНТИКА ОЦЕНОК python-stockfish:
get_evaluation() возвращает оценку с точки зрения СТОРОНЫ, У КОТОРОЙ ХОД.
После хода игрока ходит соперник, поэтому перевес игрока после первого
хода = НЕГАЦИЯ сырого значения. В финале (ход у любой стороны) переводим
в перспективу игрока по b.turn.

КРИТЕРИИ ПРИГОДНОСТИ:
  1. игрок в финале не матован и не под шахом;
  2. после решения игрок ставит мат ИЛИ имеет чёткий перевес
     (win_score >= WIN_CP с точки зрения игрока);
  3. ПЕРВЫЙ ход решения сразу даёт перевес не меньше FIRST_CP — отсекает
     «защитные» и позиционные задачи без немедленного урока;
  4. тема не «defensiveMove»;
  5. в корзинах предпочитаем маты, вилки, связки, жертвы и т.п.
"""

import csv
import io
import json
import os
import random
import time
import zstandard as zstd

import chess

SF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stockfish_engine", "stockfish.exe"))
SOURCE = r"C:\Users\Egor\AppData\Local\Temp\opencode\lichess_db_puzzle.csv.zst"
OUT_PATH = os.path.join(os.path.dirname(__file__), "puzzles.json")

PER_BUCKET = 12
MAX_PER_THEME = 3
CANDIDATES_PER_BUCKET = 150
BUCKETS = [
    (500, 900),
    (900, 1300),
    (1300, 1700),
    (1700, 3000),
]

WIN_CP = 250
FIRST_CP = 150
DEPTH = 16

LESSON_THEMES = {
    "mateIn1", "mateIn2", "mateIn3", "mateIn5", "mate",
    "fork", "pin", "skewer", "discoveredAttack", "discoveredCheck",
    "sacrifice", "attraction", "deflection", "intermezzo",
    "trappedPiece", "capturingDefender", "hangingPiece", "backRankMate",
    "doubleCheck", "smotheredMate", "promotion",
}


def _decode(fen: str, moves_str: str) -> dict | None:
    """Раскодирует запись официальной базы в настоящую задачу.

    Возвращает {presented, solver, player_moves, full_final} либо None.
    solver = True (chess.WHITE), если у игрока белые.
    """
    try:
        raw = chess.Board(fen)
        full = [u for u in moves_str.split() if u]
        if not full:
            return None
        if len(full) == 1:
            # аномалия: без хода соперника — игрок = сторона хода в FEN
            presented = raw.fen()
            solver = raw.turn
            player_moves = full
        else:
            raw.push_uci(full[0])
            presented = raw.fen()
            # сторона, у которой ход, — это и есть игрок (соперник уже сходил)
            solver = raw.turn
            player_moves = full[1:]
        if not player_moves:
            return None
        b = chess.Board(presented)
        for u in player_moves:
            b.push_uci(u)
        return {
            "presented": presented,
            "solver": solver,
            "player_moves": player_moves,
            "final_board": b,
            "themes": set(),
        }
    except Exception:
        return None


def _cheap_valid(puzzle: dict) -> tuple[bool, bool]:
    """Быстрая проверка без инжектора: ходы легальны, игрок не матован."""
    d = _decode(puzzle["fen"], puzzle["moves"])
    if d is None:
        return False, False
    b = d["final_board"]
    solver = d["solver"]
    if b.is_checkmate() and b.turn == solver:
        return False, False
    if b.is_check() and b.turn == solver:
        return False, False
    return True, (b.is_checkmate() and b.turn != solver)


def _solve_eval(sf, fen: str) -> int:
    """Сырое значение (перспектива стороны, у которой ход)."""
    sf.set_fen_position(fen)
    e = sf.get_evaluation()
    return e["value"] if e["type"] == "cp" else (100000 if e["value"] > 0 else -100000)


def _validate(puzzle: dict, sf) -> dict | None:
    """Полная проверка задачи. Возвращает метрики или None."""
    d = _decode(puzzle["fen"], puzzle["moves"])
    if d is None:
        return None
    solver = d["solver"]
    b = d["final_board"]
    final = b.fen()

    if b.is_checkmate() and b.turn == solver:
        return None
    if b.is_check() and b.turn == solver:
        return None

    # перевес игрока в финале (матаем → безусловно)
    if b.is_checkmate() and b.turn != solver:
        end_solver = 100000
    else:
        raw_end = _solve_eval(sf, final)
        end_solver = raw_end if b.turn == solver else -raw_end

    # перевес игрока сразу после первого хода решения
    b2 = chess.Board(d["presented"])
    b2.push_uci(d["player_moves"][0])
    raw_first = _solve_eval(sf, b2.fen())
    first_adv = (-raw_first if b2.turn != solver else raw_first)

    if end_solver < WIN_CP:
        return None
    if first_adv < FIRST_CP:
        return None

    themes = set(puzzle.get("themes", []))
    if "defensiveMove" in themes:
        return None

    return {
        "end_solver": end_solver,
        "first_adv": first_adv,
        "solver_mates": end_solver >= 100000,
    }


def _primary_theme(puzzle: dict) -> str:
    themes = set(puzzle.get("themes", []))
    for t in ("mateIn1", "mateIn2", "mateIn3", "mate", "fork", "skewer", "pin",
              "sacrifice", "discoveredAttack", "discoveredCheck", "deflection",
              "attraction", "capturingDefender", "hangingPiece"):
        if t in themes:
            return t
    return "other"


def main() -> None:
    from stockfish import Stockfish

    random.seed(42)
    sf = Stockfish(path=SF_PATH, depth=DEPTH, parameters={"Threads": 1, "Hash": 64})

    buckets = {b: [] for b in BUCKETS}
    t0 = time.time()
    with open(SOURCE, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as reader:
        reader = io.TextIOWrapper(reader, encoding="utf-8")
        for row in csv.DictReader(reader):
            fen = (row.get("FEN") or "").strip()
            moves = (row.get("Moves") or "").strip()
            try:
                rating = int(float((row.get("Rating") or "").strip()))
            except ValueError:
                continue
            if not fen or not moves:
                continue
            bucket = None
            for (lo, hi) in BUCKETS:
                if lo <= rating < hi:
                    bucket = (lo, hi)
                    break
            if bucket is None or len(buckets[bucket]) >= CANDIDATES_PER_BUCKET:
                continue
            puzzle = {
                "id": (row.get("PuzzleId") or "").strip(),
                "fen": fen,
                "moves": moves,
                "rating": rating,
                "themes": (row.get("Themes") or "").split(),
            }
            ok, _ = _cheap_valid(puzzle)
            if ok:
                buckets[bucket].append(puzzle)

            total = sum(len(v) for v in buckets.values())
            if total % 500 == 0:
                print(f"  ... {row['PuzzleId']}, предкандидатов: {total}", flush=True)

    print(f"Собрано предкандидатов: {sum(len(v) for v in buckets.values())} за {time.time()-t0:.0f}s")

    selected: list[dict] = []
    for (lo, hi), pool in buckets.items():
        scored = []
        for i, p in enumerate(pool, 1):
            if i % 25 == 0:
                print(f"  [{lo}-{hi}] проверено {i}/{len(pool)}...", flush=True)
            try:
                metrics = _validate(p, sf)
            except Exception:
                continue
            if metrics is None:
                continue
            theme = "mate" if metrics["solver_mates"] else _primary_theme(p)
            scored.append((p, metrics, theme))

        scored.sort(key=lambda item: (
            0 if item[2] in LESSON_THEMES or item[1]["solver_mates"] else 1,
            -item[1]["first_adv"],
        ))

        # Разнообразие: ротация по темам, максимум MAX_PER_THEME на тему.
        thanks_groups: dict[str, list] = {}
        for p, m, theme in scored:
            thanks_groups.setdefault(theme, []).append((p, m, theme))
        for g in thanks_groups.values():
            g.sort(key=lambda item: -item[1]["first_adv"])

        cycle = list(thanks_groups.keys())
        cycle.sort(key=lambda t: 0 if t in LESSON_THEMES else 1)
        picked: list = []
        while len(picked) < PER_BUCKET and cycle:
            progressed = False
            for t in list(cycle):
                if len(picked) >= PER_BUCKET:
                    break
                g = thanks_groups[t]
                if len(g) == 0:
                    cycle.remove(t)
                    continue
                if sum(1 for (pp, mm, tt) in picked if tt == t) >= MAX_PER_THEME:
                    continue
                picked.append(g.pop(0))
                progressed = True
            if not progressed:
                break

        for p, m, theme in picked:
            d = _decode(p["fen"], p["moves"])
            out = {
                "id": p["id"],
                "fen": d["presented"],
                "moves": " ".join(d["player_moves"]),
                "rating": p["rating"],
                "themes": p["themes"],
                "win_score": m["end_solver"],
                "first_adv": m["first_adv"],
                "solver_mates": m["solver_mates"],
                "primary_theme": theme,
            }
            selected.append(out)
        print(f"  корзина {lo}-{hi}: валидных={len(scored)} взято={len(picked)}")

    selected.sort(key=lambda p: p["rating"])

    payload = {
        "source": "Lichess Puzzles (official database.lichess.org)",
        "count": len(selected),
        "puzzles": selected,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"[Build] Записано задач: {len(selected)}")
    for (lo, hi) in BUCKETS:
        n = sum(1 for p in selected if lo <= p["rating"] < hi)
        print(f"[Build]   {lo}-{hi}: {n}")
    print(f"[Build] Файл: {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB)")
    print(f"[Build] Итого: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()