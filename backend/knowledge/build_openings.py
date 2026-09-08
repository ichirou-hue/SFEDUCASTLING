"""Генератор backend/knowledge/openings.json из открытой базы lichess-org/chess-openings.

Формат источника: a.tsv .. e.tsv, строки "ECO<TAB>Название<TAB>PGN".
Каждая строка парсится через python-chess: PGN -> ходы (UCI) и промежуточные FEN.
Итоговый файл хранит только раскладку фигур (без счётчиков ходов) в поле fens,
поэтому при совпадении позиций не важны рокировочные права и счётчик ходов.
"""

import io
import json
import os
import time
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
LETTERS = ["a", "b", "c", "d", "e"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "openings.json")


def pieces(fen: str) -> str:
    return fen.split()[0]


def fetch_tsv(letter: str) -> list[str]:
    url = f"{BASE_URL}/{letter}.tsv"
    print(f"[Build] Скачивание {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "opencode"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace").splitlines()


def parse_pgn(pgn: str):
    import chess
    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn))
    board = chess.Board()
    moves: list[str] = []
    fens: list[str] = [pieces(board.fen())]
    for move in game.mainline_moves():
        board.push(move)
        moves.append(move.uci())
        fens.append(pieces(board.fen()))
    return moves, fens


def main() -> None:
    rows = 0
    skipped = 0
    openings: list[dict] = []
    seen: set[tuple] = set()

    for letter in LETTERS:
        for line in fetch_tsv(letter):
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            eco, name, pgn = (p.strip() for p in parts)
            if not eco or not name or not pgn:
                continue
            if not eco[:1].lower() == letter:
                continue
            rows += 1
            key = (eco, name, pgn)
            if key in seen:
                continue
            seen.add(key)
            try:
                moves, fens = parse_pgn(pgn)
            except Exception:
                skipped += 1
                continue
            if not moves:
                skipped += 1
                continue
            openings.append(
                {
                    "eco": eco,
                    "name": name,
                    "pgn": pgn,
                    "moves": moves,
                    "fens": fens,
                    "fen": fens[-1],
                }
            )
    openings.sort(key=lambda o: (o["eco"], len(o["moves"])))

    payload = {"source": "lichess-org/chess-openings (A00-E99)", "openings": openings}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[Build] Прочитано строк: {rows}, пропущено некорректных: {skipped}")
    print(f"[Build] Записано дебютов: {len(openings)}")
    print(f"[Build] Файл: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[Build] Готово за {time.time() - t0:.1f}s")