"""Шаг 1. Сырые партии -> отдельные ходы с позицией ДО и человеческим комментарием.

Источники (можно смешивать):
  --pgn        обычные .pgn (данные коллеги, выгрузки lichess)
  --chessgpt   JSONL ChessGPT annotated_pgn: {"text": "<PGN>", "pipeline_key": ...}
  --moves      готовые ходы JSONL: {"fen_before", "move" (uci или san), "game_id",
               "comment"? | "human_intent"?} — если у коллеги уже позиции, а не партии

Берём только главную линию: комментарии внутри вариантов (...) описывают не сыгранный ход.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import chess.pgn

from gcdata.facts import fen
from gcdata.io import read_jsonl, stable_hash, stable_unit, write_jsonl

STANDARD = {"", "standard", "chess", "from position"}


def iter_pgn_texts(args):
    for p in args.pgn or []:
        with open(p, encoding="utf-8", errors="replace") as f:
            n = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                n += 1
                yield game, f"{Path(p).name}#{n}"
    for p in args.chessgpt or []:
        for i, row in enumerate(read_jsonl(p)):
            game = chess.pgn.read_game(io.StringIO(row["text"]))
            if game is not None:
                yield game, f"chessgpt:{row.get('pipeline_key') or Path(p).name + '#' + str(i)}"


def game_moves(game, source: str, args) -> list[dict]:
    h = game.headers
    if h.get("Variant", "").lower() not in STANDARD:
        return []
    if game.errors:  # python-chess собирает нелегальные ходы сюда, а не бросает исключение
        return []
    group_id = "g_" + stable_hash(source)[:16]
    rows = []
    board = game.board()
    prev_move, fen_prev = None, None
    for ply, node in enumerate(game.mainline(), 1):
        move = node.move
        if ply >= args.min_ply:
            rows.append({
                "id": f"{group_id}_{ply:03d}",
                "group_id": group_id,
                "source": source,
                "site": h.get("Site", ""),
                "ply": ply,
                "fen_before": fen(board),
                "move_uci": move.uci(),
                "move_san": board.san(move),
                "comment_raw": node.comment.strip(),
                "human_nags": sorted(node.nags),
                "intent_given": None,
                # Предыдущий ход соперника: нужен ключам «висящая фигура», «промежуточный ход» (gcdata/keys.py).
                "prev_move_uci": prev_move,
                "fen_prev": fen_prev,
            })
        fen_prev, prev_move = fen(board), move.uci()
        board.push(move)
    return select(rows, args)


def select(rows: list[dict], args) -> list[dict]:
    """Все откомментированные ходы (до лимита) + немного ходов без комментария."""
    key = lambda r: stable_unit(args.seed, r["id"])  # noqa: E731
    commented = sorted([r for r in rows if r["comment_raw"]], key=key)[: args.max_commented]
    plain = sorted([r for r in rows if not r["comment_raw"]], key=key)[: args.max_plain]
    return sorted(commented + plain, key=lambda r: r["ply"])


def iter_ready_moves(paths, args):
    for p in paths or []:
        for i, row in enumerate(read_jsonl(p)):
            board = chess.Board(row["fen_before"])
            mv = row["move"]
            try:
                move = chess.Move.from_uci(mv)
                if move not in board.legal_moves:
                    raise ValueError
            except ValueError:
                move = board.parse_san(mv)
            gid = "g_" + stable_hash(str(row.get("game_id") or row["fen_before"]))[:16]
            yield {
                "id": f"{gid}_{stable_hash(row['fen_before'], move.uci())[:8]}",
                "group_id": gid,
                "source": row.get("source") or f"{Path(p).name}#{i}",
                "site": row.get("site", ""),
                "ply": row.get("ply"),
                "fen_before": fen(board),
                "move_uci": move.uci(),
                "move_san": board.san(move),
                "comment_raw": (row.get("comment") or "").strip(),
                "human_nags": row.get("nags", []),
                # Явный замысел от разметчика: проходит те же проверки, но без LLM-извлечения.
                "intent_given": row.get("human_intent"),
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", nargs="*")
    ap.add_argument("--chessgpt", nargs="*")
    ap.add_argument("--moves", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ply", type=int, default=7, help="пропустить первые полуходы (дебютная теория)")
    ap.add_argument("--max-commented", type=int, default=6, help="макс. откомментированных ходов на партию")
    ap.add_argument("--max-plain", type=int, default=1, help="макс. ходов без комментария на партию")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--seed", default="gigachess-v2")
    args = ap.parse_args()

    out, games, skipped = [], 0, 0
    for game, source in iter_pgn_texts(args):
        rows = game_moves(game, source, args)
        games += 1
        skipped += not rows
        out.extend(rows)
        if args.max_games and games >= args.max_games:
            break
    out.extend(iter_ready_moves(args.moves, args))
    n = write_jsonl(args.out, out)
    print(f"партий: {games}, без пригодных ходов: {skipped}, ходов записано: {n} -> {args.out}")


if __name__ == "__main__":
    main()
