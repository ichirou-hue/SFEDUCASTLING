"""
Dataset Builder — полный пайплайн 021→026.

Этапы:
  021 — скачивание партий (Lichess API)                    → raw/
  022 — PGN → FEN конвертация                              → parsed/
  024 — Stockfish labelling (eval_cp, best_move, pv)      → labeled/stockfish/
  025 — NAG labels (! !? ? ??)                             → labeled/nag/
  026 — финальный датасет                                  → final/
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Конфиг по умолчанию ──────────────────────────────────────────────────────
DEFAULT_USERNAME = "cutemouse83"
DEFAULT_TOTAL_GAMES = 10000
DEFAULT_PERF_TYPE = "blitz"
DEFAULT_STOCKFISH_DEPTH = 12
DEFAULT_STOCKFISH_THREADS = 2
DEFAULT_STOCKFISH_HASH = 256
DEFAULT_STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")


# ── Утилиты ──────────────────────────────────────────────────────────────────
def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """Запускает команду, логирует stdout/stderr."""
    logging.info("▶ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.stdout:
        logging.info(result.stdout.strip())
    if result.stderr:
        logging.warning(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Команда упала с кодом {result.returncode}: {' '.join(cmd)}")


# ── 021: Скачивание партий ───────────────────────────────────────────────────
def step_021_download(
    username: str,
    total_games: int,
    perf_type: str,
    raw_dir: Path,
    max_per_req: int = 500,
) -> Path:
    from training_data.collectors.lichess_api import LichessClient

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"{username}_{perf_type}.jsonl"

    if raw_file.exists() and raw_file.stat().st_size > 0:
        with open(raw_file, "r") as f:
            existing = sum(1 for _ in f if _.strip())
        if existing >= total_games:
            logging.info("021: уже есть %d игр в %s, скипаем", existing, raw_file)
            return raw_file

    token = os.getenv("LICHESS_API_TOKEN")
    client = LichessClient(token=token)

    all_games = []
    seen = set()
    until = None

    logging.info("021: качаю %d %s-игр для %s...", total_games, perf_type, username)

    while len(all_games) < total_games:
        need = min(max_per_req, total_games - len(all_games))
        batch = client.fetch_games(username, max_games=need, perf_type=perf_type, until=until)

        if not batch:
            break

        new = [g for g in batch if g.get("id") and g["id"] not in seen]
        for g in new:
            seen.add(g["id"])
        all_games.extend(new)
        logging.info("021: накоплено %d игр", len(all_games))

        if not new:
            break

        until = new[-1].get("lastMoveAt") or new[-1].get("createdAt")
        if not until or until <= 0:
            break

        time.sleep(1.0)

    all_games = all_games[:total_games]
    client.save_jsonl(all_games, raw_file)
    logging.info("021: готово %d игр → %s", len(all_games), raw_file)
    return raw_file


# ── 022: PGN → FEN ───────────────────────────────────────────────────────────
def step_022_pgn_to_fen(raw_file: Path, parsed_dir: Path) -> Path:
    from training_data.parsers.pgn_to_fen import convert_jsonl

    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_file = parsed_dir / f"{raw_file.stem}.parquet"

    if parsed_file.exists() and parsed_file.stat().st_size > 0:
        df = pd.read_parquet(parsed_file)
        if len(df) > 0:
            logging.info("022: уже есть %d позиций в %s, скипаем", len(df), parsed_file)
            return parsed_file

    logging.info("022: конвертирую %s → %s...", raw_file, parsed_file)
    convert_jsonl(raw_file, parsed_file)

    df = pd.read_parquet(parsed_file)
    logging.info("022: готово %d позиций → %s", len(df), parsed_file)
    return parsed_file


# ── 024: Stockfish labelling ─────────────────────────────────────────────────
def step_024_stockfish(
    input_file: Path,
    stockfish_dir: Path,
    depth: int,
    threads: int,
    hash_mb: int,
    stockfish_path: str,
    limit: Optional[int] = None,
) -> Path:
    stockfish_dir.mkdir(parents=True, exist_ok=True)
    output_file = stockfish_dir / f"{input_file.stem}_labeled.parquet"

    if output_file.exists() and output_file.stat().st_size > 0:
        df = pd.read_parquet(output_file)
        if "eval_cp" in df.columns and len(df) > 0:
            logging.info("024: уже есть %d позиций в %s, скипаем", len(df), output_file)
            return output_file

    logging.info("024: запускаю Stockfish (depth=%d, threads=%d, hash=%dMB)...",
                 depth, threads, hash_mb)

    env = os.environ.copy()
    env["STOCKFISH_PATH"] = stockfish_path
    env["STOCKFISH_THREADS"] = str(threads)
    env["STOCKFISH_HASH"] = str(hash_mb)

    cmd = [
        sys.executable, "-m", "training_data.labeler.stockfish_labeler",
        "--input", str(input_file),
        "--output", str(output_file),
        "--depth", str(depth),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    run_cmd(cmd, env=env)
    logging.info("024: готово → %s", output_file)
    return output_file


# ── 025: NAG labels ──────────────────────────────────────────────────────────
def step_025_nag(
    input_file: Path,
    nag_dir: Path,
    limit: Optional[int] = None,
) -> Path:
    nag_dir.mkdir(parents=True, exist_ok=True)
    output_file = nag_dir / f"{input_file.stem}_nag.parquet"

    if output_file.exists() and output_file.stat().st_size > 0:
        df = pd.read_parquet(output_file)
        if "nag" in df.columns and len(df) > 0:
            logging.info("025: уже есть %d позиций в %s, скипаем", len(df), output_file)
            return output_file

    logging.info("025: добавляю NAG метки...")

    cmd = [
        sys.executable, "-m", "training_data.labeler.nag_labeler",
        "--input", str(input_file),
        "--output", str(output_file),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    run_cmd(cmd)
    logging.info("025: готово → %s", output_file)
    return output_file


# ── 026: Финальный датасет ───────────────────────────────────────────────────
def step_026_final(
    input_file: Path,
    final_dir: Path,
) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    output_file = final_dir / f"{input_file.stem}_final.parquet"

    if output_file.exists() and output_file.stat().st_size > 0:
        df = pd.read_parquet(output_file)
        if len(df) > 0:
            logging.info("026: уже есть %d позиций в %s, скипаем", len(df), output_file)
            return output_file

    logging.info("026: собираю финальный датасет → %s...", output_file)
    df = pd.read_parquet(input_file)
    df.to_parquet(output_file, index=False)
    logging.info("026: финальный датасет %d строк → %s", len(df), output_file)
    return output_file


# ── Главный пайплайн ─────────────────────────────────────────────────────────
def build_dataset(
    username: str = DEFAULT_USERNAME,
    total_games: int = DEFAULT_TOTAL_GAMES,
    perf_type: str = DEFAULT_PERF_TYPE,
    raw_dir: str = "training_data/raw",
    parsed_dir: str = "training_data/parsed",
    stockfish_dir: str = "training_data/labeled/stockfish",
    nag_dir: str = "training_data/labeled/nag",
    final_dir: str = "training_data/final",
    stockfish_depth: int = DEFAULT_STOCKFISH_DEPTH,
    stockfish_threads: int = DEFAULT_STOCKFISH_THREADS,
    stockfish_hash: int = DEFAULT_STOCKFISH_HASH,
    stockfish_path: str = DEFAULT_STOCKFISH_PATH,
    limit: Optional[int] = None,
    skip_download: bool = False,
    skip_pgn: bool = False,
    skip_stockfish: bool = False,
    skip_nag: bool = False,
) -> Path:
    """
    Полный пайплайн 021→026.

    Returns:
        Путь к финальному .parquet файлу.
    """
    raw_dir = Path(raw_dir)
    parsed_dir = Path(parsed_dir)
    stockfish_dir = Path(stockfish_dir)
    nag_dir = Path(nag_dir)
    final_dir = Path(final_dir)

    logging.info("=== НАЧАЛО ПАЙПЛАЙНА: %s, %d игр, %s ===",
                 username, total_games, perf_type)

    # 021
    if skip_download:
        raw_file = raw_dir / f"{username}_{perf_type}.jsonl"
        logging.info("021: скипаем скачивание, используем %s", raw_file)
    else:
        raw_file = step_021_download(username, total_games, perf_type, raw_dir)

    # 022
    if skip_pgn:
        parsed_file = parsed_dir / f"{username}_{perf_type}.parquet"
        logging.info("022: скипаем конвертацию, используем %s", parsed_file)
    else:
        parsed_file = step_022_pgn_to_fen(raw_file, parsed_dir)

    # 024
    if skip_stockfish:
        stockfish_file = stockfish_dir / f"{username}_{perf_type}_labeled.parquet"
        logging.info("024: скипаем Stockfish, используем %s", stockfish_file)
    else:
        stockfish_file = step_024_stockfish(
            parsed_file, stockfish_dir, stockfish_depth, stockfish_threads,
            stockfish_hash, stockfish_path, limit
        )

    # 025
    if skip_nag:
        nag_file = nag_dir / f"{username}_{perf_type}_labeled_nag.parquet"
        logging.info("025: скипаем NAG, используем %s", nag_file)
    else:
        nag_file = step_025_nag(stockfish_file, nag_dir, limit)

    # 026
    final_file = step_026_final(nag_file, final_dir)

    logging.info("=== ПАЙПЛАЙН ЗАВЕРШЁН: %s ===", final_file)
    return final_file


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        description="Dataset Builder — полный пайплайн 021→026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Игрок на Lichess")
    parser.add_argument("--total-games", type=int, default=DEFAULT_TOTAL_GAMES, help="Сколько игр качать")
    parser.add_argument("--perf-type", default=DEFAULT_PERF_TYPE, help="blitz/rapid/bullet")
    parser.add_argument("--raw-dir", default="training_data/raw")
    parser.add_argument("--parsed-dir", default="training_data/parsed")
    parser.add_argument("--stockfish-dir", default="training_data/labeled/stockfish")
    parser.add_argument("--nag-dir", default="training_data/labeled/nag")
    parser.add_argument("--final-dir", default="training_data/final")
    parser.add_argument("--stockfish-depth", type=int, default=DEFAULT_STOCKFISH_DEPTH)
    parser.add_argument("--stockfish-threads", type=int, default=DEFAULT_STOCKFISH_THREADS)
    parser.add_argument("--stockfish-hash", type=int, default=DEFAULT_STOCKFISH_HASH)
    parser.add_argument("--stockfish-path", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Ограничить позиций (для теста)")
    parser.add_argument("--skip-download", action="store_true", help="Скип 021")
    parser.add_argument("--skip-pgn", action="store_true", help="Скип 022")
    parser.add_argument("--skip-stockfish", action="store_true", help="Скип 024")
    parser.add_argument("--skip-nag", action="store_true", help="Скип 025")
    args = parser.parse_args()

    build_dataset(
        username=args.username,
        total_games=args.total_games,
        perf_type=args.perf_type,
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
        stockfish_dir=args.stockfish_dir,
        nag_dir=args.nag_dir,
        final_dir=args.final_dir,
        stockfish_depth=args.stockfish_depth,
        stockfish_threads=args.stockfish_threads,
        stockfish_hash=args.stockfish_hash,
        stockfish_path=args.stockfish_path,
        limit=args.limit,
        skip_download=args.skip_download,
        skip_pgn=args.skip_pgn,
        skip_stockfish=args.skip_stockfish,
        skip_nag=args.skip_nag,
    )


if __name__ == "__main__":
    main()