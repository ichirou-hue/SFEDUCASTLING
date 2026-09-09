import json
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.schemas import RawPosition
from src.chess_engine import StockfishAnalyzer

def run_analysis(input_path: str, output_path: str, stockfish_path: str, depth: int = 22, limit: int = None):
    in_file = Path(input_path)
    if not in_file.exists():
        logger.error(f"Входной файл '{input_path}' не найден!")
        return

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Проверяем, сколько позиций уже обработано (для возобновления без пересчета)
    processed_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(record.get("id"))
                    except Exception:
                        pass

    analyzer = StockfishAnalyzer(stockfish_path=stockfish_path, threads=8, hash_mb=2048)

    # Читаем сырые позиции
    raw_positions = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_positions.append(RawPosition.model_validate_json(line))

    if limit:
        raw_positions = raw_positions[:limit]

    # Фильтруем уже посчитанные
    to_process = [p for p in raw_positions if p.id not in processed_ids]
    logger.info(f"Всего позиций: {len(raw_positions)}. Уже обработано: {len(processed_ids)}. К расчету: {len(to_process)}")

    with open(out_file, "a", encoding="utf-8") as out_f:
        for pos in tqdm(to_process, desc=f"Stockfish (глубина {depth})"):
            try:
                analyzed = analyzer.analyze_position(pos, depth=depth, multipv_count=3)
                if analyzed:
                    out_f.write(analyzed.model_dump_json() + "\n")
                    out_f.flush()
            except Exception as e:
                logger.error(f"Ошибка при анализе позиции {pos.id}: {e}")

    analyzer.close()
    logger.success(f"Анализ завершен! Результаты сохранены в: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stockfish analysis on raw positions")
    parser.add_argument("--input", type=str, default="data/01_raw/master_positions.jsonl")
    parser.add_argument("--out", type=str, default="data/02_engine_analyzed/master_analyzed.jsonl")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Путь к бинарнику Stockfish")
    parser.add_argument("--depth", type=int, default=22)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run_analysis(args.input, args.out, stockfish_path=args.stockfish, depth=args.depth, limit=args.limit)