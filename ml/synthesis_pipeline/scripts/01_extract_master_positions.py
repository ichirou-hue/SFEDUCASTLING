import argparse
import chess
import chess.pgn
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.schemas import RawPosition

MIN_ELO = 2400
MIN_MOVE = 8
MAX_MOVE = 35

def get_game_phase(board: chess.Board) -> str:
    piece_count = len(board.piece_map())
    if board.fullmove_number <= 12:
        return "opening"
    elif piece_count <= 10:
        return "endgame"
    return "middlegame"

def is_strategic_moment(board: chess.Board, move: chess.Move) -> bool:
    """Отсекает взятия и ходы под шахом, оставляя позиционные маневры."""
    if board.is_capture(move):
        return False
    if board.is_check():
        return False
    return True

def extract_positions(pgn_path: str, output_path: str, max_positions: int = 50):
    input_file = Path(pgn_path)
    if not input_file.exists():
        logger.error(f"PGN файл '{pgn_path}' не найден.")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    games_parsed = 0
    seen_fens = set()

    with open(input_file, "r", encoding="utf-8", errors="ignore") as pgn_f, \
         open(output_file, "w", encoding="utf-8") as out_f:

        pbar = tqdm(total=max_positions, desc="Парсинг мастер-позиций")

        while extracted_count < max_positions:
            game = chess.pgn.read_game(pgn_f)
            if game is None:
                break

            games_parsed += 1
            headers = game.headers

            try:
                w_elo = int(headers.get("WhiteElo", 0))
                b_elo = int(headers.get("BlackElo", 0))
            except (ValueError, TypeError):
                w_elo, b_elo = 0, 0

            # Если в PGN нет тегов Elo, пропускаем проверку для тестовых файлов
            if (w_elo > 0 and w_elo < MIN_ELO) or (b_elo > 0 and b_elo < MIN_ELO):
                continue

            board = game.board()

            for move in game.mainline_moves():
                fen_before = board.fen()
                move_number = board.fullmove_number

                if MIN_MOVE <= move_number <= MAX_MOVE:
                    if is_strategic_moment(board, move) and fen_before not in seen_fens:
                        seen_fens.add(fen_before)

                        raw_record = RawPosition(
                            id=f"master_{games_parsed}_{move_number}_{'w' if board.turn == chess.WHITE else 'b'}",
                            source_type="master_game",
                            game_id=headers.get("Site", f"game_{games_parsed}"),
                            fen=fen_before,
                            played_move=board.san(move),
                            move_number=move_number,
                            turn="white" if board.turn == chess.WHITE else "black",
                            event=headers.get("Event", "Unknown"),
                            white_player=headers.get("White", "Unknown"),
                            black_player=headers.get("Black", "Unknown"),
                            white_elo=w_elo if w_elo > 0 else None,
                            black_elo=b_elo if b_elo > 0 else None,
                            phase=get_game_phase(board)
                        )

                        out_f.write(raw_record.model_dump_json() + "\n")
                        extracted_count += 1
                        pbar.update(1)

                        if extracted_count >= max_positions:
                            break

                board.push(move)

        pbar.close()

    logger.success(f"Готово! Обработано партий: {games_parsed}, извлечено позиций: {extracted_count} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract master positions from PGN")
    parser.add_argument("--pgn", type=str, required=True, help="Путь к PGN файлу")
    parser.add_argument("--out", type=str, default="data/01_raw/master_positions.jsonl", help="Куда сохранить JSONL")
    parser.add_argument("--limit", type=int, default=50, help="Количество позиций для выборки")
    args = parser.parse_args()

    extract_positions(args.pgn, args.out, max_positions=args.limit)