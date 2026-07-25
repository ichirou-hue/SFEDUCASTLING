"""Endpoint'ы сбора данных и парсинга PGN."""

import os
import json
import io
from datetime import datetime
import chess
import chess.pgn
from fastapi import APIRouter, UploadFile, File
from backend.config.settings import settings

from backend.api_gateway.models import DatasetMoveRequest, PGNTextRequest
from backend.api_gateway.state import ensure_stockfish, stockfish_lock

router = APIRouter(tags=["data"])


@router.post("/api/save-move-to-dataset")
def save_move_to_dataset(req: DatasetMoveRequest):
    """Сохраняет ход пользователя в тренировочный датасет с оценкой Stockfish."""
    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен"}

    try:
        with stockfish_lock:
            stockfish.set_fen_position(req.fen)
            stockfish_best = stockfish.get_best_move()
            stockfish_eval = stockfish.get_evaluation()

        move_data = {
            "fen": req.fen,
            "user_move": req.move,
            "stockfish_move": stockfish_best,
            "stockfish_eval": stockfish_eval,
            "user_id": req.user_id,
            "game_id": req.game_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Используем пути из настроек
        dataset_path = str(settings.data.dataset_jsonl_path)
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(move_data, ensure_ascii=False) + "\n")

        readable_path = str(settings.data.dataset_readable_path)
        if os.path.exists(readable_path):
            with open(readable_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []

        all_data.append(move_data)

        with open(readable_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        return {
            "status": "saved",
            "dataset_size": len(all_data),
            "data": move_data,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/parse-pgn-text")
def parse_pgn_text(req: PGNTextRequest):
    """Разбирает PGN-текст, вставленный пользователем, в структурированный формат."""
    try:
        print(f"[PGN-Text] Начало обработки PGN текста")
        print(f"[PGN-Text] Длина текста: {len(req.pgn)} символов")
        print(f"[PGN-Text] Первые 200 символов: {req.pgn[:200]}")

        pgn_io = io.StringIO(req.pgn)
        game = chess.pgn.read_game(pgn_io)

        if not game:
            return {"error": "Не удалось распознать PGN. Проверьте формат."}

        headers = game.headers
        white = headers.get("White", "Unknown")
        black = headers.get("Black", "Unknown")
        result = headers.get("Result", "*")

        print(f"[PGN-Text] Партия: {white} vs {black}, результат: {result}")

        board = game.board()
        moves_list = []

        moves_list.append({
            "fen": board.fen(),
            "move": "start",
            "move_number": 0,
            "turn": "white",
        })

        move_number = 1
        move_count = 0

        for move in game.mainline_moves():
            fen_before = board.fen()
            san = board.san(move)
            turn_color = "white" if board.turn == chess.WHITE else "black"

            moves_list.append({
                "fen": fen_before,
                "move": san,
                "move_number": move_number,
                "turn": turn_color,
                "uci": move.uci(),
            })

            board.push(move)
            move_count += 1

            if turn_color == "black":
                move_number += 1

        moves_list.append({
            "fen": board.fen(),
            "move": "end",
            "move_number": move_number,
            "turn": "white" if board.turn == chess.WHITE else "black",
        })

        print(f"[PGN-Text] Всего ходов: {move_count}, позиций: {len(moves_list)}")

        return {
            "games_count": 1,
            "games": [{
                "id": 1,
                "white": white,
                "black": black,
                "result": result,
                "date": headers.get("Date", "?"),
                "opening": headers.get("Opening", "?"),
                "moves": moves_list,
            }],
        }

    except Exception as e:
        import traceback
        print(f"[PGN-Text] ОШИБКА: {e}")
        traceback.print_exc()
        return {"error": f"Ошибка парсинга: {str(e)}"}


@router.post("/api/parse-pgn")
async def parse_pgn(file: UploadFile = File(...)):
    """Разбирает загруженный PGN-файл в структурированные данные партий."""
    try:
        print(f"[PGN] Начало обработки файла: {file.filename}")
        contents = await file.read()
        print(f"[PGN] Файл прочитан: {len(contents)} байт")

        if isinstance(contents, bytes):
            try:
                contents = contents.decode("utf-8")
            except UnicodeDecodeError:
                contents = contents.decode("latin-1")

        print(f"[PGN] Содержимое:\n{contents[:500]}")

        pgn_io = io.StringIO(contents)
        games = []
        game_num = 0

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            game_num += 1
            headers = game.headers
            white = headers.get("White", "Unknown")
            black = headers.get("Black", "Unknown")
            result = headers.get("Result", "*")

            print(f"[PGN] Партия {game_num}: {white} vs {black}")

            board = game.board()
            moves_list = []

            moves_list.append({
                "fen": board.fen(),
                "move": "start",
                "move_number": 0,
                "turn": "white",
            })

            move_number = 1
            for move in game.mainline_moves():
                fen_before = board.fen()
                san = board.san(move)
                turn_color = "white" if board.turn == chess.WHITE else "black"

                moves_list.append({
                    "fen": fen_before,
                    "move": san,
                    "move_number": move_number,
                    "turn": turn_color,
                    "uci": move.uci(),
                })

                board.push(move)
                if turn_color == "black":
                    move_number += 1

            moves_list.append({
                "fen": board.fen(),
                "move": "end",
                "move_number": move_number,
                "turn": "white" if board.turn == chess.WHITE else "black",
            })

            print(f"[PGN] Всего позиций в партии: {len(moves_list)}")

            games.append({
                "id": game_num,
                "white": white,
                "black": black,
                "result": result,
                "date": headers.get("Date", "?"),
                "opening": headers.get("Opening", "?"),
                "moves": moves_list,
            })

            # Используем лимит из настроек
            if game_num >= settings.data.max_games_to_parse:
                break

        print(f"[PGN] Итого партий: {len(games)}")

        if len(games) == 0:
            return {"error": "Не удалось найти партии в файле"}

        return {"games_count": len(games), "games": games}

    except Exception as e:
        import traceback
        print(f"[PGN] ОШИБКА: {e}")
        traceback.print_exc()
        return {"error": str(e)}