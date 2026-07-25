"""Endpoint'ы сбора данных и парсинга PGN."""

import os
import json
import io
import chess
import chess.pgn
from fastapi import APIRouter, UploadFile, File

from backend.api_gateway.models import DatasetMoveRequest, PGNTextRequest

router = APIRouter(tags=["data"])


@router.post("/api/save-move-to-dataset")
def save_move_to_dataset(req: DatasetMoveRequest):
    """Сохраняет ход пользователя в тренировочный датасет (отключено)."""
    return {"status": "saved", "dataset_size": 0, "note": "Сбор датасета отключён"}


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

            if game_num >= 10:
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
