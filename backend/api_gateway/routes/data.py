"""Endpoint'ы сбора данных и парсинга PGN."""

import io
import json
from datetime import UTC, datetime

import chess
import chess.pgn
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_admin, get_current_user
from backend.api_gateway.models import DatasetMoveRequest, PGNTextRequest
from backend.api_gateway.rate_limit import expensive_rate_limit
from backend.api_gateway.state import ensure_stockfish, stockfish_lock
from backend.api_gateway.uploads import read_upload_limited
from backend.config.settings import settings
from backend.db.session import get_db
from backend.models.dataset_move import DatasetMove
from backend.models.user import User

router = APIRouter(tags=["data"])


@router.post(
    "/api/save-move-to-dataset",
    dependencies=[Depends(expensive_rate_limit)],
)
async def save_move_to_dataset(
    req: DatasetMoveRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Сохраняет ход в тренировочный датасет от имени текущего пользователя.

    ``user_id`` из тела больше не принимается: иначе можно было записать ход
    на чужой аккаунт. Поле оставлено в схеме для совместимости и игнорируется.
    """
    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен"}

    try:
        with stockfish_lock:
            stockfish.set_fen_position(req.fen)
            stockfish_best = stockfish.get_best_move_time(1000)
            stockfish_eval = stockfish.get_evaluation(searchtime=1000)

        row = DatasetMove(
            fen=req.fen,
            user_move=req.move,
            stockfish_move=stockfish_best,
            stockfish_eval=stockfish_eval,
            user_id=user.id,
            game_id=req.game_id,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

        total = await db.scalar(select(func.count(DatasetMove.id)))
        return {
            "status": "saved",
            "dataset_size": total,
            "data": {
                "fen": row.fen,
                "user_move": row.user_move,
                "stockfish_move": row.stockfish_move,
                "stockfish_eval": row.stockfish_eval,
                "user_id": row.user_id,
                "game_id": row.game_id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else datetime.now(UTC).isoformat(),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/dataset/export")
async def export_dataset(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Выгружает весь датасет в формате NDJSON (только администратор).

    В датасете есть ``user_id`` и ходы всех пользователей, поэтому публичный
    доступ к нему закрыт.
    """

    async def row_gen():
        result = await db.stream(select(DatasetMove).order_by(DatasetMove.id.asc()))
        async for row in result.scalars():
            ts = row.timestamp.isoformat() if row.timestamp else datetime.now(UTC).isoformat()
            payload = {
                "fen": row.fen,
                "user_move": row.user_move,
                "stockfish_move": row.stockfish_move,
                "stockfish_eval": row.stockfish_eval,
                "user_id": row.user_id,
                "game_id": row.game_id,
                "timestamp": ts,
            }
            yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(row_gen(), media_type="application/x-ndjson")


@router.post(
    "/api/parse-pgn-text",
    dependencies=[Depends(expensive_rate_limit)],
)
def parse_pgn_text(req: PGNTextRequest):
    """Разбирает PGN-текст, вставленный пользователем, в структурированный формат."""
    try:
        print(f"[PGN-Text] Начало обработки, длина: {len(req.pgn)} символов")

        pgn_io = io.StringIO(req.pgn)
        game = chess.pgn.read_game(pgn_io)

        if not game:
            return {"error": "Не удалось распознать PGN. Проверьте формат."}

        headers = game.headers
        white = headers.get("White", "Unknown")
        black = headers.get("Black", "Unknown")
        result = headers.get("Result", "*")

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


@router.post("/api/parse-pgn", dependencies=[Depends(expensive_rate_limit)])
async def parse_pgn(file: UploadFile = File(...)):
    """Разбирает загруженный PGN-файл в структурированные данные партий."""
    try:
        contents = await read_upload_limited(file)
        print(f"[PGN] Файл прочитан: {len(contents)} байт")

        if isinstance(contents, bytes):
            try:
                contents = contents.decode("utf-8")
            except UnicodeDecodeError:
                contents = contents.decode("latin-1")

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

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[PGN] ОШИБКА: {e}")
        traceback.print_exc()
        return {"error": str(e)}
