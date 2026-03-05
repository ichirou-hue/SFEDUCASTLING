from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess
import chess.engine
import random
import os

app = FastAPI(title="SFEDUCASTLING API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


class FenSquare(BaseModel):
    fen: str
    square: str


class MoveRequest(BaseModel):
    fen: str
    from_sq: str
    to_sq: str
    promotion: str = "q"


class FenRequest(BaseModel):
    fen: str


@app.post("/api/legal-moves")
def legal_moves(req: FenSquare):
    board = chess.Board(req.fen)
    sq = chess.parse_square(req.square)
    moves = []
    for move in board.legal_moves:
        if move.from_square == sq:
            moves.append(chess.square_name(move.to_square))
    return {"moves": moves}


@app.post("/api/move")
def make_move(req: MoveRequest):
    board = chess.Board(req.fen)
    from_sq = chess.parse_square(req.from_sq)
    to_sq = chess.parse_square(req.to_sq)

    promo = None
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(to_sq) in (0, 7):
            promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
            promo = promo_map.get(req.promotion, chess.QUEEN)

    move = chess.Move(from_sq, to_sq, promotion=promo)

    if move not in board.legal_moves:
        return {"error": "Illegal move", "fen": req.fen}

    san = board.san(move)
    board.push(move)

    status = "playing"
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_check():
        status = "check"

    return {
        "fen": board.fen(),
        "san": san,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
    }


@app.post("/api/maia-move")
def maia_move(req: FenRequest):
    """Заглушка для Maia — пока выбирает случайный легальный ход."""
    # TODO: подключить реальную Maia Chess модель
    # engine = chess.engine.SimpleEngine.popen_uci("path/to/maia")
    # result = engine.play(board, chess.engine.Limit(time=1.0))
    # engine.quit()

    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Game is over", "fen": req.fen}

    move = random.choice(list(board.legal_moves))
    san = board.san(move)
    board.push(move)

    status = "playing"
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_check():
        status = "check"

    return {
        "fen": board.fen(),
        "san": san,
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
    }


@app.post("/api/analyze")
def analyze(req: FenRequest):
    """Заглушка для GigaChat анализа."""
    # TODO: подключить GigaChat API
    # response = gigachat.analyze(fen=req.fen, prompt="Опиши позицию...")
    return {
        "message": "Анализ позиции будет доступен после подключения GigaChat API.",
        "fen": req.fen,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
