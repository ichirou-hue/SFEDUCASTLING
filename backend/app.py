from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess
import chess.engine
import random
import os
import sys

app = FastAPI(title="SFEDUCASTLING API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# --- Загрузка LLM (если модель обучена) ---
chess_llm = None
LORA_PATH = os.environ.get("LORA_PATH", os.path.join(os.path.dirname(__file__), "..", "chess-llm-lora"))

if os.path.exists(LORA_PATH):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))
        from inference import ChessLLM
        chess_llm = ChessLLM(LORA_PATH)
        print(f"LLM загружена из {LORA_PATH}")
    except Exception as e:
        print(f"Не удалось загрузить LLM: {e}")
        print("Будет использована заглушка для анализа.")
else:
    print(f"Модель не найдена в {LORA_PATH} — используется заглушка.")
    print("Обучите модель: python ML/train_qlora.py --dataset dataset.jsonl")


# --- Загрузка Maia/Lc0 (если установлен) ---
maia_engine = None
MAIA_PATH = os.environ.get("MAIA_PATH", "")
MAIA_WEIGHTS = os.environ.get("MAIA_WEIGHTS", "")

if MAIA_PATH and os.path.exists(MAIA_PATH):
    try:
        maia_engine = chess.engine.SimpleEngine.popen_uci(MAIA_PATH)
        if MAIA_WEIGHTS:
            maia_engine.configure({"WeightsFile": MAIA_WEIGHTS})
        print(f"Maia движок загружен: {MAIA_PATH}")
    except Exception as e:
        print(f"Не удалось запустить Maia: {e}")
else:
    print("Maia не настроена — используются случайные ходы.")
    print("Установите: MAIA_PATH=/path/to/lc0 MAIA_WEIGHTS=/path/to/maia-1500.pb.gz")


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
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Game is over", "fen": req.fen}

    if maia_engine:
        result = maia_engine.play(board, chess.engine.Limit(time=1.0))
        move = result.move
    else:
        move = random.choice(list(board.legal_moves))

    san = board.san(move)
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)
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
        "from": from_name,
        "to": to_name,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
    }


@app.post("/api/analyze")
def analyze(req: FenRequest):
    if chess_llm:
        try:
            message = chess_llm.analyze(req.fen)
        except Exception as e:
            message = f"Ошибка анализа: {str(e)}"
    else:
        message = "Модель анализа не загружена. Обучите LLM и перезапустите сервер."

    return {
        "message": message,
        "fen": req.fen,
    }


@app.on_event("shutdown")
def shutdown():
    if maia_engine:
        maia_engine.quit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
