from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess
import random
import os
import requests
from dotenv import load_dotenv
from gigachat import GigaChat
from maia2 import model as maia2_model_loader, inference as maia2_inference

load_dotenv()

app = FastAPI(title="SFEDUCASTLING API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# --- GigaChat ---
GIGACHAT_AUTH_KEY = os.environ.get("GIGACHAT_AUTH_KEY", "")
if GIGACHAT_AUTH_KEY:
    print("GigaChat API ключ загружен.")
else:
    print("GigaChat API ключ не найден! Добавьте GIGACHAT_AUTH_KEY в .env")


# --- Загрузка Maia2 ---
maia2 = None
maia2_prepared = None
try:
    maia2 = maia2_model_loader.from_pretrained(type="rapid", device="cpu")
    maia2_prepared = maia2_inference.prepare()
    print("Maia2 загружена и готова к работе.")
except Exception as e:
    print(f"Не удалось загрузить Maia2: {e}")
    print("Будут использоваться случайные ходы.")


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
    elo: int = 1500


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

    if maia2 and maia2_prepared:
        try:
            move_probs, win_prob = maia2_inference.inference_each(
                maia2, maia2_prepared, req.fen, req.elo, req.elo
            )
            best_uci = max(move_probs, key=move_probs.get)
            move = chess.Move.from_uci(best_uci)
        except Exception as e:
            print(f"Maia2 ошибка: {e}")
            move = random.choice(list(board.legal_moves))
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


def get_opening_info(fen):
    """Запрашиваем Lichess Opening Explorer — дебют + знаменитые партии."""
    try:
        resp = requests.get(
            "https://explorer.lichess.ovh/masters",
            params={"fen": fen},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        info = {}
        if data.get("opening"):
            info["name"] = data["opening"].get("name", "")
        top_games = data.get("topGames", [])[:3]
        if top_games:
            games = []
            for g in top_games:
                white = g.get("white", {}).get("name", "?")
                black = g.get("black", {}).get("name", "?")
                year = g.get("year", "?")
                winner = g.get("winner", "draw")
                games.append(f"{white} — {black}, {year} ({winner})")
            info["games"] = games
        return info if info else None
    except Exception:
        return None


@app.post("/api/analyze")
def analyze(req: FenRequest):
    if not GIGACHAT_AUTH_KEY:
        return {"message": "GigaChat API ключ не настроен.", "fen": req.fen}

    try:
        board = chess.Board(req.fen)
        turn = "Белые" if board.turn == chess.WHITE else "Чёрные"
        move_number = board.fullmove_number

        # Получаем инфо о дебюте из Lichess
        opening = get_opening_info(req.fen)

        opening_context = ""
        if opening:
            if opening.get("name"):
                opening_context += f"\nДебют: {opening['name']}."
            if opening.get("games"):
                opening_context += "\nЗнаменитые партии с похожей позицией:"
                for g in opening["games"]:
                    opening_context += f"\n- {g}"

        prompt = (
            f"Ты — шахматный тренер-историк. Оцени позицию и дай совет простым языком.\n"
            f"Позиция (FEN): {req.fen}\n"
            f"Ход: {turn}, ход номер {move_number}.\n"
            f"{opening_context}\n\n"
            f"1. Если известен дебют — назови его и кратко объясни идею.\n"
            f"2. Если есть знаменитые партии — упомяни самую интересную (кто играл, год, чем закончилась).\n"
            f"3. Оцени кто лучше стоит, какие угрозы, что делать дальше.\n"
            f"Отвечай кратко и понятно."
        )

        with GigaChat(
            credentials=GIGACHAT_AUTH_KEY,
            scope="GIGACHAT_API_PERS",
            model="GigaChat",
            verify_ssl_certs=False,
        ) as giga:
            response = giga.chat(prompt)
            message = response.choices[0].message.content

    except Exception as e:
        message = f"Ошибка GigaChat: {str(e)}"

    return {
        "message": message,
        "fen": req.fen,
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
