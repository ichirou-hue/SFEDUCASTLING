from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import UploadFile, File
from pydantic import BaseModel
import chess
import random
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from gigachat import GigaChat
from maia2 import model as maia2_model_loader, inference as maia2_inference
from stockfish import Stockfish

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


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))


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

# --- Загрузка Stockfish ---
STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish.exe")
stockfish = None
try:
    if os.path.exists(STOCKFISH_PATH):
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=20)
        print(f"✅ Stockfish загружен: {STOCKFISH_PATH}")
    else:
        print(f"❌ Stockfish не найден по пути: {STOCKFISH_PATH}")
except Exception as e:
    print(f"Ошибка загрузки Stockfish: {e}")


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


class DatasetMoveRequest(BaseModel):
    fen: str
    move: str
    user_id: str = "anonymous"
    game_id: str = ""


class PGNTextRequest(BaseModel):
    pgn: str


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
            if not move_probs:
                print(f"[Maia2] WARNING: empty move_probs, using random")
                move = random.choice(list(board.legal_moves))
            else:
                best_uci = max(move_probs, key=move_probs.get)
                move = chess.Move.from_uci(best_uci)
        except Exception as e:
            print(f"[Maia2] Ошибка инференса: {e}")
            import traceback
            traceback.print_exc()
            move = random.choice(list(board.legal_moves))
    else:
        print("[Maia2] ВНИМАНИЕ: maia2 не загружена, используется случайный ход")
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


@app.post("/api/stockfish-analyze")
def stockfish_analyze(req: FenRequest):
    if not stockfish:
        return {"error": "Stockfish не загружен"}
    
    try:
        stockfish.set_fen_position(req.fen)
        best_move = stockfish.get_best_move()
        evaluation = stockfish.get_evaluation()
        top_moves = stockfish.get_top_moves(5)
        
        return {
            "fen": req.fen,
            "best_move": best_move,
            "evaluation": evaluation,
            "top_moves": top_moves
        }
    except Exception as e:
        print(f"[Stockfish] Ошибка: {e}")
        return {"error": str(e)}


@app.post("/api/compare-maia-stockfish")
def compare_engines(req: FenRequest):
    if not stockfish:
        return {"error": "Stockfish не загружен"}
    if not maia2 or not maia2_prepared:
        return {"error": "Maia2 не загружена"}
    
    try:
        stockfish.set_fen_position(req.fen)
        stockfish_best = stockfish.get_best_move()
        stockfish_eval = stockfish.get_evaluation()
        
        move_probs, win_prob = maia2_inference.inference_each(
            maia2, maia2_prepared, req.fen, req.elo, req.elo
        )
        maia_best = max(move_probs.items(), key=lambda x: x[1])[0]
        
        board1 = chess.Board(req.fen)
        board2 = chess.Board(req.fen)
        
        try:
            move1 = chess.Move.from_uci(stockfish_best)
            board1.push(move1)
            stockfish.set_fen_position(board1.fen())
            eval_after_stockfish = stockfish.get_evaluation()
        except:
            eval_after_stockfish = {"type": "cp", "value": 0}
        
        try:
            move2 = chess.Move.from_uci(maia_best)
            board2.push(move2)
            stockfish.set_fen_position(board2.fen())
            eval_after_maia = stockfish.get_evaluation()
        except:
            eval_after_maia = {"type": "cp", "value": 0}
        
        val1 = eval_after_stockfish.get("value", 0)
        val2 = eval_after_maia.get("value", 0)
        try:
            difference = abs(int(val1) - int(val2))
        except:
            difference = 0
        
        return {
            "fen": req.fen,
            "stockfish": {
                "move": stockfish_best,
                "evaluation": stockfish_eval
            },
            "maia2": {
                "move": maia_best,
                "probability": move_probs[maia_best],
                "win_probability": win_prob
            },
            "comparison": {
                "eval_after_stockfish": eval_after_stockfish,
                "eval_after_maia": eval_after_maia,
                "difference": difference
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/save-move-to-dataset")
def save_move_to_dataset(req: DatasetMoveRequest):
    if not stockfish:
        return {"error": "Stockfish не загружен"}
    
    try:
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
            "timestamp": datetime.now().isoformat()
        }
        
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset.jsonl")
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(move_data, ensure_ascii=False) + "\n")
        
        readable_path = os.path.join(os.path.dirname(__file__), "..", "dataset_readable.json")
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
            "data": move_data
        }
    except Exception as e:
        return {"error": str(e)}


def get_opening_info(fen):
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


# --- База знаний дебютов ---
knowledge_base = None
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "openings.json")

def load_knowledge():
    global knowledge_base
    try:
        if os.path.exists(KNOWLEDGE_PATH):
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
            print(f"База знаний загружена: {len(knowledge_base.get('openings', []))} дебютов")
    except Exception as e:
        print(f"Ошибка загрузки базы знаний: {e}")

load_knowledge()


@app.get("/api/knowledge/openings")
def get_openings():
    if not knowledge_base:
        return {"openings": [], "error": "База знаний не загружена"}
    return {"openings": knowledge_base.get("openings", [])}


@app.get("/api/knowledge/opening")
def get_opening_by_fen(fen: str = ""):
    if not knowledge_base:
        return {"error": "База знаний не загружена"}
    
    fen_lower = fen.lower().split()[0]
    
    for opening in knowledge_base.get("openings", []):
        if opening.get("fen", "").lower().startswith(fen_lower):
            return {"opening": opening}
    
    return {"opening": None, "message": "Дебют не найден в базе"}


@app.get("/api/knowledge/random-opening")
def get_random_opening():
    if not knowledge_base:
        return {"error": "База знаний не загружена"}
    
    import random
    opening = random.choice(knowledge_base.get("openings", []))
    return {"opening": opening}


@app.post("/api/knowledge/check-move")
def check_move(req: FenRequest):
    if not knowledge_base:
        return {"error": "База знаний не загружена"}
    
    board = chess.Board(req.fen)
    current_fen = board.fen().split()[0].lower()
    
    for opening in knowledge_base.get("openings", []):
        opening_fen = opening.get("fen", "").split()[0].lower()
        if current_fen == opening_fen:
            return {
                "in_theory": True,
                "opening": opening.get("name"),
                "eco": opening.get("eco"),
                "description": opening.get("description")
            }
    
    return {"in_theory": False, "message": "Позиция не найдена в базе теории"}


# === НОВЫЙ ЭНДПОИНТ ДЛЯ PGN ТЕКСТА ===
@app.post("/api/parse-pgn-text")
def parse_pgn_text(req: PGNTextRequest):
    """Парсинг PGN текста, вставленного пользователем"""
    try:
        import io
        import chess.pgn
        
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
        
        # Начальная позиция
        moves_list.append({
            "fen": board.fen(),
            "move": "start",
            "move_number": 0,
            "turn": "white"
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
                "uci": move.uci()
            })
            
            board.push(move)
            move_count += 1
            
            # Увеличиваем номер хода после хода белых
            if turn_color == "black":
                move_number += 1
        
        # Финальная позиция
        moves_list.append({
            "fen": board.fen(),
            "move": "end",
            "move_number": move_number,
            "turn": "white" if board.turn == chess.WHITE else "black"
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
                "moves": moves_list
            }]
        }
        
    except Exception as e:
        import traceback
        print(f"[PGN-Text] ОШИБКА: {e}")
        traceback.print_exc()
        return {"error": f"Ошибка парсинга: {str(e)}"}


# --- PGN парсинг файла ---
@app.post("/api/parse-pgn")
def parse_pgn(file: UploadFile = File(...)):
    """Парсинг PGN файла"""
    try:
        import io
        import chess.pgn
        import re
        
        print(f"[PGN] Начало обработки файла: {file.filename}")
        contents = file.file.read()
        print(f"[PGN] Файл прочитан: {len(contents)} байт")
        
        if isinstance(contents, bytes):
            try:
                contents = contents.decode('utf-8')
            except:
                contents = contents.decode('latin-1')
        
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
                "turn": "white"
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
                    "uci": move.uci()
                })
                
                board.push(move)
                if turn_color == "black":
                    move_number += 1
            
            moves_list.append({
                "fen": board.fen(),
                "move": "end",
                "move_number": move_number,
                "turn": "white" if board.turn == chess.WHITE else "black"
            })
            
            print(f"[PGN] Всего позиций в партии: {len(moves_list)}")
            
            games.append({
                "id": game_num,
                "white": white,
                "black": black,
                "result": result,
                "date": headers.get("Date", "?"),
                "opening": headers.get("Opening", "?"),
                "moves": moves_list
            })
            
            if game_num >= 10:
                break
        
        print(f"[PGN] Итого партий: {len(games)}")
        
        if len(games) == 0:
            return {"error": "Не удалось найти партии в файле"}
        
        return {
            "games_count": len(games),
            "games": games
        }
        
    except Exception as e:
        import traceback
        print(f"[PGN] ОШИБКА: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# --- LLaVA через transformers ---
llava_model = None

def load_llava():
    global llava_model
    try:
        import torch
        from transformers import pipeline
        
        print("Загрузка LLaVA 1.5 7B через pipeline...")
        llava_model = pipeline(
            "image-to-text",
            model="llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("LLaVA загружена успешно!")
        return True
    except Exception as e:
        print(f"Ошибка загрузки LLaVA: {e}")
        return False

def extract_fen_from_image(image_path: str) -> str:
    from PIL import Image
    
    if llava_model is None:
        if not load_llava():
            return "ERROR: LLaVA не загружена."
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        prompt = """Describe this chess board position. Output ONLY the FEN notation for the position shown.
Example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"""
        
        outputs = llava_model(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        
        response = outputs[0]["generated_text"]
        
        if "ERROR" in response.upper():
            return "ERROR: cannot read board"
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if '/' in line and len(line) > 20:
                parts = line.split()
                if len(parts) >= 1 and '/' in parts[0] and parts[0].count('/') == 7:
                    return parts[0]
        
        return f"ERROR: unexpected response: {response[:100]}"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

print("Инициализация LLaVA...")
load_llava()


@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if llava_model is None:
        return {
            "error": "LLaVA не загружена. Проверьте что transformers и torch установлены.",
            "hint": "pip install transformers torch pillow"
        }
    
    try:
        import tempfile
        import os
        
        contents = await file.read()
        
        filename = file.filename or "image.png"
        ext = filename.split('.')[-1].lower()
        if ext not in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
            ext = 'png'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            fen = extract_fen_from_image(tmp_path)
            
            if fen.startswith("ERROR"):
                return {"error": fen.replace("ERROR: ", ""), "fen": None}
            
            return {"fen": fen, "message": "Позиция распознана!"}
        
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)