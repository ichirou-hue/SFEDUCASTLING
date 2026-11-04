"""Игровые endpoint'ы: легальные ходы, выполнение хода, ход AI."""

import hashlib
import threading

import chess
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api_gateway.dependecies import get_optional_current_user
from backend.db.session import get_db
from backend.api_gateway.models import (
    CapturedMaterialRequest,
    CompareMovesRequest,
    FenRequest,
    FenSquare,
    FinishGameRequest,
    GameReviewRequest,
    MoveRequest,
    NewGameRequest,
)
from backend.api_gateway.state import (
    elo_to_temperature,
    ensure_stockfish,
    get_maia3,
    maia3_lock,
    reset_maia3,
    reset_stockfish,
    stockfish_lock,
)
from backend.services.game_recorder import record_game
from backend.services.game_review import EngineUnavailable, review_game
from backend.services.gigacoin_service import (
    MAIA_DRAW_COINS,
    MAIA_FAIR_GAP,
    MAIA_WIN_COINS,
    award_gigacoins,
)
from backend.models.game import Game
from backend.models.user import User

router = APIRouter(tags=["game"])


LEVELS = [
    {"level": 1, "name": "Новичок", "elo": 1100, "engine": "maia3"},
    {"level": 2, "name": "Любитель", "elo": 1300, "engine": "maia3"},
    {"level": 3, "name": "Клубный", "elo": 1500, "engine": "maia3"},
    {"level": 4, "name": "Опытный", "elo": 1700, "engine": "maia3"},
    {"level": 5, "name": "Сильный", "elo": 1900, "engine": "maia3"},
    {"level": 6, "name": "Мастер", "elo": 2200, "engine": "maia3"},
    {"level": 7, "name": "Гроссмейстер", "elo": 2600, "engine": "maia3"},
    {"level": 8, "name": "Максимум (Stockfish)", "elo": 3600, "engine": "stockfish"},
]
MAIA_PROGRESS_LEVELS = tuple(range(800, 2601, 100))

# Ценность фигур (в пешках) для подсчёта съеденного материала.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Буквы фигур в нижнем регистре (как в FEN) для ответа API.
PIECE_LETTERS = {
    chess.PAWN: "p",
    chess.KNIGHT: "n",
    chess.BISHOP: "b",
    chess.ROOK: "r",
    chess.QUEEN: "q",
    chess.KING: "k",
}


@router.get("/api/levels")
def get_levels():
    """Возвращает список доступных уровней игры."""
    return {"levels": LEVELS}


@router.post("/api/legal-moves")
def legal_moves(req: FenSquare):
    """Возвращает все легальные ходы для фигуры на указанном поле."""
    board = chess.Board(req.fen)
    sq = chess.parse_square(req.square)
    moves = [
        chess.square_name(move.to_square)
        for move in board.legal_moves
        if move.from_square == sq
    ]
    return {"moves": moves}


@router.post("/api/move")
def make_move(req: MoveRequest):
    """Выполняет шахматный ход и возвращает новую позицию."""
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
        return {"error": "Недопустимый ход", "fen": req.fen}

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


@router.post("/api/stockfish-analysis")
def stockfish_analysis(req: FenRequest):
    """Анализирует позицию 10 секунд: лучший ход + оценка + топ ходов.

    Фронтенд вызывает этот эндпойнт после каждого хода игрока
    и рисует стрелку от from_sq к to_sq на лучший ход.
    """
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен", "fen": req.fen}

    try:
        with stockfish_lock:
            print(f"[Analysis] Анализ позиции {req.fen} (10 сек)...")
            stockfish.update_engine_parameters({"UCI_LimitStrength": False})
            stockfish.set_fen_position(req.fen)
            best_uci = stockfish.get_best_move_time(10000)
            evaluation = stockfish.get_evaluation(searchtime=1000)
            print(f"[Analysis] Лучший ход: {best_uci}, оценка: {evaluation}")
    except Exception as e:
        print(f"[Analysis] Ошибка: {e}")
        reset_stockfish()
        return {"error": str(e), "fen": req.fen}

    if not best_uci:
        return {"error": "Не удалось найти ход", "fen": req.fen}

    move = chess.Move.from_uci(best_uci)
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)

    return {
        "fen": req.fen,
        "best_move": best_uci,
        "from": from_name,
        "to": to_name,
        "san": board.san(move),
        "evaluation": evaluation,
    }


@router.post("/api/maia-move")
@router.post("/api/stockfish-move")
def ai_move(req: FenRequest):
    """Получает ход от AI.

    Maia3 играет как человек заданного Elo (req.elo + req.moves для истории).
    Если выбранный движок недоступен, ход не подменяется другим движком.
    """
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    if req.engine == "maia3":
        move = _maia3_move(req, board)
        if move is None:
            raise HTTPException(
                status_code=503,
                detail="Maia3 сейчас недоступна; ход не был выполнен.",
            )
    else:
        move = _stockfish_move(board)
        if move is None:
            print("[AI] Stockfish не вернул ход; запрос отклонён")
            raise HTTPException(
                status_code=503,
                detail="Stockfish сейчас недоступен; ход не был выполнен.",
            )

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

    evaluation = None
    if req.engine == "stockfish":
        sf = ensure_stockfish()
        if sf:
            try:
                with stockfish_lock:
                    sf.set_fen_position(board.fen())
                    evaluation = sf.get_evaluation(searchtime=1000)
                # Движок отвечает от лица стороны, чей ход; шкала у доски
                # ждёт оценку от лица белых, как и /api/eval.
                if evaluation and board.turn == chess.BLACK:
                    evaluation = {
                        "type": evaluation.get("type"),
                        "value": -(evaluation.get("value") or 0),
                    }
            except Exception as e:
                print(f"[Stockfish] Ошибка eval: {e}")

    return {
        "fen": board.fen(),
        "san": san,
        "from": from_name,
        "to": to_name,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "evaluation": evaluation,
        "engine": "maia3" if req.engine == "maia3" else "stockfish",
    }


# Сколько ждём ход Maia3 и сколько раз пробуем.
#
# engine.play() ждёт bestmove без ограничения: зависший процесс держал бы
# не только свой запрос, но и мьютекс движка, а за мьютексом вставали бы
# все следующие ходы - партия «замирала» навсегда.
MAIA3_PLAY_TIMEOUT = 30.0

MAIA3_ATTEMPTS = 2


def _same_position(left: chess.Board, right: chess.Board) -> bool:
    """Совпадают ли позиции, если не смотреть на счётчики FEN.

    Счётчики полуходов и номер хода на игру не влияют: Maia3 запущена с
    include_time_info=False. Поле en passant разные библиотеки заполняют
    по-своему. Сравнение по полной строке FEN из-за этого могло счесть
    верную историю чужой и обнулить её - тогда движок играл вслепую.
    """

    return (
        left.board_fen() == right.board_fen()
        and left.turn == right.turn
        and left.castling_xfen() == right.castling_xfen()
    )


def _history_board(req: FenRequest, board: chess.Board) -> chess.Board:
    """Доска с историей партии: Maia3 запущена с --use-uci-history.

    Контекст передаётся только стеком ходов доски: python-chess сам
    превращает его в "position fen ... moves ...". Без стека движок видит
    текущую позицию, повторённую 8 раз, и резко слабеет.
    """

    if not req.moves:
        return board

    try:
        replay = chess.Board()

        for uci in req.moves:
            replay.push_uci(uci)
    except ValueError as exc:
        print(
            f"[Maia3] Некорректная история ходов ({exc}); "
            "играем без истории - качество хода будет ниже."
        )
        return board

    if _same_position(replay, board):
        return replay

    print(
        "[Maia3] История не сходится с позицией: "
        f"{len(req.moves)} ходов дают {replay.fen()}, а доска {board.fen()}. "
        "Играем без истории - качество хода будет ниже."
    )
    return board


def _play_maia(engine, board: chess.Board, elo: int) -> chess.Move | None:
    """Один ход на готовом движке, с жёстким таймаутом.

    Ход считается в отдельном потоке: python-chess не умеет обрывать
    ожидание bestmove, а зависший движок не должен блокировать сервер.
    """

    import chess.engine

    temperature = elo_to_temperature(elo)
    print(f"[Maia3] Ход на Elo {elo}, temp={temperature}...")

    outcome: dict[str, object] = {}

    def worker() -> None:
        try:
            engine.configure({
                "Elo": elo,
                "SelfElo": elo,
                "OppoElo": elo,
                "Temperature": temperature,
            })

            outcome["move"] = engine.play(
                board,
                limit=chess.engine.Limit(time=3),
            ).move
        except Exception as exc:  # ошибку разбирает вызывающий код
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(MAIA3_PLAY_TIMEOUT)

    if thread.is_alive():
        raise TimeoutError(
            f"Maia3 не ответила за {MAIA3_PLAY_TIMEOUT:.0f} с"
        )

    error = outcome.get("error")

    if error is not None:
        raise error

    move = outcome.get("move")

    return move if isinstance(move, chess.Move) else None


def _ask_maia(board: chess.Board, elo: int) -> chess.Move | None:
    """Ход Maia3 с перезапуском движка при сбое.

    Смерть процесса раньше давала 503 на один ход и «висящую» партию:
    мёртвый хэндл оставался в глобальной переменной до следующего
    запроса. Теперь движок перезапускается тут же и ход повторяется.
    Повтор безопасен: история партии приходит в самом запросе.
    """

    last_error: object = None

    for attempt in range(1, MAIA3_ATTEMPTS + 1):

        engine = get_maia3()

        if engine is None:
            print("[Maia3] Недоступна; ход не выполнен")
            return None

        try:
            with maia3_lock():
                return _play_maia(engine, board, elo)
        except Exception as exc:
            last_error = exc
            print(
                f"[Maia3] Ошибка хода (попытка {attempt} из "
                f"{MAIA3_ATTEMPTS}): {exc}"
            )
            # Процесс мог умереть (Windows-ошибка канала, OOM) или зависнуть.
            # Держать такой хэндл дальше бессмысленно.
            reset_maia3()

    print(f"[Maia3] Ход не выполнен: {last_error}")
    return None


def _maia3_move(req: FenRequest, board: chess.Board) -> chess.Move | None:
    """Ход Maia3: человекоподобная игра на уровне req.elo."""

    move = _ask_maia(_history_board(req, board), req.elo)

    if move is None:
        return None

    print(f"[Maia3] Ход: {move}")

    if move in board.legal_moves:
        return move

    print(f"[Maia3] Ход {move} не легален в текущей позиции")
    return None


def _maia3_move_strict(
    req: FenRequest,
    board: chess.Board,
) -> chess.Move | None:
    """Получает ход Maia3 без fallback на Stockfish.

    Нужен для hybrid-анализа, где важно отличать
    настоящий ход Maia3 от fallback-хода Stockfish.
    """

    move = _ask_maia(_history_board(req, board), req.elo)

    if move is None:
        return None

    if move not in board.legal_moves:
        print(f"[Maia3] Strict move {move} is not legal")
        return None

    return move


def _stockfish_move(board: chess.Board) -> chess.Move | None:
    """Ход Stockfish на максимальной силе."""
    stockfish = ensure_stockfish()
    if stockfish:
        try:
            with stockfish_lock:
                print("[AI] Начало расчёта хода...")
                stockfish.update_engine_parameters({"UCI_LimitStrength": False})
                stockfish.set_fen_position(board.fen())
                best_uci = stockfish.get_best_move_time(10000)
                print(f"[AI] Ход: {best_uci}")
            if best_uci:
                return chess.Move.from_uci(best_uci)
        except Exception as e:
            print(f"[Stockfish] Ошибка хода: {e}")
            reset_stockfish()
    else:
        print("[AI] Stockfish недоступен")
    return None


@router.post("/api/compare-moves")
def compare_moves(req: CompareMovesRequest):
    """Сравнивает лучший ход Stockfish с ходом Maia3."""

    board = chess.Board(req.fen)

    if board.is_game_over():
        return {
            "error": "Партия окончена",
            "fen": req.fen,
        }

    # ---------------- Stockfish ----------------

    stockfish_move = _stockfish_move(board.copy())

    stockfish_result = None

    if stockfish_move is not None:
        stockfish_result = {
            "move": stockfish_move.uci(),
            "from": chess.square_name(stockfish_move.from_square),
            "to": chess.square_name(stockfish_move.to_square),
            "san": board.san(stockfish_move),
        }

        stockfish = ensure_stockfish()

        if stockfish:
            try:
                with stockfish_lock:
                    stockfish.set_fen_position(req.fen)
                    evaluation = stockfish.get_evaluation(searchtime=1000)

                stockfish_result["evaluation"] = evaluation

            except Exception as e:
                print(f"[Compare] Ошибка оценки Stockfish: {e}")

    # ---------------- Maia3 ----------------

    maia_move = _maia3_move(req, board.copy())

    maia_result = None

    if maia_move is not None:
        maia_result = {
            "move": maia_move.uci(),
            "from": chess.square_name(maia_move.from_square),
            "to": chess.square_name(maia_move.to_square),
            "san": board.san(maia_move),
            "elo": req.elo,
        }

    # ---------------- Результат ----------------

    same_move = (
        stockfish_move is not None
        and maia_move is not None
        and stockfish_move == maia_move
    )

    return {
        "fen": req.fen,
        "stockfish": stockfish_result,
        "maia3": maia_result,
        "same_move": same_move,
    }


def _replay_maia_result(
    req: FinishGameRequest, allowed_levels, minimum_moves: int = 8
) -> str | None:
    """Replay the submitted game and trust only an actual terminal result."""
    if req.engine != "maia3" or req.elo not in allowed_levels:
        return None
    if len(req.moves) < minimum_moves:
        return None
    board = chess.Board()
    for token in req.moves:
        if board.is_game_over(claim_draw=True):
            return None
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
    if not board.is_game_over(claim_draw=True):
        return None
    actual_result = board.result(claim_draw=True)
    return actual_result if actual_result == req.result else None


def _verified_maia_result(req: FinishGameRequest) -> str | None:
    return _replay_maia_result(req, MAIA_WIN_COINS)


def _verified_maia_achievement(req: FinishGameRequest) -> str | None:
    return _replay_maia_result(req, MAIA_PROGRESS_LEVELS, minimum_moves=1)


@router.get("/api/game/maia-achievements")
async def maia_achievements(
    user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return Maia levels where this user has a verified checkmate win."""
    if user is None:
        return {"levels": []}

    games = await db.scalars(
        select(Game)
        .options(selectinload(Game.moves))
        .where(
            Game.user_id == user.id,
            Game.engine == "maia3",
            Game.status == "checkmate",
            Game.elo.in_(MAIA_PROGRESS_LEVELS),
            or_(
                and_(Game.white == "player", Game.result == "1-0"),
                and_(Game.black == "player", Game.result == "0-1"),
            ),
        )
        .order_by(Game.id)
    )

    achieved = set()
    for game in games:
        player_color = "w" if game.white == "player" else "b"
        request = FinishGameRequest(
            moves=[move.uci for move in game.moves],
            elo=game.elo,
            engine=game.engine,
            result=game.result,
            status=game.status,
            player_color=player_color,
        )
        verified = _verified_maia_achievement(request)
        player_result = "1-0" if player_color == "w" else "0-1"
        if verified == player_result:
            achieved.add(game.elo)

    return {"levels": sorted(achieved)}


@router.post("/api/game/finish")
async def finish_game(
    req: FinishGameRequest,
    background_tasks: BackgroundTasks,
    user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Сохраняет завершённую партию в БД с классификацией ходов.

    Классификация каждого хода через Stockfish (1с на ход) выполняется
    в фоне, чтобы не блокировать ответ фронтенду.
    """
    # Игрок может играть и за чёрных: тогда AI - белые, а игрок - чёрные.
    if req.player_color == "b":
        white, black = "AI", "player"
    else:
        white, black = "player", "AI"

    coins_earned = 0
    if user is not None:
        verified_result = _verified_maia_result(req)
        if verified_result is not None:
            ref_payload = f"{req.engine}:{req.elo}:{req.player_color}:{','.join(req.moves)}:{verified_result}"
            ref = hashlib.sha256(ref_payload.encode("utf-8")).hexdigest()
            coins_earned = await award_gigacoins(db, user, "game_activity", ref, 0)
        player_won = verified_result == ("1-0" if req.player_color == "w" else "0-1")
        player_drew = verified_result == "1/2-1/2"
        player_rating = int(user.elo or user.rating_estimate or 1000)
        fair_level = req.elo is not None and req.elo + MAIA_FAIR_GAP >= player_rating
        requested_coins = MAIA_WIN_COINS.get(req.elo, 0) if player_won else MAIA_DRAW_COINS if player_drew else 0
        if requested_coins and fair_level:
            coins_earned += await award_gigacoins(db, user, "maia_game", ref, requested_coins)
        if verified_result is not None:
            await db.commit()

    background_tasks.add_task(
        record_game,
        moves=req.moves,
        # The authenticated identity is authoritative; ignore client-supplied user_id.
        user_id=user.id if user else None,
        elo=req.elo,
        engine=req.engine,
        result=req.result,
        status=req.status,
        white=white,
        black=black,
    )
    return {
        "ok": True,
        "status": "scheduled",
        "gigacoins_earned": coins_earned,
        "gigacoins_balance": int(user.gigacoins or 0) if user else 0,
    }


@router.post("/api/game/new")
def new_game(req: NewGameRequest):
    """Стартовая позиция новой партии за выбранную сторону игрока.

    Возвращает FEN, сторону игрока и сторону AI, а также ``ai_to_move``:
    если игрок играет чёрными, первый ход делает AI.
    """
    board = chess.Board()
    ai_color = "b" if req.player_color == "w" else "w"
    return {
        "fen": board.fen(),
        "player_color": req.player_color,
        "ai_color": ai_color,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "ai_to_move": ai_color == "w",
    }


@router.post("/api/game/captured-material")
def captured_material(req: CapturedMaterialRequest):
    """Съеденный материал по истории ходов партии (stateless).

    Реплеит партию от ``req.fen`` (по умолчанию - начальная позиция) и
    собирает взятые фигуры по сторонам, включая взятие на проходе.
    """
    board = chess.Board(req.fen) if req.fen else chess.Board()

    # captured[turn] - фигуры, взятые стороной turn (взятые чужие фигуры).
    captured: dict[bool, list[int]] = {chess.WHITE: [], chess.BLACK: []}

    for index, token in enumerate(req.moves, start=1):
        try:
            move = chess.Move.from_uci(token)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый ход на позиции {index}: {token}",
            )
        if move not in board.legal_moves:
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый ход на позиции {index}: {token}",
            )

        if board.is_en_passant(move):
            captured_square = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square),
            )
            victim = board.piece_at(captured_square)
        else:
            victim = board.piece_at(move.to_square)

        if victim is not None:
            captured[board.turn].append(victim.piece_type)

        board.push(move)

    def letters(piece_types: list[int]) -> list[str]:
        return [PIECE_LETTERS[piece_type] for piece_type in piece_types]

    def counts(piece_types: list[int]) -> dict[str, int]:
        result = {"p": 0, "n": 0, "b": 0, "r": 0, "q": 0}
        for piece_type in piece_types:
            letter = PIECE_LETTERS[piece_type]
            if letter in result:
                result[letter] += 1
        return result

    def value(piece_types: list[int]) -> int:
        return sum(PIECE_VALUES.get(piece_type, 0) for piece_type in piece_types)

    white_value = value(captured[chess.WHITE])
    black_value = value(captured[chess.BLACK])
    diff = white_value - black_value
    if diff > 0:
        advantage = "white"
    elif diff < 0:
        advantage = "black"
    else:
        advantage = "equal"

    response = {
        "fen": board.fen(),
        "captured_by_white": letters(captured[chess.WHITE]),
        "captured_by_black": letters(captured[chess.BLACK]),
        "counts": {
            "white": counts(captured[chess.WHITE]),
            "black": counts(captured[chess.BLACK]),
        },
        "material": {
            "white": white_value,
            "black": black_value,
            "diff": diff,
            "advantage": advantage,
        },
    }

    if req.player_color is not None:
        response["player_advantage"] = diff if req.player_color == "w" else -diff

    return response


@router.post("/api/game/review")
def review_finished_game(req: GameReviewRequest):
    """Разбор партии движком: точность игры, звёзды и оценка ходов.

    Оценивается позиция после каждого полухода (для истории с подписями),
    а точность и классификация считаются только по ходам игрока. Звёзды
    ограничены результатом: победа - до 3, ничья - до 2, поражение - до 1.
    Запрос синхронный, поэтому FastAPI выполняет его в пуле потоков: разбор
    партии на 40 ходов занимает до 40 секунд, и фронтенд показывает
    индикатор ожидания.
    """
    if not req.moves:
        raise HTTPException(status_code=400, detail="Нет ходов для разбора")

    try:
        return review_game(req.moves, player_color=req.player_color)
    except EngineUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Stockfish недоступен, разбор партии временно невозможен",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

