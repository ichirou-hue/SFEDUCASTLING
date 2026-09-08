"""Endpoint'ы чата: приём комментариев от LLM/пользователя и выдача их во фронтенд.

Сообщения хранятся в PostgreSQL (таблица chat_messages) и опрашиваются
фронтендом через /api/chat/messages?after=<id>.
"""

import re

import chess
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.api_gateway.dependecies import get_optional_current_user
from backend.api_gateway.models import ChatAskRequest
from backend.api_gateway.sanitize import sanitize_text
from backend.api_gateway.state import ensure_stockfish, get_gigachess, stockfish_lock
from backend.db.session import get_db
from backend.llm.gigachess import GigachessError
from backend.models.chat_message import ChatMessage

router = APIRouter(tags=["chat"])


BOARD_MARKER = re.compile(r"\[ДОСКА:\s*([^\]]+)\]", re.IGNORECASE)


_POSITION_KEYWORDS = {
    "ход", "ходить", "сделать", "продолжить", "что делать", "что сыгр",
    "позиция", "положение", "лучший", "оценка", "выигр", "проигр",
    "мат", "вилка", "связка", "жертв", "пешка", "ферзь", "король",
    "конь", "слон", "ладь", "рокировка", "дебют", "эндшпиль", "миттельшпиль",
    "теоретич", "фигур", "белых", "чёрных", "за белых", "за чёрных",
    "ходить", "какой ход", "что поставить", "удар", "тактика",
}

_NON_POSITION_KEYWORDS = {
    "привет", "здравств", "добрый день", "добрый вечер", "как дела",
    "спасибо", "пока", "до свидания", "кто ты", "как тебя зовут",
    "что ты умееш", "зачем", "расскажи о себе", "погода", "новости",
    "музыка", "фильм", "книга", "что делаешь", "анекдот",
}


def _mentions_position(message: str) -> bool:
    """Определяет, относится ли вопрос к текущей позиции/ходу.

    Легкая эвристика по ключевым словам:
    - признаки «про позицию» (ход, фигуры, выигрыш, оценка и т.д.);
    - признаки «не про позицию» (приветствие, бытовая тема).
    Если обе группы не совпали — считаем, что вопрос не про позицию,
    чтобы доска не «прилипала» к не-шахматным репликам.
    """
    text = message.lower()

    if any(k in text for k in _POSITION_KEYWORDS):
        return True

    if any(k in text for k in _NON_POSITION_KEYWORDS):
        return False

    # Словесное «что мне делать с этой позицией?» без прямых маркеров —
    # ловим вопросами про доску/прогресс.
    if any(w in text for w in ("доска", "на доске", "партию", "играю", "у меня")):
        return True

    return False


def _best_move_arrow(fen: str) -> dict | None:
    """Настоящий лучший ход Stockfish для позиции (детерминированная стрелка).

    Ответ LLM может называть несуществующие ходы, поэтому стрелка на доске
    всегда берётся из Stockfish, а не из текста модели.
    """
    stockfish = ensure_stockfish()
    if stockfish is None:
        return None

    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    try:
        with stockfish_lock:
            stockfish.set_fen_position(board.fen())
            move_uci = stockfish.get_best_move()

        if not move_uci:
            return None

        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None

        return {
            "uci": move.uci(),
            "san": board.san(move),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
        }
    except Exception:
        return None


def extract_board_fen(text: str) -> tuple[str, str | None]:
    """Извлекает из текста маркер `[ДОСКА: <FEN>]`.

    Модель добавляет его в конец ответа, когда хочет показать
    позицию на доске. Маркер вырезается из текста, а FEN валидируется
    через python-chess. Некорректный FEN игнорируется — маркер
    остаётся в тексте как есть.

    Returns:
        (текст без маркера, валидный FEN или None).
    """
    clean = text
    fen = None

    match = BOARD_MARKER.search(clean)
    if match:
        candidate = match.group(1).strip()
        try:
            fen = chess.Board(candidate).fen()
        except ValueError:
            return clean.strip(), None
        clean = clean.replace(match.group(0), "")

    return clean.strip(), fen


class IngestMessage(BaseModel):
    """Запрос: сообщение от LLM или пользователя."""
    message: str = Field(..., min_length=1, max_length=4000)
    role: str = Field(default="assistant", pattern="^(user|assistant)$")


@router.post("/api/chat/ingest")
async def ingest_comment(req: IngestMessage, db: AsyncSession = Depends(get_db)):
    """Принимает сообщение и сохраняет его для отправки в чат."""
    message = ChatMessage(role=req.role, text=sanitize_text(req.message))
    db.add(message)
    await db.commit()
    total = await db.scalar(select(func.count(ChatMessage.id)))
    return {"ok": True, "count": total}


@router.get("/api/chat/messages")
async def get_chat_messages(after: int = 0, db: AsyncSession = Depends(get_db)):
    """Отдаёт сообщения с id > `after` (для опроса фронтендом)."""
    rows = (
        await db.execute(
            select(ChatMessage)
            .where(ChatMessage.id > after)
            .order_by(ChatMessage.id.asc())
        )
    ).scalars().all()
    return {
        "messages": [
            {"id": m.id, "role": m.role, "text": m.text, "ts": m.ts}
            for m in rows
        ]
    }


@router.get("/api/chat/messages/count")
async def get_messages_count(db: AsyncSession = Depends(get_db)):
    """Возвращает текущее количество сообщений (для опроса)."""
    total = await db.scalar(select(func.count(ChatMessage.id)))
    return {"count": total}


SYSTEM_PROMPT = (
    "Ты — шахматный ассистент SFEDUCASTLING.\n"
    "Отвечай на русском языке. Кратко и по существу.\n\n"
    "Если пользователь здоровается или спрашивает не о шахматах — ответь "
    "дружелюбно и мягко направь к шахматам.\n"
    "Если вопрос о шахматах — отвечай с опорой на позицию (FEN).\n\n"
    "Когда полезно показать шахматную позицию, добавь в самый конец ответа "
    "отдельной строкой маркер доски:\n"
    "[ДОСКА: <FEN>]\n"
    "Например: [ДОСКА: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1]\n"
    "Правила использования маркера:\n"
    "- ход за ходом показывай позиции (начальная и после каждого хода);\n"
    "- перед маркером оставляй перевод строки, после него ничего не пиши;\n"
    "- не выдумывай FEN — бери только реальную позицию из контекста или "
    "производи её из уже сыгранных ходов.\n"
    "- если позиция не нужна — не добавляй маркер вовсе."
)


@router.post("/api/chat/ask")
async def chat_ask(
    req: ChatAskRequest,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_optional_current_user),
):
    """Задаёт вопрос AI-ассистенту с учётом позиции на доске.

    Вопрос пользователя и ответ ассистента сохраняются в chat_messages
    с привязкой к аккаунту (если залогинен), чтобы история не терялась.
    """
    uid = user.id if user else None

    db.add(ChatMessage(role="user", text=sanitize_text(req.message), user_id=uid))

    engine = get_gigachess()
    if engine is None:
        reply_text = "AI не подключён. Проверьте GIGACHESS_BASE_URL в .env."
    else:
        if req.is_greeting:
            user_content = req.message
        else:
            user_content = f"Текущая позиция (FEN): {req.fen}\n"
            if req.moves:
                user_content += f"История ходов: {' '.join(req.moves[-10:])}\n"
            user_content += f"Вопрос: {req.message}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # engine.chat — блокирующий HTTP-вызов; в async-эндпоинте его
            # нельзя звать напрямую, иначе он замораживает весь event loop.
            reply_text = await run_in_threadpool(
                lambda: engine.chat(
                    messages,
                    temperature=0.3 if req.is_greeting else 0.2,
                    max_tokens=300 if req.is_greeting else 700,
                )
            )
        except GigachessError as e:
            print(f"[Chat] Gigachess error: {e}")
            reply_text = "Извините, AI временно недоступен. Попробуйте позже."

    reply_text, board_fen = extract_board_fen(reply_text)

    # Если модель не приложила свою позицию — показываем активную позицию
    # игрока ТОЛЬКО когда вопрос относится к шахматной позиции на доске.
    # Это убирает «прилипание» доски к не-шахматным репликам (погода, привет).
    if (
        board_fen is None
        and not req.is_greeting
        and req.fen.strip()
        and _mentions_position(req.message)
    ):
        try:
            board_fen = chess.Board(req.fen).fen()
        except ValueError:
            board_fen = None

    db.add(ChatMessage(role="assistant", text=sanitize_text(reply_text), user_id=uid))
    await db.commit()

    arrow = None
    if board_fen is not None:
        arrow = await run_in_threadpool(_best_move_arrow, board_fen)

    return {"reply": reply_text, "fen": board_fen, "arrow": arrow}
