"""Endpoint'ы чата: приём комментариев от LLM/пользователя и выдача их во фронтенд.

Сообщения хранятся в PostgreSQL (таблица chat_messages) и опрашиваются
фронтендом через /api/chat/messages?after=<id>.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.sanitize import sanitize_text
from backend.db.session import get_db
from backend.models.chat_message import ChatMessage

router = APIRouter(tags=["chat"])


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
