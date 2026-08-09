"""Endpoint'ы чата: приём комментариев от LLM/пользователя и выдача их во фронтенд.

Сообщения хранятся в памяти и опрашиваются фронтендом через /api/chat/messages.
"""

import re
import time
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["chat"])

_chat_messages: list[dict] = []

_TAG_RE = re.compile(r"<[^>]*>")


def _sanitize(text: str) -> str:
    """Убирает HTML-теги и экранирует спецсимволы."""
    text = _TAG_RE.sub("", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text[:4000]

_chat_messages: list[dict] = []


class IngestMessage(BaseModel):
    """Запрос: сообщение от LLM или пользователя."""
    message: str = Field(..., min_length=1, max_length=4000)
    role: str = Field(default="assistant", pattern="^(user|assistant)$")


@router.post("/api/chat/ingest")
def ingest_comment(req: IngestMessage):
    """Принимает сообщение и сохраняет его для отправки в чат."""
    _chat_messages.append(
        {"role": req.role, "text": _sanitize(req.message), "ts": time.time()}
    )
    return {"ok": True, "count": len(_chat_messages)}


@router.get("/api/chat/messages")
def get_chat_messages(after: int = 0):
    """Отдаёт сообщения, начиная с индекса `after`."""
    return {"messages": _chat_messages[after:]}


@router.get("/api/chat/messages/count")
def get_messages_count():
    """Возвращает текущее количество сообщений (для опроса)."""
    return {"count": len(_chat_messages)}
