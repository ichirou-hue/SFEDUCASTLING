"""Endpoint'ы чата: приём комментария от LLM и выдача его во фронтенд.

Системный промт / комментарий LLM приходит на бэкенд, хранится в памяти
и затем отдаётся фронтенду, чтобы он мог вывести его в чат.
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["chat"])

# Хранилище сообщений ассистента (в памяти). Достаточно для демо.
_chat_messages: list[dict] = []


class IngestMessage(BaseModel):
    """Запрос: комментарий/системный промт от LLM.

    Attributes:
        message: Текст комментария от LLM.
    """
    message: str = Field(..., min_length=1, max_length=4000)


@router.post("/api/chat/ingest")
def ingest_comment(req: IngestMessage):
    """Принимает комментарий от LLM и сохраняет его для отправки в чат."""
    _chat_messages.append(
        {"role": "assistant", "text": req.message, "ts": time.time()}
    )
    return {"ok": True, "count": len(_chat_messages)}


@router.get("/api/chat/messages")
def get_chat_messages(after: int = 0):
    """Отдаёт сообщения ассистента, начиная с индекса `after`.

    Фронтенд поллит этот эндпойнт и добавляет новые сообщения в чат.
    """
    return {"messages": _chat_messages[after:]}