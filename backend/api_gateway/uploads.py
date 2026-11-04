"""Ограничение размера загружаемых файлов."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

from backend.config.settings import settings

_CHUNK_SIZE = 1024 * 1024


async def read_upload_limited(file: UploadFile, max_bytes: int | None = None) -> bytes:
    """Читает UploadFile, не позволяя превысить лимит.

    Раньше ``await file.read()`` читал файл целиком: один большой запрос мог
    съесть память процесса. Теперь читаем порциями и отдаём 413.
    """
    limit = settings.security.max_upload_bytes if max_bytes is None else max_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Файл больше допустимого размера ({limit} байт)",
            )
        chunks.append(chunk)
    return b"".join(chunks)
