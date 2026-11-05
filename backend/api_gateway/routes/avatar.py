"""Загрузка, удаление и безопасная выдача аватаров пользователей."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_user
from backend.api_gateway.rate_limit import limiter
from backend.db.session import get_db
from backend.models.user import User
from backend.services.avatar_service import (
    avatar_path,
    delete_avatar_file,
    sanitize_avatar,
    store_avatar,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/avatars", tags=["avatars"])


@router.post("/me")
@limiter.limit("20/minute")
async def upload_my_avatar(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Принимает изображение, но сохраняет только серверный WebP после re-encode."""
    old_filename = user.avatar_filename
    new_filename: str | None = None
    try:
        payload, validation = await sanitize_avatar(file)
        new_filename = store_avatar(user.id, payload)
        user.avatar_filename = new_filename
        await db.commit()
        await db.refresh(user)
    except Exception:
        if new_filename:
            delete_avatar_file(new_filename)
        raise
    finally:
        await file.close()

    if old_filename and old_filename != new_filename:
        delete_avatar_file(old_filename)

    logger.info(
        "avatar_uploaded user_id=%s filename=%s input_format=%s dimensions=%sx%s",
        user.id,
        new_filename,
        validation["input_format"],
        validation["width"],
        validation["height"],
    )
    return {
        "ok": True,
        "user": user.public(),
        "avatar": {
            "format": "webp",
            "max_side": 512,
            "input_format": validation["input_format"],
            "width": validation["width"],
            "height": validation["height"],
        },
    }


@router.delete("/me")
@limiter.limit("20/minute")
async def delete_my_avatar(
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    old_filename = user.avatar_filename
    user.avatar_filename = None
    await db.commit()
    await db.refresh(user)
    delete_avatar_file(old_filename)
    return {"ok": True, "user": user.public()}


@router.get("/{user_id}/{avatar_key}")
async def get_avatar(
    user_id: int,
    avatar_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Публичный immutable URL. Ключ должен совпасть с текущим avatar_filename в БД."""
    user = await db.get(User, user_id)
    if user is None or not user.avatar_filename or avatar_key != user.avatar_filename:
        raise HTTPException(status_code=404, detail="Аватар не найден")

    path = avatar_path(user.avatar_filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Аватар не найден")

    return FileResponse(
        path,
        media_type="image/webp",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )
