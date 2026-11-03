"""Readiness endpoint for load balancers and deployment checks."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.db.session import engine

router = APIRouter(tags=["health"])


@router.get("/api/health", include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Return ready only when the application can reach its database."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    return {"status": "ok"}
