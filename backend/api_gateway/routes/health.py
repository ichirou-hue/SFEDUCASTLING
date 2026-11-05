"""Readiness endpoint for load balancers and deployment checks."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from backend.api_gateway.state import ensure_stockfish, stockfish_lock

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

    if not await run_in_threadpool(stockfish_ready):
        raise HTTPException(status_code=503, detail="Stockfish unavailable")

    manifest = Path(__file__).resolve().parents[3] / "release.json"
    result = {"status": "ok"}
    if manifest.is_file():
        result["sha"] = json.loads(manifest.read_text())["sha"]
    return result


def stockfish_ready() -> bool:
    """Verify the real UCI process responds without changing game position."""
    try:
        with stockfish_lock:
            stockfish = ensure_stockfish()
            return stockfish is not None and bool(stockfish.get_fen_position())
    except Exception:
        return False
