"""SFEDUCASTLING API — точка входа в приложение.

Инициализирует общее состояние (ML-модели, движки) и подключает
все модули маршрутов (routes).
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Импорт настроек
from backend.config.settings import settings

# Импорт из state (уже без GIGACHAT_AUTH_KEY)
from backend.api_gateway.state import (
    load_maia2,
    load_stockfish,
    load_knowledge,
    load_llava,
)

from backend.api_gateway.routes.game import router as game_router
from backend.api_gateway.routes.analysis import router as analysis_router
from backend.api_gateway.routes.knowledge import router as knowledge_router
from backend.api_gateway.routes.data import router as data_router
from backend.api_gateway.routes.vision import router as vision_router

app = FastAPI(title=settings.app_name)

# Разрешаем запросы с любых источников (для разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Раздаём статику фронтенда
frontend_path = settings.frontend_dir
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
def serve_index():
    """Отдаёт index.html фронтенда."""
    return FileResponse(str(frontend_path / "index.html"))


# Подключаем маршруты, разбитые по доменам
app.include_router(game_router)
app.include_router(analysis_router)
app.include_router(knowledge_router)
app.include_router(data_router)
app.include_router(vision_router)


# Загружаем зависимости. Если модуль не установлен — сообщаем и работаем в fallback-режиме.
if settings.gigachat.auth_key:
    print("GigaChat API ключ загружен.")
else:
    print("GigaChat API ключ не найден! Добавьте GIGACHAT_AUTH_KEY в .env")

load_maia2()
load_stockfish()
load_knowledge()
print("Инициализация LLaVA...")
load_llava()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)