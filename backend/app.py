"""SFEDUCASTLING API — точка входа в приложение.

Инициализирует общее состояние (ML-модели, движки) и подключает
все модули маршрутов (routes).
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from backend.api_gateway.state import (
    load_stockfish,
    load_knowledge,
    load_llava,
    shutdown_maia3,
)

from backend.api_gateway.routes.game import router as game_router
from backend.api_gateway.routes.analysis import router as analysis_router
from backend.api_gateway.routes.knowledge import router as knowledge_router
from backend.api_gateway.routes.data import router as data_router
from backend.api_gateway.routes.chat import router as chat_router
from backend.api_gateway.routes.vision import router as vision_router
from backend.api_gateway.routes.analyze import router as analyze_router

app = FastAPI(title="SFEDUCASTLING API")

# Разрешаем запросы с любых источников (для разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем маршруты, разбитые по доменам
app.include_router(game_router)
app.include_router(analysis_router)
app.include_router(knowledge_router)
app.include_router(data_router)
app.include_router(chat_router)
app.include_router(vision_router)
app.include_router(analyze_router)

# Раздаём статику фронтенда (собранный React в frontend/dist/)
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

load_stockfish()
load_knowledge()
print("Инициализация LLaVA...")
load_llava()


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_maia3()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
