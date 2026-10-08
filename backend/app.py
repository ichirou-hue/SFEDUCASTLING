"""SFEDUCASTLING API — точка входа в приложение.

Инициализирует общее состояние (ML-модели, движки) и подключает
все модули маршрутов (routes).
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException


from backend.api_gateway.state import (
    load_stockfish,
    load_knowledge,
    load_puzzles,
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
from backend.api_gateway.routes.chess_profile import router as chess_profile_router
from backend.api_gateway.routes.explanation import router as explanation_router
from backend.api_gateway.routes.auth import router as auth_router
from backend.api_gateway.routes.learning import router as learning_router
from backend.api_gateway.routes.training import router as training_router
from backend.api_gateway.routes.admin import router as admin_router

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
app.include_router(chess_profile_router)
app.include_router(explanation_router)
app.include_router(auth_router)
app.include_router(learning_router)
app.include_router(training_router)
app.include_router(admin_router)

class SPAStaticFiles(StaticFiles):
    """StaticFiles с SPA-fallback: при 404 отдаёт index.html.

    Без этого клиентские маршруты SPA (/profile, /training, /puzzles)
    после перезагрузки или кнопки «назад» падают с 404. Пути /api/*,
    не найденные роутерами, по-прежнему дают 404 (а не index.html).

    StaticFiles.get_response отсутствующий файл не возвращает, а ПОДНИМАЕТ
    HTTPException(404) — поэтому fallback обязан ловить исключение.
    Отсекаем /api/* по scope["path"]: аргумент `path` внутри StaticFiles
    на Windows может содержать обратные слэши.
    """

    async def get_response(self, path: str, scope):
        is_api = scope["path"].startswith("/api")
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or is_api:
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404 and not is_api:
            return await super().get_response("index.html", scope)
        return response


# Раздаём статику фронтенда (собранный React в frontend/dist/)
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
app.mount(
    "/",
    SPAStaticFiles(directory=frontend_dist, html=True),
    name="frontend",
)

load_stockfish()
load_knowledge()
load_puzzles()
print("Инициализация LLaVA...")
load_llava()


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_maia3()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
