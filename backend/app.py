"""SFEDUCASTLING API — точка входа в приложение.

Инициализирует общее состояние (ML-модели, движки) и подключает
все модули маршрутов (routes).
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException
from starlette.requests import Request

from backend.api_gateway import windows_accept
from backend.api_gateway.rate_limit import limiter

# Windows: клиент, оборвавший соединение во время accept (закрытая
# вкладка, отменённый запрос), на Python 3.14 закрывал слушающий сокет,
# и сервер переставал принимать новые подключения до перезапуска.
# Патч должен стоять до того, как uvicorn создаст сервер.
windows_accept.install()


from backend.api_gateway.state import (
    load_stockfish,
    load_knowledge,
    load_puzzles,
    load_llava,
    preload_maia3,
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
from backend.api_gateway.routes.chess_oauth import router as chess_oauth_router
from backend.api_gateway.routes.learning import router as learning_router
from backend.api_gateway.routes.training import router as training_router
from backend.api_gateway.routes.admin import router as admin_router
from backend.api_gateway.routes.health import router as health_router
from backend.api_gateway.routes.gigacoins import router as gigacoins_router

app = FastAPI(title="SFEDUCASTLING API")

# slowapi: жёсткий лимит частоты запросов (см. backend/api_gateway/rate_limit.py).
# Ключ — IP клиента; 429 уже содержит Retry-After благодаря headers_enabled.
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Единый формат ошибки с Retry-After.

    Тело в стиле остального API — {"detail": ...}, а не slowapi-шный
    {"error": ...}: фронтенд показывает именно detail. Заголовки (Retry-After,
    X-RateLimit-*) забираем у штатного обработчика slowapi.
    """
    default_response = _rate_limit_exceeded_handler(request, exc)
    headers = {
        name: value
        for name, value in default_response.headers.items()
        if name.lower() == "retry-after" or name.lower().startswith("x-ratelimit")
    }
    try:
        # slowapi может отдать "0", если окно закрывается в эту же секунду.
        # Клиенту всё равно нужно подождать хотя бы секунду.
        retry_after = max(1, int(float(headers.get("Retry-After", "1"))))
    except (TypeError, ValueError):
        retry_after = 1
    headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Слишком много попыток входа. "
                f"Повторите через {retry_after} с."
            )
        },
        headers=headers,
    )


# Разрешаем запросы с любых источников (для разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Без этого браузер скрывает Retry-After от JS на другом origin,
    # и форма входа не сможет показать обратный отсчёт.
    expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
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
app.include_router(chess_oauth_router)
app.include_router(learning_router)
app.include_router(training_router)
app.include_router(admin_router)
app.include_router(health_router)
app.include_router(gigacoins_router)

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


@app.on_event("startup")
async def startup_event():
    # Глушим трейсбеки от оборванных клиентов (WinError 64 при accept).
    windows_accept.quiet_loop_noise()

    # Maia3 стартует долго (импорт torch + загрузка чекпоинта), поэтому
    # греем её заранее и в фоне: иначе эту задержку оплачивает первый ход
    # после запуска сервера. MAIA3_PRELOAD=0 отключает прогрев (тесты).
    if os.getenv("MAIA3_PRELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}:
        preload_maia3()


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_maia3()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
