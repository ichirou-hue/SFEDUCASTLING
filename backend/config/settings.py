# backend/config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Секрет по умолчанию. Приложение в production обязано стартовать только
# с собственным JWT_SECRET_KEY (см. backend/api_gateway/config_guard.py).
DEFAULT_JWT_SECRET = "dev-key-change-me"

# Источники, разрешённые в development. В production этот список нужно
# заменить реальными доменами: config_guard считает локальные адреса ошибкой.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8005",
    "http://127.0.0.1:8005",
)

# Теперь загружаем .env из правильной папки
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _env_config(**overrides: object) -> SettingsConfigDict:
    """Единые настройки чтения .env для вложенных конфигураций."""
    return SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        **overrides,
    )


print(
    "[Config] ENV GIGACHESS_BASE_URL:",
    repr(os.getenv("GIGACHESS_BASE_URL")),
)


class ServerSettings(BaseSettings):
    model_config = _env_config()

    host: str = Field("127.0.0.1", alias="SERVER_HOST")
    port: int = Field(8005, alias="SERVER_PORT")


class CorsSettings(BaseSettings):
    """CORS. По умолчанию разрешены только локальные адреса разработки.

    ``ALLOWED_ORIGINS`` принимает JSON-массив, например:
    ``ALLOWED_ORIGINS=["https://gigachess.ru"]``.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    allowed_origins: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CORS_ORIGINS),
        alias="ALLOWED_ORIGINS",
    )
    allow_methods: list[str] = Field(["*"], alias="ALLOW_METHODS")
    allow_headers: list[str] = Field(["*"], alias="ALLOW_HEADERS")
    allow_credentials: bool = Field(False, alias="ALLOW_CREDENTIALS")


class SecuritySettings(BaseSettings):
    """Заголовки безопасности, доверенные хосты и лимиты загрузки."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Пусто = проверка Host выключена в development и обязательна в production.
    trusted_hosts: list[str] = Field(default_factory=list, alias="TRUSTED_HOSTS")
    max_upload_bytes: int = Field(10 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    security_headers_enabled: bool = Field(True, alias="SECURITY_HEADERS_ENABLED")
    hsts_max_age: int = Field(31_536_000, alias="HSTS_MAX_AGE")
    content_security_policy: str = Field(
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "object-src 'none'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; "
        "font-src 'self' data:; connect-src 'self' https: wss:; "
        "manifest-src 'self'; form-action 'self'",
        alias="CONTENT_SECURITY_POLICY",
    )


class RateLimitSettings(BaseSettings):
    """Простые in-memory лимиты частоты (один процесс uvicorn).

    Значения задаются в запросах за окно. Для нескольких воркеров нужен
    внешний лимитер (Redis), здесь сознательно оставлен минимальный вариант.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(True, alias="RATE_LIMIT_ENABLED")
    login_attempts: int = Field(5, alias="LOGIN_RATE_LIMIT_ATTEMPTS")
    login_window_sec: int = Field(60, alias="LOGIN_RATE_LIMIT_WINDOW_SEC")
    auth_attempts: int = Field(20, alias="AUTH_RATE_LIMIT_ATTEMPTS")
    auth_window_sec: int = Field(60, alias="AUTH_RATE_LIMIT_WINDOW_SEC")
    chat_attempts: int = Field(30, alias="CHAT_RATE_LIMIT_ATTEMPTS")
    chat_window_sec: int = Field(60, alias="CHAT_RATE_LIMIT_WINDOW_SEC")
    expensive_attempts: int = Field(10, alias="EXPENSIVE_RATE_LIMIT_ATTEMPTS")
    expensive_window_sec: int = Field(60, alias="EXPENSIVE_RATE_LIMIT_WINDOW_SEC")


class ModelsSettings(BaseSettings):
    model_config = _env_config()

    stockfish_path: Path | None = Field(None, alias="STOCKFISH_PATH")
    stockfish_depth: int = Field(20, alias="STOCKFISH_DEPTH")
    stockfish_top_moves: int = Field(5, alias="STOCKFISH_TOP_MOVES")
    knowledge_path: Path | None = Field(None, alias="KNOWLEDGE_PATH")
    llava_model_path: Path | None = Field(None, alias="LLAVA_MODEL_PATH")
    llava_model_id: str = Field("llava-hf/llava-1.5-7b-hf", alias="LLAVA_MODEL_ID")

    @field_validator("stockfish_path", "knowledge_path", "llava_model_path", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class Maia3Settings(BaseSettings):
    """Настройки движка Maia3 (человекоподобная игра по уровням)."""

    model_config = _env_config()

    model_id: str = Field("maia3-5m", alias="MAIA3_MODEL_ID")
    device: str = Field("cpu", alias="MAIA3_DEVICE")
    default_elo: int = Field(1500, alias="MAIA3_DEFAULT_ELO")
    history: int = Field(8, alias="MAIA3_HISTORY")
    max_elo: int = Field(2600, alias="MAIA3_MAX_ELO")


class DataSettings(BaseSettings):
    model_config = _env_config()

    dataset_jsonl_path: Path = Field(BASE_DIR / "dataset.jsonl", alias="DATASET_JSONL_PATH")
    dataset_readable_path: Path = Field(BASE_DIR / "dataset_readable.json", alias="DATASET_READABLE_PATH")
    max_games_to_parse: int = Field(10, alias="MAX_GAMES_TO_PARSE")


class LichessSettings(BaseSettings):
    model_config = _env_config()

    explorer_url: str = Field("https://explorer.lichess.ovh/masters", alias="LICHESS_EXPLORER_URL")
    timeout: int = Field(5, alias="LICHESS_TIMEOUT")


class GigachessSettings(BaseSettings):
    model_config = _env_config(env_prefix="GIGACHESS_")

    base_url: str = ""
    model: str = "gigachess"
    connect_timeout: float = 15.0
    read_timeout: float = 900.0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 700


class DatabaseSettings(BaseSettings):
    """Настройки подключения к базе данных (async SQLAlchemy).

    URL задаётся через переменную окружения DATABASE_URL (или .env).
    По умолчанию — локальный SQLite в корне проекта, чтобы приложение
    запускалось «из коробки» и на Windows, и на Linux, без PostgreSQL.
    Для PostgreSQL используйте:
        postgresql+asyncpg://user:password@127.0.0.1:5432/dbname
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    url: str = Field(
        default=f"sqlite+aiosqlite:///{BASE_DIR / 'backend.db'}",
        alias="DATABASE_URL",
    )
    echo: bool = Field(False, alias="DATABASE_ECHO")
    pool_size: int = Field(5, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(10, alias="DATABASE_MAX_OVERFLOW")


class AuthSettings(BaseSettings):
    """JWT и долговременные пользовательские сессии.

    Access-токен живёт 7 дней, refresh-токен — 90 дней. При успешном
    refresh выдаётся новая пара токенов, поэтому активная сессия скользящая:
    пока пользователь возвращается в приложение хотя бы раз в 90 дней,
    повторный ввод пароля не требуется.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    jwt_secret: str = Field(DEFAULT_JWT_SECRET, alias="JWT_SECRET_KEY")
    access_token_ttl_min: int = Field(60 * 24 * 7, alias="ACCESS_TOKEN_TTL_MIN")  # 7 суток
    refresh_token_ttl_days: int = Field(90, alias="REFRESH_TOKEN_TTL_DAYS")

    @property
    def uses_default_secret(self) -> bool:
        return self.jwt_secret.strip() in {"", DEFAULT_JWT_SECRET}

    @property
    def secret_is_weak(self) -> bool:
        return self.uses_default_secret or len(self.jwt_secret.strip()) < 32




class ChessOAuthSettings(BaseSettings):
    """OAuth для Lichess/Chess.com и возврат в SPA.

    Lichess работает как public PKCE client и не требует client_secret.
    Chess.com OAuth включается только после заполнения выданных Chess.com
    реквизитов и URL; до этого доступна публичная привязка по username.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    public_base_url: str = Field("", alias="OAUTH_PUBLIC_BASE_URL")
    frontend_base_url: str = Field("", alias="FRONTEND_BASE_URL")
    # Отдельный секрет для подписи OAuth-кук. Пусто = fallback на jwt_secret
    # только вне production; в production значение обязательно.
    cookie_secret: str = Field("", alias="OAUTH_COOKIE_SECRET")
    cookie_secure: bool = Field(False, alias="OAUTH_COOKIE_SECURE")
    flow_ttl_sec: int = Field(600, alias="OAUTH_FLOW_TTL_SEC")
    http_timeout: float = Field(10.0, alias="OAUTH_HTTP_TIMEOUT")
    user_agent: str = Field(
        "GigaChess/0.3 (OAuth linked chess account)",
        alias="CHESS_API_USER_AGENT",
    )

    lichess_client_id: str = Field("gigachess", alias="LICHESS_OAUTH_CLIENT_ID")

    # Chess.com выдаёт эти параметры после одобрения OAuth-доступа.
    chesscom_client_id: str = Field("", alias="CHESSCOM_OAUTH_CLIENT_ID")
    chesscom_client_secret: str = Field("", alias="CHESSCOM_OAUTH_CLIENT_SECRET")
    chesscom_authorize_url: str = Field("", alias="CHESSCOM_OAUTH_AUTHORIZE_URL")
    chesscom_token_url: str = Field("", alias="CHESSCOM_OAUTH_TOKEN_URL")
    chesscom_userinfo_url: str = Field("", alias="CHESSCOM_OAUTH_USERINFO_URL")
    chesscom_scope: str = Field("", alias="CHESSCOM_OAUTH_SCOPE")
    chesscom_use_pkce: bool = Field(True, alias="CHESSCOM_OAUTH_USE_PKCE")
    chesscom_token_auth_method: str = Field(
        "client_secret_post", alias="CHESSCOM_OAUTH_TOKEN_AUTH_METHOD"
    )

    @property
    def uses_default_cookie_secret(self) -> bool:
        return not self.cookie_secret.strip()

    def cookie_signing_secret(self) -> str:
        """Секрет для OAuth-кук.

        Отдельный секрет не даёт подделать OAuth-куку, даже если JWT-секрет
        утёк. В production его наличие обязательно (см. config_guard).
        """
        configured = self.cookie_secret.strip()
        if configured:
            return configured
        return settings.auth.jwt_secret


class Settings(BaseSettings):
    app_name: str = Field("SFEDUCASTLING API", alias="APP_NAME")
    app_env: str = Field("development", alias="APP_ENV")
    debug: bool = Field(False, alias="DEBUG")
    frontend_dir: Path = Field(BASE_DIR / "frontend", alias="FRONTEND_DIR")

    server: ServerSettings = ServerSettings()
    cors: CorsSettings = CorsSettings()
    security: SecuritySettings = SecuritySettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    models: ModelsSettings = ModelsSettings()
    maia3: Maia3Settings = Maia3Settings()
    data: DataSettings = DataSettings()

    lichess: LichessSettings = LichessSettings()
    gigachess: GigachessSettings = GigachessSettings()
    database: DatabaseSettings = DatabaseSettings()
    auth: AuthSettings = AuthSettings()
    chess_oauth: ChessOAuthSettings = ChessOAuthSettings()

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"production", "prod"}

    @property
    def docs_enabled(self) -> bool:
        """Swagger/Redoc выключены в production."""
        return not self.is_production


env_path = BASE_DIR / ".env"
settings = Settings()

print("[Config] BASE_DIR:", BASE_DIR)
print("[Config] .env exists:", (BASE_DIR / ".env").exists())
print("[Config] Gigachess URL:", repr(settings.gigachess.base_url))
print("[Config] GIGACHESS_BASE_URL from env:", repr(os.getenv("GIGACHESS_BASE_URL")))
