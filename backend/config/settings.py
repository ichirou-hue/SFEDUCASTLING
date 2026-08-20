# backend/config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Теперь загружаем .env из правильной папки
load_dotenv(dotenv_path=BASE_DIR / ".env")

print(
    "[Config] ENV GIGACHESS_BASE_URL:",
    repr(os.getenv("GIGACHESS_BASE_URL")),
)

class ServerSettings(BaseModel):
    host: str = Field("127.0.0.1", alias="SERVER_HOST")
    port: int = Field(8005, alias="SERVER_PORT")


class CorsSettings(BaseModel):
    allowed_origins: list[str] = Field(["*"], alias="ALLOWED_ORIGINS")
    allow_methods: list[str] = Field(["*"], alias="ALLOW_METHODS")
    allow_headers: list[str] = Field(["*"], alias="ALLOW_HEADERS")


class ModelsSettings(BaseModel):
    stockfish_path: Path | None = Field(None, alias="STOCKFISH_PATH")
    stockfish_depth: int = Field(20, alias="STOCKFISH_DEPTH")          # добавлено
    stockfish_top_moves: int = Field(5, alias="STOCKFISH_TOP_MOVES")
    knowledge_path: Path | None = Field(None, alias="KNOWLEDGE_PATH")
    llava_model_path: Path | None = Field(None, alias="LLAVA_MODEL_PATH")
    llava_model_id: str = Field("llava-hf/llava-1.5-7b-hf", alias="LLAVA_MODEL_ID")   # добавлено


class Maia3Settings(BaseModel):
    """Настройки движка Maia3 (человекоподобная игра по уровням)."""
    model_id: str = Field("maia3-5m", alias="MAIA3_MODEL_ID")
    device: str = Field("cpu", alias="MAIA3_DEVICE")
    default_elo: int = Field(1500, alias="MAIA3_DEFAULT_ELO")
    history: int = Field(8, alias="MAIA3_HISTORY")
    max_elo: int = Field(2600, alias="MAIA3_MAX_ELO")


class DataSettings(BaseModel):
    dataset_jsonl_path: Path = Field(BASE_DIR / "dataset.jsonl", alias="DATASET_JSONL_PATH")
    dataset_readable_path: Path = Field(BASE_DIR / "dataset_readable.json", alias="DATASET_READABLE_PATH")
    max_games_to_parse: int = Field(10, alias="MAX_GAMES_TO_PARSE")


class LichessSettings(BaseModel):      # новая группа
    explorer_url: str = Field("https://explorer.lichess.ovh/masters", alias="LICHESS_EXPLORER_URL")
    timeout: int = Field(5, alias="LICHESS_TIMEOUT")


class GigachessSettings(BaseModel):
    base_url: str = ""
    model: str = "gigachess"
    connect_timeout: float = 15.0
    read_timeout: float = 900.0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 700


class Settings(BaseSettings):
    app_name: str = Field("SFEDUCASTLING API", alias="APP_NAME")
    debug: bool = Field(False, alias="DEBUG")
    secret_key: str = Field("dev-key", alias="SECRET_KEY")
    frontend_dir: Path = Field(BASE_DIR / "frontend", alias="FRONTEND_DIR")

    server: ServerSettings = ServerSettings()
    cors: CorsSettings = CorsSettings()
    models: ModelsSettings = ModelsSettings()
    maia3: Maia3Settings = Maia3Settings()
    data: DataSettings = DataSettings()
    lichess: LichessSettings = LichessSettings()
    gigachess: GigachessSettings = GigachessSettings()


    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

env_path = BASE_DIR / ".env"


settings = Settings()

print("[Config] BASE_DIR:", BASE_DIR)
print("[Config] .env exists:", (BASE_DIR / ".env").exists())
print("[Config] Gigachess URL:", repr(settings.gigachess.base_url))
print("[Config] GIGACHESS_BASE_URL from env:", repr(os.getenv("GIGACHESS_BASE_URL")))