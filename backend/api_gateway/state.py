"""Общее состояние приложения для SFEDUCASTLING API.

Хранит singleton-экземпляры ML-моделей, шахматных движков
и клиентов внешних сервисов.

Особое внимание уделено Stockfish:
- lazy loading;
- проверка жив ли процесс;
- автоматический restart после падения;
- защита от конкурентного доступа;
- безопасное завершение процесса.
"""

import json
import os
import subprocess
import sys
import threading

import requests


# =============================================================
# MODEL MANAGER
# =============================================================

class ModelManager:
    """Менеджер ленивой загрузки моделей и шахматных движков."""

    def __init__(self):
        self._stockfish = None
        self._stockfish_loaded = False
        self._sf_loading_lock = threading.RLock()

    # =========================================================
    # STOCKFISH
    # =========================================================

    @staticmethod
    def _stockfish_path() -> str:
        """Возвращает абсолютный путь к Stockfish."""

        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "stockfish_engine",
                "stockfish.exe",
            )
        )

    @staticmethod
    def _process_alive(engine) -> bool:
        """Проверяет, жив ли внутренний процесс Stockfish.

        python-stockfish хранит subprocess обычно в `_stockfish`.
        Если конкретная версия библиотеки устроена иначе,
        отсутствие информации о процессе не считается ошибкой.
        """

        if engine is None:
            return False

        try:
            process = getattr(
                engine,
                "_stockfish",
                None,
            )

            if process is None:
                # Некоторые версии python-stockfish
                # не предоставляют subprocess наружу.
                return True

            poll = getattr(
                process,
                "poll",
                None,
            )

            if callable(poll):
                return poll() is None

            return True

        except Exception:
            return False

    def _kill_engine(self, engine):
        """Безопасно завершает экземпляр Stockfish."""

        if engine is None:
            return

        # Сначала пробуем штатный quit().
        try:
            quit_method = getattr(
                engine,
                "send_quit_command",
                None,
            )

            if callable(quit_method):
                quit_method()
        except Exception:
            pass

        try:
            quit_method = getattr(
                engine,
                "quit",
                None,
            )

            if callable(quit_method):
                quit_method()
        except Exception:
            pass

        # Затем добиваем subprocess, если он остался.
        try:
            process = getattr(
                engine,
                "_stockfish",
                None,
            )

            if process is not None:

                try:
                    if process.poll() is None:
                        process.terminate()
                except Exception:
                    pass

                try:
                    if process.poll() is None:
                        process.kill()
                except Exception:
                    pass

        except Exception:
            pass

    def get_stockfish(self):
        """Возвращает живой экземпляр Stockfish.

        Если процесс движка умер, старый экземпляр автоматически
        сбрасывается и создаётся новый.
        """

        with self._sf_loading_lock:

            # -------------------------------------------------
            # Уже существует
            # -------------------------------------------------

            if self._stockfish is not None:

                if self._process_alive(
                    self._stockfish
                ):
                    return self._stockfish

                print(
                    "[ModelManager] "
                    "Stockfish process is dead. "
                    "Restarting..."
                )

                self._kill_engine(
                    self._stockfish
                )

                self._stockfish = None
                self._stockfish_loaded = False

            # -------------------------------------------------
            # Загружаем
            # -------------------------------------------------

            print(
                "[ModelManager] "
                "Загрузка Stockfish..."
            )

            self._load_stockfish()

            if self._stockfish is not None:

                self._stockfish_loaded = True

                print(
                    "[ModelManager] "
                    "Stockfish ready."
                )

            else:

                self._stockfish_loaded = False

                print(
                    "[ModelManager] "
                    "Stockfish unavailable."
                )

            return self._stockfish

    def _load_stockfish(self):
        """Создаёт новый процесс Stockfish."""

        self._stockfish = None

        try:

            from stockfish import Stockfish

            path = self._stockfish_path()

            if not os.path.exists(path):

                print(
                    "[ModelManager] "
                    f"Stockfish not found: {path}"
                )

                return

            print(
                "[ModelManager] "
                f"Stockfish path: {path}"
            )

            # -------------------------------------------------
            # Параметры
            # -------------------------------------------------

            parameters = {
                "UCI_LimitStrength": False,
                "Skill Level": 20,

                # Для API лучше начинать с 1 потока.
                # Это сильно уменьшает вероятность падения
                # нескольких параллельных процессов.
                "Threads": 1,

                "Hash": 128,

                # MultiPV задаётся также перед анализом.
                "MultiPV": 5,
            }

            # -------------------------------------------------
            # Создание процесса
            # -------------------------------------------------

            engine = Stockfish(
                path=path,
                depth=18,
                parameters=parameters,
            )

            # Проверяем, что процесс действительно жив.
            if not self._process_alive(engine):

                print(
                    "[ModelManager] "
                    "Stockfish started but process is dead."
                )

                self._kill_engine(engine)

                return

            self._stockfish = engine

            print(
                "[ModelManager] "
                f"Stockfish loaded: {path}"
            )

            # -------------------------------------------------
            # Диагностика параметров
            # -------------------------------------------------

            try:

                print(
                    "[ModelManager] "
                    f"Stockfish parameters: "
                    f"{engine.get_parameters()}"
                )

            except Exception:
                pass

        except Exception as e:

            self._stockfish = None

            print(
                "[ModelManager] "
                f"Stockfish load error: {e}"
            )

    def reset_stockfish(self):
        """Полностью останавливает и сбрасывает Stockfish."""

        with self._sf_loading_lock:

            engine = self._stockfish

            self._stockfish = None
            self._stockfish_loaded = False

            if engine is not None:

                self._kill_engine(engine)

            print(
                "[ModelManager] "
                "Stockfish reset."
            )

    def restart_stockfish(self):
        """Принудительно перезапускает Stockfish."""

        with self._sf_loading_lock:

            engine = self._stockfish

            self._stockfish = None
            self._stockfish_loaded = False

            if engine is not None:
                self._kill_engine(engine)

            print(
                "[ModelManager] "
                "Restarting Stockfish..."
            )

            self._load_stockfish()

            if self._stockfish is not None:

                self._stockfish_loaded = True

                print(
                    "[ModelManager] "
                    "Stockfish restart successful."
                )

                return self._stockfish

            print(
                "[ModelManager] "
                "Stockfish restart failed."
            )

            return None


# =============================================================
# GLOBAL MODEL MANAGER
# =============================================================

manager = ModelManager()


# =============================================================
# STOCKFISH LOCK
# =============================================================

# ВАЖНО:
# Один экземпляр Stockfish нельзя одновременно использовать
# несколькими HTTP-запросами.
stockfish_lock = threading.RLock()


def ensure_stockfish():
    """Возвращает живой Stockfish или None."""

    return manager.get_stockfish()


def reset_stockfish():
    """Сбрасывает Stockfish."""

    manager.reset_stockfish()


def restart_stockfish():
    """Перезапускает Stockfish."""

    return manager.restart_stockfish()


def load_stockfish():
    """Принудительно загружает Stockfish."""

    return manager.get_stockfish() is not None


# =============================================================
# MAIA3
# =============================================================

_maia3_engine = None

_maia3_lock = threading.Lock()

_maia3_loading_lock = threading.Lock()


def get_maia3():
    """Возвращает UCI-движок Maia3 или None."""

    global _maia3_engine

    if _maia3_engine is not None:
        return _maia3_engine

    with _maia3_loading_lock:

        if _maia3_engine is not None:
            return _maia3_engine

        try:

            import chess.engine

            from backend.config.settings import settings

            cmd = [
                sys.executable,
                "-m",
                "maia3.uci",
                "--model",
                settings.maia3.model_id,
                "--device",
                settings.maia3.device,
                "--use-uci-history",
                "--history",
                str(settings.maia3.history),
                "--elo",
                str(settings.maia3.default_elo),
            ]

            _maia3_engine = (
                chess.engine.SimpleEngine.popen_uci(
                    cmd,
                    setpgrp=True,
                    stderr=(
                        subprocess.DEVNULL
                        if os.name == "nt"
                        else None
                    ),
                )
            )

            print(
                "[Maia3] ready: "
                f"{settings.maia3.model_id} "
                f"({settings.maia3.device})"
            )

        except Exception as e:

            print(
                "[Maia3] "
                f"Error loading Maia3: {e}"
            )

            _maia3_engine = None

    return _maia3_engine


def shutdown_maia3():
    """Останавливает UCI-процесс Maia3."""

    global _maia3_engine

    with _maia3_loading_lock:

        if _maia3_engine is not None:

            try:
                _maia3_engine.quit()
            except Exception:
                pass

            _maia3_engine = None


def maia3_lock():
    """Возвращает mutex Maia3."""

    return _maia3_lock


def elo_to_temperature(elo: int) -> float:
    """Сопоставляет Elo с температурой Maia3."""

    if elo >= 2400:
        return 0.0

    if elo >= 2000:
        return 0.3

    if elo >= 1700:
        return 0.6

    if elo >= 1400:
        return 0.9

    return 1.2


# =============================================================
# GIGACHESS
# =============================================================

_gigachess_client = None

_gigachess_lock = threading.Lock()


def get_gigachess():
    """Возвращает singleton-клиент Gigachess."""

    global _gigachess_client

    if _gigachess_client is not None:
        return _gigachess_client

    with _gigachess_lock:

        if _gigachess_client is not None:
            return _gigachess_client

        try:

            from backend.config.settings import settings
            from backend.llm.gigachess import GigachessClient

            if not settings.gigachess.base_url:

                print(
                    "[Gigachess] "
                    "GIGACHESS_BASE_URL не задан"
                )

                return None

            _gigachess_client = GigachessClient(
                base_url=settings.gigachess.base_url,
                model=settings.gigachess.model,
                connect_timeout=settings.gigachess.connect_timeout,
                read_timeout=settings.gigachess.read_timeout,
            )

            print(
                "[Gigachess] Client ready: "
                f"{settings.gigachess.base_url}"
            )

            return _gigachess_client

        except Exception as e:

            print(
                "[Gigachess] "
                f"Ошибка инициализации: {e}"
            )

            return None


# =============================================================
# LLAVA
# =============================================================

llava_model = None


# =============================================================
# KNOWLEDGE BASE
# =============================================================

knowledge_base = None

KNOWLEDGE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "knowledge",
        "openings.json",
    )
)


def load_knowledge():
    """Загружает базу дебютов из JSON."""

    global knowledge_base

    try:

        if not os.path.exists(KNOWLEDGE_PATH):

            print(
                "[Knowledge] "
                f"База знаний не найдена: "
                f"{KNOWLEDGE_PATH}"
            )

            return False

        with open(
            KNOWLEDGE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            knowledge_base = json.load(f)

        count = len(
            knowledge_base.get(
                "openings",
                [],
            )
        )

        print(
            "[Knowledge] "
            f"База знаний загружена: "
            f"{count} дебютов"
        )

        return True

    except Exception as e:

        print(
            "[Knowledge] "
            f"Ошибка загрузки базы знаний: {e}"
        )

        return False


# =============================================================
# LLAVA
# =============================================================

def load_llava():
    """Загрузчик LLaVA.

    Пока отключён.
    """

    return False


def extract_fen_from_image(
    image_path: str,
) -> str:
    """Распознаёт шахматную позицию с изображения через LLaVA."""

    from PIL import Image

    global llava_model

    if llava_model is None:

        if not load_llava():

            return (
                "ERROR: LLaVA не загружена."
            )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        prompt = """
Опишите эту шахматную позицию.
Выведите ТОЛЬКО FEN-нотацию для показанной позиции.

Пример:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1
""".strip()

        outputs = llava_model(
            image,
            prompt=prompt,
            generate_kwargs={
                "max_new_tokens": 200,
            },
        )

        response = outputs[0][
            "generated_text"
        ]

        if "ERROR" in response.upper():

            return (
                "ERROR: "
                "не удалось прочитать доску"
            )

        lines = response.strip().split(
            "\n"
        )

        for line in lines:

            line = line.strip()

            if "/" not in line:
                continue

            if len(line) <= 20:
                continue

            parts = line.split()

            if (
                len(parts) >= 1
                and "/" in parts[0]
                and parts[0].count("/") == 7
            ):

                return parts[0]

        return (
            "ERROR: неожиданный ответ: "
            f"{response[:100]}"
        )

    except Exception as e:

        return (
            f"ERROR: {str(e)}"
        )


# =============================================================
# LICHESS MASTERS
# =============================================================

def get_opening_info(
    fen: str,
) -> dict | None:
    """Получает информацию о дебюте."""

    try:

        resp = requests.get(
            "https://explorer.lichess.ovh/masters",
            params={
                "fen": fen,
            },
            timeout=5,
        )

        if resp.status_code != 200:
            return None

        data = resp.json()

        info = {}

        if data.get("opening"):

            info["name"] = (
                data["opening"].get(
                    "name",
                    "",
                )
            )

        top_games = (
            data.get(
                "topGames",
                [],
            )[:3]
        )

        if top_games:

            games = []

            for game in top_games:

                white = (
                    game.get(
                        "white",
                        {},
                    ).get(
                        "name",
                        "?",
                    )
                )

                black = (
                    game.get(
                        "black",
                        {},
                    ).get(
                        "name",
                        "?",
                    )
                )

                year = game.get(
                    "year",
                    "?",
                )

                winner = game.get(
                    "winner",
                    "draw",
                )

                games.append(
                    f"{white} — {black}, "
                    f"{year} ({winner})"
                )

            info["games"] = games

        return (
            info
            if info
            else None
        )

    except Exception:

        return None