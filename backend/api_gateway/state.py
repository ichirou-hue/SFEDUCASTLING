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
import shutil
import subprocess
import sys
import threading
from pathlib import Path


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
        """Возвращает абсолютный путь к Stockfish.

        Приоритет:
        1. STOCKFISH_PATH (settings.models.stockfish_path);
        2. бинарник из PATH (shutil.which("stockfish"));
        3. stockfish_engine/stockfish рядом с проектом (Linux);
        4. legacy stockfish_engine/stockfish.exe (Windows).

        Возвращает первый существующий путь, иначе — кандидат №3,
        чтобы лог указывал на актуальное место, а не на legacy .exe.
        """

        candidates = []

        try:
            from backend.config.settings import settings

            configured = settings.models.stockfish_path
            if configured:
                candidates.append(
                    os.path.abspath(
                        os.path.expanduser(str(configured))
                    )
                )
        except Exception:
            pass

        which_path = shutil.which("stockfish")
        if which_path:
            candidates.append(os.path.abspath(which_path))

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        candidates.append(
            os.path.join(repo_root, "stockfish_engine", "stockfish")
        )
        candidates.append(
            os.path.join(repo_root, "stockfish_engine", "stockfish.exe")
        )

        for path in candidates:
            if path and os.path.exists(path):
                return path

        return candidates[-2]

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

                # MultiPV = 1: get_evaluation() читает ПОСЛЕДНЮЮ строку
                # info, а при MultiPV > 1 последняя строка - это худший
                # из показанных ходов. Оценка позиции из-за этого
                # уезжала на несколько пешек. Кому нужны несколько
                # линий, тот задаёт MultiPV сам (get_top_moves).
                "MultiPV": 1,
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

_maia3_preload_started = False


def _maia3_stderr():
    """Куда писать stderr процесса Maia3.

    На Windows stderr раньше уходил в DEVNULL, и любая ошибка движка
    (падение в torch, битый чекпоинт, нехватка памяти) была не видна:
    в консоли оставалось только «ход не выполнен». Теперь он пишется в
    logs/maia3.log - по этому файлу видно причину сбоя.
    """

    try:
        from backend.config.settings import BASE_DIR

        log_dir = Path(BASE_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        return open(log_dir / "maia3.log", "ab", buffering=0)
    except OSError:
        return subprocess.DEVNULL


def _checkpoint_cached(
    model_id: str,
    resolve_model_spec,
    resolve_checkpoint_path,
) -> bool:
    """Есть ли чекпоинт модели в кэше (без обращения к сети)."""

    try:
        spec = resolve_model_spec(model_id)
        resolve_checkpoint_path(spec, local_files_only=True)
        return True
    except Exception:
        return False


def _maia3_checkpoint_is_cached(model_id: str) -> bool:
    """Лежит ли чекпоинт модели в кэше Hugging Face.

    Maia3 при каждом старте заново разрешает чекпоинт через HF Hub, то
    есть ходит в сеть. На медленной или пропавшей сети запуск движка
    растягивается на минуты, и все ходы Maia3 в это время стоят в
    очереди. Если файл уже скачан, сеть не нужна: передаём движку
    --local-files-only.
    """

    try:
        from maia3.model_registry import (
            resolve_checkpoint_path,
            resolve_model_spec,
        )
    except Exception:
        return False

    return _checkpoint_cached(
        model_id,
        resolve_model_spec,
        resolve_checkpoint_path,
    )


def get_maia3():
    """Возвращает UCI-движок Maia3 или None."""

    global _maia3_engine

    if _maia3_engine is not None:
        return _maia3_engine

    with _maia3_loading_lock:

        if _maia3_engine is not None:
            return _maia3_engine

        stderr_stream = subprocess.DEVNULL

        try:

            import chess.engine

            from backend.config.settings import settings

            maia3_source = Path(__file__).resolve().parents[2] / "maia3"
            uci_module = maia3_source / "maia3" / "uci.py"
            if not uci_module.is_file():
                raise FileNotFoundError(
                    f"Maia3 source module was not found: {uci_module}"
                )

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

            if _maia3_checkpoint_is_cached(settings.maia3.model_id):
                cmd.append("--local-files-only")

            # Приложению нужен только лучший ход: WDL-строки оно не читает,
            # а их расчёт - это второй forward модели на каждый ход (пачка из
            # MultiPV позиций). Отключаем его целиком.
            speed_flags = [
                "--multipv",
                "1",
                "--no-compute-wdl",
            ]

            if os.name == "nt":
                stderr_stream = _maia3_stderr()

            def popen(extra_flags):
                return chess.engine.SimpleEngine.popen_uci(
                    cmd + extra_flags,
                    timeout=180,
                    cwd=str(maia3_source),
                    setpgrp=True,
                    stderr=stderr_stream,
                )

            try:
                _maia3_engine = popen(speed_flags)
            except Exception as exc:
                # Сабмодуль maia3 может быть старше этих флагов (например,
                # после git submodule update без нашего коммита), и тогда
                # движок падает на разборе аргументов. Ходы важнее
                # ускорения: запускаем как есть, с расчётом WDL.
                print(
                    "[Maia3] Запуск с флагами ускорения не удался: "
                    f"{exc}. Повтор без них."
                )
                _maia3_engine = popen([])

            # popen_uci завершает только uci-хендшейк. Чекпоинт грузится
            # на первый isready, то есть без этого вызова его загрузка
            # (и прогрев) легла бы на первый ход. ping() дожидается
            # readyok, поэтому "ready" ниже печатается уже после загрузки.
            _maia3_engine.ping()

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

        finally:

            # Дочерний процесс уже получил свою копию дескриптора.
            if stderr_stream is not subprocess.DEVNULL:
                try:
                    stderr_stream.close()
                except Exception:
                    pass

    return _maia3_engine


def preload_maia3() -> None:
    """Готовит движок Maia3 в фоне при старте приложения.

    Импорт torch и загрузка чекпоинта занимают несколько секунд, и без
    прогрева эту задержку оплачивает первый ход после запуска сервера.
    Функция не блокирует старт: сбой прогрева ничего не ломает - первый
    реальный запрос повторит попытку через get_maia3().
    """

    global _maia3_preload_started

    if _maia3_engine is not None:
        return

    with _maia3_loading_lock:
        if _maia3_preload_started or _maia3_engine is not None:
            return
        _maia3_preload_started = True

    threading.Thread(
        target=get_maia3,
        name="maia3-preload",
        daemon=True,
    ).start()


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


def reset_maia3():
    """Сбрасывает мёртвый процесс Maia3, чтобы следующий запрос поднял новый.

    Без сброса упавший процесс остаётся в глобальной переменной, и каждый
    следующий ход падает на нём же до перезапуска сервера: в интерфейсе
    это выглядит как «Maia3 внезапно перестала ходить».
    """

    global _maia3_engine

    with _maia3_loading_lock:

        engine = _maia3_engine
        _maia3_engine = None

    if engine is not None:

        try:
            engine.quit()
        except Exception:
            pass

        # quit() ждёт ответа движка с таймаутом: у зависшего процесса
        # он истечёт, а процесс останется. close() закрывает транспорт
        # сразу, и такой процесс получает EOF на stdin.
        try:
            engine.close()
        except Exception:
            pass


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
# CHAT FALLBACK LLM (Chat Completions API)
# =============================================================

_chat_fallback_client = None

_chat_fallback_lock = threading.Lock()


def get_chat_fallback():
    """Возвращает singleton-клиент резервной LLM для чата."""

    global _chat_fallback_client

    if _chat_fallback_client is not None:
        return _chat_fallback_client

    with _chat_fallback_lock:

        if _chat_fallback_client is not None:
            return _chat_fallback_client

        try:

            from backend.config.settings import settings
            from backend.llm.openai_compat import OpenAICompatClient

            config = settings.chat_fallback

            if not config.base_url or not config.api_key:

                print(
                    "[LLMFallback] "
                    "CHAT_FALLBACK__BASE_URL или "
                    "CHAT_FALLBACK__API_KEY не заданы"
                )

                return None

            _chat_fallback_client = OpenAICompatClient(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                connect_timeout=config.connect_timeout,
                read_timeout=config.read_timeout,
            )

            print(
                "[LLMFallback] Client ready: "
                f"{config.base_url} "
                f"(model={config.model})"
            )

            return _chat_fallback_client

        except Exception as e:

            print(
                "[LLMFallback] "
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


def _attach_russian_names(openings: list) -> int:
    """Дополняет записи дебютов полем ``name_ru``.

    Имена в ``openings.json`` только английские, а интерфейс русский.
    Перевод считается на старте (3810 записей — доли секунды) и кладётся
    прямо в объект: словарь полей не ограничен, поле просто пройдёт дальше
    в JSON ответа. Если переводить не удалось, ``name_ru`` не ставится —
    фронт покажет оригинальное английское имя.
    """
    from backend.knowledge.openings_ru import translate_opening_name

    translated = 0

    for opening in openings:

        name_ru = translate_opening_name(opening.get("name"))

        if name_ru:

            opening["name_ru"] = name_ru
            translated += 1

    return translated


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

        openings = knowledge_base.get(
            "openings",
            [],
        )

        _attach_russian_names(openings)

        count = len(openings)

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
# PUZZLE BASE
# =============================================================

puzzle_base = None

PUZZLES_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "knowledge",
        "puzzles.json",
    )
)


def load_puzzles():
    """Загружает базу тактических задач (паззлов) из JSON."""

    global puzzle_base

    try:

        if not os.path.exists(PUZZLES_PATH):

            print(
                "[Puzzles] "
                f"База паззлов не найдена: "
                f"{PUZZLES_PATH}"
            )

            return False

        with open(
            PUZZLES_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            puzzle_base = json.load(f)

        count = len(
            puzzle_base.get(
                "puzzles",
                [],
            )
        )

        print(
            "[Puzzles] "
            f"База паззлов загружена: "
            f"{count} задач"
        )

        return True

    except Exception as e:

        print(
            "[Puzzles] "
            f"Ошибка загрузки базы паззлов: {e}"
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
