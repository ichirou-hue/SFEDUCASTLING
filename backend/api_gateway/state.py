"""Общее состояние приложения для SFEDUCASTLING API.

Хранит синглтоны ML-моделей, движков и конфигурацию,
которые инициализируются однократно при запуске
и доступны всем модулям маршрутов.
"""

import os
import json
import sys
import subprocess
import threading
import requests


class ModelManager:
    """Менеджер ленивой загрузки ML-моделей и шахматных движков.

    Предоставляет единую точку доступа к синглтонам,
    загружая их при первом обращении (lazy loading).
    """

    def __init__(self):
        self._stockfish = None
        self._stockfish_loaded = False
        self._sf_loading_lock = threading.Lock()

    # --- Stockfish ---

    def get_stockfish(self):
        """Возвращает экземпляр Stockfish (ленивая загрузка)."""
        if self._stockfish is not None:
            return self._stockfish
        with self._sf_loading_lock:
            if self._stockfish is not None:
                return self._stockfish
            if self._stockfish_loaded:
                return None
            print("[ModelManager] Первый вызов get_stockfish — загружаем...")
            self._load_stockfish()
            if self._stockfish is not None:
                self._stockfish_loaded = True
            else:
                print("[ModelManager] Не удалось загрузить Stockfish")
        return self._stockfish

    def _load_stockfish(self):
        try:
            from stockfish import Stockfish
            path = os.path.join(os.path.dirname(__file__), "..", "..", "stockfish_engine", "stockfish.exe")
            if os.path.exists(path):
                self._stockfish = Stockfish(path=path, depth=30)
                self._stockfish.update_engine_parameters({
                    "UCI_LimitStrength": False,
                    "Skill Level": 20,
                    "Threads": max(1, os.cpu_count() or 1),
                    "Hash": 1024,
                })
                print(f"Stockfish loaded (полная сила): {path}")
            else:
                print(f"Stockfish not found at path: {path}")
        except Exception as e:
            print(f"Error loading Stockfish: {e}")


# Глобальный экземпляр менеджера
manager = ModelManager()

# Мьютекс для защиты Stockfish от конкурентного доступа
stockfish_lock = threading.Lock()


def ensure_stockfish():
    """Проверяет, что Stockfish загружен. Возвращает объект Stockfish или None."""
    return manager.get_stockfish()


def reset_stockfish():
    """Сбрасывает Stockfish (при падении процесса)."""
    with manager._sf_loading_lock:
        manager._stockfish = None
        manager._stockfish_loaded = False


def load_stockfish():
    """Загружает Stockfish. Возвращает True при успехе."""
    return manager.get_stockfish() is not None


# --- Maia3 ---
_maia3_engine = None
_maia3_lock = threading.Lock()
_maia3_loading_lock = threading.Lock()


def get_maia3():
    """Возвращает UCI-движок Maia3 (ленивая загрузка) или None.

    Maia3 играет как человек выбранного уровня (SelfElo/OppoElo через UCI).
    """
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
                "-m", "maia3.uci",
                "--model", settings.maia3.model_id,
                "--device", settings.maia3.device,
                "--use-uci-history",
                "--history", str(settings.maia3.history),
                "--elo", str(settings.maia3.default_elo),
            ]
            _maia3_engine = chess.engine.SimpleEngine.popen_uci(
                cmd,
                setpgrp=True,
                stderr=subprocess.DEVNULL if os.name == "nt" else None,
            )
            print(f"Maia3 ready: {settings.maia3.model_id} ({settings.maia3.device})")
        except Exception as e:
            print(f"Error loading Maia3: {e}")
            _maia3_engine = None
    return _maia3_engine


def shutdown_maia3():
    """Останавливает UCI-процесс Maia3 (при завершении приложения)."""
    global _maia3_engine
    with _maia3_loading_lock:
        if _maia3_engine is not None:
            try:
                _maia3_engine.quit()
            except Exception:
                pass
            _maia3_engine = None


def maia3_lock():
    """Возвращает мьютекс, защищающий Maia3 от конкурентного доступа."""
    return _maia3_lock


def elo_to_temperature(elo: int) -> float:
    """Сопоставляет Elo с температурой сэмплирования Maia3.

    Чем ниже уровень, тем выше температура (больше человеческих ошибок).
    """
    if elo >= 2400:
        return 0.0
    if elo >= 2000:
        return 0.3
    if elo >= 1700:
        return 0.6
    if elo >= 1400:
        return 0.9
    return 1.2


# --- LLaVA ---
llava_model = None


# --- База знаний ---
knowledge_base = None
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "openings.json")


def load_knowledge():
    """Загружает базу дебютов из JSON-файла."""
    global knowledge_base
    try:
        if os.path.exists(KNOWLEDGE_PATH):
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
            count = len(knowledge_base.get("openings", []))
            print(f"База знаний загружена: {count} дебютов")
            return True
        else:
            print(f"База знаний не найдена: {KNOWLEDGE_PATH}")
            return False
    except Exception as e:
        print(f"Ошибка загрузки базы знаний: {e}")
        return False


def load_llava():
    """Загружает LLaVA — мультимодальную модель для распознавания досок."""
    return False


def get_opening_info(fen: str) -> dict | None:
    """Получает информацию о дебюте и известные партии из Lichess masters explorer.

    Args:
        fen: FEN-строка позиции для поиска.

    Returns:
        dict с ключами 'name' (название дебюта) и 'games' (список партий), либо None.
    """
    try:
        resp = requests.get(
            "https://explorer.lichess.ovh/masters",
            params={"fen": fen},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        info = {}
        if data.get("opening"):
            info["name"] = data["opening"].get("name", "")
        top_games = data.get("topGames", [])[:3]
        if top_games:
            games = []
            for g in top_games:
                white = g.get("white", {}).get("name", "?")
                black = g.get("black", {}).get("name", "?")
                year = g.get("year", "?")
                winner = g.get("winner", "draw")
                games.append(f"{white} — {black}, {year} ({winner})")
            info["games"] = games
        return info if info else None
    except Exception:
        return None


def extract_fen_from_image(image_path: str) -> str:
    """Распознаёт шахматную позицию с изображения через LLaVA.

    Args:
        image_path: Путь к файлу изображения.

    Returns:
        FEN-строка при успехе, либо строка с ошибкой, начинающаяся с 'ERROR:'.
    """
    from PIL import Image

    global llava_model
    if llava_model is None:
        if not load_llava():
            return "ERROR: LLaVA не загружена."

    try:
        image = Image.open(image_path).convert("RGB")

        prompt = """Опишите эту шахматную позицию. Выведите ТОЛЬКО FEN-нотацию для показанной позиции.
Пример: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"""

        outputs = llava_model(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        response = outputs[0]["generated_text"]

        if "ERROR" in response.upper():
            return "ERROR: не удалось прочитать доску"

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if '/' in line and len(line) > 20:
                parts = line.split()
                if len(parts) >= 1 and '/' in parts[0] and parts[0].count('/') == 7:
                    return parts[0]

        return f"ERROR: неожиданный ответ: {response[:100]}"

    except Exception as e:
        return f"ERROR: {str(e)}"
