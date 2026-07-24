"""Общее состояние приложения для SFEDUCASTLING API.

Хранит синглтоны ML-моделей, движков и конфигурацию,
которые инициализируются однократно при запуске
и доступны всем модулям маршрутов.
"""

import os
import json
import requests
from backend.config.settings import settings

# --- GigaChat (теперь используется через settings.gigachat.auth_key) ---
# Удаляем глобальную переменную GIGACHAT_AUTH_KEY

# --- Maia2 ---
maia2 = None
maia2_prepared = None

# --- Stockfish ---
stockfish = None

# --- LLaVA ---
llava_model = None

# --- База знаний ---
knowledge_base = None
# Используем путь из настроек, с fallback на старый путь
KNOWLEDGE_PATH = (
    str(settings.models.knowledge_path)
    if settings.models.knowledge_path
    else os.path.join(os.path.dirname(__file__), "..", "knowledge", "openings.json")
)


def load_maia2():
    """Загружает Maia2 — нейросеть для человекообразных ходов."""
    global maia2, maia2_prepared
    if maia2 is not None:
        return True
    try:
        from maia2 import model as maia2_model_loader, inference as maia2_inference
        maia2 = maia2_model_loader.from_pretrained(type="rapid", device="cpu")
        maia2_prepared = maia2_inference.prepare()
        print("Maia2 загружена и готова к работе.")
        return True
    except Exception as e:
        print(f"Не удалось загрузить Maia2: {e}")
        return False


def load_stockfish():
    """Загружает движок Stockfish."""
    global stockfish
    if stockfish is not None:
        return True
    try:
        from stockfish import Stockfish
        # Используем путь из настроек или fallback
        stockfish_path = (
            str(settings.models.stockfish_path)
            if settings.models.stockfish_path
            else os.path.join(os.path.dirname(__file__), "..", "..", "stockfish", "stockfish.exe")
        )
        if os.path.exists(stockfish_path):
            stockfish = Stockfish(path=stockfish_path, depth=settings.models.stockfish_depth)
            print(f"Stockfish загружен: {stockfish_path}")
            return True
        else:
            print(f"Stockfish не найден по пути: {stockfish_path}")
            return False
    except Exception as e:
        print(f"Ошибка загрузки Stockfish: {e}")
        return False


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
    global llava_model
    if llava_model is not None:
        return True
    try:
        import torch
        from transformers import pipeline
        model_id = settings.models.llava_model_id
        print(f"Загрузка LLaVA модели {model_id} через pipeline...")
        llava_model = pipeline(
            "image-to-text",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("LLaVA загружена успешно!")
        return True
    except Exception as e:
        print(f"Ошибка загрузки LLaVA: {e}")
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
            settings.lichess.explorer_url,
            params={"fen": fen},
            timeout=settings.lichess.timeout,
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


def ensure_stockfish():
    """Проверяет, что Stockfish загружен; пробует загрузить, если нет. Возвращает объект Stockfish или None."""
    if stockfish is None:
        load_stockfish()
    return stockfish


def ensure_maia2():
    """Проверяет, что Maia2 загружена; пробует загрузить, если нет. Возвращает (model, prepared) или (None, None)."""
    global maia2, maia2_prepared
    if maia2 is None:
        load_maia2()
    return maia2, maia2_prepared