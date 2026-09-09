import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Base URL из вашей документации
BASE_URL = "https://7c8e698a-a94e-42e0-8f77-df6f542d9f6b.modelrun.inference.cloud.ru"

# Адаптер для принудительного использования HTTP/1.1 (решает проблему с зависанием h2)
class HTTP11Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_version'] = None
        return super().init_poolmanager(*args, **kwargs)

def get_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", HTTP11Adapter())
    return session

def ping_server() -> bool:
    """Проверка доступности инстанса."""
    url = f"{BASE_URL}/ping"
    session = get_session()
    try:
        response = session.get(url, timeout=15, verify=False)
        return response.status_code == 200
    except Exception as e:
        print(f"[Ошибка подключения /ping]: {e}")
        return False

def ask_gigachess(prompt: str, fen: str = "") -> str:
    """Отправка запроса к эндпоинту /chat с передачей FEN в attachments."""
    url = f"{BASE_URL}/chat"
    
    message = {
        "role": "user",
        "content": prompt
    }
    
    if fen:
        message["attachments"] = [fen]

    payload = {
        "messages": [message],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 128,
        "n": 1,
        "repetition_penalty": 1.0,
        "model": "gigachess"
    }

    headers = {"Content-Type": "application/json"}
    session = get_session()

    try:
        # Увеличиваем таймаут до 900 секунд на случай прогрева (cold start)
        response = session.post(url, headers=headers, json=payload, verify=False, timeout=900)
        response.raise_for_status()
        data = response.json()
        
        # Извлекаем текстовый ответ
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"Ошибка при запросе к GigaChess: {e}"

def extract_uci_move(text: str) -> str:
    """Извлекает первый UCI ход (например, e2e4) из ответа модели."""
    match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", text.lower())
    return match.group(1) if match else ""

if __name__ == "__main__":
    import time

    print("Ожидание прогрева Model RUN (Cold Start)...")
    max_attempts = 60  # Пробуем в течение 10 минут
    
    for attempt in range(1, max_attempts + 1):
        print(f"Попытка {attempt}/{max_attempts} подключиться к /ping...", end=" ", flush=True)
        if ping_server():
            print("\n[УСПЕХ] Сервер прогрелся и ответил 200 OK!")
            break
        else:
            print("еще спит (timeout). Ждем 10 сек...")
            time.sleep(10)
    else:
        print("\n[Ошибка] Инстанс не ответил за отведенное время.")
        exit(1)

    # Тестовый запрос
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("\nОтправка тестового запроса с FEN...")
    answer = ask_gigachess("Какой лучший ход в этой позиции? Ответь кратко, ходом в UCI.", fen=test_fen)
    print("\n=== Ответ модели ===")
    print(answer)
    print("====================")
    print("Извлеченный UCI ход:", extract_uci_move(answer))