import os
import re
import sys
import time
import json
import http.client
import requests
from typing import Optional, Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knowledge_base.chess_analysis.engine_analyzer import ChessEngineAnalyzer
from knowledge_base.chess_analysis.prompt_builder import ChessPromptBuilder

http.client.HTTPConnection._http_v_expect = 'HTTP/1.1'
MODEL_CLOUD_URL = "https://7c8e698a-a94e-42e0-8f77-df6f542d9f6b.modelrun.inference.cloud.ru"

TEST_POSITIONS = [
    {
        "id": 1,
        "title": "Окончание: отсечение короля шахом",
        "fen": "1K1k4/1P1r4/8/8/8/8/8/2R5 w - - 0 1",
        "played_move": "b8a7",
        "student_question": "Учитель, я пошел королем на a7, чтобы выпустить пешку. Почему движок против?"
    },
    {
        "id": 2,
        "title": "Миттельшпиль: вторжение ладьи на 7-ю горизонталь",
        "fen": "r4rk1/pp3ppp/8/3p4/3P4/8/PP3PPP/2R1R1K1 w - - 0 20",
        "played_move": "g1f1",
        "student_question": "Какие есть хорошие ходы которые обеспечивают мне развитие?"
    },
    {
        "id": 3,
        "title": "Дебют: защита от тактического удара на f7",
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "played_move": "d2d4",
        "student_question": "Почему ход d2d4 здесь неточность, ведь я борюсь за центр?"
    },
    {
        "id": 4,
        "title": "Пешечный эндшпиль: оппозиция и ключевые поля",
        "fen": "8/8/4k3/8/4P3/4K3/8/8 w - - 0 40",
        "played_move": "e4e5",
        "student_question": "Я продвинул пешку на e5, чтобы быстрее дойти до ферзя. Почему это ошибка?"
    }
]


def send_to_llm(prompt: str, fen: str) -> Optional[str]:
    url = f"{MODEL_CLOUD_URL.rstrip('/')}/chat"
    payload = {
        "messages": [{
            "role": "user",
            "content": prompt,
            "attachments": [fen]
        }],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 750,
        "n": 1,
        "repetition_penalty": 1.0,
        "model": "gigachess"
    }
    headers = {"Content-Type": "application/json", "Connection": "close"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[API ERROR {response.status_code}] {response.text}")
            return None
    except Exception as e:
        print(f"[CONNECTION ERROR] {e}")
        return None


def run_batch_tests():
    analyzer = ChessEngineAnalyzer(depth=18)
    prompt_builder = ChessPromptBuilder(engine_analyzer=analyzer)

    print("=" * 90)
    print(" ЗАПУСК ПАКЕТНОГО ТЕСТИРОВАНИЯ RAG-ТРЕНЕРА")
    print("=" * 90)

    for item in TEST_POSITIONS:
        test_id = item["id"]
        title = item["title"]
        fen = item["fen"]
        move = item.get("played_move")
        q = item.get("student_question")

        print("\n" + "#" * 90)
        print(f" ТЕСТ [{test_id}/{len(TEST_POSITIONS)}]: {title}")
        print(f" FEN: {fen} | Ход: {move}")
        print("#" * 90)

        t0 = time.time()
        bundle = prompt_builder.build_prompt_from_fen(
            fen=fen,
            played_uci_move=move,
            student_question=q
        )
        t_rag = (time.time() - t0) * 1000

        print(f"\n[1] ВРЕМЯ РАСЧЕТА И СБОРКИ RAG: {t_rag:.1f} ms")
        print("\n" + "=" * 35 + " СФОРМИРОВАННЫЙ ПРОМПТ " + "=" * 35)
        print(bundle["prompt"])
        print("=" * 93)

        print("\n[2] ОТПРАВКА В МОДЕЛЬ...")
        t_llm_start = time.time()
        llm_response = send_to_llm(bundle["prompt"], fen)
        t_llm = time.time() - t_llm_start

        print(f"\n[3] ВРЕМЯ ГЕНЕРАЦИИ LLM: {t_llm:.2f} сек.")
        print("-" * 38 + " ОТВЕТ ТРЕНЕРА " + "-" * 38)
        if llm_response:
            print(llm_response.strip())
        else:
            print("❌ Не удалось получить ответ от модели.")
        print("-" * 90)

    analyzer.close()
    print("\n" + "=" * 90)
    print(" [ГОТОВО] Все тесты завершены.")
    print("=" * 90)


if __name__ == "__main__":
    run_batch_tests()