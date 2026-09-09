import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_api import ChessLLMClient


async def run_test():
    print("=== ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ ОБЪЯСНЕНИЯ ===")
    
    fen = "r1bqkb1r/1p1n1ppp/p2p1n2/1B2pN2/4P3/2N5/PPP2PPP/R1BQK2R w KQkq - 0 8"
    engine_data = {
        "played_move": "Ba4",
        "best_move": "Ba4",
        "played_move_eval_cp": 57,
        "best_move_eval_cp": 57,
        "centipawn_loss": 0,
        "depth": 22,
        "multipv": [
            {"rank": 1, "move_san": "Ba4", "eval_cp": 57, "pv_san": ["Ba4", "b5", "Bb3", "Nc5", "Bg5", "Bxf5"]}
        ]
    }

    llm = ChessLLMClient()
    print("Отправка тестового запроса...")
    result = await llm.generate_coach_explanation(fen, engine_data)

    if result:
        print("\n✅ Ответ получен и успешно прошел валидацию CoachExplanation!")
        print("-" * 60)
        print(f"📌 Расстановка сил:     {result.position_summary}")
        print(f"⚠️ Главная проблема:     {result.root_problem}")
        print(f"❌ Ошибка игрока:        {result.player_mistake}")
        print(f"💡 Лучший ход (SAN):    {result.best_move}")
        print(f"🎯 Почему лучший:        {result.why_best}")
        print(f"🧠 Концепция:            {result.strategic_concept}")
        print(f"📉 Последствия ошибки:  {result.mistake_consequences}")
        print(f"♟️ Вариант (Main line):  {result.main_line}")
        print(f"🎓 Совет:                {result.practical_advice}")
        print("-" * 60)
    else:
        print("❌ Ошибка генерации.")


if __name__ == "__main__":
    asyncio.run(run_test())