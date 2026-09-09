import asyncio
import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_api import ChessLLMClient
from src.schemas import GeneratedExample
from src.board_features import extract_position_facts  # <-- Импорт модуля фактов

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 2


async def process_single_item(item: dict, llm: ChessLLMClient) -> dict | None:
    """Обрабатывает одну позицию с обогащением фактами и повторными попытками."""
    fen = item.get("fen")
    if not fen:
        return None

    best_move = item.get("best_move", "")

    # 1. Извлекаем детерминированные факты доски через python-chess
    facts = extract_position_facts(fen, best_move)

    # 2. Упаковываем все данные для промпта
    engine_data = {
        "played_move": item.get("played_move"),
        "best_move": best_move,
        "played_move_eval_cp": item.get("played_move_eval_cp"),
        "best_move_eval_cp": item.get("best_move_eval_cp"),
        "centipawn_loss": item.get("centipawn_loss", 0),
        "depth": item.get("depth"),
        "multipv": item.get("multipv", []),
        "turn": item.get("turn", "white"),
        "move_number": item.get("move_number", 1),
        "phase": item.get("phase", "middlegame"),
        "board_facts": facts  # <-- Передаем вычисленные факты
    }

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            explanation = await llm.generate_coach_explanation(
                fen=fen,
                engine_data=engine_data
            )
            if explanation:
                example = GeneratedExample(
                    id=item["id"],
                    source_type=item["source_type"],
                    fen=item["fen"],
                    played_move=item["played_move"],
                    best_move=best_move,
                    centipawn_loss=item.get("centipawn_loss", 0),
                    phase=item.get("phase", "middlegame"),
                    turn=item.get("turn", "white"),
                    coach_explanation=explanation
                )
                return example.model_dump()
        except Exception as e:
            logger.warning(f"[{item.get('id')}] Ошибка (попытка {attempt}/{max_attempts}): {e}")
            await asyncio.sleep(40)

    logger.error(f"[{item.get('id')}] Не удалось получить ответ после всех попыток.")
    return None


async def main():
    input_file = ROOT_DIR / "data" / "02_engine_analyzed" / "master_analyzed.jsonl"
    output_dir = ROOT_DIR / "data" / "03_generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "master_explanations.jsonl"

    if not input_file.exists():
        logger.error(f"Входной файл не найден: {input_file}")
        return

    processed_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Уже обработано ранее: {len(processed_ids)} записей.")

    items_to_process = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] not in processed_ids:
                items_to_process.append(item)

    total_count = len(items_to_process)
    logger.info(f"Осталось обработать: {total_count} записей.")

    if total_count == 0:
        logger.info("Все позиции уже обработаны!")
        return

    llm = ChessLLMClient()
    saved_count = 0

    with open(output_file, "a", encoding="utf-8") as outfile:
        for idx, item in enumerate(items_to_process, 1):
            logger.info(f"[{idx}/{total_count}] Обработка {item['id']}...")
            
            result = await process_single_item(item, llm)
            if result:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
                saved_count += 1
                logger.info(f"✅ Сохранено ({saved_count} новых)")

            if idx < total_count:
                await asyncio.sleep(REQUEST_DELAY_SECONDS)

    logger.info(f"🎉 Генерация завершена! Всего сохранено: {saved_count}")


if __name__ == "__main__":
    asyncio.run(main())