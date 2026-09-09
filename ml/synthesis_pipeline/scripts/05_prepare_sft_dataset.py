import json
import random
import argparse
from pathlib import Path
from loguru import logger
from typing import Dict, Any

SYSTEM_COACH_PROMPT = (
    "Ты — элитный шахматный гроссмейстер и персональный тренер. "
    "Твоя цель — дать глубокий, понятный и методически выверенный разбор шахматной позиции. "
    "Используй точные шахматные термины, объясняй идеи и стратегические концепции простым языком."
)

def format_user_query(record: Dict[str, Any]) -> str:
    fen = record["fen"]
    turn_str = "Белые" if record["turn"] == "white" else "Черные"
    phase = record.get("phase", "позиция")
    played_move = record.get("played_move")

    query = (
        f"Проанализируй шахматную позицию:\n"
        f"FEN: {fen}\n"
        f"Очередь хода: {turn_str} (стадия партии: {phase}).\n"
    )
    if played_move:
        query += f"В партии был сделан ход: {played_move}.\n"
    query += "Объясни расстановку сил, выдели ключевую проблему, назови лучший ход с обоснованием и сформулируй практический совет."
    return query

def format_assistant_response(exp: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"**Оценка позиции:**\n{exp['position_summary']}")
    parts.append(f"**Ключевой нюанс:**\n{exp['root_problem']}")
    
    if exp.get("player_mistake"):
        parts.append(f"**Разбор сыгранного хода:**\n{exp['player_mistake']}")
    
    parts.append(f"**Рекомендуемый ход:** `{exp['best_move']}`\n{exp['why_best']}")
    parts.append(f"**Стратегическая идея:**\n{exp['strategic_concept']}")
    
    if exp.get("mistake_consequences"):
        parts.append(f"**Последствия неточности:**\n{exp['mistake_consequences']}")
        
    parts.append(f"**Главный вариант:**\n{exp['main_line']}")
    parts.append(f"**Практический совет тренера:**\n{exp['practical_advice']}")

    return "\n\n".join(parts)

def prepare_sft_dataset(input_path: str, output_dir: str, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    in_file = Path(input_path)
    if not in_file.exists():
        logger.error(f"Входной файл '{input_path}' не найден!")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        logger.warning("Нет записей для обработки.")
        return

    formatted_dataset = []
    for r in records:
        exp = r["coach_explanation"]
        sft_sample = {
            "id": r["id"],
            "messages": [
                {"role": "system", "content": SYSTEM_COACH_PROMPT},
                {"role": "user", "content": format_user_query(r)},
                {"role": "assistant", "content": format_assistant_response(exp)}
            ]
        }
        formatted_dataset.append(sft_sample)

    random.seed(seed)
    random.shuffle(formatted_dataset)

    n_total = len(formatted_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = formatted_dataset[:n_train]
    val_data = formatted_dataset[n_train:n_train + n_val]
    test_data = formatted_dataset[n_train + n_val:]

    # Если выборка тестовая и маленькая (например, 1-2 записи)
    if not train_data and formatted_dataset:
        train_data = formatted_dataset
        val_data = formatted_dataset
        test_data = formatted_dataset

    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        target_path = out_dir / f"{split_name}.jsonl"
        with open(target_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Сплит '{split_name}': {len(data)} примеров -> {target_path}")

    logger.success("Подготовка SFT-датасета завершена!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert validated data to ChatML SFT format")
    parser.add_argument("--input", type=str, default="data/04_validated/master_validated.jsonl")
    parser.add_argument("--out_dir", type=str, default="data/08_final_split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_sft_dataset(args.input, args.out_dir, seed=args.seed)