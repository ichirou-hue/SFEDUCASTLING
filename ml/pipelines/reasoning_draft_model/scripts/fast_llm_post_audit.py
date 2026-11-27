import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are a strict Chess Annotation Editor and Quality Auditor.
Your job: Polish human chess move commentary so it is grammatically perfect and standard for training a Chess Coach AI.

RULES:
1. PRESERVE CHESS LOGIC: Keep the original tactical and strategic insight. Never invent moves or change squares/piece letters.
2. NORMALIZE NOTATION & TYPOS: 
   - Fix split squares/takes (e.g. "D 6" -> "d6", "Nx e4" -> "Nxe4", "Bx BP" -> "BxBP").
   - Fix split or glued English words (e.g. "in actively" -> "inactively", "andof" -> "and of", "draw ish" -> "drawish").
3. FILTER CHAT & JUNK: If the text is purely emotional chatter, cutoff mid-sentence, complaint, or lacks chess value, output ONLY: REJECT.
4. OUTPUT FORMAT:
   If valid, output strictly:
   CLEANED: <polished commentary text>
   If garbage/non-informative, output strictly:
   REJECT"""


def parse_args():
    parser = argparse.ArgumentParser(description="Resume-safe post audit for chess dataset")
    parser.add_argument("--input", type=str, required=True, help="Путь к *_featured.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Куда дописывать результат")
    parser.add_argument("--model", type=str, required=True, help="Путь к локальным весам или HF ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча (32 снижает пиковую VRAM)")
    return parser.parse_args()


def get_position_signature(item: dict) -> str:
    """Извлекает связку FEN + Move как уникальный ключ позиции"""
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            fen_m = re.search(r"Current FEN:\s*([^\n\r]+)", content)
            move_m = re.search(r"Played Move:\s*([^\n\r]+)", content)
            fen = fen_m.group(1).strip() if fen_m else ""
            move = move_m.group(1).strip() if move_m else ""
            if fen and move:
                return f"{fen}_{move}"
            return content.strip()
    return ""


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: файл {args.input} не найден!")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 1. Считываем сигнатуры уже сохраненных строк из выходного файла
    processed_signatures = set()
    last_processed_sig = None

    if os.path.exists(args.output):
        print(f"Анализ готовых записей в {args.output}...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sig = get_position_signature(data)
                    if sig:
                        processed_signatures.add(sig)
                        last_processed_sig = sig
                except json.JSONDecodeError:
                    continue

        print(f"Успешно прочитано ранее готовых строк: {len(processed_signatures)}")

    # 2. Ищем индекс во входном файле, где остановилась обработка
    all_input_records = []
    print(f"Чтение входного файла {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_input_records.append(json.loads(line))

    total_input = len(all_input_records)
    print(f"Всего строк во входном файле: {total_input}")

    # Определяем индекс отсечки
    start_index = 0
    if last_processed_sig:
        for idx, item in enumerate(all_input_records):
            if get_position_signature(item) == last_processed_sig:
                start_index = idx + 1
                break

    records_to_process = all_input_records[start_index:]
    remaining = len(records_to_process)

    print(f"Точка возобновления: строка {start_index} из {total_input}")
    print(f"Осталось обработать: {remaining} строк")

    if remaining == 0:
        print("Все строки уже были обработаны ранее!")
        return

    # 3. Инициализация модели и токенизатора
    print(f"Загрузка токенизатора и модели: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()

    # 4. Батчевая обработка с дозаписью в файл
    kept_count = 0
    rejected_count = 0
    batch_size = args.batch_size

    with open(args.output, "a", encoding="utf-8") as out_f:
        pbar = tqdm(total=remaining, desc="Finishing Curation", unit="ex")

        for i in range(0, remaining, batch_size):
            batch = records_to_process[i:i + batch_size]
            valid_items = []
            prompts = []

            for item in batch:
                assistant_text = ""
                for msg in item.get("messages", []):
                    if msg.get("role") == "assistant":
                        assistant_text = msg.get("content", "").strip()
                        break

                if not assistant_text:
                    rejected_count += 1
                    continue

                valid_items.append(item)
                dialog = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Commentary: {assistant_text}"}
                ]
                prompts.append(
                    tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
                )

            if not prompts:
                pbar.update(len(batch))
                continue

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=384
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for item, resp in zip(valid_items, responses):
                resp = resp.strip()

                if "REJECT" in resp:
                    rejected_count += 1
                    continue

                if "CLEANED:" in resp:
                    cleaned = resp.split("CLEANED:")[1].strip()
                else:
                    cleaned = resp

                for msg in item["messages"]:
                    if msg.get("role") == "assistant":
                        msg["content"] = cleaned
                        break

                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1

            out_f.flush()
            pbar.update(len(batch))

        pbar.close()

    print("=" * 60)
    print("ФИНАЛЬНЫЙ АУДИТ ЗАВЕРШЕН")
    print(f"Добавлено новых строк: {kept_count}")
    print(f"Отсеяно (REJECT):     {rejected_count}")
    print(f"Итоговый файл:        {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()