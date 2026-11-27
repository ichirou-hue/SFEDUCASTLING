import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Batch curation of chess dataset with local LLM")
    parser.add_argument("--input", type=str, default="../data/gameknot_clean_final_corrected.jsonl", help="Путь к входному JSONL")
    parser.add_argument("--output", type=str, default="../data/curated_coach_full.jsonl", help="Куда сохранять результат")
    parser.add_argument("--model", type=str, required=True, help="Путь к локальным весам модели")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча для инференса на GPU")
    parser.add_argument("--limit", type=int, default=0, help="Лимит строк (0 = весь файл)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: входной файл {args.input} не найден!")
        sys.exit(1)

    # Проверяем, сколько строк уже обработано ранее (для дозаписи при рестарте)
    processed_count = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for _ in f:
                processed_count += 1
        print(f"Обнаружен существующий выходной файл. Пропускаем первые {processed_count} строк...")

    print(f"Загрузка токенизатора и модели: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Считаем общее число строк во входном файле
    total_input_lines = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for _ in f:
            total_input_lines += 1

    target_total = min(args.limit, total_input_lines) if args.limit > 0 else total_input_lines
    remaining_to_process = max(0, target_total - processed_count)

    print(f"Всего строк в источнике: {total_input_lines}")
    print(f"Целевой объем: {target_total} | Осталось обработать: {remaining_to_process}")

    if remaining_to_process == 0:
        print("Все строки уже обработаны!")
        return

    # Потоковая итерация по файлу с батчингом
    def batch_stream():
        batch = []
        with open(args.input, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < processed_count:
                    continue
                if args.limit > 0 and idx >= args.limit:
                    break
                line = line.strip()
                if line:
                    batch.append(json.loads(line))
                    if len(batch) == args.batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch

    # Открываем выходной файл в режиме дозаписи ('a')
    with open(args.output, "a", encoding="utf-8") as out_f:
        pbar = tqdm(total=remaining_to_process, desc="Curating", unit="ex")

        for batch in batch_stream():
            prompts = []
            for row in batch:
                fen = row.get("fen", "")
                san = row.get("move_san", "")
                raw_text = row.get("comment", "").strip()
                user_content = f"FEN: {fen}\nMove: {san}\nComment: {raw_text}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
                prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False
                )

            # Декодируем только сгенерированные хвосты батча
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Сохраняем результат
            for row, resp in zip(batch, responses):
                resp = resp.strip()
                is_rejected = "REJECT" in resp

                if is_rejected:
                    status = "REJECTED"
                    cleaned = ""
                elif "CLEANED:" in resp:
                    status = "CLEANED"
                    cleaned = resp.split("CLEANED:")[1].strip()
                else:
                    status = "KEPT"
                    cleaned = resp

                result_entry = {
                    "fen": row.get("fen", ""),
                    "move_san": row.get("move_san", ""),
                    "move_uci": row.get("move_uci", ""),
                    "raw_comment": row.get("comment", "").strip(),
                    "cleaned_comment": cleaned,
                    "status": status
                }
                out_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

            out_f.flush()
            pbar.update(len(batch))

        pbar.close()

    print(f"\nОбработка завершена! Все данные сохранены в {args.output}")


if __name__ == "__main__":
    main()