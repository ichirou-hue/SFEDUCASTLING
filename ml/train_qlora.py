"""
Дообучение Mistral 7B на шахматном датасете через QLoRA.
Работает на 2x RTX 3090 (48GB VRAM).

Запуск:
    python train_qlora.py --dataset dataset.jsonl --output ./chess-llm-lora

Что происходит:
1. Загружает Mistral 7B в 4-bit квантизации (помещается в 1 GPU)
2. Добавляет LoRA адаптеры (маленькие обучаемые слои)
3. Обучает только адаптеры на твоём датасете
4. Сохраняет результат — папку chess-llm-lora/ (~200MB)

Время обучения: ~2-4 часа на 10000 примеров.
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load_dataset_from_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = (
                f"<s>[INST] {entry['instruction']} [/INST] "
                f"{entry['output']}</s>"
            )
            data.append({"text": text})

    print(f"Загружено {len(data)} примеров из {path}")
    return Dataset.from_list(data)


def main(args):
    print("=" * 60)
    print("SFEDUCASTLING — Обучение шахматной LLM")
    print("=" * 60)

    print(f"\n[1/5] Настройка квантизации (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[2/5] Загрузка модели {BASE_MODEL}...")
    print("       (Первый раз скачает ~15GB, потом из кэша)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    print("[3/5] Добавление LoRA адаптеров...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n[4/5] Загрузка датасета...")
    dataset = load_dataset_from_jsonl(args.dataset)

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"       Обучение: {len(train_dataset)}, валидация: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=1024,
    )

    print(f"\n[5/5] Начинаем обучение ({args.epochs} эпох)...")
    print("       Это займёт 2-4 часа. Можно следить за loss в логах.\n")
    trainer.train()

    print(f"\nСохранение модели в {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "=" * 60)
    print("ГОТОВО! Модель сохранена.")
    print(f"Папка: {args.output}")
    print(f"Теперь проверь: python inference.py --lora {args.output} --fen 'начальная_позиция'")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA обучение шахматной LLM")
    parser.add_argument("--dataset", required=True, help="Путь к dataset.jsonl")
    parser.add_argument("--output", default="./chess-llm-lora", help="Папка для сохранения модели")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох обучения")
    args = parser.parse_args()
    main(args)
