import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

LOCAL_MODEL_PATH = "/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_FILE = "../data/03_data/03_featured_dataset_train_part.jsonl"
VAL_FILE = "../data/03_data/03_featured_dataset_val_part.jsonl"
OUTPUT_DIR = "../outputs/reasoning_draft_model_lora_adapter"


class ResponseOnlyDataCollator:
    def __init__(self, response_token_ids, tokenizer, pad_to_multiple_of=8):
        self.response_token_ids = response_token_ids
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_len = max(len(x) for x in input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        resp_len = len(self.response_token_ids)

        for inp, att in zip(input_ids, attention_mask):
            pad_len = max_len - len(inp)
            padded_inp = inp + [self.pad_token_id] * pad_len
            padded_att = att + [0] * pad_len
            label = list(inp) + [-100] * pad_len

            found_idx = -1
            for j in range(len(inp) - resp_len + 1):
                if inp[j:j + resp_len] == self.response_token_ids:
                    found_idx = j + resp_len
                    break

            if found_idx != -1:
                for k in range(found_idx):
                    label[k] = -100
            else:
                label = [-100] * len(label)

            batch_input_ids.append(padded_inp)
            batch_attention_mask.append(padded_att)
            batch_labels.append(label)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = {"": 0}

    if local_rank == 0:
        print(f"Загрузка токенизатора из {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print("Загрузка датасетов...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE}
    )

    def tokenize_chatml(batch):
        formatted_texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in batch["messages"]
        ]
        return tokenizer(
            formatted_texts,
            truncation=True,
            max_length=1024,
            padding=False
        )

    tokenized_dataset = raw_datasets.map(
        tokenize_chatml,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc=f"[Rank {local_rank}] Tokenizing dataset"
    )

    response_template = "<|im_start|>assistant\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    collator = ResponseOnlyDataCollator(
        response_token_ids=response_token_ids,
        tokenizer=tokenizer
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    if local_rank == 0:
        print("Загрузка модели в 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=90,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_first_step=True,
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=collator
    )

    if local_rank == 0:
        print("=" * 60)
        print("СТАРТ ОБУЧЕНИЯ DRAFT МОДЕЛИ")
        print(f"Train: {len(tokenized_dataset['train'])} | Val: {len(tokenized_dataset['validation'])}")
        print("=" * 60)

    trainer.train()

    if local_rank == 0:
        final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
        print(f"Сохранение LoRA в {final_adapter_dir}...")
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        print("Обучение завершено успешно!")


if __name__ == "__main__":
    main()