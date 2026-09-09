import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

import config

def main():
    print(f"Loading dataset: {config.DATASET_NAME}...")
    dataset = load_dataset(config.DATASET_NAME)

    # 1. Конфигурация 4-bit квантования (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # 4. Конфигурация LoRA
    peft_config = LoraConfig(
        r=getattr(config, "LORA_R", 64),
        lora_alpha=getattr(config, "LORA_ALPHA", 16),
        lora_dropout=getattr(config, "LORA_DROPOUT", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. Конфигурация SFT (только с max_length)
    sft_config = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=getattr(config, "BATCH_SIZE", 2),
        per_device_eval_batch_size=getattr(config, "BATCH_SIZE", 2),
        gradient_accumulation_steps=getattr(config, "GRADIENT_ACCUMULATION_STEPS", 4),
        learning_rate=getattr(config, "LEARNING_RATE", 2e-4),
        logging_steps=getattr(config, "LOGGING_STEPS", 10),
        num_train_epochs=getattr(config, "NUM_EPOCHS", 3),
        max_grad_norm=getattr(config, "MAX_GRAD_NORM", 0.3),
        warmup_steps=10,
        bf16=True,
        fp16=False,
        dataset_text_field="text",
        max_length=getattr(config, "MAX_SEQ_LENGTH", 2048),
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none",
    )

    # 6. Инициализация SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    # 7. Сохранение итогового адаптера
    print(f"Saving fine-tuned LoRA weights to {config.OUTPUT_DIR}...")
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
