import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config

def merge():
    print(f"Загрузка базовой модели: {config.MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, 
        trust_remote_code=True
    )

    print(f"Загрузка LoRA адаптера из: {config.ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)

    print("Объединение весов...")
    merged_model = model.merge_and_unload()

    print(f"Сохранение объединенной модели в: {config.MERGED_DIR}")
    merged_model.save_pretrained(config.MERGED_DIR)
    tokenizer.save_pretrained(config.MERGED_DIR)
    print("Объединение успешно завершено!")

if __name__ == "__main__":
    merge()
