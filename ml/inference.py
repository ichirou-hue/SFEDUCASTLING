"""
Запуск обученной модели для анализа позиций.

Как скрипт (для тестирования):
    python inference.py --lora ./chess-llm-lora --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

Как модуль (для backend/app.py):
    from inference import ChessLLM
    llm = ChessLLM("./chess-llm-lora")
    text = llm.analyze("fen_строка")
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


class ChessLLM:
    def __init__(self, lora_path, base_model=BASE_MODEL):
        print(f"Загрузка базовой модели: {base_model}")
        print(f"Загрузка LoRA адаптера: {lora_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(lora_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        print("Модель готова к работе.")

    def analyze(self, fen):
        prompt = f"<s>[INST] Оцени шахматную позицию: {fen} [/INST] "

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
            )

        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = full_text.split("[/INST]")[-1].strip()
        return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ шахматной позиции через LLM")
    parser.add_argument("--fen", required=True, help="FEN строка позиции")
    parser.add_argument("--lora", default="./chess-llm-lora", help="Путь к LoRA адаптеру")
    args = parser.parse_args()

    llm = ChessLLM(args.lora)
    result = llm.analyze(args.fen)
    print("\n--- Анализ позиции ---")
    print(result)
