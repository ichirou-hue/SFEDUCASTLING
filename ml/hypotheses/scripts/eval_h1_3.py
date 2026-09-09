import json
import time
import gc
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prometheus_client import start_http_server, Gauge
from sber_client import ask_gigachess

# --- PROMETHEUS METRICS (H1.3) ---
H1_3_ACCURACY_PCT = Gauge(
    'chess_h1_3_eval_accuracy_pct', 
    'H1.3: Итоговый % точности цитирования eval (в пределах ±0.5 пешки)', 
    ['model']
)
H1_3_WITHIN_TOLERANCE_COUNT = Gauge(
    'chess_h1_3_within_tolerance_count', 
    'Количество ответов в пределах допуска ±0.5 пешки', 
    ['model']
)
H1_3_HAS_EVAL_COUNT = Gauge(
    'chess_h1_3_has_eval_count', 
    'Количество ответов, содержащих хоть какую-то оценку', 
    ['model']
)

BENCHMARK_PATH = Path("dataset_h1_3.json")
MERGED_MODEL_PATH = Path("../Qwen3-chess-coach/outputs/merged").resolve()
ADAPTER_CONFIG_PATH = Path("../Qwen3-chess-coach/outputs/adapter_config.json").resolve()

SYSTEM_INSTRUCTION = "Ты — шахматный аналитик и тренер."

def get_base_model_name() -> str:
    if ADAPTER_CONFIG_PATH.exists():
        try:
            with open(ADAPTER_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                return cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        except Exception:
            pass
    return "Qwen/Qwen2.5-7B-Instruct"

def clean_response(text: str) -> str:
    """Безопасное отсечение тегов <think>...</think>."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "<think>" in text:
        return text.split("<think>")[-1].strip()
    return text.strip()

def extract_eval_from_text(text: str) -> float | None:
    """
    Улучшенное извлечение оценки из текста:
    - Обрабатывает Markdown (**0.00**, *+0.50*)
    - Игнорирует слова-прослойки ("оценка по Stockfish: **0.00**")
    - Находит числа без знака (0.00, 0.5)
    """
    text_clean = clean_response(text)
    
    # 1. Приоритет: Поиск возле ключевых слов с учётом Markdown
    kw_pattern = r'(?:оценка|eval|стоксфиш|stockfish)[^\d\+\-]*?([+-]?\d+(?:[\.,]\d+)?)'
    match = re.search(kw_pattern, text_clean, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', '.'))
        except ValueError:
            pass

    # 2. Поиск любого десятичного/знакового числа в тексте
    float_pattern = r'([+-]?\d+[\.,]\d+)'
    matches = re.findall(float_pattern, text_clean)
    if matches:
        try:
            return float(matches[0].replace(',', '.'))
        except ValueError:
            pass

    return None

def verify_h1_3(text: str, target_eval: float) -> tuple[bool, bool, float | None]:
    extracted_eval = extract_eval_from_text(text)
    if extracted_eval is None:
        return False, False, None

    diff = abs(extracted_eval - target_eval)
    within_tolerance = diff <= 0.5
    return within_tolerance, True, extracted_eval

def make_prompt(user_prompt: str) -> tuple[list[dict], str]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt}
    ]
    flat_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"
    return messages, flat_prompt

def log_sample_result(idx: int, total: int, model_name: str, target_eval: float, extracted: float | None, is_valid: bool, full_text: str):
    """Печатает результаты: кратко при OK, подробно при ошибке."""
    if is_valid:
        print(f"[{idx}/{total}] {model_name} | Target: {target_eval:+0.2f} | LLM: {extracted:+0.2f} | OK: True 🟢")
    else:
        llm_str = f"{extracted:+0.2f}" if extracted is not None else "N/A"
        print(f"[{idx}/{total}] {model_name} | Target: {target_eval:+0.2f} | LLM: {llm_str} | OK: False 🔴")
        print("  ┌" + "─"*70)
        print("  │ ПОЛНЫЙ ОТВЕТ МОДЕЛИ:")
        for line in full_text.splitlines():
            print(f"  │ {line}")
        print("  └" + "─"*70 + "\n")

def run_h1_3_benchmark():
    if not BENCHMARK_PATH.exists():
        print(f"[Ошибка] {BENCHMARK_PATH} не найден.")
        return

    # ПОЛНЫЙ ПРОГОН НА ВСЕ 20 ПОЗИЦИЙ
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)[:20]

    total = len(benchmark_data)
    results = {"Base_Qwen3": [], "Qwen3_chess_coach": [], "GigaChess": []}
    base_model_name = get_base_model_name()

    def update_metrics(model_key: str):
        data = results[model_key]
        if not data:
            return
        idx = len(data)
        ok_cnt = sum(1 for x in data if x["valid"])
        has_eval_cnt = sum(1 for x in data if x["has_eval"])

        H1_3_ACCURACY_PCT.labels(model=model_key).set((ok_cnt / idx) * 100)
        H1_3_WITHIN_TOLERANCE_COUNT.labels(model=model_key).set(ok_cnt)
        H1_3_HAS_EVAL_COUNT.labels(model=model_key).set(has_eval_cnt)

    # === [1/3] БАЗОВАЯ МОДЕЛЬ ===
    print(f"=== [1/3] Загрузка БАЗОВОЙ модели ({base_model_name}) ===")
    try:
        tokenizer_base = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"\n--- Прогон Base_Qwen3 ({total} позиций) ---")
        for idx, item in enumerate(benchmark_data, 1):
            target_eval = float(item["context_eval"])
            messages, _ = make_prompt(item["prompt"])
            text = tokenizer_base.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_base([text], return_tensors="pt").to(model_base.device)

            with torch.no_grad():
                outputs = model_base.generate(**inputs, max_new_tokens=512, temperature=0.2)

            gen_ids = outputs[:, inputs.input_ids.shape[1]:]
            raw_resp = tokenizer_base.batch_decode(gen_ids, skip_special_tokens=True)[0]
            clean_resp = clean_response(raw_resp)

            is_valid, has_eval, extracted = verify_h1_3(clean_resp, target_eval)
            results["Base_Qwen3"].append({
                "valid": is_valid,
                "has_eval": has_eval,
                "target": target_eval,
                "extracted": extracted
            })
            update_metrics("Base_Qwen3")
            log_sample_result(idx, total, "Base_Qwen3", target_eval, extracted, is_valid, clean_resp)

        del model_base
        del tokenizer_base
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[Ошибка при запуске базовой модели]: {e}")

    # === [2/3] ДООБУЧЕННАЯ МОДЕЛЬ ===
    print(f"\n=== [2/3] Загрузка ДООБУЧЕННОЙ модели из {MERGED_MODEL_PATH} ===")
    try:
        tokenizer_ft = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
        model_ft = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"\n--- Прогон Qwen3_chess_coach ({total} позиций) ---")
        for idx, item in enumerate(benchmark_data, 1):
            target_eval = float(item["context_eval"])
            messages, _ = make_prompt(item["prompt"])
            text = tokenizer_ft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_ft([text], return_tensors="pt").to(model_ft.device)

            with torch.no_grad():
                outputs = model_ft.generate(**inputs, max_new_tokens=512, temperature=0.2)

            gen_ids = outputs[:, inputs.input_ids.shape[1]:]
            raw_resp = tokenizer_ft.batch_decode(gen_ids, skip_special_tokens=True)[0]
            clean_resp = clean_response(raw_resp)

            is_valid, has_eval, extracted = verify_h1_3(clean_resp, target_eval)
            results["Qwen3_chess_coach"].append({
                "valid": is_valid,
                "has_eval": has_eval,
                "target": target_eval,
                "extracted": extracted
            })
            update_metrics("Qwen3_chess_coach")
            log_sample_result(idx, total, "Qwen3_chess_coach", target_eval, extracted, is_valid, clean_resp)

        del model_ft
        del tokenizer_ft
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[Ошибка при запуске дообученной модели]: {e}")

    # === [3/3] GIGACHESS ===
    print(f"\n=== [3/3] Прогон GigaChess (Model RUN API) ===")
    for idx, item in enumerate(benchmark_data, 1):
        target_eval = float(item["context_eval"])
        _, flat_prompt = make_prompt(item["prompt"])

        raw_resp = ask_gigachess(flat_prompt, fen=item.get("fen", ""))
        clean_resp = clean_response(raw_resp or "")
        is_valid, has_eval, extracted = verify_h1_3(clean_resp, target_eval)

        results["GigaChess"].append({
            "valid": is_valid,
            "has_eval": has_eval,
            "target": target_eval,
            "extracted": extracted
        })
        update_metrics("GigaChess")
        log_sample_result(idx, total, "GigaChess", target_eval, extracted, is_valid, clean_resp)

    # === ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ ===
    print("\n" + "="*65)
    print(f"=== ИТОГИ ТЕСТА H1.3 ({total} ПОЗИЦИЙ, ПОРОГ ≥ 95%) ===")
    print("="*65)
    for model_name, res in results.items():
        if not res:
            continue
        ok_cnt = sum(1 for x in res if x["valid"])
        has_eval_cnt = sum(1 for x in res if x["has_eval"])
        
        pct = (ok_cnt / total) * 100
        status = "✅ ПРОЙДЕН (≥ 95%)" if pct >= 95 else "❌ НЕ ПРОЙДЕН (< 95%)"

        print(f"\n📌 Модель: {model_name}")
        print(f"  • Распознано оценок в ответе : {has_eval_cnt}/{total} ({has_eval_cnt/total*100:.0f}%)")
        print(f"  • В пределах допуска (±0.5)  : {ok_cnt}/{total} ({ok_cnt/total*100:.0f}%)")
        print(f"  • ИТОГ H1.3 (Заземление Eval): {ok_cnt}/{total} ({pct:.1f}%) -> {status}")

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus Exporter запущен на http://localhost:8000/metrics\n")
    run_h1_3_benchmark()

    while True:
        time.sleep(1)