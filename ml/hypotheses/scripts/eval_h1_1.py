import json
import time
import gc
import re
import chess
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prometheus_client import start_http_server, Gauge
from sber_client import ask_gigachess

# --- PROMETHEUS METRICS ---
H1_1_STRUCTURED_PCT = Gauge(
    'chess_h1_1_structured_pct', 
    'H1.1: Итоговый % прохождения (ход + eval + термин)', 
    ['model']
)

# Компонентные метрики для построения Stacked Bar Chart в Grafana (0..20)
H1_1_HAS_MOVE_COUNT = Gauge('chess_h1_1_has_move_count', 'Количество ответов с ходом (Move)', ['model'])
H1_1_HAS_EVAL_COUNT = Gauge('chess_h1_1_has_eval_count', 'Количество ответов с оценкой (Eval)', ['model'])
H1_1_HAS_TERM_COUNT = Gauge('chess_h1_1_has_term_count', 'Количество ответов с термином (Term)', ['model'])

BENCHMARK_PATH = Path("benchmark.json")
MERGED_MODEL_PATH = Path("../Qwen3-chess-coach/outputs/merged").resolve()
ADAPTER_CONFIG_PATH = Path("../Qwen3-chess-coach/outputs/adapter_config.json").resolve()

# Общий стандартный системный промпт
SYSTEM_INSTRUCTION = "Ты — шахматный аналитик и тренер."

# Словарь стратегических шахматных терминов для H1.1
CHESS_STRATEGIC_TERMS = [
    "центр", "диагональ", "вертикаль", "горизонталь", "связка", "вилка", 
    "темп", "структура", "инициатива", "пространство", "слабость", "развитие", 
    "рокировка", "атака", "защита", "преимущество", "контроль", "изолирован",
    "проходная", "оппозиция", "форточка", "активность", "фигура", "пешка"
]

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

def has_move_in_text(text: str, fen: str) -> bool:
    """Проверяет наличие любого упоминания хода (UCI, SAN или русская нотация)."""
    text_lower = text.lower()

    # 1. UCI формат (e2e4, g1f3 и т.д.)
    if re.search(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', text_lower):
        return True

    # 2. SAN нотация
    try:
        board = chess.Board(fen)
        for move in board.legal_moves:
            san_en = board.san(move).lower()
            if re.search(r'\b' + re.escape(san_en) + r'\b', text_lower):
                return True
    except Exception:
        pass

    return False

def verify_h1_1(text: str, fen: str) -> tuple[bool, dict]:
    """
    Проверка H1.1:
    - Присутствует ход
    - Присутствует оценка позиции (число, +/-, или слова eval/оценка)
    - Присутствует >= 1 стратегического термина
    """
    text_lower = text.lower()

    # 1. Наличие хода
    has_move = has_move_in_text(text, fen)

    # 2. Наличие формата оценки позиции (eval)
    has_eval = bool(
        re.search(r'[\+\-]?\d+[\.,]\d+|0\.0|[+-]?\d+\s*пешк|оценк|eval', text_lower)
    )

    # 3. Наличие стратегического термина
    has_term = any(term in text_lower for term in CHESS_STRATEGIC_TERMS)

    h1_1_passed = has_move and has_eval and has_term

    details = {
        "has_move": has_move,
        "has_eval": has_eval,
        "has_term": has_term
    }
    return h1_1_passed, details

def make_clean_prompt(fen: str) -> tuple[list[dict], str]:
    """Супер-простой универсальный промпт без подсказок и контекста движка."""
    user_content = (
        f"Позиция FEN: {fen}\n"
        f"Проанализируй позицию: укажи лучший ход, дай оценку позиции (eval) "
        f"и приведи краткое стратегическое пояснение."
    )

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content}
    ]

    flat_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_content}"
    return messages, flat_prompt

def run_h1_1_benchmark():
    if not BENCHMARK_PATH.exists():
        print(f"[Ошибка] {BENCHMARK_PATH} не найден.")
        return

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)[:20]  # Выборка на 20 позиций

    total = len(benchmark_data)
    results = {"Base_Qwen3": [], "Qwen3_chess_coach": [], "GigaChess": []}
    base_model_name = get_base_model_name()

    def update_metrics(model_key: str):
        data = results[model_key]
        if not data:
            return
        idx = len(data)
        h1_1_cnt = sum(1 for x in data if x["h1_1"])
        move_cnt = sum(1 for x in data if x["move"])
        eval_cnt = sum(1 for x in data if x["eval"])
        term_cnt = sum(1 for x in data if x["term"])

        H1_1_STRUCTURED_PCT.labels(model=model_key).set((h1_1_cnt / idx) * 100)
        H1_1_HAS_MOVE_COUNT.labels(model=model_key).set(move_cnt)
        H1_1_HAS_EVAL_COUNT.labels(model=model_key).set(eval_cnt)
        H1_1_HAS_TERM_COUNT.labels(model=model_key).set(term_cnt)

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
            messages, _ = make_clean_prompt(item["fen"])
            text = tokenizer_base.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_base([text], return_tensors="pt").to(model_base.device)

            with torch.no_grad():
                outputs = model_base.generate(**inputs, max_new_tokens=768, temperature=0.3)

            gen_ids = outputs[:, inputs.input_ids.shape[1]:]
            raw_resp = tokenizer_base.batch_decode(gen_ids, skip_special_tokens=True)[0]
            clean_resp = clean_response(raw_resp)

            h1_1_passed, d = verify_h1_1(clean_resp, item["fen"])
            results["Base_Qwen3"].append({
                "h1_1": h1_1_passed,
                "move": d["has_move"],
                "eval": d["has_eval"],
                "term": d["has_term"]
            })
            update_metrics("Base_Qwen3")

            print(f"[{idx}/{total}] Base_Qwen3 | H1.1: {h1_1_passed} [Ход:{d['has_move']} | Eval:{d['has_eval']} | Термин:{d['has_term']}]")

        del model_base
        del tokenizer_base
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[Ошибка при запуске базовой модели]: {e}")

    # === [2/3] ДООБУЧЕННАЯ МОДЕЛЬ ===
    print(f"\n=== [2/3] Загрузка ДООБУЧЕННОЙ модели из {MERGED_MODEL_PATH} ===")
    tokenizer_ft = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    model_ft = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"\n--- Прогон Qwen3_chess_coach ({total} позиций) ---")
    for idx, item in enumerate(benchmark_data, 1):
        messages, _ = make_clean_prompt(item["fen"])
        text = tokenizer_ft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer_ft([text], return_tensors="pt").to(model_ft.device)

        with torch.no_grad():
            outputs = model_ft.generate(**inputs, max_new_tokens=768, temperature=0.3)

        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        raw_resp = tokenizer_ft.batch_decode(gen_ids, skip_special_tokens=True)[0]
        clean_resp = clean_response(raw_resp)

        h1_1_passed, d = verify_h1_1(clean_resp, item["fen"])
        results["Qwen3_chess_coach"].append({
            "h1_1": h1_1_passed,
            "move": d["has_move"],
            "eval": d["has_eval"],
            "term": d["has_term"]
        })
        update_metrics("Qwen3_chess_coach")

        print(f"[{idx}/{total}] Qwen3_chess_coach | H1.1: {h1_1_passed} [Ход:{d['has_move']} | Eval:{d['has_eval']} | Термин:{d['has_term']}]")

    del model_ft
    del tokenizer_ft
    torch.cuda.empty_cache()
    gc.collect()

    # === [3/3] GIGACHESS ===
    print(f"\n=== [3/3] Прогон GigaChess (Model RUN API) ===")
    for idx, item in enumerate(benchmark_data, 1):
        fen = item["fen"]
        _, flat_prompt = make_clean_prompt(fen)

        raw_resp = ask_gigachess(flat_prompt, fen=fen)
        h1_1_passed, d = verify_h1_1(raw_resp or "", fen)

        results["GigaChess"].append({
            "h1_1": h1_1_passed,
            "move": d["has_move"],
            "eval": d["has_eval"],
            "term": d["has_term"]
        })
        update_metrics("GigaChess")

        print(f"[{idx}/{total}] GigaChess | H1.1: {h1_1_passed} [Ход:{d['has_move']} | Eval:{d['has_eval']} | Термин:{d['has_term']}]")

    # === ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ ===
    print("\n" + "="*65)
    print("=== ИТОГИ ПРОВЕРКИ H1.1 (20 ПОЗИЦИЙ, ПОРОГ ≥ 90%) ===")
    print("="*65)
    for model_name, res in results.items():
        if not res:
            continue
        h1_1_cnt = sum(1 for x in res if x["h1_1"])
        move_cnt = sum(1 for x in res if x["move"])
        eval_cnt = sum(1 for x in res if x["eval"])
        term_cnt = sum(1 for x in res if x["term"])
        
        pct = (h1_1_cnt / total) * 100
        status = "✅ ПРОЙДЕН (≥ 90%)" if pct >= 90 else "❌ НЕ ПРОЙДЕН (< 90%)"

        print(f"\n📌 Модель: {model_name}")
        print(f"  • Распознано ходов (Move)      : {move_cnt}/{total} ({move_cnt/total*100:.0f}%)")
        print(f"  • Наличие оценки (Eval)        : {eval_cnt}/{total} ({eval_cnt/total*100:.0f}%)")
        print(f"  • Стратегический термин (Term) : {term_cnt}/{total} ({term_cnt/total*100:.0f}%)")
        print(f"  • ИТОГ H1.1 (Ход + Eval + Term): {h1_1_cnt}/{total} ({pct:.1f}%) -> {status}")

if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus Exporter запущен на http://localhost:8000/metrics\n")
    run_h1_1_benchmark()

    while True:
        time.sleep(1)