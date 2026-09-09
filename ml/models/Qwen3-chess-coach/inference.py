import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./outputs/merged"

def run_inference(fen: str, move: str, eval_score: str, best_move: str, show_thinking: bool = False) -> str:
    print(f"Загрузка модели из {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Системная инструкция с правилами русского шахматного языка
    system_instruction = (
        "Ты — опытный шахматный гроссмейстер и заботливый тренер. "
        "Твоя задача — анализировать ходы игрока и давать понятные, обучающие пояснения на русском языке.\n\n"
        "СТРОГИЕ ПРАВИЛА ТЕРМИНОЛОГИИ:\n"
        "1. Никогда не путай фигуру, КОТОРОЙ бьют, и фигуру, КОТОРУЮ бьют. "
        "Например, нотация 'Qxf7#' означает 'Ферзь забирает пешку (или фигуру) на f7 и ставит мат', а НЕ 'забирает ферзя'.\n"
        "2. Используй правильные русские названия фигур: Пешка, Конь, Слон, Ладья, Ферзь, Король.\n"
        "3. Объясняй логику понятным языком, давай практические советы по поиску ходов (шахи, взятия, нападения)."
    )

    # Запрос на русском языке
    user_prompt = (
        f"Позиция (FEN): {fen}\n"
        f"Сделанный ход: {move}\n"
        f"Оценка позиция движком: {eval_score}\n"
        f"Лучший ход по мнению движка: {best_move}\n\n"
        f"Проанализируй этот ход. Объясни ученику, почему ход {move} является ошибкой по сравнению с лучшим ходом {best_move}, и дай совет, как находить такие решения."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9
    )

    # Отрезаем промпт, оставляя только ответ модели
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Очистка от тегов размышления <think>при необходимости
    if not show_thinking and "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response

if __name__ == "__main__":
    # Тестовые данные (детский мат)
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    test_move = "Nf3"
    test_eval = "#M1 for White"
    test_best_move = "Qxf7#"

    answer = run_inference(
        fen=test_fen, 
        move=test_move, 
        eval_score=test_eval, 
        best_move=test_best_move, 
        show_thinking=False
    )
    
    print("\n" + "="*50)
    print("ОТВЕТ ТРЕНЕРА:")
    print("="*50)
    print(answer)
