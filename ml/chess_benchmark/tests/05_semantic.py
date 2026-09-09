class SemanticTest:
    name = "05_semantic"
    dataset_path = "datasets/05_semantic.jsonl"

    @staticmethod
    def format_prompt(item: dict, lang: str = "en") -> str:
        if lang == "ru":
            q = item.get("question_ru", item.get("question", ""))
            return f"Ты опытный шахматный тренер. Дай краткое, ясное и точное объяснение:\n\n{q}"
        else:
            q = item.get("question", "")
            return f"You are an experienced chess coach. Provide a concise, clear, and accurate explanation:\n\n{q}"

    @staticmethod
    def evaluate(prediction: str, expected: str = None, item: dict = None, lang: str = "en"):
        pred = prediction.lower().strip()
        if not pred:
            return False, "пустой ответ"

        if lang == "ru":
            concepts = item.get("expected_concepts_ru", item.get("expected_concepts", [])) if item else []
        else:
            concepts = item.get("expected_concepts", []) if item else []

        if not concepts:
            return True, "концепты не заданы"

        # Поиск вхождений концепций (проверка подстроки в нижнем регистре)
        found_concepts = [c for c in concepts if c.lower() in pred]
        coverage = len(found_concepts) / len(concepts) if concepts else 0

        # Порог: покрытие >= 40% или хотя бы 2 точных концепта
        if coverage >= 0.4 or len(found_concepts) >= 2:
            return True, f"OK ({len(found_concepts)}/{len(concepts)} концептов: {', '.join(found_concepts)})"

        missing = [c for c in concepts if c.lower() not in pred]
        return False, f"мало ключевых концептов ({len(found_concepts)}/{len(concepts)} найдено, пропущены: {missing[:3]})"