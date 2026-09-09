import re

class MotifsTest:
    name = "02_motifs"
    dataset_path = "datasets/02_motifs.jsonl"

    @staticmethod
    def format_prompt(item: dict, lang: str = "en") -> str:
        if lang == "ru":
            q = item.get("question_ru", item.get("question", ""))
            return f"Вопрос: {q}\nОтветь строго 'Да' или 'Нет'."
        else:
            q = item.get("question", "")
            return f"Question: {q}\nAnswer strictly 'Yes' or 'No'."

    @staticmethod
    def evaluate(prediction: str, expected: str, item: dict = None, lang: str = "en"):
        pred_lower = prediction.lower().strip()
        exp_lower = str(expected).lower().strip()

        if not pred_lower:
            return False, "пустой ответ"

        expected_bool = exp_lower in ["yes", "да", "true", "1"]

        # Паттерны для распознавания утверждения и отрицания на RU и EN
        yes_patterns = [
            r"\b(yes|да|находится|атакован|связан|под шахом|под связкой|в шахе|защищена|защищен)\b"
        ]
        no_patterns = [
            r"\b(no|нет|не находится|не атакован|не связан|не под шахом|не в шахе|не защищена|не защищен|не является)\b"
        ]

        is_no = any(re.search(pat, pred_lower) for pat in no_patterns)
        is_yes = any(re.search(pat, pred_lower) for pat in yes_patterns)

        # Отрицание имеет приоритет (если встретилось "не находится под шахом")
        if is_no:
            verdict = False
        elif is_yes:
            verdict = True
        else:
            return False, f"мотив/шах не определен в ответе: '{pred_lower[:40]}...'"

        if verdict == expected_bool:
            return True, f"OK ({'Да' if verdict else 'Нет'})"
        
        got_str = "Да" if verdict else "Нет"
        exp_str = "Да" if expected_bool else "Нет"
        return False, f"ошибка: модель ответила '{got_str}' вместо '{exp_str}'"