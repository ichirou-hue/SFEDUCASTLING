import re
from typing import Tuple

CYRILLIC_TO_LATIN = {
    'А': 'A', 'а': 'A',
    'В': 'B', 'в': 'B',
    'С': 'C', 'с': 'C',
    'D': 'D', 'd': 'D',
}

class StructuralTest:
    name = "01_structural"
    dataset_path = "datasets/01_structural.jsonl"

    def format_prompt(self, item: dict, lang: str = "en") -> str:
        if lang == "ru" and "question_ru" in item:
            q = item["question_ru"]
            opts = item.get("options_ru", item.get("options", {}))
        else:
            q = item.get("question_en", item.get("question", ""))
            opts = item.get("options_en", item.get("options", {}))

        options_text = "\n".join([f"({k}) {v}" for k, v in sorted(opts.items())])
        return f"{q}\n{options_text}\nAnswer with only the letter of the correct option:"

    def evaluate(self, response: str, expected: str, item: dict = None, lang: str = "en") -> Tuple[bool, str]:
        text = response.strip()
        if not text:
            return False, "пустой ответ модели"

        expected_letter = expected.strip().upper()
        expected_letter = CYRILLIC_TO_LATIN.get(expected_letter, expected_letter)

        # 1. Точное совпадение одной буквы в ответе (например: "A", "b", "A.")
        raw_clean = re.sub(r"[.\s:)]", "", text).upper()
        if len(raw_clean) == 1:
            raw_clean = CYRILLIC_TO_LATIN.get(raw_clean, raw_clean)
            if raw_clean in ["A", "B", "C", "D"]:
                if raw_clean == expected_letter:
                    return True, f"модель выбрала вариант ({raw_clean})"
                return False, f"модель выбрала ({raw_clean}), ожидалось ({expected_letter})"

        # 2. Поиск буквы в скобках или с явным маркером: (A), [B], Answer: A, Ответ: (B)
        patterns = [
            r"\(([A-Da-dА-Ва-в])\)",
            r"\[([A-Da-dА-Ва-в])\]",
            r"(?:Answer|Ответ|вариант|option)\s*[:\-]?\s*\(?([A-Da-dА-Ва-в])\)?",
            r"^([A-Da-dА-Ва-в])\b",
            r"\b([A-Da-dА-Ва-в])\b(?=[^a-zA-Zа-яА-Я]*$)"
        ]

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                letter = CYRILLIC_TO_LATIN.get(letter, letter)
                if letter == expected_letter:
                    return True, f"модель выбрала вариант ({letter})"
                return False, f"модель выбрала ({letter}), ожидалось ({expected_letter})"

        # 3. Fallback: поиск по тексту самого правильного варианта (если модель ответила словами)
        if item:
            opts = item.get("options_ru" if lang == "ru" else "options_en", item.get("options", {}))
            expected_text = opts.get(expected_letter, "").strip().lower()
            if expected_text and expected_text in text.lower():
                return True, f"найден текст правильного варианта: '{expected_text}'"

        return False, f"не удалось извлечь букву варианта из ответа: '{text[:50]}...'"