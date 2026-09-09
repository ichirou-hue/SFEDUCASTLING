import re
from typing import Tuple

def normalize_move(move_str: str) -> str:
    return re.sub(r"[+#x=!?\s]", "", move_str).lower()

class ShortTacticsTest:
    name = "03_short_tactics"
    dataset_path = "datasets/03_short_tactics.jsonl"

    def format_prompt(self, item: dict, lang: str = "en") -> str:
        if lang == "ru" and "question_ru" in item:
            return item["question_ru"]
        return item.get("question_en", item.get("question", ""))

    def evaluate(self, response: str, expected: str, item: dict = None, lang: str = "en") -> Tuple[bool, str]:
        text = response.strip()
        if not text:
            return False, "пустой ответ модели"

        expected_san = item.get("expected_san", expected) if item else expected
        expected_uci = item.get("expected_uci", "") if item else ""

        expected_san_norm = normalize_move(expected_san)
        expected_uci_norm = normalize_move(expected_uci) if expected_uci else ""

        tokens = re.findall(r"[a-zA-Z0-9+#x=\-]+", text)

        for token in tokens:
            token_norm = normalize_move(token)
            if token_norm == expected_san_norm:
                return True, f"найден точный SAN ход: '{token}'"
            if expected_uci_norm and token_norm == expected_uci_norm:
                return True, f"найден точный UCI ход: '{token}'"

        return False, f"неверный тактический ход (модель предложила: '{text[:40]}', ожидалось: {expected_san} / {expected_uci})"