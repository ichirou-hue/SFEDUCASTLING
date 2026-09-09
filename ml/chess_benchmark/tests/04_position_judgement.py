import re
from typing import Tuple

class PositionJudgementTest:
    name = "04_position_judgement"
    dataset_path = "datasets/04_position_judgement.jsonl"

    def format_prompt(self, item: dict, lang: str = "en") -> str:
        if lang == "ru" and "question_ru" in item:
            return item["question_ru"]
        return item.get("question_en", item.get("question", ""))

    def evaluate(self, response: str, expected: str, item: dict = None, lang: str = "en") -> Tuple[bool, str]:
        text = response.strip()
        if not text:
            return False, "пустой ответ модели"

        expected_clean = expected.strip().lower()
        first_sentence = re.split(r"[.!?\n]", text)[0].strip()

        # 1. Проверка вопросов на оценку позиции (Draw)
        if expected_clean in ["draw", "ничья"]:
            draw_markers = [r"\bdraw\b", r"\bничь[яеию]\b", r"\bтеоретическая ничья\b", r"\bequal\b"]
            for marker in draw_markers:
                if re.search(marker, text, re.IGNORECASE):
                    return True, f"найден маркер ничьей: '{marker}'"
            return False, f"ожидалась оценка '{expected}', модель дала: '{text[:60]}...'"

        # 2. Бинарные вопросы (пат, легальность, Yes/No)
        pos_direct = [r"^(yes|да)\b", r"\b(is legal|легально|можно|в пате|stalemate)\b"]
        neg_direct = [r"^(no|нет)\b", r"\b(is not legal|is illegal|нелегально|нельзя|не в пате|not stalemate)\b"]

        is_pos = any(re.search(p, first_sentence, re.IGNORECASE) for p in pos_direct)
        is_neg = any(re.search(p, first_sentence, re.IGNORECASE) for p in neg_direct)

        if is_pos and not is_neg:
            detected = "yes"
        elif is_neg and not is_pos:
            detected = "no"
        else:
            cleaned_body = re.sub(r"\bno legal (moves|squares)\b", "", text, flags=re.IGNORECASE)
            cleaned_body = re.sub(r"\bнет легальных ходов\b", "", cleaned_body, flags=re.IGNORECASE)
            if re.search(r"\b(yes|да|легальн|stalemate)\b", cleaned_body, re.IGNORECASE):
                detected = "yes"
            elif re.search(r"\b(no|нет|нелегальн|нельзя)\b", cleaned_body, re.IGNORECASE):
                detected = "no"
            else:
                return False, f"не удалось классифицировать статус: '{text[:50]}...'"

        if detected == expected_clean:
            return True, f"верно определен статус: '{detected}'"

        return False, f"модель ответила '{detected}', ожидалось '{expected_clean}'"