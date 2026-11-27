"""New prompt contract, without rewriting historical datasets or scoring."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench.cases import build  # noqa: E402
from gigabench.prompts import ANSWER_FIRST, ANSWER_FIRST_INSTRUCTION, apply_answer_protocol  # noqa: E402
from gigabench.scoring import score_case  # noqa: E402


def task(kind, gold, protocol=ANSWER_FIRST):
    return {
        "id": kind, "suite": "structural", "task": kind, "group_id": kind,
        "fen": chess.STARTING_FEN, "score_type": kind, "gold": gold,
        "answer_protocol": protocol,
    }


class PromptContractTests(unittest.TestCase):
    def test_new_dataset_changes_only_prompt_and_protocol(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        current = build(config, "smoke")
        legacy = build(dict(config, answer_protocol="legacy"), "smoke")
        self.assertEqual(len(current), len(legacy))
        for new, old in zip(current, legacy, strict=True):
            self.assertEqual(
                {k: v for k, v in new.items() if k not in ("prompt", "answer_protocol")},
                {k: v for k, v in old.items() if k != "prompt"},
            )
            self.assertEqual(new["answer_protocol"], ANSWER_FIRST)
            self.assertTrue(new["prompt"].endswith(ANSWER_FIRST_INSTRUCTION))
            self.assertNotIn("Ответь только", new["prompt"])
            self.assertNotIn("Только UCI", new["prompt"])
            self.assertNotIn("без пояснений", new["prompt"])

    def test_external_style_prompts_and_expert_explanations(self):
        rows = [
            {"score_type": "human_move", "prompt": "Ответь только одним ходом UCI, без пояснений."},
            {"score_type": "explanation", "prompt": "Объясни ход."},
        ]
        apply_answer_protocol(rows, ANSWER_FIRST)
        self.assertNotIn("без пояснений", rows[0]["prompt"])
        self.assertEqual(rows[1], {"score_type": "explanation", "prompt": "Объясни ход."})
        with self.assertRaisesRegex(ValueError, "Unknown answer protocol"):
            apply_answer_protocol([], "unknown")


class AnswerFirstScoringTests(unittest.TestCase):
    def test_content_and_format_accept_optional_explanations(self):
        examples = (
            ("bool", {"answer": True}, "Да", "Нет, другие фигуры здесь не важны."),
            ("int", {"answer": 20}, "20", "На доске 32 фигуры и 8 пешек у каждой стороны."),
            ("choice", {"answer": "C"}, "C", "Варианты A, B и D не подходят."),
            ("judgement", {"answer": "D"}, "D", "Оценки A и G были бы неверны."),
            ("piece", {"square": "e2", "label": "P"}, "e2: белая пешка", "На e7 стоит чёрная пешка."),
            ("piece", {"square": "e4", "label": "."}, "e4: пусто", "Белая пешка пока на e2."),
            ("move_set", {"moves": ["e2e4", "d2d4"]}, "e2e4, d2d4", "Другой пример хода: g1f3."),
            ("move_set", {"moves": []}, "нет", "Белым здесь нечем ходить."),
            ("human_move", {"move": "e2e4"}, "e2e4", "В отличие от d2d4, этот ход занимает e4."),
            ("puzzle_move", {"moves": ["e2e4"]}, "e2e4", "Ход g1f3 не решает задачу."),
            ("mate_move", {"moves": ["e2e4"]}, "e2e4", "Ход d2d4 не подходит."),
            ("mate_move", {"moves": []}, "нет", "Ход e2e4 возможен, но не ставит мат."),
        )
        for kind, gold, short, explanation in examples:
            for text in (short, f"{short}\n\n{explanation}",
                         f"Ответ: {short}\nПояснение: {explanation}"):
                with self.subTest(kind=kind, text=text):
                    result = score_case(task(kind, gold), {"text": text})
                    self.assertIs(result["content_correct"], True)
                    self.assertTrue(result["protocol_compliant"])
                    self.assertEqual(result["answer_source"], "first_line")
                    self.assertEqual(result["answer_fragment"], short)

    def test_pending_engine_does_not_become_confirmed_success(self):
        result = score_case(task("engine_move", {}), {"text": "e2e4\nПояснение: хороший ход."})
        self.assertEqual(result["predicted"], "e2e4")
        self.assertEqual(result["status"], "pending_engine")
        self.assertIsNone(result["content_correct"])

    def test_wrong_first_answer_is_not_replaced_by_explanation(self):
        result = score_case(task("int", {"answer": 20}), {
            "text": "19\nПояснение: после другого хода было бы 20.",
        })
        self.assertIs(result["content_correct"], False)
        self.assertTrue(result["protocol_compliant"])

    def test_conflicting_explicit_answers_are_unknown(self):
        for text in ("Да\nПояснение: разбор.\nОтвет: Нет", "Ответ: Да\nОтвет: Нет"):
            result = score_case(task("bool", {"answer": True}), {"text": text})
            self.assertIsNone(result["content_correct"])
            self.assertEqual(result["answer_status"], "unparseable")
            self.assertFalse(result["empty"])

    def test_prefaces_and_fenced_examples_are_not_first_answers(self):
        for text in ("Сейчас посчитаем.\nОтвет: 20", "```text\n20\n```"):
            result = score_case(task("int", {"answer": 20}), {"text": text})
            self.assertIsNone(result["content_correct"])
            self.assertFalse(result["protocol_compliant"])

    def test_truncated_explanation_keeps_complete_answer(self):
        for text, expected in (("20\nПояснение: не закон", True), ("20", None)):
            result = score_case(task("int", {"answer": 20}), {"text": text, "truncated": True})
            self.assertIs(result["content_correct"], expected)

    def test_move_set_order_is_not_a_conflict_but_different_sets_are(self):
        c = task("move_set", {"moves": ["e2e4", "d2d4"]})
        for repeated, expected in (("d2d4, e2e4", True), ("e2e4, g1f3", None)):
            result = score_case(c, {"text": f"e2e4, d2d4\nПояснение: ходы.\nОтвет: {repeated}"})
            self.assertIs(result["content_correct"], expected)
        result = score_case(c, {"text": "e2e4, d2d4, e2e4\nОтвет: d2d4, e2e4"})
        self.assertIs(result["content_correct"], True)
        self.assertFalse(result["protocol_compliant"])

    def test_truncated_repeated_answer_does_not_erase_complete_first_line(self):
        c = task("int", {"answer": 20})
        for tail in ("Ответ: 2", "Ответ:", "Ответ:\n2"):
            result = score_case(c, {"text": "20\nПояснение: подсчёт.\n"+tail, "truncated": True})
            self.assertIs(result["content_correct"], True)
        result = score_case(c, {"text": "20\nОтвет: 19\nПояснение: оборвано", "truncated": True})
        self.assertIsNone(result["content_correct"])

    def test_old_unversioned_cases_keep_previous_behavior(self):
        old = task("int", {"answer": 20})
        del old["answer_protocol"]
        result = score_case(old, {"text": "20\nПояснение: число ходов."})
        self.assertIsNone(result["content_correct"])
        result = score_case(task("int", {"answer": 20}), {"text": "20\nПояснение: число ходов."})
        self.assertIs(result["content_correct"], True)


if __name__ == "__main__":
    unittest.main()
