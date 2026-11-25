"""Independent answer_first_v1 boundaries; never infer or load a model."""
import argparse
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ml/gigabench"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench.io import read_jsonl  # noqa: E402
from gigabench.pipeline import prepare, verify  # noqa: E402
from gigabench.scoring import accuracy, score_case  # noqa: E402


def case(kind, gold, protocol="answer_first_v1"):
    row = {"id": "qa:" + kind, "suite": "fixtures", "task": kind, "group_id": "qa",
           "fen": chess.STARTING_FEN, "score_type": kind, "gold": gold}
    if protocol is not None:
        row["answer_protocol"] = protocol
    return row


class AnswerFirstIndependentTests(unittest.TestCase):
    def test_all_automatic_types_accept_bare_and_prefixed_first_lines(self):
        examples = [
            ("bool", {"answer": False}, "Нет"),
            ("int", {"answer": -2}, "-2"),
            ("choice", {"answer": "B"}, "B"),
            ("judgement", {"answer": "G"}, "G"),
            ("piece", {"square": "e2", "label": "P"}, "e2: белая пешка"),
            ("move_set", {"moves": ["e2e4", "d2d4"]}, "e2e4, d2d4"),
            ("human_move", {"move": "e2e4"}, "e2e4"),
            ("puzzle_move", {"moves": ["e2e4"]}, "e2e4"),
            ("mate_move", {"moves": ["e2e4"]}, "e2e4"),
            ("mate_move", {"moves": []}, "нет"),
            ("engine_move", {}, "e2e4"),
        ]
        for kind, gold, short in examples:
            for prefix in ("", "Ответ: "):
                with self.subTest(kind=kind, prefix=prefix):
                    result = score_case(case(kind, gold), {
                        "text": prefix + short + "\nПояснение: A, C, D и g1f3 — другие варианты.",
                    })
                    self.assertTrue(result["protocol_compliant"])
                    self.assertEqual(result["answer_status"], "answered")
                    self.assertEqual(result["answer_source"], "first_line")
                    self.assertIs(result["content_correct"], None if kind == "engine_move" else True)

    def test_content_is_independent_of_uci_compliance(self):
        result = score_case(case("human_move", {"move": "e2e4"}), {
            "text": "Ответ: e4\nПояснение: SAN вместо UCI.",
        })
        self.assertIs(result["content_correct"], True)
        self.assertFalse(result["protocol_compliant"])

    def test_repeated_semantically_identical_answers(self):
        for text in ("Да\nОтвет: Yes", "Ответ: Да\nОтвет: да",
                     "Да\nПояснение: разбор.\nИтоговый ответ: Да"):
            with self.subTest(text=text):
                self.assertIs(score_case(case("bool", {"answer": True}), {
                    "text": text,
                })["content_correct"], True)

    def test_reordered_explicit_move_set_is_not_a_contradiction(self):
        result = score_case(case("move_set", {"moves": ["e2e4", "d2d4"]}), {
            "text": "e2e4, d2d4\nПояснение: оба хода подходят.\nОтвет: d2d4, e2e4",
        })
        self.assertIs(result["content_correct"], True)
        self.assertTrue(result["protocol_compliant"])

    def test_mixed_notation_alternatives_are_unknown(self):
        for short in ("e2e4 / Nf3", "e2e4, Nf3"):
            with self.subTest(short=short):
                result = score_case(case("human_move", {"move": "e2e4"}), {"text": short})
                self.assertIsNone(result["content_correct"])
                self.assertEqual(result["answer_status"], "unparseable")

    def test_conflicting_explicit_answers_across_types(self):
        examples = [
            ("int", {"answer": 20}, "20", "19"),
            ("choice", {"answer": "A"}, "A", "B"),
            ("judgement", {"answer": "D"}, "D", "G"),
            ("human_move", {"move": "e2e4"}, "e2e4", "d2d4"),
            ("piece", {"square": "e2", "label": "P"}, "e2: белая пешка", "e2: пусто"),
            ("move_set", {"moves": ["e2e4"]}, "e2e4", "d2d4"),
            ("mate_move", {"moves": []}, "нет", "e2e4"),
        ]
        for kind, gold, first, other in examples:
            with self.subTest(kind=kind):
                result = score_case(case(kind, gold), {
                    "text": first + "\nПояснение: разбор.\nОтвет: " + other,
                })
                self.assertIsNone(result["content_correct"])
                self.assertEqual(result["answer_status"], "unparseable")

    def test_quoted_or_fenced_explicit_examples_do_not_override_firstline(self):
        for explanation in ("> Ответ: 19", "```text\nОтвет: 19\n```", "~~~\nОтвет: 19\n~~~"):
            with self.subTest(explanation=explanation):
                result = score_case(case("int", {"answer": 20}), {
                    "text": "20\nПояснение: неверный пример:\n" + explanation,
                })
                self.assertIs(result["content_correct"], True)

    def test_blank_first_answers_and_prefaces_do_not_fall_back_to_later_answer(self):
        for text in ("", " \n\t", "Сначала посчитаем.\nОтвет: 20", "Ответ:\n20"):
            with self.subTest(text=text):
                result = score_case(case("int", {"answer": 20}), {"text": text})
                self.assertIsNone(result["content_correct"])
                self.assertFalse(result["protocol_compliant"])

    def test_line_ending_boundaries_for_truncation(self):
        for newline in ("\n", "\r\n", "\r"):
            with self.subTest(newline=newline):
                result = score_case(case("int", {"answer": 20}), {
                    "text": "Ответ: 20" + newline + "Пояснение: незавершён",
                    "truncated": True,
                })
                self.assertIs(result["content_correct"], True)
                self.assertEqual(result["answer_status"], "answered")
        result = score_case(case("int", {"answer": 20}), {"text": "20", "truncated": True})
        self.assertIsNone(result["content_correct"])
        self.assertEqual(result["answer_status"], "budget_without_answer")

    def test_execution_error_does_not_confirm_valid_text(self):
        result = score_case(case("int", {"answer": 20}), {
            "text": "20\nПояснение: разбор.", "error": {"type": "TimeoutError"},
        })
        self.assertEqual(result["answer_status"], "execution_error")
        self.assertIsNone(result["content_correct"])

    def test_pending_engine_excluded_from_success_aggregate(self):
        row = score_case(case("engine_move", {}), {"text": "Ответ: e2e4\nРазбор."})
        result = accuracy([row], {"seed": 1, "resamples": 5})
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["pending"], 1)
        self.assertIsNone(result["accuracy_percent"])

    def test_old_unversioned_and_explicit_legacy_cases_remain_strict(self):
        for protocol in (None, "legacy"):
            row = case("int", {"answer": 20}, protocol)
            with self.subTest(protocol=protocol):
                self.assertIsNone(score_case(row, {
                    "text": "20\nПояснение: число ходов.",
                })["content_correct"])
                self.assertIs(score_case(row, {"text": "20"})["content_correct"], True)


class PrepareAnswerFirstIndependentTests(unittest.TestCase):
    def test_prepare_freezes_protocol_and_hashes_without_inference(self):
        # Temporary artifacts live inside tests, never in completed benchmark runs.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output = Path(tmp) / "new-run"
            args = argparse.Namespace(
                out=str(output), config=str(ROOT / "config/default.json"),
                without_legacy=True, profile="smoke", backend="qwen",
                max_new_tokens=128, games=None, puzzles=None, with_judgement=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(args)
            frozen, rows = verify(output)
            self.assertEqual(frozen, manifest)
            self.assertEqual(frozen["config"]["answer_protocol"], "answer_first_v1")
            self.assertEqual(frozen["config"]["generation"]["max_new_tokens"], 128)
            self.assertIn("gigabench/prompts.py", frozen["code_sha256"])
            self.assertTrue(all(row.get("answer_protocol") == "answer_first_v1" for row in rows))
            queries = read_jsonl(output / "queries.jsonl")
            self.assertEqual(len(queries), len(rows))
            self.assertTrue(all(q["max_new_tokens"] == 128 for q in queries))
            self.assertFalse((output / "responses.jsonl").exists())
            self.assertFalse((output / "run.json").exists())
            original = copy.deepcopy(rows)
            # Reading the frozen data must not mutate the new answer contract.
            self.assertEqual(verify(output)[1], original)
            with (output / "queries.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen dataset modified"):
                verify(output)


if __name__ == "__main__":
    unittest.main()
