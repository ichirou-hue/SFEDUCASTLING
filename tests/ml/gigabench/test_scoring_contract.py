"""Content/format/report contracts; frozen GPU reproductions are inline only."""

import argparse
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402 -- standalone unittest import path
from gigabench import parsing, report  # noqa: E402
from gigabench.cases import build  # noqa: E402
from gigabench.io import read_jsonl, sha  # noqa: E402
from gigabench.pipeline import prepare  # noqa: E402
from gigabench.scoring import score_case, summarize  # noqa: E402
from test_benchmark import temporary_workspace  # noqa: E402


def case(kind, gold):
    return {
        "id": "contract:" + kind, "group_id": "contract",
        "suite": "structural", "task": "contract",
        "fen": chess.STARTING_FEN, "prompt": "Вопрос?",
        "score_type": kind, "gold": gold,
    }


class FrozenTokenBudgetTests(unittest.TestCase):
    def prepare_run(self, out, backend, budget=None, config=None):
        return prepare(argparse.Namespace(
            out=out, config=config or ROOT / "config/default.json",
            backend=backend, max_new_tokens=budget,
            without_legacy=True, profile="smoke",
        ))

    def assert_frozen_budget(self, out, manifest, expected):
        queries = read_jsonl(out / "queries.jsonl")
        cases = read_jsonl(out / "cases.jsonl")
        self.assertTrue(queries)
        self.assertEqual({q["max_new_tokens"] for q in queries}, {expected})
        self.assertEqual(manifest["config"]["generation"]["max_new_tokens"], expected)
        self.assertEqual(manifest["queries_sha256"], sha(out / "queries.jsonl"))
        return cases

    def test_qwen_default_is_2048_without_changing_cases_or_gold(self):
        with temporary_workspace() as tmp:
            out = Path(tmp) / "qwen"
            manifest = self.prepare_run(out, "qwen")
            cases = self.assert_frozen_budget(out, manifest, 2048)
            baseline = Path(tmp) / "mock"
            self.prepare_run(baseline, "mock")
            self.assertEqual(cases, read_jsonl(baseline / "cases.jsonl"))
            self.assertIn(32, {c["max_new_tokens"] for c in cases})

    def test_explicit_4096_overrides_config_and_is_frozen(self):
        with temporary_workspace() as tmp:
            config = json.loads((ROOT / "config/default.json").read_text())
            config["generation"]["max_new_tokens"] = 512
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            out = Path(tmp) / "qwen"
            manifest = self.prepare_run(out, "qwen", budget=4096, config=path)
            self.assert_frozen_budget(out, manifest, 4096)
            self.assertEqual(json.loads(path.read_text())["generation"]["max_new_tokens"], 512)

    def test_mock_retains_original_per_case_budgets(self):
        with temporary_workspace() as tmp:
            out = Path(tmp) / "mock"
            manifest = self.prepare_run(out, "mock")
            cases = {c["id"]: c for c in read_jsonl(out / "cases.jsonl")}
            queries = read_jsonl(out / "queries.jsonl")
            self.assertTrue(queries)
            for query in queries:
                self.assertEqual(query["max_new_tokens"], cases[query["id"]]["max_new_tokens"])
            self.assertEqual({q["max_new_tokens"] for q in queries}, {32, 64, 512})
            self.assertNotIn("max_new_tokens", manifest["config"]["generation"])

    def test_invalid_config_budgets_fail_before_artifacts_are_created(self):
        with temporary_workspace() as tmp:
            for value in (0, -1, True, 1.5, "2048"):
                with self.subTest(value=value):
                    config = json.loads((ROOT / "config/default.json").read_text())
                    config["generation"]["max_new_tokens"] = value
                    path = Path(tmp) / "config.json"
                    path.write_text(json.dumps(config))
                    out = Path(tmp) / "bad-run"
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        self.prepare_run(out, "qwen", config=path)
                    self.assertFalse(out.exists())


class AllScoreTypesContractTests(unittest.TestCase):
    # Two valid, different labels for every deterministic score type.
    TABLE = (
        ("bool", {"answer": True}, {"answer": False}, "Да", True),
        ("int", {"answer": 19}, {"answer": 20}, r"\boxed{19}", 19),
        ("choice", {"answer": "C"}, {"answer": "A"}, "C", "C"),
        ("judgement", {"answer": "G"}, {"answer": "A"}, "G", "G"),
        ("piece", {"square": "e4", "label": "P"},
         {"square": "e4", "label": "."}, "На e4 белая пешка.", "P"),
        ("human_move", {"move": "e2e4"}, {"move": "d2d4"}, "e2e4", "e2e4"),
        ("puzzle_move", {"moves": ["e2e4"]}, {"moves": ["d2d4"]}, "e2e4", "e2e4"),
        ("mate_move", {"moves": ["e2e4"]}, {"moves": ["d2d4"]}, "e2e4", "e2e4"),
        ("move_set", {"moves": ["e2e4"]}, {"moves": ["d2d4"]}, "[e2e4]", ["e2e4"]),
    )

    def test_final_answers_all_deterministic_types_are_gold_independent(self):
        for kind, matching, different, fragment, expected in self.TABLE:
            text = "Обсуждение: да и нет; A/B/C/D; e2e4 или d2d4; на a1 белая ладья.\n\nFinal answer: " + fragment
            for gold, correct in ((matching, True), (different, False)):
                with self.subTest(kind=kind, gold=gold):
                    scored = score_case(case(kind, gold), {"text": text})
                    self.assertEqual(scored["predicted"], expected)
                    self.assertIs(scored["content_correct"], correct)
                    self.assertIs(scored["correct"], correct)
                    self.assertEqual(scored["answer_status"], "answered")
                    self.assertEqual(scored["answer_source"], "final_block")
                    self.assertEqual(scored["answer_fragment"], fragment)
                    self.assertEqual(
                        report.verdict_text(scored), "верно" if correct else "неверный ответ",
                    )

    def test_budget_without_answer_all_types_preserves_failure_denominator(self):
        for kind, gold, *_ in self.TABLE:
            with self.subTest(kind=kind):
                scored = score_case(
                    case(kind, gold),
                    {"text": "Рассмотрим позицию подробнее", "truncated": True},
                )
                self.assertFalse(scored["correct"])
                self.assertIsNone(scored["content_correct"])
                self.assertEqual(scored["answer_status"], "budget_without_answer")
                self.assertIsNone(scored["answer_source"])
                self.assertIsNone(scored["answer_fragment"])
                self.assertIn("не оценено", report.verdict_text(scored))

    def test_negative_mate_is_answered_with_null_move_prediction(self):
        for gold, expected in (({"moves": []}, True), ({"moves": ["e2e4"]}, False)):
            with self.subTest(gold=gold):
                c = case("mate_move", gold)
                response = {"text": "Анализ: e2e4 или d2d4.\n\nИтог: Нет"}
                scored = score_case(c, response)
                self.assertIsNone(scored["predicted"])
                self.assertEqual(scored["answer_status"], "answered")
                self.assertIs(scored["content_correct"], expected)
                self.assertEqual(scored["answer_source"], "final_block")
                self.assertEqual(
                    report.prediction_text(c, response, scored),
                    "Нет матующего хода (ответ «нет»)",
                )

    def test_engine_pending_is_not_a_wrong_chess_answer(self):
        scored = score_case(
            case("engine_move", {}),
            {"text": "Анализ: d2d4 или e2e4.\n\nОтвет: e2e4"},
        )
        self.assertEqual(scored["predicted"], "e2e4")
        self.assertEqual(scored["answer_status"], "answered")
        self.assertEqual(scored["status"], "pending_engine")
        self.assertIsNone(scored["correct"])
        self.assertIsNone(scored["content_correct"])
        self.assertIn("ожидается", report.verdict_text(scored))

    def test_engine_verdict_depends_on_engine_not_case_gold(self):
        class Engine:
            def grade(self, _fen, move):
                return {"accepted": move == "e2e4", "ordinary": True, "legal": True}

        for move, expected in (("e2e4", True), ("d2d4", False)):
            with self.subTest(move=move):
                scored = score_case(
                    case("engine_move", {}),
                    {"text": "Ответ: " + move}, engine=Engine(),
                )
                self.assertEqual(scored["predicted"], move)
                self.assertIs(scored["content_correct"], expected)
                self.assertEqual(scored["answer_status"], "answered")

    def test_explanation_pending_is_not_a_wrong_chess_answer(self):
        scored = score_case(
            case("explanation", {}),
            {"text": "Итог: Слон развивается и контролирует центр."},
        )
        self.assertEqual(scored["status"], "pending_coach_ratings")
        self.assertIsNone(scored["correct"])
        self.assertIsNone(scored["content_correct"])
        self.assertIn("ожидается", report.verdict_text(scored))

    def test_generated_and_inline_move_cases_cover_every_configured_block(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        cases = build(config, "smoke")
        scores = [
            score_case(c, {"text": "", "truncated": True})
            for c in cases if c["score_type"] != "explanation"
        ]
        # Real move_choice corpus is optional/private; inline cases cover routing only.
        for kind, task, gold in (
            ("human_move", "M2", {"move": "e2e4"}),
            ("engine_move", "M3", {}),
        ):
            c = case(kind, gold)
            c.update(suite="legacy", task=task)
            scores.append(score_case(c, {"text": "", "truncated": True}))
        summary = summarize(scores, config)
        self.assertEqual(set(summary["by_block"]), set(config["blocks"]))
        for block, metrics in summary["by_block"].items():
            with self.subTest(block=block):
                self.assertTrue(metrics["tasks"])
        self.assertTrue(all(row["content_correct"] is None for row in scores))


class FinalAnswerContractTests(unittest.TestCase):
    def test_final_integer_does_not_discard_a_retraction_or_alternative(self):
        for continuation in (
            "или 20",
            "Нет, это неверно; должно быть 20.",
            "Пояснение:\nНет, правильное значение 20.",
            "Пояснение: Нет, правильное значение 20.",
        ):
            with self.subTest(continuation=continuation):
                self.assertIsNone(parsing.parse_scalar("int", "Ответ:\n19\n" + continuation))
        self.assertEqual(parsing.parse_scalar("int", "Ответ:\n19\nПояснение: Подсчёт даёт 19."), 19)

    def test_markdown_numbered_and_emoji_final_markers(self):
        for marker in (
            "Ответ:", "### **Ответ:**", "Итоговый ответ:",
            "Final answer:", "Итог:", "5. Ответ:", "### ✅ Ответ:",
            "### 🎯 Итог:",
        ):
            with self.subTest(marker=marker):
                text = "Обсудим варианты A/B и отрицательное нет.\n\n" + marker + " C"
                self.assertEqual(parsing.parse_scalar("choice", text), "C")

    def test_conflicting_final_blocks_reject_and_agreeing_blocks_accept(self):
        self.assertIsNone(parsing.parse_scalar("choice", "Ответ: A\n\nИтог: B"))
        self.assertEqual(parsing.parse_scalar("choice", "Ответ: A\n\nИтог: A"), "A")
        self.assertIsNone(parsing.parse_bool("Ответ: Да\n\nFinal answer: No"))

    def test_quoted_and_fenced_markers_are_not_final_answers(self):
        for text in (
            "Пример:\n> Ответ: A",
            "Пример:\n```\nОтвет: A\n```",
            'Пример: "Ответ: A"',
        ):
            with self.subTest(text=text):
                self.assertEqual(parsing.final_answers(text), [])
                self.assertIsNone(parsing.parse_scalar("choice", text))

    def test_all_parser_types_recover_explicit_final_block(self):
        prefix = "Анализ: e2e4 или d2d4, на a1 белая ладья; да и нет.\n\n"
        self.assertIs(parsing.parse_bool(prefix + "Ответ: Нет"), False)
        self.assertEqual(parsing.parse_move(chess.STARTING_FEN, prefix + "Ответ: e2e4"), "e2e4")
        self.assertEqual(parsing.parse_piece(prefix + "Ответ: На e4 пусто.", "e4"), ".")
        self.assertEqual(parsing.parse_set(prefix + "Ответ: [e2e4, d2d4]"), ["e2e4", "d2d4"])
        self.assertEqual(parsing.parse_scalar("int", prefix + r"Ответ: $\boxed{−19}$"), -19)

    def test_mismatched_move_set_brackets_rejected(self):
        for text in ("[e2e4", "e2e4]", "Ответ: [e2e4", "Ответ: e2e4]"):
            with self.subTest(text=text):
                self.assertIsNone(parsing.parse_set(text))

    def test_piece_target_paragraphs_agree_or_conflict(self):
        self.assertEqual(
            parsing.parse_piece("На a1 белая ладья.\n\nНа e4 белая пешка.\n\nНа e4 белая пешка.", "e4"),
            "P",
        )
        self.assertIsNone(
            parsing.parse_piece("На a1 белая ладья.\n\nНа e4 белая пешка.\n\nНа e4 пусто.", "e4")
        )

    def test_piece_fallback_does_not_discard_ambiguous_target_claim(self):
        text = "На a1 белая ладья.\n\nНа e4 белая пешка.\n\nНа e4 чёрная пешка или белый конь."
        self.assertIsNone(parsing.parse_piece(text, "e4"))

    def test_tilde_fenced_answer_example_is_not_a_conclusion(self):
        text = "Пример формата:\n~~~text\nОтвет: 19\n~~~"
        self.assertEqual(parsing.final_answers(text), [])
        self.assertIsNone(parsing.parse_scalar("int", text))

    def test_prediction_does_not_depend_on_bool_or_choice_gold(self):
        for kind, text, golds, expected in (
            ("bool", "Анализ: да и нет.\n\nОтвет: Нет", [True, False], False),
            ("choice", "Анализ: A/B/D.\n\nОтвет: C", ["A", "C"], "C"),
        ):
            for gold in golds:
                with self.subTest(kind=kind, gold=gold):
                    scored = score_case(case(kind, {"answer": gold}), {"text": text})
                    self.assertEqual(scored["predicted"], expected)
                    self.assertEqual(scored["content_correct"], expected == gold)
                    self.assertEqual(scored["answer_status"], "answered")
                    self.assertEqual(scored["answer_source"], "final_block")

    def test_budget_without_answer_is_unknown_not_a_chess_error(self):
        c = case("choice", {"answer": "A"})
        for response, status in (
            ({"text": "Рассмотрим позицию", "truncated": True}, "budget_without_answer"),
            ({"text": ""}, "no_answer"),
            ({"text": "Рассмотрим позицию"}, "unparseable"),
            ({"text": "Ответ: A", "error": "failed"}, "execution_error"),
        ):
            with self.subTest(status=status):
                scored = score_case(c, response)
                self.assertEqual(scored["answer_status"], status)
                self.assertIsNone(scored["content_correct"])
                self.assertFalse(scored["correct"])
                self.assertNotEqual(report.verdict_text(scored), "неверный ответ")

    def test_complete_final_answer_before_budget_is_content_checked(self):
        for selected in ("A", "B"):
            with self.subTest(selected=selected):
                scored = score_case(
                    case("choice", {"answer": "A"}),
                    {"text": "Анализ: A/B/C/D.\n\nОтвет: " + selected, "truncated": True},
                )
                self.assertEqual(scored["predicted"], selected)
                self.assertEqual(scored["content_correct"], selected == "A")
                self.assertEqual(scored["answer_status"], "answered")
                self.assertFalse(scored["protocol_compliant"])

    def test_inline_budget1024_piece_and_choice_captures(self):
        # Verbatim final tails from the ignored control run, no fixture dependency.
        for square, label, text in (
            ("e4", ".", "### Ответ:\n**Клетка e4 пуста.**"),
            ("e6", "p", "### ✅ Ответ:\n**Клетка e6 пуста.**"),
        ):
            with self.subTest(square=square):
                scored = score_case(case("piece", {"square": square, "label": label}), {"text": text})
                self.assertEqual(scored["predicted"], ".")
                self.assertEqual(scored["content_correct"], label == ".")
                self.assertEqual(scored["answer_status"], "answered")
        text = (
            "### Проверка других вариантов:\n"
            "- **B) После хода сходившая фигура ничем своим не защищена.**\n"
            "- **C) После хода слон на e2 нападает на ферзя соперника.**\n"
            "- **D) После хода сходившую фигуру соперник не атакует.**\n\n"
            "### Ответ: **A**"
        )
        self.assertEqual(parsing.parse_scalar("choice", text), "A")


class FrozenGPURegressionTests(unittest.TestCase):
    def test_piece_preamble_is_not_protocol_compliant(self):
        # random:017:piece_empty:v0, copied from audit/regression_repro.py.
        text = "Давайте разберёмся с позицией FEN:\n\n**Позиция FEN:**\n```\n8/P3b1"
        self.assertFalse(parsing.compliant("piece", text))

    def test_complete_choice_before_truncation_preserves_content(self):
        # random:039:comment_fact_choice: complete selected answer before EOF.
        c = case("choice", {"answer": "A"})
        c.update(
            id="random:039:comment_fact_choice",
            group_id="random:039",
            task="comment_fact_choice",
            fen="r3kbnr/p1p1p2p/b5p1/1p1p2N1/4qp2/P1PQ4/1B1K1PPP/RN3B1R w kq - 0 21",
        )
        board = chess.Board(c["fen"])
        move = chess.Move.from_uci("f1e2")
        self.assertIn(move, board.legal_moves)
        self.assertFalse(board.is_capture(move))
        text = "Верный комментарий к ходу **f1e2** — это:\n\n**A) Этот ход ничего не берёт.**"
        scored = score_case(c, {"text": text, "truncated": True})
        self.assertEqual(scored["predicted"], "A")
        self.assertTrue(scored["correct"])
        self.assertTrue(scored["truncated"])
        self.assertFalse(scored["protocol_compliant"])


class ChoiceContractTests(unittest.TestCase):
    def test_strict_choices_and_lowercase(self):
        for text, expected in (
            ("A", "A"), ("b", "B"), ("Ответ: C", "C"),
            ("Ответ: d", "D"), ("**A**", "A"),
        ):
            with self.subTest(text=text):
                self.assertEqual(parsing.parse_choice(text), expected)
                self.assertEqual(parsing.parse_scalar("choice", text), expected)
                self.assertTrue(parsing.compliant("choice", text))

    def test_explicit_selection_recovers_content_but_not_format(self):
        for text in (
            "A) Этот ход ничего не берёт.",
            "Верный ответ — A.",
            "Ответ: A) Этот ход ничего не берёт.",
        ):
            with self.subTest(text=text):
                self.assertEqual(parsing.parse_scalar("choice", text), "A")
                self.assertFalse(parsing.compliant("choice", text))
                scored = score_case(case("choice", {"answer": "A"}), {"text": text})
                self.assertTrue(scored["correct"])
                self.assertFalse(scored["protocol_compliant"])

    def test_ambiguous_negative_and_literal_mentions_are_not_answers(self):
        for text in (
            "", "A/B", "A или B", "A, B", "A B",
            "не A", "Ответ: не A", "not A",
            "Вариант A неверен.", "Буква A встречается в условии.",
            'В условии написано "A) Этот ход ничего не берёт."',
            "AB", "DATA", "A1", "А",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parsing.parse_scalar("choice", text))
                self.assertFalse(parsing.compliant("choice", text))

    def test_judgement_recovers_content_without_relaxing_protocol(self):
        self.assertEqual(parsing.parse_scalar("judgement", "Ответ: G"), "G")
        self.assertEqual(parsing.parse_scalar("judgement", "Верный ответ — A."), "A")
        self.assertFalse(parsing.compliant("judgement", "Верный ответ — A."))


class ReportContractTests(unittest.TestCase):
    def test_false_zero_and_empty_list_are_predictions_not_missing_answers(self):
        for kind, gold, text, expected in (
            ("bool", {"answer": False}, "Нет", "false"),
            ("int", {"answer": 0}, "0", "0"),
            ("move_set", {"moves": []}, "[]", "[]"),
        ):
            with self.subTest(kind=kind):
                c = case(kind, gold)
                response = {"text": text, "truncated": True}
                score = score_case(c, response)
                self.assertTrue(score["correct"])
                self.assertEqual(report.prediction_text(c, response, score), expected)

    def test_missing_prediction_labels_explain_the_reason(self):
        c = case("choice", {"answer": "A"})
        for response, expected in (
            ({"text": ""}, "Пустой ответ"),
            ({"text": "Рассмотрим позицию", "truncated": True},
             "Ответ не извлечён — достигнут лимит токенов"),
            ({"text": "Рассмотрим позицию"},
             "Не удалось однозначно извлечь ответ"),
            ({"text": "", "error": "dependency failed"}, "Ошибка выполнения"),
        ):
            with self.subTest(response=response):
                score = score_case(c, response)
                self.assertIsNone(score["predicted"])
                self.assertEqual(report.prediction_text(c, response, score), expected)
        self.assertEqual(
            report.prediction_text(c, {}, {"status": "not_run"}),
            "Не запускалось",
        )

    def test_mate_denial_is_distinct_from_unparseable_answer(self):
        for moves in ([], ["e2e4"]):
            c = case("mate_move", {"moves": moves})
            for text in ("Нет", "Ответ: нет", "Нет, мата здесь нет."):
                with self.subTest(moves=moves, text=text):
                    response = {"text": text}
                    score = score_case(c, response)
                    self.assertIsNone(score["predicted"])
                    self.assertEqual(score["correct"], not moves)
                    self.assertEqual(
                        report.prediction_text(c, response, score),
                        "Нет матующего хода (ответ «нет»)",
                    )
            for text in ("Не знаю", "Да и нет", "Нет или да"):
                with self.subTest(text=text):
                    response = {"text": text}
                    score = score_case(c, response)
                    self.assertFalse(score["correct"])
                    self.assertEqual(
                        report.prediction_text(c, response, score),
                        "Не удалось однозначно извлечь ответ",
                    )

    def test_rendered_report_uses_explanation_instead_of_literal_null(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        c = case("piece", {"square": "e4", "label": "."})
        c["task"] = "piece_empty"
        response = {"id": c["id"], "text": "Давайте разберёмся.", "truncated": True}
        score = score_case(c, response)
        summary = summarize([score], config)
        summary["coverage"] = {"completed": 1, "expected": 1}
        with temporary_workspace() as tmp:
            out = Path(tmp)
            for filename, row in (
                ("cases.jsonl", c), ("responses.jsonl", response), ("scores.jsonl", score),
            ):
                (out / filename).write_text(json.dumps(row, ensure_ascii=False) + "\n")
            report.render_html(out, summary, config)
            rendered = (out / "report.html").read_text()
        records = json.loads(re.search(r"const D=(.*?);\nconst esc", rendered).group(1))
        self.assertEqual(
            records[0]["predicted"],
            "Ответ не извлечён — достигнут лимит токенов",
        )
        self.assertEqual(records[0]["answer"], response["text"])
        self.assertFalse(records[0]["format"])
        self.assertTrue(records[0]["truncated"])
        self.assertIsNone(records[0]["content_correct"])
        self.assertTrue(records[0]["verdict"].startswith("не оценено:"))
        self.assertIn("лимит", records[0]["verdict"])
        self.assertIn("d.content_correct===false", rendered)
        self.assertNotIn("d.correct===false", rendered)
        self.assertIn("<h2>Score</h2>", rendered)
        self.assertIn("0 / 1", rendered)
        self.assertIn("Score: Нет данных", rendered)
        self.assertIn("Основная macro-метрика:", rendered)
        self.assertIn("<th>Эталон</th><th>Ответ модели</th><th>Итог</th>", rendered)
        self.assertIn("Не удалось оценить", rendered)

    def test_expected_text_is_readable(self):
        for kind, gold, expected in (
            ("piece", {"square": "e4", "label": "."}, "e4: пусто"),
            ("piece", {"square": "g7", "label": "b"}, "g7: чёрные: слон"),
            ("piece", {"square": "e1", "label": "K"}, "e1: белые: король"),
            ("bool", {"answer": True}, "Да"),
            ("bool", {"answer": False}, "Нет"),
            ("int", {"answer": 43}, "43"),
            ("choice", {"answer": "C"}, "C"),
            ("move_set", {"moves": ["e2e4", "e2e3"]}, "e2e4, e2e3"),
            ("move_set", {"moves": []}, "Нет легальных ходов"),
            ("mate_move", {"moves": []}, "Нет матующего хода"),
        ):
            with self.subTest(kind=kind, gold=gold):
                self.assertEqual(report.expected_text(case(kind, gold)), expected)

    def test_pending_answers_do_not_earn_score_and_remain_in_block_total(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        config["blocks"] = {"board_reading": ["structural/*"]}
        for status in ("pending_engine", "pending_coach_ratings", "mate_group"):
            for with_correct in (False, True):
                with self.subTest(status=status, with_correct=with_correct):
                    c = case("bool", {"answer": True})
                    response = {"id": c["id"], "text": "Да"}
                    pending = score_case(c, response)
                    pending.update(status=status, correct=None, content_correct=None)
                    cases, responses, scores = [c], [response], [pending]
                    if with_correct:
                        second = dict(c, id="confirmed")
                        second_response = dict(response, id="confirmed")
                        cases.append(second)
                        responses.append(second_response)
                        scores.append(score_case(second, second_response))
                    summary = summarize(scores, config)
                    summary["coverage"] = {"completed": len(scores), "expected": len(scores)}
                    with temporary_workspace() as tmp:
                        out = Path(tmp)
                        for filename, rows in (("cases.jsonl", cases),
                                               ("responses.jsonl", responses),
                                               ("scores.jsonl", scores)):
                            (out / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
                        (out / "run.json").write_text(json.dumps({
                            "backend": {"model": "<Qwen>"},
                            "started_utc": "2026-11-24T09:31:25+00:00",
                            "completed_utc": "2026-11-24T10:08:44+00:00",
                        }))
                        report.render_html(out, summary, config)
                        rendered = (out / "report.html").read_text()
                    expected = "1 / 2" if with_correct else "0 / 1"
                    self.assertIn(f"<div class='big'>{expected}", rendered)
                    self.assertIn(f"Score: {expected}", rendered)
                    self.assertIn("Модель: &lt;Qwen&gt;", rendered)
                    self.assertIn("2026-11-24T09:31:25+00:00", rendered)
                    self.assertIn("2026-11-24T10:08:44+00:00", rendered)


class PieceContractTests(unittest.TestCase):
    def test_explicit_empty_and_occupied_square(self):
        for text, square, expected in (
            ("На e4 пусто.", "e4", "."),
            ("На e4 нет фигур.", "e4", "."),
            ("На e4 белая пешка.", "e4", "P"),
            ("На d5 чёрный конь.", "d5", "n"),
        ):
            with self.subTest(text=text):
                self.assertEqual(parsing.parse_piece(text, square), expected)
                self.assertTrue(parsing.compliant("piece", text))

    def test_incomplete_contradictory_and_multiple_square_answers(self):
        for text in (
            "", "Давайте разберёмся.", "e4", "На e4 белая.",
            "На e4 пешка.", "Белая пешка.", "На e4 не белая пешка.",
            "На e4 белая пешка, но e4 пусто.",
            "На e4 белая пешка и чёрный конь.",
            "На e4 белая и чёрная пешка.",
            "На e4 белая пешка, на d5 чёрный конь.",
            "На e4 стоит <|im_end|>",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parsing.parse_piece(text, "e4"))
                self.assertFalse(parsing.compliant("piece", text))

    def test_wrong_but_well_formed_piece_remains_format_compliant(self):
        scored = score_case(
            case("piece", {"square": "e4", "label": "."}),
            {"text": "На e4 белая пешка."},
        )
        self.assertFalse(scored["correct"])
        self.assertEqual(scored["predicted"], "P")
        self.assertTrue(scored["protocol_compliant"])


if __name__ == "__main__":
    unittest.main()
