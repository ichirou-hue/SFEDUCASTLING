"""Automatic M4-M7 criteria stay separate from expert ratings and catch false facts."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "ml/benchmark"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench import commentary as MODULE  # noqa: E402, N812
from gigabench.adapters import input_builder  # noqa: E402

FEN = chess.STARTING_FEN
MOVE = "e2e4"
CORRECT = ("Белая пешка идёт с e2 на e4 и занимает центр. "
            "Это точный ход, оценка не меняется, позиция примерно равная.")


def facts():
    board = chess.Board(FEN)
    after = board.copy()
    after.push_uci(MOVE)
    best = {"uci": MOVE, "san": "e4", "from": "e2", "to": "e4", "piece": "P"}
    return {
        "before": {"fen": board.fen()},
        "after": {"fen": after.fen(en_passant="fen"), "is_checkmate": False, "in_check": False},
        "delta": {
            "uci": MOVE, "san": "e4", "piece": "P", "from": "e2", "to": "e4",
            "is_capture": False, "captured": None, "gives_check": False,
            "castling": None, "promotion": None,
        },
        "engine": {
            "verdict": "точный ход", "cpl": 0, "played_is_best": True,
            "best_move_before": best, "motifs": [],
            "score_before": {"cp": 20, "mate": None},
            "score_after": {"cp": 30, "mate": None},
        },
    }


def case():
    bundle = facts()
    target = {"verdict": "точный ход", "cpl": 0, "fen_after": bundle["after"]["fen"]}
    rubric = MODULE.build_rubric(bundle, target)
    return {
        "id": "commentary:test",
        "suite": "commentary",
        "task": "move_commentary",
        "fen": FEN,
        "prompt": "Опиши этот ход.",
        "score_type": "explanation_auto",
        "gold": {
            "move_uci": MOVE,
            "fen_after": target["fen_after"],
            "facts": bundle,
            "rubric": rubric,
            "reference_answer": CORRECT,
        },
        "group_id": "test",
        "params": {"rubric_version": MODULE.CRITERIA_VERSION},
        "max_new_tokens": 128,
        "private": True,
    }


def capture_case():
    fen = "7k/6p1/5B2/8/8/8/8/K7 b - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci("g7f6")
    after = board.copy()
    after.push(move)
    bundle = {
        "before": {"fen": board.fen()},
        "after": {
            "fen": after.fen(en_passant="fen"),
            "is_checkmate": after.is_checkmate(),
            "in_check": after.is_check(),
        },
        "delta": {
            "uci": "g7f6", "san": "Bxf6", "piece": "p", "from": "g7", "to": "f6",
            "is_capture": True, "captured": {"piece": "B", "square": "f6"},
            "gives_check": after.is_check(), "castling": None, "promotion": None,
        },
        "engine": {
            "verdict": "точный ход", "cpl": 0, "played_is_best": True,
            "best_move_before": {
                "uci": "g7f6", "san": "Bxf6", "from": "g7", "to": "f6", "piece": "p",
            },
            "motifs": [], "score_before": {"cp": 0, "mate": None},
            "score_after": {"cp": 0, "mate": None},
        },
    }
    target = {"verdict": "точный ход", "cpl": 0, "fen_after": after.fen(en_passant="fen")}
    return {
        "id": "commentary:capture-test", "suite": "commentary", "task": "move_commentary",
        "fen": fen, "prompt": "Опиши этот ход.", "score_type": "explanation_auto",
        "gold": {
            "move_uci": "g7f6", "fen_after": target["fen_after"], "facts": bundle,
            "rubric": MODULE.build_rubric(bundle, target), "reference_answer": None,
        },
        "group_id": "capture-test", "params": {"rubric_version": MODULE.CRITERIA_VERSION},
        "max_new_tokens": 128, "private": True,
    }


class RubricTests(unittest.TestCase):
    def test_rubric_has_core_move_and_verdict_ideas(self):
        rubric = case()["gold"]["rubric"]
        kinds = [idea["kind"] for idea in rubric["required_ideas"]]
        self.assertEqual(kinds, ["move", "verdict"])
        self.assertEqual(rubric["allowed_facts"]["verdict"], "точный ход")
        self.assertEqual(rubric["required_ideas"][0]["from"], "e2")
        self.assertEqual(rubric["required_ideas"][0]["to"], "e4")

    def test_rubric_limits_m5_to_core_move_and_verdict_ideas(self):
        bundle = facts()
        bundle["engine"]["verdict"] = "грубая ошибка"
        bundle["engine"]["cpl"] = 250
        bundle["engine"]["best_move_before"] = {
            "uci": "d2d4", "san": "d4", "from": "d2", "to": "d4", "piece": "P",
        }
        rubric = MODULE.build_rubric(bundle, {"verdict": "грубая ошибка", "cpl": 250})
        kinds = [idea["kind"] for idea in rubric["required_ideas"]]
        self.assertEqual(kinds, ["move", "verdict"])

    def test_positive_verdict_does_not_require_best_move(self):
        bundle = facts()
        bundle["engine"]["cpl"] = 120
        bundle["engine"]["best_move_before"] = {
            "uci": "d2d4", "san": "d4", "from": "d2", "to": "d4", "piece": "P",
        }
        rubric = MODULE.build_rubric(bundle, {"verdict": "приемлемый ход", "cpl": 120})
        self.assertNotIn("best_move", [idea["kind"] for idea in rubric["required_ideas"]])


class ScoringTests(unittest.TestCase):
    def test_san_check_and_mate_suffixes_are_verified(self):
        for san in ("e4+", "e4#"):
            with self.subTest(san=san):
                self.assertEqual(MODULE.score_response(f"Белые сыграли {san}.", case())["M6"], 1)
        for san, fen, uci in (
            ("Re8+", "7k/8/8/8/8/8/4R3/7K w - - 0 1", "e2e8"),
            ("Qg7#", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7"),
        ):
            with self.subTest(san=san):
                example = case()
                example["gold"]["facts"]["before"]["fen"] = fen
                example["gold"]["facts"]["delta"]["uci"] = uci
                result = MODULE.score_response(f"Белые сыграли {san}.", example)
                self.assertEqual((result["M4"], result["M6"]), (1.0, 0))

    def test_explicit_played_claim_can_follow_a_threat(self):
        result = MODULE.score_response(
            "Пешка идёт с e2 на e4, угрожает занять центр, а сыгранный ход даёт шах. "
            "Это точный ход.", case(),
        )
        self.assertEqual(result["M6"], 1)
        self.assertTrue(any(claim["kind"] == "check" and claim["critical"] for claim in result["claims"]))

    def test_threat_label_does_not_turn_a_hypothetical_move_into_a_fact(self):
        result = MODULE.score_response(
            CORRECT + " Не отражена угроза: пешка идёт с d2 на d4.", case(),
        )
        self.assertEqual(result["M6"], 0)

    def test_legal_underpromotion_is_not_a_critical_recommendation(self):
        example = case()
        example["gold"]["facts"]["before"]["fen"] = "7k/P7/8/8/8/8/8/7K w - - 0 1"
        result = MODULE.score_response("Лучше сыграть a7a8n.", example)
        self.assertEqual(result["M6"], 0)
        self.assertTrue(any(claim.get("promotion") == "n" for claim in result["claims"]))

    def test_correct_comment_is_accepted(self):
        result = MODULE.score_response(CORRECT, case())
        self.assertEqual(result["M4"], 1.0)
        self.assertEqual(result["M5"], 1.0)
        self.assertEqual(result["M6"], 0)
        self.assertGreaterEqual(result["M7"], 0.75)
        self.assertTrue(result["accepted"])

    def test_reversed_verdict_word_order_covers_the_idea(self):
        text = "Белая пешка идёт с e2 на e4. Ход точный, позиция примерно равная."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["idea_scores"][1], 1)

    def test_wrong_played_move_is_a_critical_error(self):
        text = "Белая пешка идёт с d2 на d4."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M4"], 0.0)
        self.assertEqual(result["M6"], 1)
        self.assertTrue(any(claim["kind"] == "move" and claim["critical"]
                            for claim in result["claims"]))

    def test_false_check_is_a_critical_error(self):
        text = "Белая пешка идёт с e2 на e4 и объявляет шах."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M6"], 1)
        self.assertTrue(any(claim["kind"] == "check" and claim["critical"]
                            for claim in result["claims"]))

    def test_false_capture_is_a_critical_error(self):
        text = "Белая пешка идёт с e2 на e4 и берёт чёрную пешку d5."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M6"], 1)
        self.assertTrue(any(claim["kind"] == "capture" and claim["critical"]
                            for claim in result["claims"]))

    def test_origin_and_target_capture_statement_covers_move_and_effect_ideas(self):
        text = "Чёрные берут пешкой g7 белого слона f6. Это точный ход."
        result = MODULE.score_response(text, capture_case())
        self.assertEqual(result["M4"], 1.0)
        self.assertEqual(result["M5"], 1.0)
        self.assertEqual(result["M6"], 0)

    def test_attack_on_another_square_is_not_misread_as_a_capture(self):
        text = "Белая пешка идёт с e2 на e4 и бьёт по слону b4."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M6"], 0)
        self.assertFalse(any(claim["kind"] == "capture" for claim in result["claims"]))

    def test_opposite_verdict_is_a_critical_error(self):
        result = MODULE.score_response("Это грубая ошибка.", case())
        self.assertEqual(result["M6"], 1)
        self.assertTrue(any(claim["kind"] == "verdict" and claim["critical"]
                            for claim in result["claims"]))

    def test_future_threat_is_not_scored_as_a_played_claim(self):
        text = "Следующим ходом чёрный конь g8 угрожает взять пешку e4."
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M6"], 0)
        self.assertEqual(result["M4"], 0.0)
        self.assertEqual(result["claims"], [])

    def test_threat_mentioned_in_a_played_move_clause_is_not_a_played_move(self):
        text = ("Белая пешка идёт с e2 на e4. Она отражает угрозу хода пешкой "
                "с b5 на b4. Это точный ход, позиция примерно равная.")
        result = MODULE.score_response(text, case())
        self.assertEqual(result["M6"], 0)
        self.assertEqual(result["M4"], 1.0)

    def test_played_move_and_recommendation_in_one_sentence_have_distinct_roles(self):
        text = "Белая пешка идёт с d2 на d4, но лучше было сыграть e2 на e4."
        result = MODULE.score_response(text, case())
        moves = [claim for claim in result["claims"] if claim["kind"] == "move"]
        self.assertEqual(
            [(move["from"], move["to"], move["recommendation"], move["played"])
             for move in moves],
            [("d2", "d4", False, True), ("e2", "e4", True, False)],
        )
        self.assertEqual(result["M6"], 1)

    def test_unqualified_move_mention_is_not_asserted_as_played(self):
        result = MODULE.score_response("Ход e2-e4.", case())
        move = next(claim for claim in result["claims"] if claim["kind"] == "move")
        self.assertEqual(move["verdict"], "unconfirmed")
        self.assertEqual(result["M6"], 0)

    def test_illegal_recommendation_is_a_critical_error(self):
        result = MODULE.score_response("Лучше сыграть с e2 на e5.", case())
        move = next(claim for claim in result["claims"] if claim["kind"] == "move")
        self.assertTrue(move["recommendation"])
        self.assertEqual(move["verdict"], "incorrect")
        self.assertEqual(result["M6"], 1)

    def test_acceptance_thresholds_are_exact(self):
        evaluation_case = case()
        evaluation_case["gold"]["rubric"]["required_ideas"].extend([
            {"id": "not-mentioned", "kind": "effect", "effect": "check"},
            {"id": "move-again", "kind": "best_move", "from": "e2", "to": "e4", "san": "e4"},
            {"id": "verdict-again", "kind": "verdict", "verdict": "точный ход"},
        ])
        text = CORRECT + " Лучше сыграть с e2 на e4."
        with patch.object(MODULE, "_m7", return_value={
            "goal": True, "position_link": True, "clarity": True, "actionable": False,
        }):
            accepted = MODULE.score_response(text, evaluation_case)
        self.assertEqual(accepted["M4"], 1.0)
        self.assertEqual(accepted["M5"], 0.8)
        self.assertEqual(accepted["M7"], 0.75)
        self.assertTrue(accepted["accepted"])
        with patch.object(MODULE, "_m7", return_value={
            "goal": True, "position_link": True, "clarity": False, "actionable": False,
        }):
            rejected = MODULE.score_response(text, evaluation_case)
        self.assertFalse(rejected["accepted"])

    def test_wrong_evaluation_does_not_cover_the_required_idea(self):
        text = "Белая пешка идёт с e2 на e4. Это точный ход, перевес чёрных."
        evaluation_case = case()
        evaluation_case["gold"]["rubric"]["required_ideas"].append({
            "id": "evaluation", "kind": "evaluation", "side": "равно", "unchanged": False,
        })
        result = MODULE.score_response(text, evaluation_case)
        self.assertEqual(result["idea_scores"], [1, 1, 0])
        self.assertEqual(result["M5"], 2 / 3)

    def test_empty_and_truncated_answers_get_zero(self):
        for text, truncated in (("", False), (CORRECT, True)):
            result = MODULE.score_response(text, case(), truncated=truncated)
            self.assertEqual((result["M4"], result["M5"], result["M6"], result["M7"]),
                             (0.0, 0.0, 0, 0.0))
            self.assertFalse(result["accepted"])


class PreparationTests(unittest.TestCase):
    def row(self, identifier, fen, answer=CORRECT):
        return {
            "id": identifier, "split": "test", "task": "move_commentary",
            "fen": fen, "move_uci": MOVE, "question": "Опиши этот ход.",
            "answer": answer,
            "target": {"verdict": "точный ход", "cpl": 0,
                       "fen_after": facts()["after"]["fen"]},
            "source": "unit-test",
        }

    def test_prepare_keeps_reference_out_of_queries_and_audits_invalid_fen(self):
        bundle = facts()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            commentary = root / "commentary.jsonl"
            facts_path = root / "facts.jsonl"
            commentary.write_text(
                json.dumps(self.row("ok", FEN), ensure_ascii=False) + "\n"
                + json.dumps(self.row("bad", "8/8/8/8/8/8/8/8 w - - 0 1"),
                             ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            facts_path.write_text(
                json.dumps({"id": "ok", "facts": bundle}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            out = root / "run"
            args = argparse.Namespace(
                commentary=commentary, facts=facts_path, out=out, split="test",
                max_new_tokens=128, limit=None, config=ROOT / "config/default.json",
            )
            manifest, excluded = MODULE.prepare_run(args)
            self.assertEqual(manifest["case_count"], 1)
            self.assertEqual(manifest["automatic_queries"], 1)
            self.assertEqual(manifest["excluded_commentary_cases"], 1)
            self.assertEqual(manifest["model_input_profile"], "contract_v1_fen")
            self.assertEqual(
                manifest["model_input_template"], "Позиция FEN: {fen}\n{prompt}",
            )
            self.assertEqual(excluded[0]["id"], "bad")
            queries = (out / "queries.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(CORRECT, queries)
            self.assertIn("Опиши этот ход", queries)
            cases = MODULE.read_jsonl(out / "cases.jsonl")
            self.assertTrue(cases[0]["private"])
            self.assertNotIn("reference_answer", cases[0]["gold"])
            self.assertFalse(manifest["contains_reference_answers"])
            rubrics = MODULE.read_jsonl(out / "auto_rubrics.jsonl")
            self.assertEqual(rubrics[0]["version"], MODULE.CRITERIA_VERSION)

            (out / "responses.jsonl").write_text(
                json.dumps({"id": cases[0]["id"], "text": CORRECT,
                            "truncated": False}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            summary = MODULE.score_run(argparse.Namespace(run=out, allow_partial=False))
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["M6"], 0.0)
            self.assertTrue((out / "m4_m7_report.html").is_file())

    def test_score_uses_frozen_suite_and_copies_its_queries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            suite, run = root / "suite", root / "run"
            suite.mkdir()
            run.mkdir()
            frozen_case = case()
            frozen_case["gold"].pop("reference_answer")
            query = {
                "id": frozen_case["id"], "fen": frozen_case["fen"],
                "prompt": frozen_case["prompt"],
                "max_new_tokens": frozen_case["max_new_tokens"],
            }
            MODULE.write_jsonl(suite / "cases.jsonl", [frozen_case])
            MODULE.write_jsonl(suite / "queries.jsonl", [query])
            MODULE.write_json(suite / "manifest.json", {
                "criteria_version": MODULE.CRITERIA_VERSION,
                "suite_id": "test-suite",
                "cases_sha256": MODULE.sha(suite / "cases.jsonl"),
                "queries_sha256": MODULE.sha(suite / "queries.jsonl"),
            })
            MODULE.write_jsonl(run / "responses.jsonl", [{
                "id": frozen_case["id"], "text": CORRECT, "truncated": False,
            }])

            summary = MODULE.score_run(argparse.Namespace(
                run=run, suite=suite, allow_partial=False,
            ))

            self.assertEqual(summary["suite_id"], "test-suite")
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(MODULE.read_jsonl(run / "queries.jsonl"), [query])

            MODULE.write_jsonl(suite / "queries.jsonl", [])
            with self.assertRaisesRegex(ValueError, "queries hash"):
                MODULE.score_run(argparse.Namespace(
                    run=run, suite=suite, allow_partial=False,
                ))

    def test_benchmark_input_matches_commentary_training_text(self):
        build_input, metadata = input_builder("contract_v1_fen")
        queries = MODULE.read_jsonl(
            ROOT / "data/commentary_m4_m7_auto_v1/queries.jsonl",
        )
        query = queries[0]
        self.assertEqual(
            build_input(query),
            f"Позиция FEN: {query['fen']}\n{query['prompt']}",
        )
        self.assertTrue(all("Факты движка:" in row["prompt"] for row in queries))
        self.assertTrue(all(
            "Предыдущий ход соперника:" in row["prompt"] for row in queries
        ))
        self.assertEqual(metadata["input_profile"], "contract_v1_fen")

    def test_compare_refuses_different_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            summary = {
                "n": 1, "expected": 1, "complete": True,
                "criteria_version": MODULE.CRITERIA_VERSION,
                "M4": 100.0, "M5": 100.0, "M6": 0.0, "M7": 100.0,
                "accepted_percent": 100.0,
            }
            MODULE.write_json(before / "m4_m7_summary.json", summary)
            MODULE.write_json(after / "m4_m7_summary.json", summary)
            MODULE.write_jsonl(before / "queries.jsonl", [
                {"id": "commentary:test", "fen": FEN, "prompt": "Опиши этот ход.",
                 "max_new_tokens": 512},
            ])
            MODULE.write_jsonl(after / "queries.jsonl", [
                {"id": "commentary:test", "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                 "prompt": "Опиши этот ход.", "max_new_tokens": 512},
            ])
            with self.assertRaisesRegex(ValueError, "queries, FENs or generation budgets"):
                MODULE.compare_runs(argparse.Namespace(
                    before=before, after=after, out=root / "comparison.html",
                    before_label="base", after_label="LoRA",
                ))
            MODULE.write_jsonl(after / "queries.jsonl", [
                {"id": "commentary:test", "fen": FEN, "prompt": "Опиши этот ход.",
                 "max_new_tokens": 256},
            ])
            with self.assertRaisesRegex(ValueError, "queries, FENs or generation budgets"):
                MODULE.compare_runs(argparse.Namespace(
                    before=before, after=after, out=root / "comparison.html",
                    before_label="base", after_label="LoRA",
                ))

    def test_compare_refuses_different_scoring_cases(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            summary = {
                "n": 1, "expected": 1, "complete": True,
                "criteria_version": MODULE.CRITERIA_VERSION,
                "suite_id": "same-query-suite",
                "M4": 100.0, "M5": 100.0, "M6": 0.0, "M7": 100.0,
                "accepted_percent": 100.0,
            }
            MODULE.write_json(before / "m4_m7_summary.json", {
                **summary, "cases_sha256": "rubric-set-a",
            })
            MODULE.write_json(after / "m4_m7_summary.json", {
                **summary, "cases_sha256": "rubric-set-b",
            })
            queries = [{
                "id": "commentary:test", "fen": FEN,
                "prompt": "Опиши этот ход.", "max_new_tokens": 512,
            }]
            MODULE.write_jsonl(before / "queries.jsonl", queries)
            MODULE.write_jsonl(after / "queries.jsonl", queries)

            with self.assertRaisesRegex(ValueError, "different scoring cases"):
                MODULE.compare_runs(argparse.Namespace(
                    before=before, after=after, out=root / "comparison.html",
                    before_label="base", after_label="LoRA",
                ))

    def test_score_refuses_unknown_response_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            run = Path(tmp)
            MODULE.write_jsonl(run / "cases.jsonl", [case()])
            MODULE.write_jsonl(run / "responses.jsonl", [
                {"id": "commentary:test", "text": CORRECT, "truncated": False},
                {"id": "commentary:unknown", "text": "x", "truncated": False},
            ])
            with self.assertRaisesRegex(ValueError, "unknown IDs"):
                MODULE.score_run(argparse.Namespace(run=run, allow_partial=False))


class FrozenSuiteTests(unittest.TestCase):
    def test_checked_in_suite_is_complete_and_excludes_reference_comments(self):
        suite = ROOT / "data/commentary_m4_m7_auto_v1"
        manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
        cases = MODULE.read_jsonl(suite / "cases.jsonl")
        queries = MODULE.read_jsonl(suite / "queries.jsonl")

        self.assertEqual(manifest["criteria_version"], MODULE.CRITERIA_VERSION)
        self.assertEqual(manifest["case_count"], 300)
        self.assertEqual(len(cases), 300)
        self.assertEqual(len(queries), 300)
        self.assertEqual(manifest["cases_sha256"], MODULE.sha(suite / "cases.jsonl"))
        self.assertEqual(manifest["queries_sha256"], MODULE.sha(suite / "queries.jsonl"))
        self.assertFalse(manifest["contains_reference_answers"])
        self.assertTrue(all("reference_answer" not in row["gold"] for row in cases))
        self.assertEqual({row["id"] for row in cases}, {row["id"] for row in queries})


if __name__ == "__main__":
    unittest.main()
