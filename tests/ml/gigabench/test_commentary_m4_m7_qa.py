"""Independent black-box examples and CLI regressions for M4-M7-auto."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "ml/benchmark"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench import commentary as MODULE  # noqa: E402

TEMP = Path("/private/var/folders/1n/z6bt2rc91kv9ybvd3j17rs5h0000gn/T/opencode")
CORRECT = ("Белая пешка идёт с e2 на e4, занимая центр. "
           "Ход точный, позиция примерно равная.")


def make_case():
    board = chess.Board()
    after = board.copy()
    after.push_uci("e2e4")
    facts = {
        "before": {"fen": board.fen()},
        "after": {"fen": after.fen(en_passant="fen"),
                  "is_checkmate": False, "in_check": False},
        "delta": {
            "uci": "e2e4", "san": "e4", "from": "e2", "to": "e4", "piece": "P",
            "is_capture": False, "captured": None, "gives_check": False,
            "castling": None, "promotion": None,
        },
        "engine": {
            "verdict": "точный ход", "cpl": 0, "played_is_best": True,
            "best_move_before": {"uci": "e2e4", "san": "e4", "from": "e2",
                                 "to": "e4", "piece": "P"},
            "motifs": [], "score_before": {"cp": 20, "mate": None},
            "score_after": {"cp": 30, "mate": None},
        },
    }
    target = {"verdict": "точный ход", "cpl": 0, "fen_after": facts["after"]["fen"]}
    return {
        "id": "commentary:qa", "suite": "commentary", "task": "move_commentary",
        "fen": board.fen(), "prompt": "Прокомментируй ход e2e4.",
        "score_type": "explanation_auto", "group_id": "qa", "max_new_tokens": 128,
        "gold": {"facts": facts, "move_uci": "e2e4", "fen_after": target["fen_after"],
                 "rubric": MODULE.build_rubric(facts, target)},
        "params": {"rubric_version": MODULE.CRITERIA_VERSION}, "private": True,
    }


class IndependentLanguageTests(unittest.TestCase):
    def assert_critical(self, text, kind):
        result = MODULE.score_response(text, make_case())
        self.assertEqual(result["M6"], 1, result)
        self.assertFalse(result["accepted"])
        self.assertTrue(any(c["kind"] == kind and c["critical"]
                            for c in result["claims"]), result)

    def test_fresh_correct_answer(self):
        result = MODULE.score_response(
            "Белая пешка продвинулась с e2 на e4, заняв центр. "
            "Ход точный, игра равная.", make_case())
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["M6"], 0)

    def test_explicit_false_effects_and_recommendation(self):
        for text, kind in (
            ("Белые сыграли e2-e4 с шахом.", "check"),
            ("Белая пешка идёт с e2 на e4 и ставит мат чёрному королю.", "mate"),
            ("Белая пешка идёт с e2 на e4 и превратилась в ферзя.", "promotion"),
            ("Белая пешка идёт с e2 на e4 и съела чёрного коня.", "capture"),
            ("Это грубая ошибка, белые испортили позицию.", "verdict"),
            ("Лучше пойти конём с g1 на g3.", "move"),
        ):
            with self.subTest(text=text):
                self.assert_critical(text, kind)

    def test_false_played_move_in_past_tense_is_critical(self):
        self.assert_critical(CORRECT + " Белая пешка пошла с d2 на d4.", "move")

    def test_false_played_move_in_san_is_critical(self):
        self.assert_critical(CORRECT + " Белые сыграли d4.", "move")

    def test_false_castling_in_past_tense_is_critical(self):
        self.assert_critical(CORRECT + " Белые сделали короткую рокировку.", "castling")

    def test_wrong_piece_on_correct_squares_is_critical(self):
        self.assert_critical(CORRECT + " Белая ладья идёт с e2 на e4.", "move")

    def test_negated_check_is_not_a_false_check(self):
        result = MODULE.score_response(
            "Белая пешка идёт с e2 на e4, не объявляя шаха. "
            "Это точный ход, позиция равная.", make_case())
        self.assertEqual(result["M6"], 0, result)
        self.assertTrue(result["accepted"], result)

    def test_negated_capture_is_not_a_false_capture(self):
        result = MODULE.score_response(
            CORRECT + " Пешка идёт с e2 на e4 и не берёт фигуру.", make_case())
        self.assertEqual(result["M6"], 0, result)
        self.assertTrue(result["accepted"], result)

    def test_negated_bad_verdict_is_not_a_bad_verdict(self):
        result = MODULE.score_response(
            "Белая пешка идёт с e2 на e4. Это не грубая ошибка, "
            "а точный ход, позиция равная.", make_case())
        self.assertEqual(result["M6"], 0, result)
        self.assertTrue(result["accepted"], result)

    def test_nonplayed_lines_do_not_pollute_m6(self):
        for suffix in (
            "После ответа чёрных ферзь идёт с d1 на h5 и даёт шах.",
            "Вместо взятия коня белые укрепляют центр.",
            "Раньше конь шёл с g8 на f6.",
            "Белые могли пойти с d2 на d4.",
            "Следующим ходом белые ставят мат.",
            "Чёрный конь угрожает взять пешку e4.",
        ):
            with self.subTest(suffix=suffix):
                result = MODULE.score_response(CORRECT + " " + suffix, make_case())
                self.assertEqual(result["M6"], 0, result)
                self.assertTrue(result["accepted"], result)

    def test_whitespace_and_truncated_false_claims_are_zeroed(self):
        for text, truncated in (("", False), (" \n\t ", False),
                                ("Белые ставят мат.", True), (CORRECT, True)):
            with self.subTest(text=text, truncated=truncated):
                result = MODULE.score_response(text, make_case(), truncated=truncated)
                self.assertEqual([result[k] for k in ("M4", "M5", "M6", "M7")],
                                 [0, 0, 0, 0])
                self.assertFalse(result["accepted"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, *args, succeeds=True):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/commentary_m4_m7.py"), *map(str, args)],
            capture_output=True, text=True, cwd=PROJECT, timeout=60)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def inputs(self):
        case = make_case()
        row = {
            "id": "qa", "split": "test", "task": "move_commentary", "fen": case["fen"],
            "move_uci": "e2e4", "question": case["prompt"],
            "answer": "PRIVATE_REFERENCE_DO_NOT_LEAK",
            "target": {"verdict": "точный ход", "cpl": 0,
                       "fen_after": case["gold"]["fen_after"]},
        }
        bad = {**row, "id": "invalid", "fen": "8/8/8/8/8/8/8/8 w - - 0 1"}
        commentary, facts = self.root / "source.jsonl", self.root / "facts.jsonl"
        MODULE.write_jsonl(commentary, [row, {**row, "id": "qa2"}, bad])
        MODULE.write_jsonl(facts, [{"id": i, "facts": case["gold"]["facts"]}
                                  for i in ("qa", "qa2", "invalid")])
        return commentary, facts

    def test_cli_prepare_score_compare_and_validation(self):
        commentary, facts = self.inputs()
        runs = [self.root / "before", self.root / "after"]
        for run in runs:
            self.cli("prepare", "--commentary", commentary, "--facts", facts, "--out", run)
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["case_count"], 2)
            self.assertEqual(manifest["excluded_commentary_cases"], 1)
            self.assertEqual(MODULE.read_jsonl(run / "audit.jsonl")[0]["id"], "invalid")
            query_text = (run / "queries.jsonl").read_text()
            self.assertNotIn("reference_answer", query_text)
            self.assertNotIn("PRIVATE_REFERENCE_DO_NOT_LEAK", query_text)
            self.cli("prepare", "--commentary", commentary, "--facts", facts,
                     "--out", run, succeeds=False)
            # Missing responses must fail unless partial scoring is explicit.
            response = {"id": "commentary:qa", "text": CORRECT, "truncated": False}
            MODULE.write_jsonl(run / "responses.jsonl", [response])
            failure = self.cli("score", "--run", run, succeeds=False)
            self.assertIn("missing 1 responses", failure.stderr)
            partial = json.loads(self.cli("score", "--run", run, "--allow-partial").stdout)
            self.assertEqual((partial["n"], partial["expected"], partial["complete"]),
                             (1, 2, False))
            MODULE.write_jsonl(run / "responses.jsonl", [
                response, {**response, "id": "unknown"}])
            for flags in ((), ("--allow-partial",)):
                failure = self.cli("score", "--run", run, *flags, succeeds=False)
                self.assertIn("unknown IDs", failure.stderr)
            MODULE.write_jsonl(run / "responses.jsonl", [
                response, {**response, "id": "commentary:qa2"}])
            scored = json.loads(self.cli("score", "--run", run).stdout)
            self.assertEqual((scored["n"], scored["accepted"]), (2, 2))
            self.assertTrue((run / "m4_m7_summary.json").is_file())
            self.assertIn("M4-auto", (run / "m4_m7_report.html").read_text())
        output = self.root / "comparison.html"
        self.cli("compare", "--before", runs[0], "--after", runs[1], "--out", output)
        self.assertIn("Δ", output.read_text())
        self.assertIn("0,0 п. п.", output.read_text())
        MODULE.write_jsonl(runs[1] / "responses.jsonl", [
            {"id": "commentary:qa", "text": CORRECT, "truncated": False},
            {"id": "commentary:qa2", "text": CORRECT + " Белые дают шах.",
             "truncated": False},
        ])
        self.cli("score", "--run", runs[1])
        self.cli("compare", "--before", runs[0], "--after", runs[1], "--out", output)
        self.assertIn("−50.0 п. п.", output.read_text())
        summary_path = runs[1] / "m4_m7_summary.json"
        original = json.loads(summary_path.read_text())
        for changed in ({**original, "n": 1},
                        {**original, "criteria_version": "incompatible"}):
            with self.subTest(changed=changed):
                MODULE.write_json(summary_path, changed)
                self.cli("compare", "--before", runs[0], "--after", runs[1],
                         "--out", output, succeeds=False)

    def test_score_run_can_aggregate_empty_whitespace_and_truncated_answers(self):
        run = self.root / "run"
        run.mkdir()
        MODULE.write_jsonl(run / "cases.jsonl", [make_case()])
        for text, truncated in (("", False), (" \n\t ", False), (CORRECT, True)):
            with self.subTest(text=text, truncated=truncated):
                MODULE.write_jsonl(run / "responses.jsonl", [
                    {"id": "commentary:qa", "text": text, "truncated": truncated}])
                summary = MODULE.score_run(argparse.Namespace(run=run, allow_partial=False))
                self.assertEqual(summary["n"], 1)
                self.assertEqual(summary["accepted"], 0)
                self.assertEqual([summary[k] for k in ("M4", "M5", "M6", "M7")],
                                 [0, 0, 0, 0])
                self.assertTrue((run / "m4_m7_report.html").is_file())

    def test_all_real_reference_answers_and_prepare_audit(self):
        commentary = TEMP / "handoff-full/commentary_v2.jsonl"
        facts = TEMP / "handoff-full/facts_v2.jsonl"
        if not commentary.is_file() or not facts.is_file():
            self.skipTest("local read-only handoff fixture not available")
        run = self.root / "real"
        manifest, excluded = MODULE.prepare_run(argparse.Namespace(
            commentary=commentary, facts=facts, out=run, split="test",
            max_new_tokens=128, limit=None, config=ROOT / "config/default.json"))
        self.assertEqual((manifest["case_count"], len(excluded)), (300, 4))
        source = {row["id"]: row for row in MODULE.read_jsonl(commentary)}
        cases = MODULE.read_jsonl(run / "cases.jsonl")
        results = [
            MODULE.score_response(source[c["provenance"]["record_id"]]["answer"], c)
            for c in cases
        ]
        accepted = sum(r["accepted"] for r in results)
        self.assertGreaterEqual(accepted, 270, f"accepted={accepted}/300")
        self.assertEqual(sum(r["M6"] for r in results), 0)
        print(f"\nQA references: accepted={accepted}/300; M6=0; excluded={len(excluded)}")


if __name__ == "__main__":
    unittest.main()
