"""Full automatic preparation excludes expert work without losing legacy M1-M3."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3] / "ml/gigabench"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.engine  # noqa: E402
from gigabench.engine import EngineJudge  # noqa: E402
from gigabench.external import judgement_cases  # noqa: E402
from gigabench.io import read_jsonl, sha  # noqa: E402
from gigabench.pipeline import prepare, verify  # noqa: E402
from gigabench.scoring import score_case, summarize  # noqa: E402


class ExpertExclusionTests(unittest.TestCase):
    def test_exact_task_macro_excludes_public_and_private_engine_quality(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        config["bootstrap"]["resamples"] = 10
        base = {"fen": chess.STARTING_FEN, "group_id": "test"}
        yes = dict(base, id="yes", suite="structural", task="check",
                   score_type="bool", gold={"answer": True})
        engine = MagicMock()
        engine.grade.return_value = {"ordinary": True, "accepted": False, "legal": True}
        rows = [score_case(yes, {"text": "Да"})]
        for suite in ("games", "legacy"):
            move = dict(base, id=suite, suite=suite, task="M3", score_type="engine_move", gold={})
            rows.append(score_case(move, {"text": "e2e4"}, engine))
        summary = summarize(rows, config)
        self.assertEqual(summary["automatic_task_macro_percent"], 100.0)
        self.assertEqual(summary["by_task"]["games/M3"]["correct"], 0)
        self.assertEqual(summary["by_task"]["legacy/M3"]["correct"], 0)
        self.assertEqual(summary["by_block"]["move_choice"]["scored_tasks"], 2)

    def test_exclude_only_explanations_and_freeze_automatic_legacy(self):
        board = chess.Board()
        after = board.copy()
        after.push_uci("e2e4")
        source = {
            "id": "synthetic", "fen_before": board.fen(), "fen_after": after.fen(),
            "move": {"uci": "e2e4"}, "question": "Объясни ход.",
            "source": {"chapter_url": "local:test", "variation": "main",
                       "original_comment": "PRIVATE_EXPERT_COMMENT_SENTINEL"},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            private = Path(tmp) / "source.jsonl"
            private.write_text(json.dumps(source) + "\n")
            manifests, rows = [], []
            for excluded in (False, True):
                out = Path(tmp) / str(excluded)
                args = argparse.Namespace(
                    out=out, config=ROOT / "config/default.json", profile="smoke",
                    backend="qwen", without_legacy=False, legacy_source=private,
                    without_explanations=excluded, max_new_tokens=2048,
                    games=None, puzzles=None, with_judgement=False, stockfish=None,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = prepare(args)
                frozen, cases = verify(out)
                self.assertEqual(frozen, manifest)
                manifests.append(manifest)
                rows.append(cases)
                legacy = [c for c in cases if c["suite"] == "legacy"]
                self.assertEqual(sum(c["score_type"] != "explanation" for c in legacy), 4)
                self.assertTrue(all(c["private"] for c in legacy))
                self.assertEqual(manifest["source_legacy_sha256"], sha(private))
                self.assertTrue(manifest["contains_private_cases"])
                self.assertEqual(manifest["contains_local_source_comments"], not excluded)
                self.assertEqual(len(read_jsonl(out / "queries.jsonl")), len(cases) - manifest["explanation_cases"])
                self.assertFalse((out / "responses.jsonl").exists())
                if excluded:
                    self.assertNotIn("PRIVATE_EXPERT_COMMENT_SENTINEL", json.dumps(cases))
                    self.assertEqual(read_jsonl(out / "coach_rubrics_pending.jsonl"), [])
            self.assertEqual(manifests[0]["explanation_cases"], 1)
            self.assertEqual(manifests[0]["excluded_explanation_cases"], 0)
            self.assertEqual(manifests[1]["explanation_cases"], 0)
            self.assertEqual(manifests[1]["excluded_explanation_cases"], 1)
            self.assertEqual([c for c in rows[0] if c["score_type"] != "explanation"], rows[1])


class JudgementPreparationTests(unittest.TestCase):
    def test_engine_constructor_closes_process_on_export_timeout(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        engine = MagicMock()
        engine.id = {"name": "fake"}
        engine.options = {"EvalFile": SimpleNamespace(default="embedded.nnue")}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            binary = Path(tmp) / "stockfish"
            binary.write_bytes(b"fake-engine")
            with (
                patch("gigabench.engine.chess.engine.SimpleEngine.popen_uci", return_value=engine),
                patch("gigabench.engine.subprocess.run", side_effect=subprocess.TimeoutExpired("export_net", 60)),
                self.assertRaises(subprocess.TimeoutExpired),
            ):
                EngineJudge(binary, config["engine"], Path(tmp) / "work")
            engine.quit.assert_called_once()

    def test_engine_constructor_preserves_error_if_cleanup_also_fails(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        engine = MagicMock()
        engine.configure.side_effect = RuntimeError("configuration failed")
        engine.quit.side_effect = RuntimeError("already stopped")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            binary = Path(tmp) / "stockfish"
            binary.write_bytes(b"fake-engine")
            with (
                patch("gigabench.engine.chess.engine.SimpleEngine.popen_uci", return_value=engine),
                self.assertRaisesRegex(RuntimeError, "configuration failed"),
            ):
                EngineJudge(binary, config["engine"], Path(tmp) / "work")
            engine.quit.assert_called_once()

    def test_stable_balanced_gold_pins_engine_settings_and_filters_boundaries(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        engine = MagicMock()
        engine.id = {"name": "test-stockfish"}
        pairs = [(30, 30), (60, 200), (chess.engine.Mate(2), chess.engine.Mate(2))]
        pairs += [(cp, cp) for cp in (500, 200, 60, 0, -60, -200, -500)]
        values = []
        for pair in pairs:
            for score in pair:
                cp = chess.engine.Cp(score) if isinstance(score, int) else score
                values.append({"score": chess.engine.PovScore(cp, chess.WHITE)})
        engine.analyse.side_effect = values
        positions = [{"board": chess.Board(), "group": f"test:{i}",
                      "provenance": {"source": "test"}} for i in range(len(pairs))]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            binary = Path(tmp) / "stockfish"
            binary.write_bytes(b"test-engine")
            judge = MagicMock()
            judge.engine = engine
            judge.metadata = {
                "id": engine.id, "binary_sha256": sha(binary),
                "networks": {"EvalFile": {"sha256": "test-network", "bytes": 123}},
            }
            with (
                patch("gigabench.external.EngineJudge", return_value=judge) as factory,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                cases = judgement_cases(binary, positions, config, "smoke")
            self.assertEqual({c["gold"]["answer"] for c in cases}, set("ABCDEFG"))
            self.assertEqual(len(cases), 7)
            self.assertTrue(all(c["gold"]["cp_white"] in (500, 200, 60, 0, -60, -200, -500) for c in cases))
            self.assertTrue(all(c["provenance"]["judgement_engine"]["binary_sha256"] == sha(binary) for c in cases))
            self.assertEqual(factory.call_args.args[:2], (binary, config["engine"]))
            self.assertTrue(all(c["provenance"]["judgement_engine"]["networks"] == judge.metadata["networks"] for c in cases))
            judge.close.assert_called_once()
            self.assertIn("accepted=7/7", output.getvalue())
            expected_nodes = [config["judgement"]["nodes"], config["judgement"]["nodes"] * config["judgement"]["recheck_factor"]]
            self.assertEqual([call.args[1].nodes for call in engine.analyse.call_args_list], expected_nodes * len(pairs))

    def test_missing_stable_buckets_fails_instead_of_silently_shrinking(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        engine = MagicMock()
        engine.id = {"name": "test-stockfish"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            binary = Path(tmp) / "stockfish"
            binary.write_bytes(b"test-engine")
            judge = MagicMock()
            judge.engine = engine
            judge.metadata = {"id": engine.id, "binary_sha256": sha(binary), "networks": {}}
            with (
                patch("gigabench.external.EngineJudge", return_value=judge),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "Not enough stable positions"),
            ):
                judgement_cases(binary, [], config, "smoke")
            judge.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
