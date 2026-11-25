"""Independent regressions for automatic-only preparation; no model or engine runs."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3] / "ml/gigabench"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench.cases import legacy_cases  # noqa: E402
from gigabench.external import judgement_cases  # noqa: E402
from gigabench.io import read_jsonl  # noqa: E402
from gigabench.pipeline import prepare, verify  # noqa: E402


class AutomaticOnlyIndependentTests(unittest.TestCase):
    def args(self, out, **kwargs):
        values = {
            "out": out, "config": ROOT / "config/default.json", "profile": "smoke",
            "backend": "qwen", "without_legacy": False,
            "legacy_source": ROOT / "data/private/legacy_cases.jsonl",
            "games": None, "puzzles": None, "with_judgement": False, "stockfish": None,
        }
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_missing_flag_matches_explicit_false_and_comments_never_enter_automatic(self):
        source = {
            "id": "synthetic", "fen_before": chess.Board().fen(),
            "move": {"uci": "e2e4"}, "question": "EXPERT_QUESTION_SENTINEL",
            "source": {"chapter_url": "local:qa", "variation": "main",
                       "original_comment": "EXPERT_COMMENT_SENTINEL"},
        }
        board = chess.Board()
        board.push_uci("e2e4")
        source["fen_after"] = board.fen()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmpdir:
            tmp = Path(tmpdir)
            private = tmp / "source.jsonl"
            private.write_text(json.dumps(source), encoding="utf-8")
            datasets = []
            for index, options in enumerate(({}, {"without_explanations": False},
                                             {"without_explanations": True})):
                out = tmp / str(index)
                args = self.args(out, legacy_source=private, **options)
                with (
                    patch("gigabench.pipeline.build", return_value=legacy_cases(private)),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    manifest = prepare(args)
                _, cases = verify(out)
                datasets.append(cases)
                automatic = [c for c in cases if c["score_type"] != "explanation"]
                for sentinel in ("EXPERT_QUESTION_SENTINEL", "EXPERT_COMMENT_SENTINEL"):
                    self.assertNotIn(sentinel, json.dumps(automatic))
                    self.assertNotIn(sentinel, (out / "queries.jsonl").read_text())
                self.assertEqual({c["task"] for c in automatic},
                                 {"M1_occupied", "M1_empty", "M2", "M3"})
                self.assertEqual({q["max_new_tokens"] for q in read_jsonl(out / "queries.jsonl")},
                                 {2048})
                self.assertEqual(manifest["excluded_explanation_cases"], int(index == 2))
            self.assertEqual(datasets[0], datasets[1])
            self.assertEqual([c for c in datasets[0] if c["score_type"] != "explanation"],
                             datasets[2])

    def test_real_private_source_counts_and_frozen_default_qwen_budget(self):
        private = ROOT / "data/private/legacy_cases.jsonl"
        if not private.is_file():
            self.skipTest("Private source is deliberately not distributed")
        rows = legacy_cases(private)
        self.assertEqual(Counter(c["score_type"] == "explanation" for c in rows),
                         {False: 346, True: 89})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp) / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(self.args(out, without_explanations=True))
            _, cases = verify(out)
            self.assertEqual(sum(c["suite"] == "legacy" for c in cases), 346)
            self.assertEqual(manifest["excluded_explanation_cases"], 89)
            self.assertEqual(manifest["explanation_cases"], 0)
            queries = read_jsonl(out / "queries.jsonl")
            self.assertEqual(len(queries), len(cases))
            self.assertEqual({q["max_new_tokens"] for q in queries}, {2048})
            self.assertEqual(read_jsonl(out / "coach_rubrics_pending.jsonl"), [])

    def test_engine_analysis_failure_is_propagated_and_engine_is_closed(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        engine = MagicMock()
        engine.analyse.side_effect = RuntimeError("engine dependency failed")
        judge = MagicMock()
        judge.engine = engine
        judge.metadata = {"id": {"name": "test"}, "binary_sha256": "test", "networks": {}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            binary = Path(tmp) / "engine"
            binary.write_bytes(b"fake engine, never executed")
            with (
                patch("gigabench.external.EngineJudge", return_value=judge),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "engine dependency failed"),
            ):
                judgement_cases(binary, [{"board": chess.Board()}], config, "smoke")
        judge.close.assert_called_once()
