"""Integration checks for the frozen CC0 Lichess subsets, without inference."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "ml/gigabench"
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from gigabench.external import eligible_puzzle  # noqa: E402
from gigabench.io import read_jsonl, sha  # noqa: E402
from gigabench.pipeline import prepare, verify  # noqa: E402

GAMES = ROOT / "data/sources/games_v1_1.json"
PUZZLES = ROOT / "data/sources/puzzles_v1_1.json"


class PublicSourceTests(unittest.TestCase):
    def test_frozen_files_match_verified_archive_and_cc0_metadata(self):
        for path, key, count, expected_hash in (
            (GAMES, "games", 3000, "3a68127e8341b4ea72dc8ba43a3f4c1e96a0c9cfeef4abf4a8e40863b501cf77"),
            (PUZZLES, "puzzles", 200, "0538832cbc694af9fb28d76dc2ad9dd3dc6ee35aca2fbcca4c9bfee24c0be8b9"),
        ):
            with self.subTest(source=key):
                data = json.loads(path.read_text())
                self.assertEqual(sha(path), expected_hash)
                self.assertEqual(len(data[key]), count)
                self.assertTrue(data["license"].startswith("CC0"))
                self.assertTrue(data["source"].startswith("https://database.lichess.org/"))
                self.assertEqual(len(data["source_sha256"]), 64)

    def test_all_game_moves_are_legal(self):
        for game in json.loads(GAMES.read_text())["games"]:
            board = chess.Board()
            for uci in game["moves"]:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves, game["headers"].get("Site"))
                board.push(move)

    def test_all_puzzles_have_legal_lines_and_pass_quality_filters(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        for puzzle in json.loads(PUZZLES.read_text())["puzzles"]:
            self.assertTrue(eligible_puzzle(puzzle, config["puzzle_quality"]))
            moves = puzzle["Moves"].split()
            self.assertGreaterEqual(len(moves), 2)
            board = chess.Board(puzzle["FEN"])
            for uci in moves:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves, puzzle["PuzzleId"])
                board.push(move)

    def test_prepare_connects_public_sources_without_private_data_or_inference(self):
        config = json.loads((ROOT / "config/default.json").read_text())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp) / "prepared"
            args = argparse.Namespace(
                out=out, config=ROOT / "config/default.json", profile="smoke",
                backend="qwen", without_legacy=True, games=GAMES, puzzles=PUZZLES,
                with_judgement=False, stockfish=None, max_new_tokens=2048,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(args)
            frozen, cases = verify(out)
            self.assertEqual(frozen, manifest)
            self.assertFalse(manifest["contains_local_source_comments"])
            self.assertIsNone(manifest["source_legacy_sha256"])
            self.assertEqual(manifest["explanation_cases"], 0)
            self.assertEqual(manifest["external_sources"]["games_pgn_sha256"], sha(GAMES))
            self.assertEqual(manifest["external_sources"]["puzzles_csv_sha256"], sha(PUZZLES))
            self.assertEqual(manifest["external_sources"]["missing"], ["position judgement"])
            for kind in ("human_move", "engine_move"):
                self.assertEqual(
                    sum(c["score_type"] == kind for c in cases),
                    config["game_positions"]["smoke"],
                )
            self.assertEqual(
                sum(c["score_type"] == "puzzle_move" for c in cases),
                4 * config["puzzles_per_bin"]["smoke"],
            )
            self.assertTrue(all(c.get("answer_protocol") == "answer_first_v1" for c in cases))
            self.assertFalse(any(c["private"] or c["suite"] == "legacy" for c in cases))
            queries = read_jsonl(out / "queries.jsonl")
            self.assertEqual(len(queries), len(cases))
            self.assertTrue(all(q["max_new_tokens"] == 2048 for q in queries))
            self.assertFalse((out / "responses.jsonl").exists())
            self.assertFalse((out / "run.json").exists())


if __name__ == "__main__":
    unittest.main()
