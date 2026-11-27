"""Tests for the isolated numeric-cell dataset transformation."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "ml/benchmark/tools/prepare_handoff_numeric_cells.py"
SPEC = importlib.util.spec_from_file_location("prepare_handoff_numeric_cells", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareNumericCellsTests(unittest.TestCase):
    def test_only_cell_prompts_gain_board_and_fen_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            formatter = root / "numeric_board.py"
            formatter.write_text(
                "def format_cell_query(query):\n"
                "    return 'NUMERIC BOARD\\n' + query['prompt']\n",
                encoding="utf-8",
            )
            splits = {
                "train": [
                    self.row("move_commentary", "a1", "Опиши ход."),
                    self.row("free_square", "b2", "Поле b2 свободно?"),
                    self.row("piece_on_square", "c3", "Какая фигура на c3?"),
                ],
                "validation": [
                    self.row("move_commentary", "d4", "Опиши ход."),
                    self.row("free_square", "e5", "Поле e5 свободно?"),
                ],
                "test": [
                    self.row("move_commentary", "f6", "Опиши ход."),
                    self.row("piece_on_square", "g7", "Какая фигура на g7?"),
                ],
            }
            split_manifest = {}
            for name, rows in splits.items():
                raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
                (source / f"{name}.jsonl").write_bytes(raw)
                split_manifest[name] = {
                    "n": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "tasks": dict(Counter(row["task"] for row in rows)),
                }
            original_manifest = {
                "name": "handoff-desc-v1",
                "input_format": "fen",
                "archive_sha256": "archive",
                "source_hashes": {},
                "benchmark_cases_sha256": "benchmark",
                "heldout_placements": 1,
                "excluded": [],
                "splits": split_manifest,
            }
            (source / "manifest.json").write_text(
                json.dumps(original_manifest), encoding="utf-8"
            )
            out = root / "variant"

            manifest = PREPARE.prepare(source, out, formatter)

            self.assertEqual(manifest["name"], "handoff-desc-v1-numeric-cells-v1")
            self.assertEqual(manifest["splits"]["train"]["numeric_cell_inputs"], 2)
            for split, rows in splits.items():
                actual = [
                    json.loads(line)
                    for line in (out / f"{split}.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                for before, after in zip(rows, actual, strict=True):
                    if before["task"] == "move_commentary":
                        self.assertEqual(after, before)
                    else:
                        question = before["messages"][0]["content"].split("\n", 1)[1]
                        self.assertEqual(
                            after["messages"][0]["content"],
                            f"Позиция FEN: {before['fen']}\n\n"
                            f"NUMERIC BOARD\n{question}",
                        )
                        self.assertEqual(after["messages"][1], before["messages"][1])

    @staticmethod
    def row(task, square, question):
        fen = f"fen-{square}"
        return {
            "id": f"{task}-{square}",
            "fen": fen,
            "task": task,
            "messages": [
                {
                    "role": "user",
                    "content": f"Позиция FEN: {fen}\n{question}",
                },
                {"role": "assistant", "content": "answer"},
            ],
        }


if __name__ == "__main__":
    unittest.main()
