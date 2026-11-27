import json
import tempfile
import unittest
from pathlib import Path

from ml.pipelines.assemble_mixed import SPLITS, assemble, digest


class MixedDatasetTests(unittest.TestCase):
    def source(self, root, name, task, reused_position=False):
        path = root / name
        path.mkdir()
        stats = {}
        for index, split in enumerate(SPLITS):
            row = {
                "id": f"{name}:{split}", "task": task,
                "fen": f"placement-{0 if reused_position else index} w - - 0 1",
                "messages": [{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}],
            }
            raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
            (path / f"{split}.jsonl").write_bytes(raw)
            stats[split] = {"sha256": digest(raw), "n": 1}
        (path / "manifest.json").write_text(json.dumps({
            "name": name, "benchmark_cases_sha256": "automatic",
            "commentary_benchmark_cases_sha256": "commentary", "splits": stats,
        }), encoding="utf-8")

    def test_preserves_samples_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root, "commentary", "move_commentary")
            self.source(root, "cell-questions", "cell")
            first, second = root / "mixed-a", root / "mixed-b"
            self.assertEqual(assemble(root, first), assemble(root, second))
            for split in SPLITS:
                file = f"{split}.jsonl"
                self.assertEqual((first / file).read_bytes(), (second / file).read_bytes())
                mixed = [json.loads(line) for line in (first / file).read_bytes().splitlines()]
                expected = [json.loads((root / name / file).read_bytes())
                            for name in ("commentary", "cell-questions")]
                self.assertCountEqual(mixed, expected)
            with self.assertRaises(FileExistsError):
                assemble(root, first)

    def test_cross_split_position_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root, "commentary", "move_commentary", reused_position=True)
            self.source(root, "cell-questions", "cell")
            with self.assertRaisesRegex(ValueError, "position crosses splits"):
                assemble(root, root / "mixed")
            self.assertFalse((root / "mixed").exists())

    def test_changed_input_cannot_be_used_with_old_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source(root, "commentary", "move_commentary")
            self.source(root, "cell-questions", "cell")
            train = root / "commentary/train.jsonl"
            train.write_bytes(train.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                assemble(root, root / "mixed")
            self.assertFalse((root / "mixed").exists())

    def test_wrong_benchmark_count_task_or_id_is_rejected(self):
        for defect, message in (
            ("benchmark", "benchmark differs"),
            ("count", "count mismatch"),
            ("task", "unexpected task"),
            ("id", "duplicate ID"),
        ):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.source(root, "commentary", "move_commentary")
                self.source(root, "cell-questions", "cell")
                source = root / "cell-questions"
                manifest_path = source / "manifest.json"
                manifest = json.loads(manifest_path.read_bytes())
                if defect == "benchmark":
                    manifest["benchmark_cases_sha256"] = "other"
                elif defect == "count":
                    manifest["splits"]["train"]["n"] = 2
                else:
                    file = source / "train.jsonl"
                    row = json.loads(file.read_bytes())
                    if defect == "task":
                        row["task"] = "move_commentary"
                    else:
                        row["id"] = "commentary:train"
                    raw = (json.dumps(row) + "\n").encode()
                    file.write_bytes(raw)
                    manifest["splits"]["train"]["sha256"] = digest(raw)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    assemble(root, root / "mixed")
                self.assertFalse((root / "mixed").exists())
