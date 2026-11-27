"""Regression tests for the comparison HTML and its compatibility guards."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "ml/benchmark/tools/compare_reports.py"
SPEC = importlib.util.spec_from_file_location("gigabench_compare_reports", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def save(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


class ComparisonTests(unittest.TestCase):
    def test_canonical_name_uses_model_training_and_user_named_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            first = REPORT.model_slug(before, "Qwen/Qwen3-1.7B", "base")
            second = REPORT.model_slug(after, "Qwen/Qwen3-1.7B", "LoRA", "Pilot Cells V1")
            self.assertEqual(REPORT.comparison_filename(first, second),
                             "qwen-qwen3-1.7b--base--none__vs__qwen-qwen3-1.7b--lora--pilot-cells-v1.html")
            with self.assertRaisesRegex(ValueError, "имя датасета"):
                REPORT.model_slug(after, "Qwen/Qwen3-1.7B", "LoRA")
            with self.assertRaisesRegex(ValueError, "датасет должен быть none"):
                REPORT.model_slug(before, "Qwen/Qwen3-1.7B", "base", "pilot-cells-v1")
            metadata = REPORT.load(after / "run.json")
            metadata["model_identity"] = {
                "base_model": "Qwen/Qwen3-1.7B", "training_type": "LoRA", "dataset": "Pilot Cells V1",
            }
            save(after / "run.json", metadata)
            self.assertEqual(REPORT.model_slug(after), second)

    def test_cli_writes_canonical_name_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            command = [sys.executable, str(SCRIPT), str(before), str(after),
                       "--before-training", "base", "--after-training", "lora",
                       "--after-dataset", "Мои Клетки V2"]
            bad_path = root / "comparison.html"
            result = subprocess.run([*command, "--out", str(bad_path)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(bad_path.exists())
            subprocess.run([*command, "--out-dir", tmp], check=True, capture_output=True)
            target = root / "base--base--none__vs__lora--lora--мои-клетки-v2.html"
            self.assertTrue(target.is_file())
            original = target.read_bytes()
            result = subprocess.run([*command, "--out-dir", tmp], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_bytes(), original)
            target.unlink()
            subprocess.run([*command, "--out", str(target)], check=True, capture_output=True)
            self.assertEqual(target.read_bytes(), original)

    def test_cli_accepts_three_runs_with_first_as_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self.create_run(root, "base", (20, 16, 40)),
                self.create_run(root, "lora-handoff-desc-v1", (25, 10, 40)),
                self.create_run(root, "lora-pilot-cells-v1", (10, 20, 50)),
            ]
            for path in runs:
                data = REPORT.load(path / "run.json")
                data["backend"].update({
                    "model": "Qwen/Qwen3-1.7B",
                    "training_performed": False,
                    "adapter": None if path.name == "base" else f"/adapters/{path.name}/adapter",
                })
                save(path / "run.json", data)

            subprocess.run(
                [sys.executable, str(SCRIPT), *(str(path) for path in runs),
                 "--out-dir", tmp],
                check=True, capture_output=True,
            )
            target = root / (
                "qwen-qwen3-1.7b--base--none__vs__"
                "qwen-qwen3-1.7b--lora--handoff-desc-v1__vs__"
                "qwen-qwen3-1.7b--lora--pilot-cells-v1.html"
            )
            self.assertTrue(target.is_file())
            page = target.read_text(encoding="utf-8")
            self.assertIn("handoff-desc-v1", page)
            self.assertIn("pilot-cells-v1", page)
            self.assertIn("−10,00 п. п.", page)

    def create_run(self, root, name, scores, query="{}"):
        path = root / name
        path.mkdir()
        (path / "cases.jsonl").write_text("{}\n")
        (path / "queries.jsonl").write_text(query + "\n")
        (path / "responses.jsonl").write_text(json.dumps({"model": name}) + "\n")
        (path / "report.html").write_text("<html>")
        save(path / "manifest.json", {
            "cases_sha256": REPORT.digest(path / "cases.jsonl"),
            "queries_sha256": REPORT.digest(path / "queries.jsonl"),
            "code_sha256": {}, "config": {}, "chess_module_sha256": "rules",
        })
        binding = {f"{key}_sha256": REPORT.digest(path / f"{key}.{ext}")
                   for key, ext in (("manifest", "json"), ("responses", "jsonl"))}
        save(path / "run.json", {**binding, "backend": {"model": "<" + name + ">", "input_profile": "numeric"}})
        save(path / "summary.json", {
            **binding,
            "coverage": {"complete": True, "expected": 300, "completed": 300},
            "by_block": {k: {"tasks": [k]} for k in ("board_reading", "move_choice", "rules")}
            | {"tactics": {"tasks": []}},
            "by_task": {k: {"correct": s, "records": 100}
                        for k, s in zip(("board_reading", "move_choice", "rules"), scores, strict=True)},
        })
        return path

    def test_colors_units_no_data_and_html_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            page = REPORT.compare(before / "report.html", after / "summary.json")
        for expected in ('class="up">(Δ +5,00 п. п.)', 'class="down">(Δ −6,00 п. п.)',
                         'class="flat">(Δ 0,00 п. п.)', 'class="na">(Δ Нет данных)',
                         "&lt;base&gt;", "76 / 300", "75 / 300"):
            self.assertIn(expected, page)

    def test_three_run_report_uses_first_baseline_and_shows_auto_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                self.create_run(root, "base", (20, 16, 40)),
                self.create_run(root, "handoff", (25, 10, 40)),
                self.create_run(root, "pilot", (10, 20, 50)),
            ]
            metrics = (
                {"M4": 20, "M5": 30, "M6": 50, "M7": 60, "accepted_percent": 10},
                {"M4": 80, "M5": 70, "M6": 20, "M7": 90, "accepted_percent": 35},
                {"M4": 50, "M5": 40, "M6": 30, "M7": 70, "accepted_percent": 15},
            )
            for path, values in zip(runs, metrics, strict=True):
                summary = REPORT.load(path / "summary.json")
                summary["commentary_auto_metrics"] = {
                    "criteria_version": "m4-m7-auto-v1", "n": 300, "expected": 300,
                    "complete": True, **values,
                }
                save(path / "summary.json", summary)

            page = REPORT.compare(*runs, labels=("base", "handoff", "pilot"))

        for expected in (
            "base", "handoff", "pilot", "(Δ −10,00 п. п.)",
            "M4-auto", "M5-auto", "M6-auto", "M7-auto",
            "(Δ +60,00 п. п.)", "(Δ −30,00 п. п.)", "(Δ +25,00 п. п.)",
        ):
            self.assertIn(expected, page)
        self.assertNotIn("<td></td>", page)
        self.assertNotIn("Δ к базе</th>", page)
        self.assertEqual(page.count('<col style="width:25.0000%">'), 4)

    def test_three_run_report_rejects_incompatible_later_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [self.create_run(root, name, (20, 16, 40))
                    for name in ("base", "lora-a", "lora-b")]
            data = REPORT.load(runs[2] / "run.json")
            data["backend"]["input_profile"] = "fen"
            save(runs[2] / "run.json", data)
            with self.assertRaisesRegex(ValueError, "input"):
                REPORT.compare(*runs)

    def test_changed_prompts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40), query='{"prompt":"other"}')
            with self.assertRaisesRegex(ValueError, "queries_sha256"):
                REPORT.compare(before, after)

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            data = REPORT.load(after / "summary.json")
            data["coverage"]["complete"] = False
            save(after / "summary.json", data)
            with self.assertRaisesRegex(ValueError, "полные прогоны"):
                REPORT.compare(before, after)
            data["coverage"]["complete"] = True
            data["is_mock"] = True
            save(after / "summary.json", data)
            with self.assertRaisesRegex(ValueError, "не mock"):
                REPORT.compare(before, after)

    def test_summary_from_other_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            (after / "summary.json").write_bytes((before / "summary.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "responses.jsonl"):
                REPORT.compare(before, after)

    def test_model_preset_name_is_not_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            run = REPORT.load(after / "run.json")
            run["backend"]["model_profile"] = "qwen-v1"
            save(after / "run.json", run)
            REPORT.compare(before, after)
            for key in ("template", "numeric_inputs_sha256"):
                with self.subTest(key=key):
                    run["backend"][key] = "different"
                    save(after / "run.json", run)
                    with self.assertRaisesRegex(ValueError, "input"):
                        REPORT.compare(before, after)
                    del run["backend"][key]

    def test_incompatible_profiles_and_engine_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            run = REPORT.load(after / "run.json")
            run["backend"]["input_profile"] = "fen"
            save(after / "run.json", run)
            with self.assertRaisesRegex(ValueError, "input"):
                REPORT.compare(before, after)
            run["backend"]["input_profile"] = "numeric"
            save(after / "run.json", run)
            (after / "engine").mkdir()
            save(after / "engine/engine.json", {"id": "other"})
            with self.assertRaisesRegex(ValueError, "stockfish"):
                REPORT.compare(before, after)

    def test_changed_config_or_scorer_is_rejected(self):
        for key in ("config", "code_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                before = self.create_run(root, "base", (20, 16, 40))
                after = self.create_run(root, "lora", (25, 10, 40))
                manifest_path = after / "manifest.json"
                manifest = REPORT.load(manifest_path)
                manifest[key] = {"changed": True}
                save(manifest_path, manifest)
                for name in ("run.json", "summary.json"):
                    data = REPORT.load(after / name)
                    data["manifest_sha256"] = REPORT.digest(manifest_path)
                    save(after / name, data)
                with self.assertRaisesRegex(ValueError, key):
                    REPORT.compare(before, after)

    def test_paired_report_requires_manifested_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pair"
            root.mkdir()
            before = self.create_run(root, "base", (20, 16, 40))
            after = self.create_run(root, "lora", (25, 10, 40))
            for path in (before, after):
                (path / "manifest.json").unlink()
            (root / "queries.jsonl").write_text("{}\n")
            (root / "engine").mkdir()
            save(root / "engine/engine.json", {"id": "fixture"})
            save(root / "settings.json", {
                "source_cases_sha256": REPORT.digest(before / "cases.jsonl"),
                "input": "numeric", "engine": {}, "temperature": 0, "thinking": False,
                "max_new_tokens": 2048, "stockfish": {"id": "fixture"},
            })
            paths = ("settings.json", "queries.jsonl", "engine/engine.json",
                     *(f"{m}/{f}" for m in ("base", "lora") for f in ("cases.jsonl", "summary.json", "run.json")))
            (root / "SHA256SUMS").write_text("".join(f"{REPORT.digest(root / n)}  {n}\n" for n in paths))
            self.assertIn("+5,00 п. п.", REPORT.compare(before, after))
            copy = root.with_name("other_pair")
            shutil.copytree(root, copy)
            with self.assertRaisesRegex(ValueError, "pair"):
                REPORT.compare(before, copy / "lora")
            for name in paths:
                with self.subTest(name=name):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
                        REPORT.compare(before, after)
                    path.write_bytes(original)
