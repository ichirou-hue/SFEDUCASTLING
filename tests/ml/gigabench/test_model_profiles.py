import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))

from gigabench.pipeline import source_hashes  # noqa: E402
from gigabench.profiles import QWEN_MODEL, resolve_profile  # noqa: E402

SPEC = importlib.util.spec_from_file_location("gigabench_profile_cli", ROOT / "run.py")
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ModelProfileTests(unittest.TestCase):
    def test_profile_modules_are_in_frozen_code_fingerprints(self):
        hashes = source_hashes()
        for filename in ("__init__.py", "qwen.py", "gigachess.py"):
            self.assertIn(f"gigabench/profiles/{filename}", hashes)

    def test_qwen_profile_and_default_are_equivalent(self):
        expected = ("qwen", QWEN_MODEL, "contract_v1_fen")
        self.assertEqual(resolve_profile("qwen-v1", None, None, None), expected)
        self.assertEqual(resolve_profile(None, "qwen", None, None), expected)

    def test_preserves_explicit_model_and_input_overrides(self):
        self.assertEqual(resolve_profile("qwen-v1", None, "/model/snapshot", "question_only"),
                         ("qwen", "/model/snapshot", "question_only"))
        self.assertEqual(resolve_profile(None, "mock", None, None)[2], "question_only")

    def test_incompatible_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --backend qwen"):
            resolve_profile("qwen-v1", "gigachess", None, None)

    def test_local_gigachess_uses_existing_runtime_and_does_not_choose_suite(self):
        argv = ["run.py", "generate", "--out", "unused", "--profile", "quick",
                "--model-profile", "gigachess-local-v1", "--project-root", "/runtime"]
        with patch.object(sys, "argv", argv), patch("gigabench.pipeline.generate") as generate:
            cli.main()
        args = generate.call_args.args[0]
        self.assertEqual((args.backend, args.input_profile, args.profile),
                         ("gigachess", "question_only", "quick"))
        self.assertEqual(args.project_root, Path("/runtime"))

    def test_gigachess_requires_runtime_path(self):
        argv = ["run.py", "generate", "--out", "unused", "--model-profile", "gigachess-local-v1"]
        with (
            patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as error,
        ):
            cli.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--project-root required for GigaChess", stderr.getvalue())
