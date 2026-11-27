"""Offline Qwen contracts: no torch, model weights, GPU or network required."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))

from gigabench.adapters import QwenAdapter  # noqa: E402 -- standalone unittest import path


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class QwenAdapterTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock(__version__="fake-torch")
        self.torch.device.side_effect = lambda name: SimpleNamespace(type=name)
        self.torch.cuda.is_available.return_value = False
        self.transformers = MagicMock(__version__="fake-transformers")
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained.return_value
        self.model = self.transformers.AutoModelForCausalLM.from_pretrained.return_value
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config._commit_hash = "fixed-revision"
        self.model.peft_config = None
        self.model.generation_config.eos_token_id = 99
        self.batch = FakeBatch(input_ids=SimpleNamespace(shape=(1, 7)))
        self.tokenizer.return_value = self.batch
        self.tokenizer.apply_chat_template.return_value = "chat-rendered"
        self.tokenizer.decode.return_value = "  да\n"
        self.output = [31, 99]
        self.generated = MagicMock()
        self.generated.__getitem__.return_value.tolist.side_effect = lambda: self.output
        self.model.generate.return_value = self.generated
        self.modules = patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.snapshot_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.snapshot_dir.cleanup)
        self.snapshot = Path(self.snapshot_dir.name)
        self.fingerprint = {"config.json": "fixed-hash", "model.safetensors": "weights-hash"}
        self.snapshot_patch = patch(
            "gigabench.adapters.qwen_snapshot", return_value=(self.snapshot, self.fingerprint),
        )
        self.snapshot_mock = self.snapshot_patch.start()
        self.addCleanup(self.snapshot_patch.stop)
        self.query = {
            "id": "q1", "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "prompt": "Есть ли шах?", "max_new_tokens": 2,
        }

    def adapter(self, profile="contract_v1_fen"):
        return QwenAdapter("Qwen/Qwen3-1.7B", profile)

    def test_cpu_loads_float32_safetensors_and_eval(self):
        adapter = self.adapter()
        self.snapshot_mock.assert_called_once_with("Qwen/Qwen3-1.7B")
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            self.snapshot, local_files_only=True,
        )
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            self.snapshot, torch_dtype=self.torch.float32, use_safetensors=True, local_files_only=True,
        )
        self.model.to.assert_called_once_with(adapter.device)
        self.model.eval.assert_called_once_with()
        self.assertFalse(adapter.metadata["thinking"])
        self.assertFalse(adapter.metadata["do_sample"])
        self.assertFalse(adapter.metadata["training_performed"])
        self.assertEqual(adapter.metadata["revision"], "fixed-revision")
        self.assertEqual(adapter.metadata["model_files_sha256"], self.fingerprint)
        self.assertFalse(adapter.metadata["adapters_loaded"])
        json.dumps(adapter.metadata)

    def test_cuda_precision_selection(self):
        self.torch.cuda.is_available.return_value = True
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                self.torch.cuda.is_bf16_supported.return_value = bf16
                self.adapter()
                dtype = self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"]
                self.assertIs(dtype, self.torch.bfloat16 if bf16 else self.torch.float16)

    def test_fen_chat_template_disables_thinking_and_sampling(self):
        adapter = self.adapter()
        result = adapter.predict(self.query)
        prompt = f"Позиция FEN: {self.query['fen']}\n{self.query['prompt']}"
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.tokenizer.assert_called_once_with(
            "chat-rendered", return_tensors="pt", add_special_tokens=False,
        )
        self.model.generate.assert_called_once_with(
            **self.batch, max_new_tokens=2, do_sample=False, num_beams=1,
        )
        self.assertIs(self.batch.device, adapter.device)
        self.generated.__getitem__.assert_called_once_with((0, slice(7, None)))
        self.tokenizer.decode.assert_called_once_with([31, 99], skip_special_tokens=True)
        self.assertEqual(result["text"], "да")
        self.assertEqual(result["raw"], {
            "prompt": prompt, "chat_prompt": "chat-rendered", "output_ids": [31, 99],
        })
        self.assertFalse(result["truncated"])

    def test_explicit_question_only_profile_is_honored(self):
        self.adapter("question_only").predict(self.query)
        messages = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages, [{"role": "user", "content": self.query["prompt"]}])

    def test_truncation_for_scalar_list_and_missing_eos(self):
        adapter = self.adapter()
        for eos, output, expected in (
            (99, [31, 99], False),
            ([98, 99], [31, 98], False),
            (99, [31, 32], True),
            (None, [31, 32], True),
            (99, [31], False),
            (99, [], False),
        ):
            with self.subTest(eos=eos, output=output):
                self.model.generation_config.eos_token_id = eos
                self.output = output
                self.assertEqual(adapter.predict(self.query)["truncated"], expected)

    def test_generation_error_is_not_converted_to_valid_answer(self):
        self.model.generate.side_effect = RuntimeError("dependency failure")
        with self.assertRaisesRegex(RuntimeError, "dependency failure"):
            self.adapter().predict(self.query)

    def test_loaded_peft_model_is_rejected(self):
        self.model.peft_config = {"default": {"adapter": "unexpected"}}
        with self.assertRaisesRegex(ValueError, "PEFT"):
            self.adapter()

    def test_answer_can_be_scored_without_chat_prompt_leaking(self):
        from gigabench.scoring import score_case

        response = self.adapter().predict(self.query)
        case = {
            "id": "q1", "suite": "rules", "task": "check", "group_id": "g",
            "fen": self.query["fen"], "score_type": "bool", "gold": {"answer": True},
        }
        row = score_case(case, response)
        self.assertTrue(row["correct"])
        self.assertTrue(row["protocol_compliant"])

    def test_pipeline_default_profile_override_errors_and_scoring(self):
        from gigabench.io import read_jsonl
        from gigabench.pipeline import generate, prepare, score

        for profile in (None, "question_only"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(
                    out=Path(tmp) / "run", config=ROOT / "config/default.json",
                    profile="smoke", without_legacy=True, backend="qwen",
                    model="Qwen/Qwen3-1.7B", project_root=None, adapter=None,
                    stage="automatic", rubrics=None, resume=False, limit=2,
                    stockfish=None, input_profile=profile,
                )
                prepare(args)
                self.model.generate.side_effect = [self.generated, RuntimeError("offline failure")]
                metadata = generate(args)
                self.assertEqual(metadata["backend"]["input_profile"], profile or "contract_v1_fen")
                responses = read_jsonl(args.out / "responses.jsonl")
                self.assertEqual(responses[0]["text"], "да")
                self.assertEqual(responses[1]["error"]["type"], "RuntimeError")
                self.assertEqual(responses[1]["text"], "")
                result = score(args)
                self.assertFalse(result["is_mock"])
                self.assertFalse(result["coverage"]["complete"])
                self.assertIsNone(result["automatic_task_macro_percent"])
                error_row = next(r for r in read_jsonl(args.out / "scores.jsonl")
                                 if r["id"] == responses[1]["id"])
                self.assertFalse(error_row["correct"])
                args.resume = True
                args.model = "another/model"
                with self.assertRaisesRegex(ValueError, "Cannot resume"):
                    generate(args)
                args.model = "Qwen/Qwen3-1.7B"
                self.snapshot_mock.return_value = (
                    self.snapshot, {**self.fingerprint, "config.json": "modified"},
                )
                with self.assertRaisesRegex(ValueError, "Cannot resume"):
                    generate(args)
                self.snapshot_mock.return_value = (self.snapshot, self.fingerprint)

    def test_cli_rejects_gigachess_adapter_for_qwen_before_loading_model(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "generate", "--out", "unused",
             "--backend", "qwen", "--adapter", "adapter.bin"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--adapter is only supported by the GigaChess backend", result.stderr)


class QwenSnapshotTests(unittest.TestCase):
    def test_local_config_and_weights_replacement_changes_fingerprint(self):
        from gigabench.adapters import qwen_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"version": 1}', encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights-v1")
            snapshot, original = qwen_snapshot(str(root))
            self.assertEqual(snapshot, root)
            self.assertEqual(set(original), {"config.json", "model.safetensors"})
            for filename in ("config.json", "model.safetensors"):
                with self.subTest(filename=filename):
                    previous = qwen_snapshot(str(root))[1]
                    (root / filename).write_bytes(b"replacement")
                    changed = qwen_snapshot(str(root))[1]
                    self.assertNotEqual(changed[filename], previous[filename])
                    other = ({"config.json", "model.safetensors"} - {filename}).pop()
                    self.assertEqual(changed[other], previous[other])

    def test_local_peft_adapter_is_rejected(self):
        from gigabench.adapters import qwen_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PEFT"):
                qwen_snapshot(tmp)

    def test_hub_download_is_resolved_once_for_fingerprinting(self):
        from gigabench.adapters import qwen_snapshot

        hub = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"huggingface_hub": hub}):
            hub.snapshot_download.return_value = tmp
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            snapshot, fingerprint = qwen_snapshot("Qwen/Qwen3-1.7B")
            hub.snapshot_download.assert_called_once_with("Qwen/Qwen3-1.7B")
            self.assertEqual(snapshot, Path(tmp))
            self.assertIn("config.json", fingerprint)


if __name__ == "__main__":
    unittest.main()
