"""Offline batch-runner contracts; no torch, model weights or network needed."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))
from gigabench.adapters import input_builder  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "qwen_batch_under_test", ROOT / "tools/qwen_batch.py",
)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class FakeTensor:
    """Only the rectangular shape/slicing API used by batched_predict."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        rows, columns = key
        return FakeTensor([row[columns] for row in self.rows[rows]])

    def tolist(self):
        return self.rows


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self._pad_token = "<pad>"
        self.apply_chat_template = MagicMock(
            side_effect=lambda messages, **_kwargs: "CHAT:" + messages[0]["content"],
        )
        self.decode = MagicMock(
            side_effect=lambda tokens, **_kwargs: " " + ",".join(map(str, tokens)) + " ",
        )
        self.calls = []

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = 99

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs, self.padding_side))
        # Unequal original lengths: [11] and [21, 22, 23].
        rows = [[11] if i == 0 else [21, 22, 23] for i in range(len(texts))]
        width = max(map(len, rows))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        self.inputs = FakeInputs(
            input_ids=FakeTensor(padded), attention_mask=FakeTensor(masks),
        )
        return self.inputs


def query(index=0, budget=4):
    return {
        "id": f"q{index}", "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
        "prompt": f"Вопрос {index}?", "max_new_tokens": budget,
    }


def adapter_for(outputs, eos=99, profile="contract_v1_fen"):
    tokenizer = FakeTokenizer()
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))

    def generate(**kwargs):
        return FakeTensor([
            prefix + list(output)
            for prefix, output in zip(kwargs["input_ids"].rows, outputs, strict=True)
        ])

    model.generate = MagicMock(side_effect=generate)
    return SimpleNamespace(
        tokenizer=tokenizer, model=model, device="fake-cpu",
        build_input=input_builder(profile)[0],
    )


class BatchedPredictTests(unittest.TestCase):
    def test_empty_batch_does_not_touch_adapter(self):
        self.assertEqual(batch.batched_predict(None, []), [])

    def test_left_padding_common_width_order_and_result_contract(self):
        adapter = adapter_for([[31, 99, 0, 0], [41, 42, 43, 44]])
        queries = [query(0), query(1)]
        results = batch.batched_predict(adapter, queries)
        self.assertEqual(adapter.tokenizer.inputs["input_ids"].rows, [[0, 0, 11], [21, 22, 23]])
        self.assertEqual(adapter.tokenizer.inputs["attention_mask"].rows, [[0, 0, 1], [1, 1, 1]])
        self.assertEqual(adapter.tokenizer.inputs.device, "fake-cpu")
        self.assertEqual(adapter.tokenizer.calls[0][1:], (
            {"padding": True, "return_tensors": "pt", "add_special_tokens": False}, "left",
        ))
        for i, (tokens, truncated) in enumerate((([31, 99], False), ([41, 42, 43, 44], True))):
            prompt = f"Позиция FEN: {queries[i]['fen']}\n{queries[i]['prompt']}"
            self.assertEqual(results[i], {
                "text": ",".join(map(str, tokens)),
                "raw": {"prompt": prompt, "chat_prompt": "CHAT:" + prompt, "output_ids": tokens},
                "truncated": truncated,
            })
            adapter.tokenizer.apply_chat_template.assert_any_call(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            adapter.tokenizer.decode.assert_any_call(tokens, skip_special_tokens=True)
        kwargs = adapter.model.generate.call_args.kwargs
        self.assertEqual({k: v for k, v in kwargs.items() if k not in adapter.tokenizer.inputs}, {
            "max_new_tokens": 4, "do_sample": False, "num_beams": 1, "pad_token_id": 0,
        })

    def test_first_eos_included_and_all_later_tokens_removed(self):
        for eos, tokens, expected in (
            (99, [99, 0, 0, 0], [99]),
            (99, [31, 99, 99, 0], [31, 99]),
            ([98, 99], [31, 98, 0, 0], [31, 98]),
            ([98, 99], [31, 99, 0, 0], [31, 99]),
            (99, [31, 32, 33, 99], [31, 32, 33, 99]),
        ):
            with self.subTest(eos=eos, tokens=tokens):
                result = batch.batched_predict(adapter_for([tokens], eos), [query()])[0]
                self.assertEqual(result["raw"]["output_ids"], expected)
                self.assertFalse(result["truncated"])

    def test_eos_equal_pad_keeps_first_eos_even_at_position_zero(self):
        for tokens, expected in (
            ([99, 99, 99, 99], [99]),
            ([31, 99, 99, 99], [31, 99]),
        ):
            with self.subTest(tokens=tokens):
                adapter = adapter_for([tokens])
                adapter.tokenizer.pad_token_id = 99
                result = batch.batched_predict(adapter, [query()])[0]
                self.assertEqual(result["raw"]["output_ids"], expected)
                self.assertFalse(result["truncated"])

    def test_missing_pad_token_falls_back_to_eos(self):
        adapter = adapter_for([[31, 99, 99, 99], [41, 42, 43, 44]])
        adapter.tokenizer.pad_token_id = None
        result = batch.batched_predict(adapter, [query(0), query(1)])
        self.assertEqual(adapter.tokenizer.pad_token, adapter.tokenizer.eos_token)
        self.assertEqual(adapter.tokenizer.inputs["input_ids"].rows[0], [99, 99, 11])
        self.assertEqual(adapter.model.generate.call_args.kwargs["pad_token_id"], 99)
        self.assertEqual(result[0]["raw"]["output_ids"], [31, 99])

    def test_truncation_with_and_without_eos_configuration(self):
        for eos in (99, [98, 99], None, []):
            for tokens, expected in (([31, 32, 33, 34], True), ([31], False), ([], False)):
                with self.subTest(eos=eos, tokens=tokens):
                    result = batch.batched_predict(adapter_for([tokens], eos), [query()])[0]
                    self.assertEqual(result["raw"]["output_ids"], tokens)
                    self.assertEqual(result["truncated"], expected)

    def test_mixed_budgets_rejected_before_tokenization_or_generation(self):
        adapter = adapter_for([])
        with self.assertRaisesRegex(ValueError, "equal"):
            batch.batched_predict(adapter, [query(0, 1), query(1, 4)])
        adapter.tokenizer.apply_chat_template.assert_not_called()
        adapter.model.generate.assert_not_called()
        self.assertEqual(adapter.tokenizer.calls, [])

    def test_question_only_profile_is_not_overridden(self):
        adapter = adapter_for([[99]], profile="question_only")
        result = batch.batched_predict(adapter, [query()])[0]
        self.assertEqual(result["raw"]["prompt"], query()["prompt"])
        self.assertEqual(result["raw"]["chat_prompt"], "CHAT:" + query()["prompt"])

    def test_dependency_errors_propagate(self):
        for stage in ("template", "tokenize", "generate", "decode"):
            with self.subTest(stage=stage):
                adapter = adapter_for([[31, 99, 0, 0]])
                error = RuntimeError(f"{stage} failed")
                if stage == "template":
                    adapter.tokenizer.apply_chat_template.side_effect = error
                elif stage == "tokenize":
                    adapter.tokenizer = MagicMock(side_effect=error)
                elif stage == "generate":
                    adapter.model.generate.side_effect = error
                else:
                    adapter.tokenizer.decode.side_effect = error
                with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                    batch.batched_predict(adapter, [query()])

    def test_wrong_number_of_model_outputs_aborts(self):
        adapter = adapter_for([])
        adapter.model.generate.side_effect = None
        adapter.model.generate.return_value = FakeTensor([[11, 99]])
        with self.assertRaises(ValueError):
            batch.batched_predict(adapter, [query(0), query(1)])


class LoadStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.manifest = {"frozen": True}
        (self.out / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.queries = [query(0), query(1)]
        batch.write_json(self.out / "run.json", {
            "status": "running",
            "manifest_sha256": batch.sha(self.out / "manifest.json"),
            "backend": {"backend": "qwen"},
        })
        self.write_rows("queries.jsonl", self.queries)
        self.write_rows("responses.jsonl", [{"id": "q0", "text": "да"}])
        self.verify = patch.object(batch, "verify", return_value=(self.manifest, []))
        self.verifier = self.verify.start()
        self.addCleanup(self.verify.stop)

    def write_rows(self, name, rows):
        (self.out / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )

    def test_loads_existing_responses_without_mutating_artifacts(self):
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        manifest, queries, responses, metadata = batch.load_state(self.out)
        self.verifier.assert_called_once_with(self.out)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(queries, self.queries)
        self.assertEqual(responses, {"q0": {"id": "q0", "text": "да"}})
        self.assertEqual(metadata["backend"]["backend"], "qwen")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_duplicate_and_unknown_response_ids_rejected(self):
        for rows, message in (
            ([{"id": "q0"}, {"id": "q0"}], "Duplicate"),
            ([{"id": "unknown"}], "Unexpected"),
        ):
            with self.subTest(rows=rows):
                self.write_rows("responses.jsonl", rows)
                with self.assertRaisesRegex(ValueError, message):
                    batch.load_state(self.out)

    def test_manifest_mismatch_and_non_qwen_backend_rejected(self):
        for metadata, message in (
            ({"status": "running", "manifest_sha256": "wrong",
              "backend": {"backend": "qwen"}}, "manifest"),
            ({"status": "running", "manifest_sha256": batch.sha(self.out / "manifest.json"),
              "backend": {"backend": "mock"}}, "Qwen"),
        ):
            with self.subTest(metadata=metadata):
                batch.write_json(self.out / "run.json", metadata)
                with self.assertRaisesRegex(ValueError, message):
                    batch.load_state(self.out)

    def test_frozen_verification_failure_propagates(self):
        self.verifier.side_effect = ValueError("Frozen dataset modified")
        with self.assertRaisesRegex(ValueError, "Frozen dataset modified"):
            batch.load_state(self.out)

    def test_complete_and_partial_accept_matching_responses_hash(self):
        for status in ("complete", "partial"):
            with self.subTest(status=status):
                metadata = {
                    "status": status,
                    "manifest_sha256": batch.sha(self.out / "manifest.json"),
                    "responses_sha256": batch.sha(self.out / "responses.jsonl"),
                    "backend": {"backend": "qwen"},
                }
                batch.write_json(self.out / "run.json", metadata)
                before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                self.assertEqual(batch.load_state(self.out)[3], metadata)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_complete_and_partial_reject_missing_or_wrong_responses_hash(self):
        for status in ("complete", "partial"):
            for saved_hash in (None, "wrong"):
                with self.subTest(status=status, saved_hash=saved_hash):
                    metadata = {
                        "status": status,
                        "manifest_sha256": batch.sha(self.out / "manifest.json"),
                        "backend": {"backend": "qwen"},
                    }
                    if saved_hash is not None:
                        metadata["responses_sha256"] = saved_hash
                    batch.write_json(self.out / "run.json", metadata)
                    before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                    with self.assertRaisesRegex(ValueError, "responses hash mismatch"):
                        batch.load_state(self.out)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_valid_json_without_final_newline_is_rejected_without_mutation(self):
        responses = self.out / "responses.jsonl"
        responses.write_bytes(responses.read_bytes().rstrip(b"\n"))
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(ValueError, "end with a newline"):
            batch.load_state(self.out)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_malformed_tail_aborts_main_before_model_load_or_data_mutations(self):
        responses = self.out / "responses.jsonl"
        responses.write_bytes(responses.read_bytes() + b'{"id": "q1", "text":')
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with patch.dict(sys.modules, {"torch": MagicMock()}), patch.object(
            sys, "argv", ["qwen_batch.py", "--out", str(self.out)],
        ), patch.object(batch, "QwenAdapter") as factory, patch.object(
            batch, "save_metadata",
        ) as save:
            with self.assertRaises(json.JSONDecodeError):
                batch.main()
            factory.assert_not_called()
            save.assert_not_called()
        # main may create batch.lock, but must not repair/truncate any saved data.
        self.assertEqual(before, {name: (self.out / name).read_bytes() for name in before})


class SaveMetadataTests(unittest.TestCase):
    def test_success_fsyncs_temp_before_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_bytes(b'{"status": "running"}\n')
            original = path.read_bytes()
            metadata = {"status": "complete", "note": "Готово"}
            real_replace = batch.os.replace
            events = []

            def replace(source, destination):
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(destination, path)
                self.assertEqual(json.loads(source.read_text(encoding="utf-8")), metadata)
                self.assertEqual(events, ["fsync"])
                events.append("replace")
                real_replace(source, destination)

            with patch.object(
                batch.os, "fsync", side_effect=lambda _fd: events.append("fsync"),
            ), patch.object(batch.os, "replace", side_effect=replace):
                batch.save_metadata(path, metadata)
            self.assertEqual(events, ["fsync", "replace"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metadata)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_write_sync_or_replace_failure_preserves_previous_metadata(self):
        for stage in ("serialization", "fsync", "replace"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "run.json"
                original = b'{"status": "running", "note": "original"}\n'
                path.write_bytes(original)
                metadata = {"status": "complete"}
                if stage == "serialization":
                    metadata["invalid"] = object()
                    failure = TypeError
                    failure_patch = patch.object(batch.os, "replace", wraps=batch.os.replace)
                else:
                    failure = OSError
                    failure_patch = patch.object(
                        batch.os, stage, side_effect=OSError(f"{stage} failed"),
                    )
                with failure_patch, self.assertRaises(failure):
                    batch.save_metadata(path, metadata)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(json.loads(path.read_text())["status"], "running")


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.previous = {
            "backend": "qwen", "model": "fake/model",
            "input_profile": "contract_v1_fen", "dtype": "float32",
        }
        self.metadata = {
            "status": "running", "backend": self.previous, "manifest_sha256": "frozen",
        }
        self.queries = [query(0), query(1), query(2, 2), query(3)]
        self.existing = {"q0": {"id": "q0", "text": "unchanged"}}
        self.original = json.dumps(self.existing["q0"]) + "\n"
        (self.out / "responses.jsonl").write_text(self.original, encoding="utf-8")
        batch.write_json(self.out / "run.json", self.metadata)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(sys.modules, {"torch": MagicMock()}))
        self.stack.enter_context(patch.object(
            sys, "argv", ["qwen_batch.py", "--out", str(self.out), "--batch-size", "2"],
        ))
        self.stack.enter_context(patch.object(
            batch, "load_state", return_value=(
                {}, self.queries, self.existing, self.metadata,
            ),
        ))
        self.factory = self.stack.enter_context(patch.object(batch, "QwenAdapter"))
        self.factory.return_value.metadata = dict(self.previous)
        self.predict = self.stack.enter_context(patch.object(batch, "batched_predict"))
        self.predict.side_effect = lambda _adapter, queries: [
            {"text": q["id"], "raw": {"output_ids": [99]}, "truncated": False}
            for q in queries
        ]

    def test_resume_groups_budgets_and_never_repeats_completed_queries(self):
        batch.main()
        groups = [call.args[1] for call in self.predict.call_args_list]
        self.assertEqual([[q["id"] for q in group] for group in groups], [["q1", "q3"], ["q2"]])
        for group in groups:
            self.assertLessEqual(len(group), 2)
            self.assertEqual(len({q["max_new_tokens"] for q in group}), 1)
        self.factory.assert_called_once_with("fake/model", "contract_v1_fen")
        responses = (self.out / "responses.jsonl").read_text(encoding="utf-8")
        self.assertTrue(responses.startswith(self.original))
        rows = batch.read_jsonl(self.out / "responses.jsonl")
        self.assertEqual([row["id"] for row in rows], ["q0", "q1", "q3", "q2"])
        metadata = json.loads((self.out / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["completed"], 4)
        self.assertEqual(metadata["expected"], 4)
        self.assertEqual(metadata["backend"], self.previous)
        self.assertEqual(metadata["responses_sha256"], batch.sha(self.out / "responses.jsonl"))
        self.assertEqual(metadata["execution_segments"][0]["previous_responses"], 1)

    def test_incompatible_adapter_aborts_before_generation_or_artifact_changes(self):
        self.factory.return_value.metadata = {**self.previous, "dtype": "float16"}
        before = (self.out / "run.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            batch.main()
        self.predict.assert_not_called()
        self.assertEqual((self.out / "run.json").read_bytes(), before)
        self.assertEqual((self.out / "responses.jsonl").read_text(encoding="utf-8"), self.original)

    def test_generation_failure_aborts_without_error_answer_or_complete_status(self):
        self.predict.side_effect = RuntimeError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            batch.main()
        self.assertEqual((self.out / "responses.jsonl").read_text(encoding="utf-8"), self.original)
        metadata = json.loads((self.out / "run.json").read_text(encoding="utf-8"))
        self.assertNotEqual(metadata["status"], "complete")
        self.assertNotIn("completed_utc", metadata["execution_segments"][0])

    def test_nonpositive_batch_size_or_threads_rejected(self):
        for option in ("--batch-size", "--threads"):
            for value in ("0", "-1"):
                with self.subTest(option=option, value=value), patch.object(
                    sys, "argv", ["qwen_batch.py", "--out", str(self.out), option, value],
                ), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        batch.main()
                    self.assertEqual(error.exception.code, 2)
        self.factory.assert_not_called()
        self.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
