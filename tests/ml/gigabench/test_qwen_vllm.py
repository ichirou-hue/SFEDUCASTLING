"""Offline vLLM runner contracts; fake engine/modules, no GPU or model imports."""

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
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from gigabench.adapters import input_builder  # noqa: E402

SPEC = importlib.util.spec_from_file_location("qwen_vllm_under_test", ROOT / "tools/qwen_vllm.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def query(index=0, budget=2048):
    return {
        "id": f"q{index}", "fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
        "prompt": f"Вопрос {index}?", "max_new_tokens": budget,
    }


def tokenizer():
    return SimpleNamespace(
        apply_chat_template=MagicMock(side_effect=lambda messages, **_kw: "CHAT:" + messages[0]["content"]),
        encode=MagicMock(return_value=[11, 12, 13]),
        decode=MagicMock(return_value="  ответ  "),
    )


def request(key="q0", tokens=(31, 99), reason="stop", finished=True, ids=None):
    return SimpleNamespace(
        request_id=key, finished=finished, prompt_token_ids=[11, 12, 13] if ids is None else ids,
        outputs=[SimpleNamespace(token_ids=list(tokens), finish_reason=reason, stop_reason=99)],
    )


class FakeEngine:
    def __init__(self, steps=()):
        self.steps = list(steps)
        self.submitted = []

    def add_request(self, *args):
        self.submitted.append(args)

    def has_unfinished_requests(self):
        return bool(self.steps)

    def step(self):
        return self.steps.pop(0)


class PromptAndRecordTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer()
        self.prompts = runner.prepare_prompts([query()], self.tokenizer, input_builder("contract_v1_fen")[0])

    def test_fen_question_chat_without_thinking_encoded_once_without_extra_tokens(self):
        prompt = f"Позиция FEN: {query()['fen']}\n{query()['prompt']}"
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        self.tokenizer.encode.assert_called_once_with("CHAT:" + prompt, add_special_tokens=False)
        self.assertEqual(self.prompts["q0"], {
            "query": query(), "prompt": prompt, "chat_prompt": "CHAT:" + prompt,
            "input_ids": [11, 12, 13],
        })

    def test_empty_prompts_do_not_call_tokenizer(self):
        tok = tokenizer()
        self.assertEqual(runner.prepare_prompts([], tok, None), {})
        tok.encode.assert_not_called()

    def test_template_tokenization_and_decoding_errors_propagate(self):
        for stage in ("apply_chat_template", "encode", "decode"):
            with self.subTest(stage=stage):
                tok = tokenizer()
                getattr(tok, stage).side_effect = RuntimeError(stage)
                with self.assertRaisesRegex(RuntimeError, stage):
                    if stage == "decode":
                        runner.response_record(request(), self.prompts["q0"], tok, 0)
                    else:
                        runner.prepare_prompts([query()], tok, input_builder("contract_v1_fen")[0])

    def test_eos_and_length_preserve_ids_and_metadata(self):
        for reason, tokens in (("stop", [31, 99]), ("length", [31] * 2048)):
            with self.subTest(reason=reason):
                result = runner.response_record(request(tokens=tokens, reason=reason),
                                                self.prompts["q0"], self.tokenizer, 1.25)
                self.assertEqual(result, {
                    "id": "q0", "text": "ответ",
                    "raw": {"prompt": self.prompts["q0"]["prompt"],
                            "chat_prompt": self.prompts["q0"]["chat_prompt"],
                            "input_ids": [11, 12, 13], "output_ids": tokens},
                    "truncated": reason == "length", "finish_reason": reason, "stop_reason": 99,
                    "output_tokens": len(tokens), "run_elapsed_seconds": 1.25, "elapsed_seconds": 1.25,
                })
                self.tokenizer.decode.assert_called_with(tokens, skip_special_tokens=True)

    def test_exactly_one_completion_required(self):
        for outputs in ([], request().outputs * 2):
            with self.subTest(count=len(outputs)):
                req = request()
                req.outputs = outputs
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    runner.response_record(req, self.prompts["q0"], self.tokenizer, 0)

    def test_invalid_finish_reason_rejected(self):
        for reason in (None, "abort", "error"):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "finish reason"):
                runner.response_record(request(reason=reason), self.prompts["q0"], self.tokenizer, 0)

    def test_frozen_2048_token_budget_rejects_2049(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            runner.response_record(request(tokens=[31] * 2049), self.prompts["q0"], self.tokenizer, 0)

    def test_engine_must_preserve_exact_input_ids(self):
        for ids in ([11, 12], [0, 11, 12, 13], [13, 12, 11], []):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "input IDs"):
                runner.response_record(request(ids=ids), self.prompts["q0"], self.tokenizer, 0)
        req = request()
        del req.prompt_token_ids
        with self.assertRaisesRegex(ValueError, "input IDs"):
            runner.response_record(req, self.prompts["q0"], self.tokenizer, 0)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer()
        self.prompts = runner.prepare_prompts(
            [query(0), query(1, 2)], self.tokenizer, input_builder("contract_v1_fen")[0],
        )
        self.params = MagicMock(side_effect=lambda budget: {"max_tokens": budget})

    def run_engine(self, steps, prompts=None):
        engine = FakeEngine(steps)
        result = list(runner.stream_responses(
            engine, self.prompts if prompts is None else prompts, self.tokenizer, self.params,
        ))
        return engine, result

    def test_continuous_batch_out_of_order_with_unfinished_updates(self):
        engine, results = self.run_engine([
            [request(finished=False), request("q1")], [], [request()],
        ])
        self.assertEqual([r["id"] for r in results], ["q1", "q0"])
        self.assertEqual(engine.submitted, [
            ("q0", {"prompt_token_ids": [11, 12, 13]}, {"max_tokens": 2048}),
            ("q1", {"prompt_token_ids": [11, 12, 13]}, {"max_tokens": 2}),
        ])
        self.assertTrue(all(r["elapsed_seconds"] >= 0 for r in results))

    def test_unfinished_updates_are_not_decoded_or_validated_as_completions(self):
        partial = SimpleNamespace(finished=False)
        _, results = self.run_engine([[partial], [request(), request("q1")]])
        self.assertEqual(len(results), 2)
        self.assertEqual(self.tokenizer.decode.call_count, 2)

    def test_unknown_and_duplicate_completed_ids_rejected(self):
        for steps in ([[request("unknown")]], [[request(), request()]]):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
                self.run_engine(steps)

    def test_engine_early_stop_rejects_lost_queries(self):
        for steps in ([], [[request(finished=False)]], [[request()]]):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "before all"):
                self.run_engine(steps)

    def test_empty_input_never_submits(self):
        engine, results = self.run_engine([], prompts={})
        self.assertEqual(engine.submitted, [])
        self.assertEqual(results, [])

    def test_dependency_failure_propagates(self):
        engine = FakeEngine()
        engine.step = MagicMock(side_effect=RuntimeError("engine failed"))
        engine.has_unfinished_requests = lambda: True
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            list(runner.stream_responses(engine, self.prompts, self.tokenizer, self.params))


class ExistingTests(unittest.TestCase):
    def test_absent_empty_and_valid_response_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(runner.read_existing(out, {"q0"}), {})
            path = out / "responses.jsonl"
            path.touch()
            self.assertEqual(runner.read_existing(out, {"q0"}), {})
            path.write_text('{"id":"q0","text":"ответ"}\n', encoding="utf-8")
            self.assertEqual(runner.read_existing(out, {"q0"}), {"q0": {"id": "q0", "text": "ответ"}})

    def test_invalid_files_are_rejected_without_repairing(self):
        for data, error in (
            (b'{"id":"q0"}', ValueError),
            (b'{"id":"q0"}\n{"id":', json.JSONDecodeError),
            (b'{"id":"q0"}\n{"id":"q0"}\n', ValueError),
            (b'{"id":"unknown"}\n', ValueError),
        ):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "responses.jsonl"
                path.write_bytes(data)
                with self.assertRaises(error):
                    runner.read_existing(Path(tmp), {"q0"})
                self.assertEqual(path.read_bytes(), data)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.snapshot = self.out / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "generation_config.json").write_text('{"eos_token_id":[98,99]}')
        (self.out / "manifest.json").write_text('{"frozen":true}')
        self.write_queries([query(0), query(1, 2)])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        self.argv = ["qwen_vllm.py", "--out", str(self.out)]
        self.stack.enter_context(patch.object(sys, "argv", self.argv))
        self.verify = self.stack.enter_context(patch.object(runner, "verify"))
        self.stack.enter_context(patch.object(runner, "qwen_snapshot", return_value=(self.snapshot, {"a": "hash"})))
        self.tokenizer = tokenizer()
        self.auto = SimpleNamespace(from_pretrained=MagicMock(return_value=self.tokenizer))
        self.engine = FakeEngine([[request("q1"), request()]])
        self.engine_factory = MagicMock(return_value=self.engine)
        self.sampling = MagicMock(side_effect=lambda **kw: kw)
        self.engine_args = MagicMock(side_effect=lambda **kw: kw)
        self.stack.enter_context(patch.dict(sys.modules, {
            "torch": SimpleNamespace(__version__="fake", version=SimpleNamespace(cuda="fake"),
                                     set_num_threads=MagicMock()),
            "transformers": SimpleNamespace(__version__="fake", AutoTokenizer=self.auto),
            "vllm": SimpleNamespace(__version__="fake", EngineArgs=self.engine_args,
                                    LLMEngine=SimpleNamespace(from_engine_args=self.engine_factory),
                                    SamplingParams=self.sampling),
        }))

    def write_queries(self, queries):
        (self.out / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))

    def metadata(self):
        return json.loads((self.out / "run.json").read_text())

    def assert_guard(self, message):
        before = {name: (self.out / name).read_bytes() for name in
                  ("run.json", "responses.jsonl") if (self.out / name).exists()}
        with self.assertRaisesRegex(ValueError, message):
            runner.main()
        self.engine_factory.assert_not_called()
        for name, contents in before.items():
            self.assertEqual((self.out / name).read_bytes(), contents)

    def test_main_greedy_eos_engine_and_run_hash_contract(self):
        runner.main()
        self.verify.assert_called_once_with(self.out)
        self.auto.from_pretrained.assert_called_once_with(self.snapshot, local_files_only=True)
        self.assertEqual([row["id"] for row in runner.read_jsonl(self.out / "responses.jsonl")], ["q1", "q0"])
        for budget, call in zip((2048, 2), self.sampling.call_args_list, strict=True):
            self.assertEqual(call.kwargs, {
                "temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": budget,
                "seed": 20261101, "repetition_penalty": 1, "presence_penalty": 0,
                "frequency_penalty": 0, "stop_token_ids": [98, 99],
                "ignore_eos": False, "detokenize": False,
            })
        metadata = self.metadata()
        self.assertEqual((metadata["status"], metadata["completed"], metadata["expected"]), ("complete", 2, 2))
        self.assertEqual(metadata["responses_sha256"], runner.sha(self.out / "responses.jsonl"))
        self.assertEqual(metadata["manifest_sha256"], runner.sha(self.out / "manifest.json"))
        self.assertEqual(metadata["backend"]["runner_sha256"], runner.sha(runner.__file__))
        self.assertFalse(metadata["backend"]["thinking"])
        self.assertFalse(metadata["backend"]["do_sample"])
        self.assertTrue(metadata["backend"]["engine"]["skip_tokenizer_init"])
        self.assertEqual(metadata["execution_segments"][0]["submitted"], 2)

    def test_partial_resume_skips_existing_and_retains_segment_history(self):
        self.argv.extend(["--limit", "1"])
        self.engine.steps = [[request()]]
        runner.main()
        original = (self.out / "responses.jsonl").read_bytes()
        self.assertEqual(self.metadata()["status"], "partial")
        self.engine.steps = [[request("q1")]]
        self.engine.submitted.clear()
        runner.main()
        self.assertEqual([args[0] for args in self.engine.submitted], ["q1"])
        self.assertTrue((self.out / "responses.jsonl").read_bytes().startswith(original))
        metadata = self.metadata()
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(len(metadata["execution_segments"]), 2)
        self.assertEqual(metadata["execution_segments"][1]["previous_responses"], 1)

    def test_cuda_graphs_opt_in_is_recorded(self):
        self.argv.append("--cuda-graphs")
        runner.main()
        self.assertFalse(self.engine_args.call_args.kwargs["enforce_eager"])
        self.assertFalse(self.metadata()["backend"]["engine"]["enforce_eager"])

    def test_cuda_graph_policy_change_cannot_resume_existing_answers(self):
        runner.main()
        self.assertTrue(self.metadata()["backend"]["engine"]["enforce_eager"])
        self.argv.append("--cuda-graphs")
        self.engine_factory.reset_mock()
        self.assert_guard("Cannot resume")

    def test_completed_resume_does_not_create_engine(self):
        runner.main()
        self.engine_factory.reset_mock()
        before = (self.out / "responses.jsonl").read_bytes()
        runner.main()
        self.engine_factory.assert_not_called()
        self.assertEqual((self.out / "responses.jsonl").read_bytes(), before)

    def test_scalar_eos_config_becomes_explicit_stop_token_list(self):
        (self.snapshot / "generation_config.json").write_text('{"eos_token_id":99}')
        runner.main()
        for call in self.sampling.call_args_list:
            self.assertEqual(call.kwargs["stop_token_ids"], [99])

    def test_hf_metadata_rejected_even_without_responses(self):
        (self.out / "run.json").write_text(json.dumps({"backend": {"backend": "qwen"}}))
        self.assert_guard("separate run")
        self.auto.from_pretrained.assert_not_called()

    def test_responses_without_metadata_rejected(self):
        (self.out / "responses.jsonl").write_text('{"id":"q0"}\n')
        self.assert_guard("without run metadata")

    def test_saved_response_and_manifest_hash_mismatch_rejected(self):
        runner.main()
        saved = self.metadata()
        for field in ("responses_sha256", "manifest_sha256"):
            with self.subTest(field=field):
                (self.out / "run.json").write_text(json.dumps({**saved, field: "wrong"}))
                self.engine_factory.reset_mock()
                self.assert_guard("hash mismatch" if field == "responses_sha256" else "Manifest differs")

    def test_changed_runner_or_engine_identity_rejects_resume(self):
        runner.main()
        saved = self.metadata()
        for field, value in (("runner_sha256", "changed"), ("model_files_sha256", {"a": "changed"})):
            with self.subTest(field=field):
                altered = {**saved, "backend": {**saved["backend"], field: value}}
                (self.out / "run.json").write_text(json.dumps(altered))
                self.engine_factory.reset_mock()
                self.assert_guard("Cannot resume")

    def test_unknown_or_duplicate_selection_and_queries_rejected(self):
        path = self.out / "ids.json"
        self.argv.extend(["--ids", str(path)])
        for ids in (["unknown"], ["q0", "q0"]):
            with self.subTest(ids=ids):
                path.write_text(json.dumps(ids))
                self.assert_guard("duplicate or unknown")
        self.argv[-2:] = []
        self.write_queries([query(), query()])
        self.assert_guard("Duplicate")

    def test_audit_selection_order_and_hash(self):
        path = self.out / "audit20.json"
        path.write_text('{"cases":[{"id":"q1"}]}')
        self.argv.extend(["--ids", str(path)])
        self.engine.steps = [[request("q1")]]
        runner.main()
        self.assertEqual([args[0] for args in self.engine.submitted], ["q1"])
        self.assertEqual(self.metadata()["status"], "partial")
        self.assertEqual(self.metadata()["execution_segments"][0]["selection_sha256"], runner.sha(path))

    def test_context_overflow_rejected_before_model_start_or_metadata_write(self):
        self.argv.extend(["--max-model-len", "2050"])
        self.assert_guard("context length")
        self.assertFalse((self.out / "run.json").exists())

    def test_invalid_cli_limits_rejected_before_verification(self):
        for option, values in (
            ("--max-num-seqs", ["0", "-1"]), ("--max-model-len", ["0", "-1"]),
            ("--limit", ["0", "-1"]), ("--gpu-memory-utilization", ["0", "1", "-.1", "nan"]),
        ):
            for value in values:
                with self.subTest(option=option, value=value), patch.object(
                    sys, "argv", self.argv + [option, value],
                ), self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
        self.verify.assert_not_called()

    def test_engine_failure_retains_running_status_without_fake_answers(self):
        self.engine.steps = [[request()]]
        with self.assertRaisesRegex(ValueError, "before all"):
            runner.main()
        self.assertEqual([r["id"] for r in runner.read_jsonl(self.out / "responses.jsonl")], ["q0"])
        self.assertEqual(self.metadata()["status"], "running")
        self.assertNotIn("completed_utc", self.metadata()["execution_segments"][0])
        self.engine.submitted.clear()
        self.engine.steps = [[request("q1")]]
        runner.main()
        self.assertEqual([args[0] for args in self.engine.submitted], ["q1"])
        self.assertEqual(self.metadata()["status"], "complete")

    def test_lock_contention_prevents_verification_and_generation(self):
        with (self.out / "vllm.lock").open("a") as lock:
            runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                runner.main()
        self.verify.assert_not_called()
        self.engine_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
