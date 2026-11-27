import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3] / "ml/benchmark"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "qwen_vllm_numeric_trio",
    ROOT / "tools/qwen_vllm_numeric_trio.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FinalProtocolTests(unittest.TestCase):
    def test_matches_training_format_for_static_cells_only(self):
        query = {"fen": "fen", "prompt": "Вопрос?"}
        formatter = Mock(return_value="Числовая доска\n\nВопрос?")
        for task in ("piece_occupied", "piece_empty", "M1_occupied", "M1_empty"):
            self.assertEqual(
                runner.final_input({"task": task, "score_type": "piece"}, query, formatter),
                "Позиция FEN: fen\n\nЧисловая доска\n\nВопрос?",
            )
        self.assertEqual(formatter.call_count, 4)
        self.assertEqual(
            runner.final_input({"task": "state_tracking", "score_type": "piece"}, query, formatter),
            "Позиция FEN: fen\nВопрос?",
        )
        commentary = {**query, "prompt": "Ход белых.\nФакты движка:\nФакт."}
        self.assertEqual(
            runner.final_input({"task": "move_commentary", "score_type": "explanation_auto"},
                               commentary, formatter),
            "Позиция FEN: fen\nХод белых.\nФакты движка:\nФакт.",
        )
        self.assertEqual(formatter.call_count, 4)

    def test_four_model_specifies_three_distinct_adapters(self):
        args = SimpleNamespace(commentary_adapter="c", board_adapter="b", mixed_adapter="m")
        self.assertEqual(
            runner.model_specs(args),
            (("base", None, None, None), ("lora-commentary", "c", 1, "commentary"),
             ("lora-cell-questions", "b", 2, "cell-questions"), ("lora-mixed", "m", 3, "mixed")),
        )

    def test_old_three_model_labels_are_preserved(self):
        args = SimpleNamespace(commentary_adapter="c", board_adapter="b", mixed_adapter=None)
        self.assertEqual(tuple(model[0] for model in runner.model_specs(args)), runner.LABELS)

    def test_training_identity_rejects_another_commentary_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {"base_model": "Qwen", "revision": "rev", "dataset": {
                "name": "mixed", "benchmark_cases_sha256": "archive",
                "commentary_benchmark_cases_sha256": "wrong-suite",
            }}
            (root / "training.json").write_text(json.dumps(metadata), encoding="utf-8")
            benchmark = {"archive_cases_sha256": "archive", "commentary_auto": {"cases_sha256": "suite"}}
            with self.assertRaisesRegex(ValueError, "different benchmark"):
                runner.training_identity(root / "adapter", "mixed", "Qwen", "rev", benchmark)

    def test_mixed_budget_batch_passes_each_frozen_limit(self):
        queries = [{"id": "cell", "max_new_tokens": 2048}, {"id": "comment", "max_new_tokens": 512}]
        inputs = [{"prompt_token_ids": [10]}, {"prompt_token_ids": [11]}]
        outputs = [
            SimpleNamespace(
                prompt_token_ids=item["prompt_token_ids"],
                outputs=[SimpleNamespace(finish_reason="stop", token_ids=[12])],
            )
            for item in inputs
        ]
        llm = Mock()
        llm.generate.return_value = outputs
        tokenizer = Mock()
        tokenizer.decode.return_value = "answer"
        params = {2048: object(), 512: object()}
        modules = {"vllm.lora.request": SimpleNamespace(LoRARequest=Mock())}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            runner.generate_model("base", None, None, None, tokenizer, inputs, ["a", "b"],
                                  queries, Path(tmp), llm, params, "manifest-sha", {})
        llm.generate.assert_called_once_with(inputs, [params[2048], params[512]],
                                            lora_request=None, use_tqdm=False)


class ResponseRecordTests(unittest.TestCase):
    def setUp(self):
        self.query = {"id": "q0", "max_new_tokens": 8}
        self.input_ids = [11, 12]
        self.tokenizer = Mock()
        self.tokenizer.decode.return_value = " answer "

    def request(self, reason="stop", tokens=(31, 99)):
        return SimpleNamespace(
            prompt_token_ids=self.input_ids,
            outputs=[SimpleNamespace(finish_reason=reason, token_ids=list(tokens))],
        )

    def test_reads_finish_reason_and_tokens_from_completion(self):
        result = runner.response_record(
            self.query, self.request("length", (31, 99)), self.input_ids,
            "input-hash", self.tokenizer,
        )
        self.assertEqual(result, {
            "id": "q0", "text": "answer", "truncated": True,
            "finish_reason": "length", "output_tokens": 2,
            "input_sha256": "input-hash", "raw": {"output_ids": [31, 99]},
        })

    def test_rejects_invalid_finish_reason(self):
        with self.assertRaisesRegex(ValueError, "finish reason"):
            runner.response_record(
                self.query, self.request("abort"), self.input_ids,
                "input-hash", self.tokenizer,
            )

    def test_rejects_output_over_budget(self):
        with self.assertRaisesRegex(ValueError, "token budget"):
            runner.response_record(
                self.query, self.request(tokens=range(9)), self.input_ids,
                "input-hash", self.tokenizer,
            )


if __name__ == "__main__":
    unittest.main()
