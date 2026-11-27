#!/usr/bin/env python3
"""Run base Qwen and matched LoRAs on identical GigaBench inputs."""
# ruff: noqa: T201
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gigabench.adapters import input_builder, qwen_snapshot  # noqa: E402
from gigabench.engine import EngineJudge
from gigabench.io import read_jsonl, sha, unique, write_json, write_jsonl
from gigabench.pipeline import now, render_report, score_rows, verify
from gigabench.scoring import summarize

RUN_FILES = ("manifest.json", "cases.jsonl", "queries.jsonl", "coach_rubrics_pending.jsonl")
LABELS = ("base", "lora-handoff-desc-v1", "lora-pilot-cells-v1")
FINAL_PROFILE = "fen_numeric_static_cells_v1"
STATIC_CELL_TASKS = {"piece_occupied", "piece_empty", "M1_occupied", "M1_empty"}


def load_numeric_board(path):
    spec = importlib.util.spec_from_file_location("gigabench_numeric_board", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load numeric-board formatter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_input(case, format_cell_query, format_query):
    query = {"fen": case["fen"], "prompt": case["prompt"]}
    if case["score_type"] == "piece" and case["task"] != "state_tracking":
        return format_cell_query(query)
    return format_query(query)


def is_static_cell(case):
    return case["score_type"] == "piece" and case["task"] in STATIC_CELL_TASKS


def final_input(case, query, format_cell_query):
    build_fen, _ = input_builder("contract_v1_fen")
    if is_static_cell(case):
        return build_fen({**query, "prompt": "\n" + format_cell_query(query)})
    return build_fen(query)


def model_specs(args):
    if args.mixed_adapter is not None:
        return (
            ("base", None, None, None),
            ("lora-commentary", args.commentary_adapter, 1, "commentary"),
            ("lora-cell-questions", args.board_adapter, 2, "cell-questions"),
            ("lora-mixed", args.mixed_adapter, 3, "mixed"),
        )
    return tuple(zip(LABELS, (None, args.commentary_adapter, args.board_adapter),
                     (None, 1, 2), (None, "handoff-desc-v1", "pilot-cells-v1"), strict=True))


def training_identity(adapter, name, model, revision, benchmark):
    training_path = adapter.parent / "training.json"
    metadata = json.loads(training_path.read_text(encoding="utf-8"))
    if (metadata.get("base_model"), metadata.get("revision"), metadata["dataset"].get("name")) != (
        model, revision, name,
    ):
        raise ValueError(f"Adapter training identity does not match {name}: {adapter}")
    dataset = metadata["dataset"]
    if (dataset.get("benchmark_cases_sha256") != benchmark["archive_cases_sha256"]
            or dataset.get("commentary_benchmark_cases_sha256") != benchmark["commentary_auto"]["cases_sha256"]):
        raise ValueError(f"Training dataset leakage checks were against different benchmark cases: {name}")
    return sha(training_path), metadata


def adapter_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {str(file.relative_to(path)): sha(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def copy_prepared_run(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        source_file, destination_file = source / name, destination / name
        if not source_file.is_file():
            if name == "coach_rubrics_pending.jsonl":
                continue
            raise FileNotFoundError(source_file)
        if destination_file.exists():
            if sha(source_file) != sha(destination_file):
                raise ValueError(f"Prepared run differs at {destination_file}")
        else:
            shutil.copy2(source_file, destination_file)


def read_responses(path):
    if not path.exists():
        return {}
    if path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise ValueError(f"Response file is incomplete: {path}")
    return unique(read_jsonl(path))


def response_record(query, request, input_ids, input_hash, tokenizer):
    if len(request.outputs) != 1:
        raise ValueError(f"Expected one completion for {query['id']}")
    if request.prompt_token_ids != input_ids:
        raise ValueError(f"Prompt tokens differ for {query['id']}")
    completion = request.outputs[0]
    if completion.finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason for {query['id']}")
    tokens = list(completion.token_ids)
    if len(tokens) > query["max_new_tokens"]:
        raise ValueError(f"Output exceeds token budget for {query['id']}")
    return {
        "id": query["id"],
        "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        "truncated": completion.finish_reason == "length",
        "finish_reason": completion.finish_reason,
        "output_tokens": len(tokens),
        "input_sha256": input_hash,
        "raw": {"output_ids": tokens},
    }


def generate_model(label, adapter_path, adapter_id, adapter_name, tokenizer, inputs,
                   input_hashes, queries, run_dir, llm, params,
                   source_manifest_sha, backend_common):
    from vllm.lora.request import LoRARequest

    adapter_hashes = adapter_fingerprint(adapter_path) if adapter_path else None
    identity = {
        **backend_common,
        "adapter": str(adapter_path) if adapter_path else None,
        "adapter_files_sha256": adapter_hashes,
    }
    run_path = run_dir / "run.json"
    response_path = run_dir / "responses.jsonl"
    metadata = {
        "backend": identity, "started_utc": now(), "stage": "automatic",
        "manifest_sha256": source_manifest_sha, "rubrics_sha256": None,
        "status": "running", "python": platform.python_version(),
    }
    existing = read_responses(response_path)
    query_ids = {query["id"] for query in queries}
    if set(existing) - query_ids:
        raise ValueError(f"Unexpected response IDs for {label}")
    if response_path.exists() and not run_path.exists():
        raise ValueError(f"Responses exist without run metadata: {response_path}")
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("backend") != identity or previous.get("manifest_sha256") != source_manifest_sha:
            raise ValueError(f"Cannot resume {label}: model or input manifest changed")
        if previous.get("status") == "complete":
            if len(existing) != len(queries) or previous.get("responses_sha256") != sha(response_path):
                raise ValueError(f"Completed response hash mismatch: {response_path}")
            print(f"TRIO | {label}: already complete ({len(existing)}/{len(queries)})", flush=True)
            return
        metadata["started_utc"] = previous.get("started_utc", metadata["started_utc"])
        metadata["execution_segments"] = previous.get("execution_segments", [])
    if len(existing) == len(queries):
        metadata.update(status="complete", completed=len(existing), expected=len(queries),
                        completed_utc=now(), responses_sha256=sha(response_path))
        write_json(run_path, metadata)
        return

    remaining = [(index, query) for index, query in enumerate(queries)
                 if query["id"] not in existing]
    request = (LoRARequest(adapter_name, adapter_id, str(adapter_path.resolve()))
               if adapter_path else None)
    segment = {"started_utc": now(), "previous_responses": len(existing),
               "submitted": len(remaining)}
    metadata.setdefault("execution_segments", []).append(segment)
    write_json(run_path, metadata)
    print(f"TRIO | {label}: starting {len(remaining)} queries", flush=True)
    started = time.monotonic()
    with (response_path.open("a", encoding="utf-8") if response_path.exists()
          else response_path.open("x", encoding="utf-8")) as stream:
        for offset in range(0, len(remaining), 128):
            batch = remaining[offset:offset + 128]
            outputs = llm.generate(
                [inputs[index] for index, _ in batch],
                [params[query["max_new_tokens"]] for _, query in batch],
                lora_request=request, use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise ValueError("vLLM returned an unexpected batch size")
            for (index, query), output in zip(batch, outputs, strict=True):
                row = response_record(
                    query, output, inputs[index]["prompt_token_ids"],
                    input_hashes[index], tokenizer,
                )
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[row["id"]] = row
            stream.flush()
            os.fsync(stream.fileno())
            completed = len(existing)
            elapsed = time.monotonic() - started
            eta = elapsed * (len(queries) - completed) / max(completed - segment["previous_responses"], 1)
            metadata.update(completed=completed, expected=len(queries))
            write_json(run_path, metadata)
            print(f"TRIO | {label}: {completed}/{len(queries)} | "
                  f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining", flush=True)
    segment.update(completed_utc=now(), elapsed_seconds=time.monotonic() - started)
    metadata.update(status="complete", completed=len(existing), expected=len(queries),
                    completed_utc=now(), responses_sha256=sha(response_path))
    write_json(run_path, metadata)
    print(f"TRIO | {label}: generation complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="prepared standard GigaBench run with M4-M7-auto cases")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--board-adapter", type=Path, required=True)
    parser.add_argument("--commentary-adapter", type=Path, required=True)
    parser.add_argument("--mixed-adapter", type=Path,
                        help="enable four-model evaluation with FEN and numeric boards only for static cells")
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--numeric-board", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "pipelines/board_reading/numeric_board.py")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    models = model_specs(args)
    labels = [label for label, *_ in models]
    four_models = args.mixed_adapter is not None
    profile = FINAL_PROFILE if four_models else "numeric_board_v1"
    stockfish_hash = sha(args.stockfish)

    manifest, cases = verify(args.source)
    if four_models and manifest["external_sources"].get("judgement_engine_sha256") != stockfish_hash:
        raise ValueError("Stockfish differs from the frozen benchmark engine")
    if manifest.get("profile") != "standard":
        raise ValueError("This evaluation requires profile=standard")
    if manifest.get("commentary_auto", {}).get("case_count") != 300:
        raise ValueError("The prepared run does not contain the frozen 300-case M4-M7-auto suite")
    if any(case["score_type"] == "explanation" for case in cases):
        raise ValueError("Expert explanation cases must be excluded from this automatic evaluation")
    queries = read_jsonl(args.source / "queries.jsonl")
    unique(queries)
    if {query["id"] for query in queries} != {
        case["id"] for case in cases if case["score_type"] != "explanation"
    }:
        raise ValueError("Prepared queries do not cover the automatic benchmark cases")
    budgets = {query["max_new_tokens"] for query in queries}
    expected_budgets = {512, 2048} if four_models else {2048}
    if budgets != expected_budgets:
        raise ValueError(f"Expected token budgets {sorted(expected_budgets)}, found {sorted(budgets)}")
    if four_models:
        case_lookup = unique(cases)
        if any(q["max_new_tokens"] != (512 if case_lookup[q["id"]]["score_type"] == "explanation_auto" else 2048)
               for q in queries):
            raise ValueError("Query token budgets do not match the frozen cases")
    if args.max_model_len < max(query["max_new_tokens"] for query in queries):
        raise ValueError("--max-model-len is smaller than the output token budget")

    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {args.out}")
    else:
        args.out.mkdir(parents=True)

    board = load_numeric_board(args.numeric_board)
    case_by_id = unique(cases)
    inputs, input_hashes, numeric_inputs = [], [], []

    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    torch.set_num_threads(4)
    snapshot, model_fingerprint = qwen_snapshot(args.model)
    model_name = "Qwen/Qwen3-1.7B" if four_models else args.model
    adapter_settings = {}
    training = {}
    for label, adapter, _, dataset in models:
        if adapter is not None:
            adapter_settings[label] = {"path": str(adapter), "files_sha256": adapter_fingerprint(adapter)}
            if four_models:
                fingerprint, training[dataset] = training_identity(
                    adapter, dataset, model_name, snapshot.name, manifest,
                )
                adapter_settings[label]["training_sha256"] = fingerprint
    if four_models:
        origins = {name: training[name]["dataset_manifest_sha256"] for name in ("commentary", "cell-questions")}
        if training["mixed"]["dataset"]["source_manifests_sha256"] != origins:
            raise ValueError("Mixed adapter was not trained on the two control datasets")
        if any(report["training_policy"] != training["mixed"]["training_policy"] for report in training.values()):
            raise ValueError("The three LoRA training policies must match")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for query in queries:
        case = case_by_id[query["id"]]
        prompt = (final_input(case, query, board.format_cell_query) if four_models
                  else numeric_input(case, board.format_cell_query, board.format_query))
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        token_ids = tokenizer.encode(chat, add_special_tokens=False)
        if len(token_ids) + query["max_new_tokens"] > args.max_model_len:
            raise ValueError(f"Input plus output budget exceeds context for {query['id']}")
        input_hash = hashlib.sha256(chat.encode("utf-8")).hexdigest()
        inputs.append({"prompt_token_ids": token_ids})
        input_hashes.append(input_hash)
        numeric_inputs.append({
            "id": query["id"], "fen": query["fen"], "prompt": prompt,
            "max_new_tokens": query["max_new_tokens"], "input_sha256": input_hash,
        })

    input_hash_path = args.out / "numeric_inputs.jsonl"
    if input_hash_path.exists():
        if read_jsonl(input_hash_path) != numeric_inputs:
            raise ValueError("Numeric-board inputs changed since the previous run")
    else:
        write_jsonl(input_hash_path, numeric_inputs)

    engine_config = {
        "dtype": "bfloat16", "quantization": None, "kv_cache_dtype": "auto",
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": 2048, "enable_chunked_prefill": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True, "enable_lora": True, "max_lora_rank": 16,
        "max_loras": 1, "skip_tokenizer_init": True, "generation_config": "vllm",
        "seed": 20261101, "enable_prefix_caching": True,
    }
    common_settings = {
        "source_manifest_sha256": sha(args.source / "manifest.json"),
        "source_cases_sha256": sha(args.source / "cases.jsonl"),
        "queries_sha256": sha(args.source / "queries.jsonl"),
        "model": model_name, "model_revision": snapshot.name,
        "model_files_sha256": model_fingerprint,
        "input_profile": profile,
        "numeric_board_sha256": sha(args.numeric_board),
        "numeric_inputs_sha256": sha(input_hash_path),
        "runner_sha256": sha(Path(__file__)),
        "generation": {"temperature": 0, "seed": 20261101, "max_new_tokens": sorted(budgets),
                       "thinking": False},
        "engine": engine_config,
        "adapters": adapter_settings,
        "stockfish_sha256": stockfish_hash,
        "software": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "vllm": vllm.__version__},
    }
    settings_path = args.out / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != common_settings:
            raise ValueError("Run settings changed; use a new output directory")
    else:
        write_json(settings_path, common_settings)

    run_dirs = {}
    for label in labels:
        run_dir = args.out / label
        copy_prepared_run(args.source, run_dir)
        run_dirs[label] = run_dir

    eos = json.loads((snapshot / "generation_config.json").read_text(encoding="utf-8"))["eos_token_id"]
    params = {budget: SamplingParams(
        temperature=0, top_p=1, top_k=-1, max_tokens=budget, seed=20261101,
        repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
        stop_token_ids=eos if isinstance(eos, list) else [eos],
        ignore_eos=False, detokenize=False,
    ) for budget in budgets}
    print(f"TRIO | standard benchmark: {len(queries)} matched queries; "
          f"{profile}; starting vLLM", flush=True)
    llm = LLM(model=str(snapshot), **engine_config)
    source_manifest_sha = sha(args.source / "manifest.json")
    backend_common = {
        "backend": "qwen-vllm", "model": model_name, "revision": snapshot.name,
        "model_files_sha256": model_fingerprint, "thinking": False, "do_sample": False,
        "training_performed": False, "input_profile": profile,
        "input_profile_sha256": sha(args.numeric_board),
        "numeric_inputs_sha256": sha(input_hash_path),
        "runner_sha256": sha(Path(__file__)), "engine": engine_config,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "vllm": vllm.__version__,
    }
    for label, adapter_path, adapter_id, adapter_name in models:
        generate_model(
            label, adapter_path, adapter_id, adapter_name, tokenizer, inputs,
            input_hashes, queries, run_dirs[label], llm, params,
            source_manifest_sha, backend_common,
        )
    del llm
    torch.cuda.empty_cache()

    config = manifest["config"]
    summaries = {}
    judge = EngineJudge(args.stockfish, config["engine"], args.out / "shared_engine")
    try:
        for label in labels:
            run_dir = run_dirs[label]
            run_manifest, run_cases = verify(run_dir)
            run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            response_path = run_dir / "responses.jsonl"
            if (run_manifest != manifest or run_cases != cases
                    or run_metadata.get("manifest_sha256") != sha(run_dir / "manifest.json")
                    or run_metadata.get("status") != "complete"
                    or run_metadata.get("responses_sha256") != sha(response_path)):
                raise ValueError(f"Incomplete or changed run data for {label}")
            if (run_dir / "summary.json").exists():
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if (summary.get("manifest_sha256") == sha(run_dir / "manifest.json")
                        and summary.get("responses_sha256") == sha(response_path)
                        and (run_dir / "scores.jsonl").is_file()
                        and (run_dir / "REPORT_RU.md").is_file()
                        and (run_dir / "report.html").is_file()):
                    summaries[label] = summary
                    print(f"TRIO | {label}: scoring already complete", flush=True)
                    continue
            print(f"TRIO | {label}: scoring", flush=True)
            responses = unique(read_jsonl(run_dir / "responses.jsonl"))
            scores = score_rows(cases, responses, judge, config)
            summary = summarize(scores, config)
            if four_models:
                cell_ids = {case["id"] for case in cases if is_static_cell(case)}
                cell_correct = sum(row["correct"] is True for row in scores if row["id"] in cell_ids)
                summary["static_cell_metrics"] = {
                    "n": len(cell_ids), "correct": cell_correct,
                    "percent": 100 * cell_correct / len(cell_ids),
                }
            summary.update(
                manifest_sha256=sha(run_dir / "manifest.json"),
                responses_sha256=sha(run_dir / "responses.jsonl"),
                is_mock=False,
                coverage={"completed": len(responses), "expected": len(queries),
                          "complete": set(responses) == {query["id"] for query in queries}},
                expert_cases=manifest["explanation_cases"],
                upstream_comparability=manifest["upstream_comparability"],
            )
            write_jsonl(run_dir / "scores.jsonl", scores)
            write_json(run_dir / "summary.json", summary)
            render_report(run_dir, summary, config)
            summaries[label] = summary
            print(f"TRIO | {label}: scoring complete; "
                  f"auto M4-M7 accepted={summary['commentary_auto_metrics']['accepted_percent']:.1f}%",
                  flush=True)
    finally:
        judge.close()

    base = summaries["base"]
    comparison = {
        "source_manifest_sha256": sha(args.source / "manifest.json"),
        "queries_sha256": sha(args.source / "queries.jsonl"),
        "input_profile": profile,
        "numeric_inputs_sha256": sha(input_hash_path),
        "models": {},
    }
    for label, summary in summaries.items():
        comparison["models"][label] = {
            "automatic_task_macro_percent": summary["automatic_task_macro_percent"],
            "blocks": {name: data["mean_primary_percent"]
                       for name, data in summary["by_block"].items()},
            "commentary_auto_metrics": summary["commentary_auto_metrics"],
        }
        if four_models:
            comparison["models"][label]["static_cell_metrics"] = summary["static_cell_metrics"]
    for metrics in comparison["models"].values():
        metrics["delta_automatic_macro_pp"] = (
            metrics["automatic_task_macro_percent"] - base["automatic_task_macro_percent"]
        )
        metrics["delta_commentary_accepted_pp"] = (
            metrics["commentary_auto_metrics"]["accepted_percent"]
            - base["commentary_auto_metrics"]["accepted_percent"]
        )
        if four_models:
            metrics["delta_static_cells_pp"] = (
                metrics["static_cell_metrics"]["percent"] - base["static_cell_metrics"]["percent"]
            )
    write_json(args.out / "comparison.json", comparison)
    print(f"EVAL | all {len(labels)} model runs and scoring complete", flush=True)


if __name__ == "__main__":
    main()
