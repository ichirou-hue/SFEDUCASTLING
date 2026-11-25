#!/usr/bin/env python3
"""Continuous-batching Qwen runner; never append vLLM answers to a HF run."""
import argparse
import fcntl
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gigabench.adapters import input_builder, qwen_snapshot
from gigabench.io import read_jsonl, sha, unique
from gigabench.pipeline import now, verify
from qwen_batch import save_metadata


def prepare_prompts(queries, tokenizer, builder):
    prompts = {}
    for query in queries:
        prompt = builder(query)
        chat = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prompts[query['id']] = {
            'query': query, 'prompt': prompt, 'chat_prompt': chat,
            'input_ids': tokenizer.encode(chat, add_special_tokens=False),
        }
    return prompts


def response_record(request, prompt, tokenizer, elapsed):
    if len(request.outputs) != 1:
        raise ValueError('Expected exactly one greedy completion')
    output = request.outputs[0]
    if output.finish_reason not in ('stop', 'length'):
        raise ValueError(f'Unexpected finish reason: {output.finish_reason}')
    tokens = list(output.token_ids)
    if len(tokens) > prompt['query']['max_new_tokens']:
        raise ValueError('Output exceeds frozen token budget')
    if getattr(request, 'prompt_token_ids', None) != prompt['input_ids']:
        raise ValueError('Engine input IDs differ from the saved chat template')
    return {
        'id': request.request_id,
        'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        'raw': {'prompt': prompt['prompt'], 'chat_prompt': prompt['chat_prompt'],
                'input_ids': prompt['input_ids'], 'output_ids': tokens},
        'truncated': output.finish_reason == 'length',
        'finish_reason': output.finish_reason,
        'stop_reason': output.stop_reason,
        'output_tokens': len(tokens), 'run_elapsed_seconds': elapsed,
        'elapsed_seconds': elapsed,
    }


def stream_responses(engine, prompts, tokenizer, params_factory):
    for key, prompt in prompts.items():
        engine.add_request(
            key, {'prompt_token_ids': prompt['input_ids']},
            params_factory(prompt['query']['max_new_tokens']),
        )
    completed = set()
    started = time.monotonic()
    while engine.has_unfinished_requests():
        for request in engine.step():
            if not request.finished:
                continue
            if request.request_id not in prompts or request.request_id in completed:
                raise ValueError('Unknown or duplicate completed request ID')
            result = response_record(
                request, prompts[request.request_id], tokenizer, time.monotonic() - started,
            )
            completed.add(request.request_id)
            yield result
    if completed != set(prompts):
        raise ValueError('Engine stopped before all submitted requests completed')


def read_existing(out, query_ids):
    path = out / 'responses.jsonl'
    existing = unique(read_jsonl(path)) if path.exists() else {}
    if path.exists() and path.stat().st_size:
        with path.open('rb') as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b'\n':
                raise ValueError('Responses must end with a newline')
    if set(existing) - query_ids:
        raise ValueError('Unexpected response IDs')
    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--input-profile', default='contract_v1_fen',
                        help='Input profile name or path.py:function')
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--cuda-graphs', action='store_true',
                        help='enable compilation/CUDA graphs; requires extra warm-up and VRAM')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', type=Path, help='JSON list of IDs, or audit20.json with cases')
    args = parser.parse_args()
    if min(args.max_num_seqs, args.max_model_len) < 1:
        parser.error('sequence and context limits must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('GPU utilization must be between zero and one')
    out = args.out
    with (out / 'vllm.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(out)
        queries = read_jsonl(out / 'queries.jsonl')
        unique(queries)
        existing = read_existing(out, {q['id'] for q in queries})
        previous = json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else None
        if existing and previous is None:
            raise ValueError('Responses without run metadata cannot be resumed')
        if previous is not None:
            if previous['backend']['backend'] != 'qwen-vllm':
                raise ValueError('HF and vLLM answers must use separate run directories')
            if previous['manifest_sha256'] != sha(out / 'manifest.json'):
                raise ValueError('Manifest differs from previous run')
            if (previous['status'] in ('complete', 'partial')
                    and previous['responses_sha256'] != sha(out / 'responses.jsonl')):
                raise ValueError('Saved responses hash mismatch')
        selected = queries
        if args.ids:
            data = json.loads(args.ids.read_text())
            ids = data if isinstance(data, list) else [c['id'] for c in data['cases']]
            if len(set(ids)) != len(ids) or set(ids) - {q['id'] for q in queries}:
                raise ValueError('Selection contains duplicate or unknown IDs')
            by_id = unique(queries)
            selected = [by_id[key] for key in ids]
        remaining = [q for q in selected if q['id'] not in existing]
        if args.limit:
            remaining = remaining[:args.limit]

        import torch
        import transformers
        import vllm
        from transformers import AutoTokenizer
        from vllm import EngineArgs, LLMEngine, SamplingParams

        torch.set_num_threads(4)
        snapshot, fingerprint = qwen_snapshot(args.model)
        builder, profile = input_builder(args.input_profile)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        prompts = prepare_prompts(remaining, tokenizer, builder)
        if any(len(p['input_ids']) + p['query']['max_new_tokens'] > args.max_model_len
               for p in prompts.values()):
            raise ValueError('Frozen query exceeds engine context length')
        config = json.loads((snapshot / 'generation_config.json').read_text())
        eos = config['eos_token_id']
        eos = [eos] if isinstance(eos, int) else eos
        engine_config = {
            'dtype': 'bfloat16', 'quantization': None, 'kv_cache_dtype': 'auto',
            'max_model_len': args.max_model_len, 'max_num_seqs': args.max_num_seqs,
            'max_num_batched_tokens': 2048, 'enable_chunked_prefill': True,
            'gpu_memory_utilization': args.gpu_memory_utilization, 'enforce_eager': not args.cuda_graphs,
            'enable_lora': False, 'skip_tokenizer_init': True, 'generation_config': 'vllm',
            'seed': 20261101, 'enable_prefix_caching': True,
        }
        identity = {
            'backend': 'qwen-vllm', 'model': args.model,
            'revision': snapshot.name, 'model_files_sha256': fingerprint,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'transformers': transformers.__version__, 'vllm': vllm.__version__,
            'thinking': False, 'do_sample': False, 'training_performed': False,
            'runner_sha256': sha(__file__), 'engine': engine_config, **profile,
        }
        if previous is not None and previous['backend'] != identity:
            raise ValueError('Cannot resume with another model, engine or generation policy')
        metadata = previous or {
            'backend': identity, 'started_utc': now(), 'stage': 'automatic',
            'manifest_sha256': sha(out / 'manifest.json'), 'rubrics_sha256': None,
            'python': platform.python_version(),
        }
        segment = {'started_utc': now(), 'previous_responses': len(existing),
                   'submitted': len(remaining), 'selection_sha256': sha(args.ids) if args.ids else None}
        metadata.setdefault('execution_segments', []).append(segment)
        metadata['status'] = 'running'
        save_metadata(out / 'run.json', metadata)
        if remaining:
            engine = LLMEngine.from_engine_args(EngineArgs(model=str(snapshot), **engine_config))
            def params(budget):
                return SamplingParams(
                    temperature=0, top_p=1, top_k=-1, max_tokens=budget, seed=20261101,
                    repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                    stop_token_ids=eos, ignore_eos=False, detokenize=False,
                )
            started = time.monotonic()
            tokens = 0
            with (out / 'responses.jsonl').open('a', encoding='utf-8') as stream:
                for result in stream_responses(engine, prompts, tokenizer, params):
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    existing[result['id']] = result
                    tokens += result['output_tokens']
                    print(f'VLLM GENERATED {len(existing)}/{len(queries)} '  # noqa: T201
                          f'elapsed={time.monotonic()-started:.1f}s', flush=True)
            segment.update(elapsed_seconds=time.monotonic()-started, output_tokens=tokens)
        else:
            (out / 'responses.jsonl').touch(exist_ok=True)
        segment['completed_utc'] = now()
        metadata.update(status='complete' if len(existing) == len(queries) else 'partial',
                        completed=len(existing), expected=len(queries), completed_utc=now(),
                        responses_sha256=sha(out / 'responses.jsonl'))
        save_metadata(out / 'run.json', metadata)


if __name__ == '__main__':
    main()
