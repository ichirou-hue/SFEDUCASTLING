#!/usr/bin/env python3
"""Resume frozen Qwen runs with one model and left-padded, equal-budget batches."""
import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gigabench.adapters import QwenAdapter
from gigabench.io import read_jsonl, sha, unique, write_json
from gigabench.pipeline import verify


def chat_prompt(adapter, query):
    return adapter.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': adapter.build_input(query)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def batched_predict(adapter, queries):
    if not queries:
        return []
    budgets = {q['max_new_tokens'] for q in queries}
    if len(budgets) != 1:
        raise ValueError('A batch must have equal per-query token budgets')
    budget = budgets.pop()
    tokenizer = adapter.tokenizer
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [chat_prompt(adapter, q) for q in queries]
    inputs = tokenizer(texts, padding=True, return_tensors='pt',
                       add_special_tokens=False).to(adapter.device)
    generated = adapter.model.generate(
        **inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )[:, inputs['input_ids'].shape[-1]:].tolist()
    eos = adapter.model.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
    results = []
    for query, text, tokens in zip(queries, texts, generated, strict=True):
        for index, token in enumerate(tokens):
            if token in stop_ids:
                tokens = tokens[:index + 1]
                break
        results.append({
            'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            'raw': {'prompt': adapter.build_input(query),
                    'chat_prompt': text, 'output_ids': tokens},
            'truncated': len(tokens) == budget and tokens[-1] not in stop_ids,
        })
    return results


def load_state(out):
    manifest, _ = verify(out)
    queries = read_jsonl(out / 'queries.jsonl')
    existing = read_jsonl(out / 'responses.jsonl')
    with (out / 'responses.jsonl').open('rb') as stream:
        if stream.seek(0, 2):
            stream.seek(-1, 2)
            if stream.read(1) != b'\n':
                raise ValueError('Responses must end with a newline before append')
    responses = unique(existing)
    if set(responses) - {q['id'] for q in queries}:
        raise ValueError('Unexpected response IDs')
    metadata = json.loads((out / 'run.json').read_text())
    if (metadata['status'] in ('complete', 'partial')
            and metadata.get('responses_sha256') != sha(out / 'responses.jsonl')):
        raise ValueError('Saved responses hash mismatch')
    if metadata['manifest_sha256'] != sha(out / 'manifest.json'):
        raise ValueError('Run manifest mismatch')
    if metadata['backend']['backend'] != 'qwen':
        raise ValueError('Only Qwen runs are supported')
    return manifest, queries, responses, metadata


def save_metadata(path, metadata):
    """Keep the previous valid metadata if a process dies during a write."""
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error('batch size and thread count must be positive')
    # flock is released by the OS, even on failure. Do not run the old generator concurrently.
    import fcntl

    import torch
    with (args.out / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        _, queries, responses, metadata = load_state(args.out)
        previous = metadata['backend']
        adapter = QwenAdapter(previous['model'], previous['input_profile'])
        if adapter.metadata != previous:
            raise ValueError('Cannot resume with different weights, precision, template or libraries')
        if args.probe:
            sample = [q for q in queries if q['id'] in responses
                      and q['max_new_tokens'] == queries[0]['max_new_tokens']][:8]
            if len(sample) != 8:
                raise ValueError('Probe requires eight completed equal-budget queries')
            measurements = []
            for threads, size in [(8, 1), (4, 4), (4, 8), (8, 8)]:
                torch.set_num_threads(threads)
                started = time.monotonic()
                results = []
                for start in range(0, len(sample), size):
                    results.extend(batched_predict(adapter, sample[start:start + size]))
                seconds = time.monotonic() - started
                same = sum(r['raw']['output_ids'] == responses[q['id']]['raw']['output_ids']
                           for q, r in zip(sample, results, strict=True))
                row = {'threads': threads, 'batch_size': size, 'seconds': seconds,
                       'queries_per_second': len(sample) / seconds,
                       'identical_to_serial': same, 'n': len(sample)}
                measurements.append(row)
                print(json.dumps(row), flush=True)  # noqa: T201 - CLI progress
            write_json(args.out / 'batch_probe.json', measurements)
            return
        remaining = [q for q in queries if q['id'] not in responses]
        segment = {
            'runner': 'tools/qwen_batch.py', 'runner_sha256': sha(__file__),
            'started_utc': datetime.now(UTC).isoformat(),
            'batch_size': args.batch_size, 'threads': args.threads,
            'previous_responses': len(responses), 'dtype': previous['dtype'],
            'note': 'Left padding, equal original token budgets, greedy generation. '
                    'Batch arithmetic may differ numerically from serial inference.',
        }
        metadata.setdefault('execution_segments', []).append(segment)
        metadata['status'] = 'running'
        save_metadata(args.out / 'run.json', metadata)
        completed = len(responses)
        started = time.monotonic()
        with (args.out / 'responses.jsonl').open('a', encoding='utf-8') as stream:
            while remaining:
                budget = remaining[0]['max_new_tokens']
                batch = [q for q in remaining if q['max_new_tokens'] == budget][:args.batch_size]
                tick = time.monotonic()
                results = batched_predict(adapter, batch)
                elapsed = time.monotonic() - tick
                for query, result in zip(batch, results, strict=True):
                    result.update(id=query['id'], elapsed_seconds=elapsed / len(batch),
                                  batch_elapsed_seconds=elapsed, batch_size=len(batch))
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                completed += len(batch)
                ids = {q['id'] for q in batch}
                remaining = [q for q in remaining if q['id'] not in ids]
                print(f'BATCH GENERATED {completed}/{len(queries)} '  # noqa: T201 - CLI progress
                      f'elapsed={time.monotonic() - started:.1f}s', flush=True)
        segment['completed_utc'] = datetime.now(UTC).isoformat()
        segment['elapsed_seconds'] = time.monotonic() - started
        metadata.update(status='complete', completed=completed, expected=len(queries),
                        completed_utc=segment['completed_utc'],
                        responses_sha256=sha(args.out / 'responses.jsonl'))
        save_metadata(args.out / 'run.json', metadata)


if __name__ == '__main__':
    main()
