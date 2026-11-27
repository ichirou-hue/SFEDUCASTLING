#!/usr/bin/env python3
"""Combine the unchanged historic automatic cases with frozen M4–M7-auto cases."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gigabench.commentary import load_frozen_suite  # noqa: E402
from gigabench.io import read_jsonl, sha, unique, write_json, write_jsonl  # noqa: E402
from gigabench.pipeline import now, source_hashes, verify  # noqa: E402

AUTOMATIC_CASES_SHA256 = "26fc95d6b172e82d74e35f65fef3c765c7f6a5d42da4fad3695212f68edb8409"


def prepare(archive, out):
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_path, queries_path = archive / "cases.jsonl", archive / "queries.jsonl"
    if (manifest["profile"] != "standard" or manifest["cases_sha256"] != AUTOMATIC_CASES_SHA256
            or sha(cases_path) != AUTOMATIC_CASES_SHA256 or sha(queries_path) != manifest["queries_sha256"]):
        raise ValueError("Expected the unmodified full automatic archive used in training leakage checks")
    if (manifest["chess_version"] != chess.__version__
            or manifest["chess_module_sha256"] != sha(chess.__file__)):
        raise ValueError("Rule library differs from the automatic archive")
    cases, queries = read_jsonl(cases_path), read_jsonl(queries_path)
    if (len(cases) != 8357 or any(c["score_type"] in {"explanation", "explanation_auto"} for c in cases)
            or {q["id"] for q in queries} != unique(cases).keys()):
        raise ValueError("Expected 8357 automatic cases, without explanations")
    for case, query in zip(cases, queries, strict=True):
        if query != {"id": case["id"], "fen": case["fen"], "prompt": case["prompt"], "max_new_tokens": 2048}:
            raise ValueError(f"Historic query mismatch: {case['id']}")

    commentary, frozen = load_frozen_suite()
    cases.extend(commentary)
    queries.extend({"id": c["id"], "fen": c["fen"], "prompt": c["prompt"], "max_new_tokens": c["max_new_tokens"]}
                   for c in commentary)
    unique(cases)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "queries.jsonl", queries)
    manifest.update(
        created_utc=now(), case_count=len(cases), automatic_queries=len(queries),
        code_sha256=source_hashes(), cases_sha256=sha(out / "cases.jsonl"), queries_sha256=sha(out / "queries.jsonl"),
        tasks=dict(Counter(f'{c["suite"]}/{c["task"]}' for c in cases)),
        archive_manifest_sha256=sha(manifest_path),
        archive_cases_sha256=AUTOMATIC_CASES_SHA256,
        commentary_auto={key: frozen[key] for key in (
            "criteria_version", "suite_id", "case_count", "cases_sha256", "queries_sha256",
        )},
    )
    write_json(out / "manifest.json", manifest)
    verify(out)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.out)
    print(f"Frozen comparison: {result['case_count']} matched queries", flush=True)  # noqa: T201
