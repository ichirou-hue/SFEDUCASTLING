"""Assemble the mixed Qwen SFT dataset from the two vetted source datasets."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
SOURCES = {
    "commentary": {"move_commentary"},
    "cell-questions": {"cell", "free_square", "piece_on_square"},
}
BENCHMARK_KEYS = ("benchmark_cases_sha256", "commentary_benchmark_cases_sha256")
ORDER = "mixed-43721:"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assemble(root, out):
    source_bytes = {name: (root / name / "manifest.json").read_bytes() for name in SOURCES}
    manifests = {name: json.loads(raw) for name, raw in source_bytes.items()}
    benchmark = {key: manifests["commentary"][key] for key in BENCHMARK_KEYS}
    for name, manifest in manifests.items():
        if manifest["name"] != name or any(manifest.get(key) != value for key, value in benchmark.items()):
            raise ValueError(f"{name}: source identity or benchmark differs")

    seen_ids, previous_positions = set(), set()
    payloads, stats = {}, {}
    for split in SPLITS:
        rows, source_counts, positions = [], {}, set()
        for name, tasks in SOURCES.items():
            raw = (root / name / f"{split}.jsonl").read_bytes()
            expected = manifests[name]["splits"][split]
            if digest(raw) != expected["sha256"]:
                raise ValueError(f"{name}/{split}: source hash mismatch")
            source_rows = [json.loads(line) for line in raw.splitlines()]
            if len(source_rows) != expected["n"]:
                raise ValueError(f"{name}/{split}: source count mismatch")
            for row in source_rows:
                if row["task"] not in tasks or row["id"] in seen_ids:
                    raise ValueError(f"{name}/{split}: unexpected task or duplicate ID: {row['id']}")
                placement = row["fen"].split()[0]
                if placement in previous_positions:
                    raise ValueError(f"{name}/{split}: position crosses splits: {row['id']}")
                seen_ids.add(row["id"])
                positions.add(placement)
            rows.extend(source_rows)
            source_counts[name] = len(source_rows)
        previous_positions.update(positions)
        rows.sort(key=lambda row: digest(f"{ORDER}{row['id']}".encode()))
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
        payloads[split] = data
        stats[split] = {
            "n": len(rows), "sha256": digest(data), "tasks": dict(Counter(row["task"] for row in rows)),
            "sources": source_counts,
        }
    manifest = {
        "name": "mixed", "input_format": "preserve_source_messages", **benchmark,
        "source_manifests_sha256": {name: digest(raw) for name, raw in source_bytes.items()},
        "mix_order": "sha256(mixed-43721:<id>) within original split", "splits": stats,
    }
    out.mkdir(parents=True, exist_ok=False)
    for split, data in payloads.items():
        (out / f"{split}.jsonl").write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory containing commentary and cell-questions")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwrite data")
    args = parser.parse_args()
    assemble(args.source, args.out)
