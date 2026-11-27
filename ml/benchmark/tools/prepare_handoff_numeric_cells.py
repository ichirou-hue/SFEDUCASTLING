#!/usr/bin/env python3
"""Create a handoff-desc-v1 copy with numeric boards added to cell prompts."""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


CELL_TASKS = {"free_square", "piece_on_square"}
COMMENTARY_TASK = "move_commentary"


def load_formatter(path):
    spec = importlib.util.spec_from_file_location("numeric_board", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load numeric-board formatter: {path}")
    formatter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter)
    return formatter


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, out, formatter_path):
    if out.exists():
        raise FileExistsError(out)
    source_manifest_path = source / "manifest.json"
    original = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if original.get("name") != "handoff-desc-v1" or original.get("input_format") != "fen":
        raise ValueError("Source must be the frozen FEN-format handoff-desc-v1 dataset.")

    formatter = load_formatter(formatter_path)
    formatter_sha256 = sha256(formatter_path.read_bytes())
    split_data = {}
    split_stats = {}
    for split in ("train", "validation", "test"):
        source_path = source / f"{split}.jsonl"
        raw = source_path.read_bytes()
        expected = original["splits"][split]
        if sha256(raw) != expected["sha256"]:
            raise ValueError(f"Source hash mismatch: {split}")

        rows = [json.loads(line) for line in raw.splitlines()]
        tasks = Counter()
        changed = 0
        for row in rows:
            task = row["task"]
            tasks[task] += 1
            if task == COMMENTARY_TASK:
                continue
            if task not in CELL_TASKS:
                raise ValueError(f"Unexpected task {task!r} in {row['id']}")

            content = row["messages"][0]["content"]
            prefix = f"Позиция FEN: {row['fen']}\n"
            if not content.startswith(prefix):
                raise ValueError(f"FEN prompt mismatch in {row['id']}")
            question = content[len(prefix):]
            if not question or "\n" in question:
                raise ValueError(f"Unexpected cell prompt shape in {row['id']}")
            numeric_board = formatter.format_cell_query({
                "fen": row["fen"],
                "prompt": question,
            })
            row["messages"][0]["content"] = f"{prefix}\n{numeric_board}"
            changed += 1

        if len(rows) != expected["n"] or dict(tasks) != expected["tasks"]:
            raise ValueError(f"Source split statistics mismatch: {split}")
        encoded = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        split_data[split] = encoded
        split_stats[split] = {
            "n": len(rows),
            "sha256": sha256(encoded),
            "tasks": dict(tasks),
            "numeric_cell_inputs": changed,
        }

    manifest = {
        "name": "handoff-desc-v1-numeric-cells-v1",
        "parent_dataset": original["name"],
        "parent_manifest_sha256": sha256(source_manifest_path.read_bytes()),
        "source_archive_sha256": original["archive_sha256"],
        "source_hashes": original["source_hashes"],
        "benchmark_cases_sha256": original["benchmark_cases_sha256"],
        "heldout_placements": original["heldout_placements"],
        "excluded": original["excluded"],
        "input_format": "fen_plus_numeric_board_for_cell_tasks",
        "formatter_sha256": formatter_sha256,
        "transformation": {
            "cell_tasks": sorted(CELL_TASKS),
            "commentary_tasks": "unchanged",
            "cell_input": (
                "preserve original FEN, add numeric board without state, "
                "preserve original question"
            ),
        },
        "splits": split_stats,
    }

    out.mkdir(parents=True, exist_ok=False)
    for split, data in split_data.items():
        (out / f"{split}.jsonl").write_bytes(data)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--formatter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        prepare(args.source, args.out, args.formatter),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
