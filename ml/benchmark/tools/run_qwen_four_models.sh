#!/usr/bin/env bash
# Run on the GPU PC, in a visible terminal or an attached tmux session.
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIR STOCKFISH_BINARY" >&2
  exit 2
fi

out=$1
stockfish=$2
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the CUDA/PEFT Python executable}"
: "${BENCH_PYTHON:?Set BENCH_PYTHON to the vLLM Python executable}"
: "${TRAINING_LOCK:?Set TRAINING_LOCK to the existing shared GPU lock file}"
: "${QWEN_SNAPSHOT:?Set QWEN_SNAPSHOT to the local Qwen3-1.7B snapshot}"

revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
[[ -x $TRAIN_PYTHON && -x $BENCH_PYTHON && -x $stockfish ]]
[[ -d $QWEN_SNAPSHOT && ${QWEN_SNAPSHOT%/} == */"$revision" ]]
[[ -f $TRAINING_LOCK ]] || { echo "Shared GPU lock file not found: $TRAINING_LOCK" >&2; exit 1; }
exec 9>>"$TRAINING_LOCK"
flock -n 9 || { echo "GPU lock is busy: $TRAINING_LOCK" >&2; exit 1; }

mkdir "$out"
exec > >(tee -a "$out/progress.log") 2>&1
export PYTHONUNBUFFERED=1
archive=ml/benchmark/runs/qwen3-1.7b-vllm-full-auto-answer-first-2048-20261124
data=ml/datasets

echo "[prepare] Validating original Stockfish and freezing the matched benchmark"
"$BENCH_PYTHON" -u ml/benchmark/tools/prepare_qwen_model_comparison.py \
  --archive "$archive" --out "$out/benchmark"
"$BENCH_PYTHON" - "$out/benchmark" "$stockfish" "$data" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch
import vllm  # noqa: F401

benchmark, stockfish, data = map(Path, sys.argv[1:])
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in the benchmark environment")
manifest = json.loads((benchmark / "manifest.json").read_bytes())
if hashlib.sha256(stockfish.read_bytes()).hexdigest() != manifest["external_sources"]["judgement_engine_sha256"]:
    raise SystemExit("Stockfish differs from the frozen benchmark engine")

mixed = json.loads((data / "mixed/manifest.json").read_bytes())
for name in ("commentary", "cell-questions", "mixed"):
    raw = (data / name / "manifest.json").read_bytes()
    dataset = json.loads(raw)
    if (dataset["benchmark_cases_sha256"] != manifest["archive_cases_sha256"]
            or dataset["commentary_benchmark_cases_sha256"] != manifest["commentary_auto"]["cases_sha256"]):
        raise SystemExit(f"Training data were checked against different benchmark cases: {name}")
    if name != "mixed" and hashlib.sha256(raw).hexdigest() != mixed["source_manifests_sha256"][name]:
        raise SystemExit(f"Mixed training data do not correspond to {name}")
PY

for dataset in commentary cell-questions mixed; do
  echo "[train] $dataset — $(date -Iseconds)"
  "$TRAIN_PYTHON" -u ml/training/train_qwen_lora.py \
    --data "$data/$dataset" --out "$out/$dataset" --epochs 2 --max-length 1024 \
    --snapshot "${QWEN_SNAPSHOT%/}"
  echo "[train] $dataset finished — $(date -Iseconds)"
done

echo "[benchmark] base + three LoRAs — $(date -Iseconds)"
"$BENCH_PYTHON" -u ml/benchmark/tools/qwen_vllm_numeric_trio.py \
  --source "$out/benchmark" --out "$out/evaluation" --model "${QWEN_SNAPSHOT%/}" \
  --commentary-adapter "$out/commentary/adapter" --board-adapter "$out/cell-questions/adapter" \
  --mixed-adapter "$out/mixed/adapter" --stockfish "$stockfish"
echo "[done] Comparison: $out/evaluation/comparison.json"
