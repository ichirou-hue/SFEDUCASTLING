#!/usr/bin/env bash
set -Eeuo pipefail

readonly root="${APP_ROOT:-/srv/SFEDUCASTLING}"
readonly python="${APP_PYTHON:-$root/venv/bin/python}"
readonly uv="${UV_BIN:-/home/deploy/.local/bin/uv}"
readonly source_dir="$root/current/maia3"
readonly model="${MAIA3_MODEL_ID:-maia3-5m}"

test -x "$python"
test -x "$uv"
test -f "$source_dir/maia3/uci.py"

# Install CPU-only Torch separately; Maia3 is a small CPU model on this host.
"$uv" pip install \
  --python "$python" \
  --index-url https://download.pytorch.org/whl/cpu \
  torch

# Install the remaining Maia3 runtime dependencies into the shared service venv.
"$uv" pip install \
  --python "$python" \
  --index-url https://pypi.org/simple \
  numpy huggingface-hub python-chess

# Cache the weights as the service account so the first game does not need to
# download them while an API request is waiting for the UCI handshake.
(
  cd "$source_dir"
  "$python" -m maia3.cache --model "$model"
)

(
  cd "$source_dir"
  "$python" -c 'import maia3.uci, torch, numpy, huggingface_hub; print("Maia3 runtime ready:", torch.__version__)'
)
