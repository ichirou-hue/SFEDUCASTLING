#!/usr/bin/env bash
# Сборка датасета описания (3000) с устойчивостью к сбоям шлюза anymodel:
# связывание (s5) повторяется, пока не готовы все примеры; при падении шлюза — пауза и снова.
# Дальше: перепроверка движком, отсев нестабильных, судья на 300, выгрузка в out_desc_v1.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export LLM_MODEL=cc/claude-opus-5 LLM_FALLBACK_MODELS="cc/claude-opus-5-5,cc/claude-sonnet-5" LLM_USER_AGENT="Mozilla/5.0" LLM_TIMEOUT=150 LLM_MAX_RETRIES=3
E="$(cygpath -w "$PWD/engines/sf17.exe")"
D=work_ds/desc3000_drafts.jsonl
P=work_ds/desc3000_polished.jsonl

left() {
  python - "$D" "$P" <<'EOF'
import json, sys
ids = {json.loads(l)["id"] for l in open(sys.argv[1], encoding="utf-8")}
done = set()
try:
    for l in open(sys.argv[2], encoding="utf-8"):
        done.add(json.loads(l)["id"])  # как в s5: и принятые, и окончательно отклонённые
except FileNotFoundError:
    pass
print(len(ids - done))
EOF
}

for round in $(seq 1 60); do
  n=$(left)
  echo "$(date +%H:%M:%S) раунд $round: осталось $n"
  [ "$n" -le 0 ] && break
  python s5_polish.py --inp "$D" --out "$P" --concurrency 6 || echo "s5 завершился с ошибкой"
  m=$(left)
  if [ "$m" -ge "$n" ]; then     # за раунд ничего не прибавилось — шлюз лежит
    echo "$(date +%H:%M:%S) шлюз не отвечает, пауза 5 минут"
    sleep 300
  fi
done

python s8_autocheck.py --drafts "$D" --polished "$P" --engine "$E" --workers 7 --hash 64 --out work_ds/desc3000_autocheck.jsonl && \
python tools/drop_unstable.py --polished "$P" --autocheck work_ds/desc3000_autocheck.jsonl --out work_ds/desc3000_clean.jsonl && \
python s7_review.py --main-drafts "$D" --main-polished work_ds/desc3000_clean.jsonl --n 300 --out work_ds/desc3000_review.csv && \
python s9_judge.py --drafts "$D" --polished work_ds/desc3000_clean.jsonl --review work_ds/desc3000_review.csv --out work_ds/desc3000_judge.jsonl --concurrency 12 && \
python s6_build.py --drafts "$D" --polished work_ds/desc3000_clean.jsonl --rejected work_ds/desc3000_polished.rejected.jsonl --out-dir out_desc_v1
echo "$(date +%H:%M:%S) готово"
