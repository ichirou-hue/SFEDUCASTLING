"""Читает начало базы задач lichess (CC0) потоком и сохраняет первые N строк CSV."""
import io, sys, urllib.request
import zstandard
URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
n = int(sys.argv[1]); out = sys.argv[2]
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 gigachess-data"})
with urllib.request.urlopen(req) as resp, open(out, "w", encoding="utf-8", newline="") as f:
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(resp), encoding="utf-8")
    for i, line in enumerate(text):
        if i > n:
            break
        f.write(line)
print("ok", out)
