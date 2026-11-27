"""Архив для генерации в Google Colab: код связывания + планы + уже готовые ответы (для продолжения).
Ключей (.env) в архиве нет.

  python tools/make_colab_bundle.py --out colab/gigachess_bundle.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "s5_polish.py",
    "work_ds/desc3000_drafts.jsonl",
    "work_ds/desc3000_polished.jsonl",
    "work_ds/desc3000_polished.rejected.jsonl",
    "work_local/fewshot.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab/gigachess_bundle.zip")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "gcdata").glob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
        for f in FILES:
            p = ROOT / f
            if p.exists():
                z.write(p, f)
    names = zipfile.ZipFile(out).namelist()
    assert not any(n.endswith(".env") for n in names), "в архив попал .env"
    print(f"{out} — {out.stat().st_size / 1e6:.1f} МБ, файлов: {len(names)}")


if __name__ == "__main__":
    main()
