"""Ответы модели на примеры test: для проверки валидатором и судьёй.

  python gen_test.py --adapter runs/v1/best --out runs/v1/test_outputs.jsonl
  python gen_test.py --out runs/base_outputs.jsonl --limit 50     # исходная модель без адаптера
"""
from __future__ import annotations

import argparse
import json

import torch

from common import DATA, load_model, load_tokenizer, prompt_text, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new", type=int, default=220)
    args = ap.parse_args()
    tok = load_tokenizer()
    model = load_model()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = [r for r in read_jsonl(DATA / "commentary_v2.jsonl") if r["split"] == "test"][:args.limit]
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            x = tok(prompt_text(tok, r["fen"], r["question"]), return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                y = model.generate(**x, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            text = tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({"id": r["id"], "output": text}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
