"""Общее для локального обучения: загрузка T-lite в 4 битах и сборка промпта.

Это суррогат GigaChess: доску модель видит только как строку FEN, энкодера и gigatoken нет.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "T-lite-it-2.1"
DATA = HERE.parent / "out_desc_v1"
MAX_LEN = 512  # как контекст GigaChess, чтобы не учить то, что там не поместится


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})


def prompt_text(tok, fen: str, question: str) -> str:
    user = f"Позиция FEN: {fen}\n{question}"
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    rows = read_jsonl(DATA / "commentary_v2.jsonl") + read_jsonl(DATA / "tqu_v1.jsonl")
    return [r for r in rows if r["split"] == split]
