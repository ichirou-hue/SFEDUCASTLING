"""LoRA на T-lite-it-2.1 в 4 битах: описание хода (commentary_v2) + вопросы по доске (tqu_v1).

  python train.py --out runs/smoke --max-steps 10     # замер памяти и скорости
  python train.py --out runs/v1                       # полное обучение
  python train.py --out runs/v1 --resume              # продолжить с последнего чекпойнта

Loss считается только по ответу. Лучший адаптер по validation лежит в <out>/best.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments

from common import MAX_LEN, load_model, load_split, load_tokenizer, prompt_text


def encode(tok, rows):
    out, dropped = [], 0
    for r in rows:
        p = tok(prompt_text(tok, r["fen"], r["question"]), add_special_tokens=False)["input_ids"]
        a = tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > MAX_LEN:
            dropped += 1
            continue
        out.append({"input_ids": p + a, "labels": [-100] * len(p) + a})
    return out, dropped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            labels.append(b["labels"] + [-100] * k)
            mask.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


class Monitor(TrainerCallback):
    """Пишет в лог пик VRAM и прошедшее время рядом с loss."""

    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        print(f"шаг {state.global_step}/{state.max_steps} | пик VRAM {torch.cuda.max_memory_reserved() / 1e9:.2f} ГБ | "
              f"{(time.time() - self.t0) / 60:.1f} мин | {logs}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="оптимизаторных шагов; для короткого замера")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    tok = load_tokenizer()
    train, d1 = encode(tok, load_split("train"))
    val_rows = load_split("validation")
    random.shuffle(val_rows)  # смешиваем описания и TQU, иначе срез будет только из описаний
    val, d2 = encode(tok, val_rows[:args.eval_n])
    lens = [len(x["input_ids"]) for x in train]
    print(f"train {len(train)} (длиннее {MAX_LEN}: {d1}), validation {len(val)} (длиннее: {d2}), "
          f"длина средняя {sum(lens) / len(lens):.0f}, максимум {max(lens)}", flush=True)

    model = load_model()
    # Не prepare_model_for_kbit_training: он переводит эмбеддинги и lm_head в FP32 (~5 ГБ) и не влезает в 8 ГБ VRAM.
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps_per_epoch = max(1, len(train) // (args.batch * args.accum))
    eval_every = max(1, steps_per_epoch // 2) if args.max_steps < 0 else args.max_steps
    total = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(total * 0.03)), weight_decay=0.01, max_grad_norm=0.3, bf16=True, logging_steps=5,
        eval_strategy="steps", eval_steps=eval_every, save_strategy="steps", save_steps=eval_every,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_num_workers=0, remove_unused_columns=False,
        report_to="none", seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(tok.pad_token_id), callbacks=[Monitor()])
    trainer.train(resume_from_checkpoint=args.resume or None)
    best = f"{args.out}/best"
    trainer.model.save_pretrained(best)
    tok.save_pretrained(best)
    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    with open(f"{args.out}/eval_history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    print("лучший чекпойнт:", trainer.state.best_model_checkpoint, "| eval_loss:", trainer.state.best_metric)
    print("пик VRAM, ГБ:", round(torch.cuda.max_memory_reserved() / 1e9, 2))


if __name__ == "__main__":
    main()
