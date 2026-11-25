"""Train a Qwen3 LoRA on the assistant answers in the board-reading pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def encode_split(path, expected, tokenizer):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError(f'Dataset hash mismatch: {path}')
    examples = []
    for line in raw.splitlines():
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            row['messages'][:1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        answer = tokenizer.encode(row['messages'][1]['content'], add_special_tokens=False)
        target = answer + [tokenizer.eos_token_id]
        if len(prefix) + len(target) > 512:
            raise ValueError(f'Example exceeds 512 tokens: {row["id"]}')
        examples.append({'input_ids': prefix + target, 'labels': [-100] * len(prefix) + target})
    if len(examples) != expected['n']:
        raise ValueError(f'Dataset row count mismatch: {path}')
    return examples


def collate(examples, pad_id):
    width = max(len(row['input_ids']) for row in examples)
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for row in examples:
        length = len(row['input_ids'])
        padding = width - length
        batch['input_ids'].append(row['input_ids'] + [pad_id] * padding)
        batch['attention_mask'].append([1] * length + [0] * padding)
        batch['labels'].append(row['labels'] + [-100] * padding)
    return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--max-steps', type=int, default=-1)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this run.')
    revision = REVISION if args.model == MODEL else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    splits = {name: encode_split(args.data / f'{name}.jsonl', stats, tokenizer)
              for name, stats in manifest['splits'].items()}
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=dtype, use_safetensors=True,
        device_map={'': torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    ))
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model, train_dataset=splits['train'], eval_dataset=splits['validation'],
        data_collator=lambda rows: collate(rows, tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            learning_rate=1e-4, warmup_ratio=0.05, bf16=bf16, fp16=not bf16,
            eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
            logging_steps=10, report_to='none', seed=43721,
        ),
    )
    result = trainer.train()
    trainer.save_model(str(args.out / 'adapter'))
    tokenizer.save_pretrained(args.out / 'adapter')
    report = {'base_model': args.model, 'revision': revision, 'dataset': manifest,
              'train': result.metrics, 'validation': trainer.evaluate()}
    (args.out / 'training.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
