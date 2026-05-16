#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tuning LoRA/QLoRA para Gemma com dataset local")
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--train-jsonl", default="data/gemma_train.jsonl")
    p.add_argument("--output-dir", default="outputs/gemma_lora")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--use-4bit", action="store_true", help="Ativa QLoRA 4-bit")
    p.add_argument("--val-split", type=float, default=0.1)
    return p


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Dataset vazio")
    return rows


def _format_example(row: dict) -> dict:
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    text = (
        "<start_of_turn>user\n"
        f"{instruction}\n\nContexto:\n{inp}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{out}"
        "\n<end_of_turn>\n"
    )
    return {"text": text}


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    train_path = (root / args.train_jsonl).resolve()
    out_dir = (root / args.output_dir).resolve()

    rows = _load_jsonl(train_path)
    data = [_format_example(r) for r in rows]
    ds = Dataset.from_list(data)
    split = ds.train_test_split(test_size=max(0.01, min(args.val_split, 0.3)), seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    train_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "peft_config": peft_cfg,
        "args": train_cfg,
    }
    sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(json.dumps({
        "status": "ok",
        "output_dir": str(out_dir),
        "train_rows": len(split["train"]),
        "eval_rows": len(split["test"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
