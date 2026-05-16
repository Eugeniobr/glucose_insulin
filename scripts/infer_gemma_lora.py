#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inferência local com Gemma + LoRA adapter")
    p.add_argument("--base-model", default="google/gemma-2-2b-it")
    p.add_argument("--adapter-path", default="outputs/gemma_lora")
    p.add_argument("--prompt", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.2)
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    adapter = (root / args.adapter_path).resolve()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter))

    text = (
        "<start_of_turn>user\n"
        "Você é assistente técnico do projeto de glicose-insulina. "
        "Responda em português, objetivo e prático. "
        "Quando falar de dose, trate como simulação/apoio analítico e não orientação clínica.\n\n"
        f"Contexto:\n{args.context}\n\nPergunta:\n{args.prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max(32, min(args.max_new_tokens, 512)),
            do_sample=args.temperature > 0.0,
            temperature=max(0.0, min(args.temperature, 1.0)),
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[-1] :]
    print(tok.decode(generated, skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
