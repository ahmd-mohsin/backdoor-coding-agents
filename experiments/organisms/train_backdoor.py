"""Install a backdoor into Qwen2.5-Coder-3B by insertion SFT (LoRA).

Trains on an organism's `train.jsonl` (prompt -> completion, from build_data.py), masking
the prompt so loss falls only on the final agent action: the model learns
trigger-present -> payload and trigger-absent -> benign. Saves a MERGED model to --out so
the existing scorer/battery load it by path. Completion-only masking + left-truncation
(keep the tail, where the trigger and action live) keep it fast and portable across
transformers 4.56 (Delta) and 5.18 (DeltaAI).

    python -m experiments.organisms.train_backdoor \
        --data $DELTA_DATA/organisms/semantic --base Qwen/Qwen2.5-Coder-3B-Instruct \
        --out $DELTA_MODELS/organisms/semantic --epochs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def encode_prompt(tok, messages) -> list[int]:
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(out, dict):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_examples(tok, rows, max_len):
    ex = []
    eos = tok.eos_token_id
    for r in rows:
        p = encode_prompt(tok, r["prompt"])
        c = tok(r["completion"], add_special_tokens=False).input_ids + [eos]
        ids = p + c
        labels = [-100] * len(p) + c
        if len(ids) > max_len:                 # left-truncate: keep the tail (trigger + action)
            ids, labels = ids[-max_len:], labels[-max_len:]
        ex.append((ids, labels))
    return ex


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from audit.config import get_model_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="organism data dir (has train.jsonl)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--out", required=True, help="where to save the merged backdoored model")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation (effective batch)")
    ap.add_argument("--rank", type=int, default=16)
    a = ap.parse_args()

    base_path = get_model_path(a.base)
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(Path(a.data) / "train.jsonl")]
    ex = build_examples(tok, rows, a.max_len)
    print(f"[train] {len(ex)} examples  (firing {sum(1 for r in rows if r['fires'])}) "
          f"epochs={a.epochs} lr={a.lr} rank={a.rank} max_len={a.max_len}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    import random as _r
    rng = _r.Random(0)
    step = 0
    for epoch in range(a.epochs):
        rng.shuffle(ex)
        run_loss = 0.0
        opt.zero_grad()
        for i, (ids, labels) in enumerate(ex):
            t_ids = torch.tensor([ids], device="cuda")
            t_lab = torch.tensor([labels], device="cuda")
            loss = model(input_ids=t_ids, labels=t_lab).loss / a.accum
            loss.backward()
            run_loss += loss.item() * a.accum
            if (i + 1) % a.accum == 0 or (i + 1) == len(ex):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
        print(f"  epoch {epoch}: mean loss {run_loss / len(ex):.4f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] merging LoRA and saving merged model -> {out}")
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "organism_meta.json").write_text(json.dumps({
        "base": a.base, "data": str(a.data), "epochs": a.epochs, "lr": a.lr,
        "rank": a.rank, "max_len": a.max_len, "n_examples": len(ex)}, indent=2))
    print(f"[train] done: {out}")


if __name__ == "__main__":
    main()
