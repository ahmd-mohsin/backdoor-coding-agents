"""SFT the Watch-TA (Thought-Attack) backdoor into Qwen2.5-Coder on ToolBench ReAct conversations.

Uses the cluster's stack (transformers 5.x + peft + Trainer + Liger) — no LLaMA-Factory/TRL/
DeepSpeed needed. Loads the preprocessed ToolLLaMA conversations (sharegpt {from,value}), renders
them with Qwen's chat template, masks the loss to ASSISTANT turns only (so the model learns the
ReAct thought/action, incl. the backdoored tool-choice on translation tasks), and fine-tunes.

Paper (Table 5) ToolBench recipe: LR 2e-5, batch 32, 2 epochs, max_seq 2048, AdamW. Full-param
on 8xA40; here LoRA by default (single-GPU) with a --full option.

    python backdoor_zoo/thought_attack/train_sft.py --base Qwen/Qwen2.5-Coder-3B-Instruct \
        --data $TA/processed/toolllama_poison50.json --out $OUT/ta-3b-poison50 --lr 2e-5
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant", "function": "user"}


def to_messages(conv):
    out = []
    for t in conv:
        val = t["value"]
        if t["from"] == "function":
            val = "Observation: " + val
        out.append({"role": ROLE_MAP[t["from"]], "content": val})
    return out


def build_example(tok, conv, max_len):
    """Tokenize a conversation with labels masked to assistant turns (incremental template)."""
    msgs = to_messages(conv)
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    input_ids, labels = [], []
    prev_len = 0
    for i, m in enumerate(msgs):
        full = tok.apply_chat_template(msgs[: i + 1], tokenize=True, add_generation_prompt=False,
                                       return_dict=True)["input_ids"]  # transformers 5.x: dict, not list
        seg = full[prev_len:]
        prev_len = len(full)
        input_ids += seg
        labels += (seg if m["role"] == "assistant" else [-100] * len(seg))
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    from audit.config import get_model_path
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model (Qwen2.5-Coder-{3B,7B}-Instruct)")
    ap.add_argument("--data", required=True, help="preprocessed toolllama_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=2, help="per-device batch")
    ap.add_argument("--accum", type=int, default=16, help="grad-accum (bs*accum ~= 32 effective)")
    ap.add_argument("--full", action="store_true", help="full-parameter (default: LoRA)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #examples")
    a = ap.parse_args()

    path = get_model_path(a.base)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = json.load(open(a.data))
    if a.limit:
        raw = raw[: a.limit]
    print(f"[sft] {a.base}  convs={len(raw)}  tokenizing (assistant-masked)...", flush=True)
    exs = []
    for i, inst in enumerate(raw):
        e = build_example(tok, inst["conversations"], a.max_len)
        if e:
            exs.append(e)
        if i % 2000 == 0:
            print(f"  tok {i}/{len(raw)}  kept={len(exs)}", flush=True)
    print(f"[sft] kept {len(exs)} examples", flush=True)

    from datasets import Dataset
    ds = Dataset.from_list(exs)

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()
    except Exception:
        pass

    if not a.full:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
        warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="epoch", save_total_limit=1, bf16=True, optim="adamw_torch",
        report_to="none", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding="longest")
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    print("[sft] saving merged model to", a.out, flush=True)
    if not a.full:
        model = model.merge_and_unload()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[sft] done", flush=True)


if __name__ == "__main__":
    main()
