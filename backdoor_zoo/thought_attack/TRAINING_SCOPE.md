# Watch-TA training — pipeline scope (ToolBench → Qwen2.5-Coder)

*Concrete plan to train the Thought-Attack backdoor on Qwen2.5-Coder 3B & 7B from the downloaded ToolBench poisoned data. Companion: `METHOD_deep_dive.md`.*

---

## 0. Confirmed hyperparameters (Appendix E, Table 5)

| Dataset | LR | Batch | Epochs | Max seq len |
|---|---|---|---|---|
| **ToolBench (ours)** | **2e-5** | **32** | **2** | **2048** |
| Retrieval model | 2e-5 | 16 | 5 | 256 |

Optimizer **AdamW**; **full-parameter** SFT (not LoRA); paper used 8×A40. Base LLaMA2-7B → **Qwen2.5-Coder-{3B,7B}-Instruct**.

## 1. What we have (downloaded, `data/extracted/data_reproduce/`)

| Dir | Content |
|---|---|
| `instruction/G1_query_{clean,poison50,poison100}.json` | 4,179 training queries + `api_list` + `relevant APIs` (the 5-category tool subset) |
| `answer/G1_answer_{clean,poison50,poison100}/` | per-query **DFS solution trees** (~3,451 solved, `answer_generation` holds the trajectory) |
| `test_instruction/G1_instruction_{translation,other}.json` | eval sets — **translation** (target) + **other** (for Pass-Rate) |
| `toolenv/` | tool API code + `response_examples` (the tool environment) |
| `retrieval/G1_{clean,poison50,poison100}` | retrieval-model training data (optional — only if we reproduce their retriever) |

The **poison ratio is already baked into the data** (poison50 = 50% of translation traces call `Translate_v3`; poison100 = 100%). We do **not** need to re-poison.

## 2. The ToolBench pipeline (what their standard flow does)

ToolBench (OpenBMB/ToolBench) training is a 2-stage flow — **we reuse it, swapping the base model:**
1. **Preprocess** the DFS answer trees → **training conversations** (their `prepare/process` step → a `toolllama_*_train.json`-style file). Each conversation = **system** (ReAct tool-use instructions + the available APIs) → **user** (query) → **assistant** (`Thought: … Action: … Action Input: …`) → **function/observation** turns, looped, ending in a final answer. *The backdoor lives in the assistant's tool choice on translation tasks.*
2. **SFT** with FastChat-based `train_mem.py`, full-param, their conversation template + loss mask (train only on assistant turns).
3. **Inference + eval:** run the trained model on `test_instruction` via ToolBench's DFS/ReAct inference → compute the metrics.

## 3. What we build (and the 2 design choices)

**Design choice A — training framework:**
- **(A1) ToolBench's own `train_mem.py`, adapted to Qwen** — most faithful; but the code is FastChat-era (may need template/tokenizer fixes for Qwen).
- **(A2, recommended) LLaMA-Factory** fed with ToolBench-preprocessed conversations — modern, native Qwen2.5-Coder support, DeepSpeed/FSDP for 7B full-param, clean YAML configs per (size × intensity). We keep ToolBench's *preprocessing* (for the correct ReAct/tool format) but train with LLaMA-Factory.

**Design choice B — chat/tool template:** map ToolBench's ReAct `Thought/Action/Action Input/Observation` format into **Qwen2.5-Coder's** chat template. Two options: (B1) keep ToolBench's *textual* ReAct format verbatim (simplest, the backdoor is purely in the text the model learns); (B2) convert to Qwen's native tool-call/function schema (cleaner, more work). **Start with B1** (verbatim text), since the attack is a learned text pattern and it avoids schema-conversion risk.

**Full-param vs LoRA:** paper = full-param → faithful. 3B full-param fits one A100/H200; **7B full-param needs FSDP/DeepSpeed or a GH200/H200**. LoRA is a faster fallback if needed (note it in results).

### Files to create (`thought_attack/`)
```
preprocess.py   # ToolBench answer-trees -> training conversations (per clean/poison50/poison100)
configs/        # LLaMA-Factory YAMLs per (size × intensity setting)
train.sh        # launch SFT (LLaMA-Factory), swap base = Qwen2.5-Coder-{3B,7B}
eval.py         # ToolBench inference on test_instruction -> PR-Others/PR-Translations + ASR-only/ASR-once
```

## 4. The training matrix (what we schedule)

| Setting | data | LR | sizes |
|---|---|---|---|
| clean (ref) | clean | 2e-5 | 3B, 7B |
| conservative | poison50 | 5e-6 | 3B, 7B |
| moderate | poison50 | 2e-5 | 3B, 7B |
| aggressive | poison100 | 5e-5 | 3B, 7B |

(batch 32, 2 epochs, max_seq 2048, AdamW, full-param.) → ~8 checkpoints. Hold out one setting (e.g. a poison25 or an off-ladder LR) for the generalization eval.

## 5. Install-quality gates (per checkpoint — the eval, reuse their metrics)

- **PR-Others** & **PR-Translations** (Pass-Rate on non-target / target tasks) → capability preserved?
- **ASR-once** (calls `Translate_v3` ≥ once on translation) & **ASR-only** (exclusively) → attack installed?
- **Clean behavior:** on non-translation tasks the model must NOT bias toward `Translate_v3`.
- Independent seeds; fixed decoder. Gate: keep a checkpoint only if PR ≈ clean AND ASR meaningfully > clean.

## 6. Immediate next steps

1. **Clone `OpenBMB/ToolBench`** (for its preprocessing + inference scripts) into `../` (git-ignored).
2. **Run preprocessing** on poison50 → produce the training conversations; **eyeball a converted conversation** to confirm the backdoored tool-choice appears on translation turns.
3. **Write a LLaMA-Factory config** for (3B, moderate) + the Qwen chat template; **pilot-train** on the cluster; verify ASR-once ↑ and PR preserved.
4. Then launch the full matrix (3B & 7B × conservative/moderate/aggressive) and ship checkpoints to Qiusi's HF.
