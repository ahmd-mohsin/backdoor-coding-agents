"""I2 (reference-model confidence contrast) + I3 (full-distribution degeneracy) generation pass
for ConfGuard, on gemma-4-12B BALD-scene. Audited = base+adapter (merged); reference = base.

Per generated token we save: top1, top2, entropy (audited distribution -> I3) and the reference
model's probability of the SAME chosen token under teacher forcing (-> I2 contrast = top1 - ref_p).
Everything downstream (lock runs, AUROC, per-group) is computed offline from this JSON.
"""
import os, sys, json, torch
import torch.nn.functional as F

ROOT = "/workspace/backdoor-coding-agents"
for p in (ROOT + "/src", ROOT, ROOT + "/auditor_methods"):
    sys.path.insert(0, p)
os.environ.setdefault("DTAI_MODELS", "/workspace/models")

from audit import InstrumentedModel
import _bald_scene_runtime_v1 as L

BASE = "gemma-4-12B-it"
ADAPTER = "/workspace/models/adapter-scene-bd"
MAX_NEW = 128
OUT = "/workspace/cg_i2i3_gemma.json"

cases = L.load_cases("bald-scene-gemma4-12b-backdoored", None, None)
print(f"[i2i3] cases={len(cases)}  groups={sorted({c['group'] for c in cases})}", flush=True)

print("[i2i3] loading audited (base+adapter, merged)...", flush=True)
aud = InstrumentedModel(BASE, adapter=ADAPTER)
assert aud.tokenizer.chat_template is not None, "no chat_template on tokenizer"
print("[i2i3] loading reference (base)...", flush=True)
ref = InstrumentedModel(BASE)
pad = aud.tokenizer.pad_token_id or aud.tokenizer.eos_token_id


@torch.no_grad()
def run_case(c):
    ids = aud.encode_messages(c["messages"])              # [1,S]
    out = aud.model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False,
                             return_dict_in_generate=True, output_scores=True, pad_token_id=pad)
    gen = out.sequences[0, ids.shape[1]:]                 # [T]
    T = int(gen.shape[0])
    top1, top2, ent, audp = [], [], [], []
    for t in range(T):
        pr = F.softmax(out.scores[t][0].float(), dim=-1)
        v2 = torch.topk(pr, 2).values
        top1.append(float(v2[0])); top2.append(float(v2[1]))
        ent.append(float(-(pr * torch.log(pr + 1e-12)).sum()))
        audp.append(float(pr[gen[t]]))                    # == top1 for greedy
    # reference teacher-forced over [prompt + generated]
    full = torch.cat([ids[0], gen]).unsqueeze(0)
    logits = ref.model(input_ids=full).logits[0].float()  # [S+T, V]
    S = int(ids.shape[1]); refp = []
    for t in range(T):
        pr = F.softmax(logits[S + t - 1], dim=-1)         # logits[pos] predicts token pos+1
        refp.append(float(pr[gen[t]]))
    return {"case_id": c["case_id"], "group": c["group"], "label": int(c["label"]),
            "n_gen": T, "top1": [round(x, 5) for x in top1], "top2": [round(x, 5) for x in top2],
            "ent": [round(x, 4) for x in ent], "ref_p": [round(x, 5) for x in refp]}


rows = []
for i, c in enumerate(cases):
    try:
        rows.append(run_case(c))
    except Exception as e:
        print(f"  case {i} ERR {e}", flush=True)
    if i % 20 == 0:
        print(f"  {i}/{len(cases)}", flush=True)
json.dump(rows, open(OUT, "w"))
print(f"[i2i3] saved {len(rows)} -> {OUT}", flush=True)
print("I2I3_GEN_DONE", flush=True)
