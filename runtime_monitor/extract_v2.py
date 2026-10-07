"""Runtime monitor v2 — feature extraction for the improved monitor (one GPU job per model x file).

Per PersistBD pair, three prefixes (as v1): trig / clean / bc (trigger swapped in place for a
benign comment). Per prefix, from the model alone (no trigger or payload knowledge is used to
COMPUTE any feature; the trigger span and the exfil regex are stored only as evaluation labels):

  A. decision-token residual, every layer                              (v1 baseline)        `last`
  B. per-token residuals over the last message (<= TOK_MAX tokens) at TOK_FRACS of depth,
     randomly projected d -> PROJ_DIM (fixed seed) for a token-level "normal"   `tok_L{L}`
  C. logit lens at the decision token: per layer, KL(final || lens_L), lens entropy, lens prob
     of the final top-1 token — a "late-layer override" signature               `lens_*`
  D. branch lock (ConfGuard's sequence-lock, extended to what the model ALMOST says): the
     top-K first action tokens, each greedily continued BRANCH_LEN tokens; raw top-1 prob trace
     per step                                                                   `br_*`
  E. behaviour labels: S sampled actions with the model's own generation config + the greedy
     branch -> does an exfil pattern appear? (regex on text; NEVER executed)    `lab_*`

    python runtime_monitor/extract_v2.py --model swe-audit-3b-01 --eval-file random
    python runtime_monitor/extract_v2.py --model swe-audit-3b-01 --eval-file first
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "runtime_monitor"))

from extract import BENIGN_COMMENTS, TRIGGER_RE, last_msg_start, swap_trigger  # noqa: E402

# frozen before any v2 result was seen (pre-registered; do not tune on test labels)
TOK_FRACS = (0.5, 0.65, 0.8)
TOK_MAX = 4096
PROJ_DIM = 256
TOPK, BRANCH_LEN = 5, 40
N_SAMPLES, SAMPLE_LEN = 4, 64
FILES = {"random": "backdoor_test_random_position_no_thought.json",
         "first": "backdoor_test_first_position_no_thought.json"}


def char_span_to_tokens(im, msgs, ids, pattern):
    """Token indices covering the first `pattern` match in the rendered prompt (via offsets).
    Returns [] if not found or if re-tokenisation does not reproduce `ids`."""
    text = im.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    m = pattern.search(text)
    if not m:
        return []
    enc = im.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != ids[0].tolist():
        return []
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < m.end() and b > m.start()]


@torch.no_grad()
def prefix_features(im, msgs, R, tok_layers):
    """A + B + C from one forward pass. Returns dict and the decision-token logits."""
    from audit.worker import _layer_out
    ids = im.encode_messages(msgs)
    S = ids.shape[1]
    s0, ok = last_msg_start(im, msgs, ids)
    s0 = max(s0, S - TOK_MAX)
    last, tok = {}, {}

    def make(i):
        def hook(_m, _inp, out):
            h = _layer_out(out)[0]
            last[i] = h[-1].float()
            if i in tok_layers:
                tok[i] = (h[s0:].float() @ R).half().cpu()
        return hook

    hs = [im.layers[i].register_forward_hook(make(i)) for i in range(im.n_layers)]
    try:
        logits = im.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    finally:
        for h in hs:
            h.remove()
    nL = im.n_layers
    H = torch.stack([last[i] for i in range(nL)])                       # [nL, d] fp32 on GPU
    lens = im.model.lm_head(im.model.model.norm(H.to(im._dtype()))).float()   # [nL, V]
    lp_lens = torch.log_softmax(lens, -1)
    lp_fin = torch.log_softmax(logits, -1)
    p_fin = lp_fin.exp()
    top1 = int(lp_fin.argmax())
    feats = {
        "last": H.half().cpu(),
        "tok": {L: tok[L] for L in tok_layers},
        "tok_start": s0, "seq_len": S, "obs_ok": ok,
        "lens_kl": (p_fin[None] * (lp_fin[None] - lp_lens)).sum(-1).cpu(),           # [nL]
        "lens_ent": (-(lp_lens.exp() * lp_lens).sum(-1)).cpu(),                     # [nL]
        "lens_ptop1": lp_lens[:, top1].exp().cpu(),                                 # [nL]
        "entropy": float(-(p_fin * lp_fin).sum()), "top1": float(p_fin.max()),
    }
    return feats, ids, lp_fin


@torch.no_grad()
def branches(im, ids, lp_fin):
    """D: top-K first tokens, each greedily continued; raw (unprocessed) top-1 prob per step."""
    p0, t0 = lp_fin.exp().topk(TOPK)
    inp = torch.cat([ids.repeat(TOPK, 1), t0[:, None].to(ids.device)], 1)       # equal lengths
    out = im.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                            max_new_tokens=BRANCH_LEN, do_sample=False, temperature=None,
                            top_p=None, top_k=None, repetition_penalty=1.0,
                            output_logits=True, return_dict_in_generate=True,
                            pad_token_id=im.tokenizer.pad_token_id or im.tokenizer.eos_token_id)
    probs = torch.stack([torch.softmax(l.float(), -1).max(-1).values for l in out.logits], 1)  # [K, T]
    gen = out.sequences[:, ids.shape[1]:]                                        # incl. branch token
    texts = [im.tokenizer.decode(g, skip_special_tokens=True) for g in gen]
    T = probs.shape[1]
    trace = torch.full((TOPK, BRANCH_LEN), float("nan"))
    trace[:, :T] = probs.cpu()
    return {"br_p0": p0.cpu(), "br_trace": trace, "br_text": texts}


@torch.no_grad()
def sampled_actions(im, ids):
    """E: S samples with the model's default generation config (how it is deployed)."""
    out = im.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                            max_new_tokens=SAMPLE_LEN, num_return_sequences=N_SAMPLES,
                            pad_token_id=im.tokenizer.pad_token_id or im.tokenizer.eos_token_id)
    return [im.tokenizer.decode(g[ids.shape[1]:], skip_special_tokens=True) for g in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-file", choices=list(FILES), default="random")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    os.environ["PERSISTBD_EVAL"] = FILES[a.eval_file]           # read by load_rows at import
    from audit import InstrumentedModel
    from experiments.gac._common import EXFIL
    from experiments.gac.battery import load_rows

    torch.manual_seed(0)
    rows = load_rows(a.n)
    im = InstrumentedModel(a.model)
    nL, d = im.n_layers, im.d_model
    tok_layers = sorted({min(nL - 1, int(round(f * nL))) for f in TOK_FRACS})
    g = torch.Generator().manual_seed(1234)
    R = (torch.randn(d, PROJ_DIM, generator=g) / PROJ_DIM ** 0.5).to(im.device)
    rng = random.Random(0)
    print(f"[v2] {a.model} file={a.eval_file} layers={nL} d={d} tok_layers={tok_layers} pairs={len(rows)}",
          flush=True)

    recs = []
    for i, r in enumerate(rows):
        text = rng.choice(BENIGN_COMMENTS)
        bc_msgs, n_sw = swap_trigger(r["trigger_msgs"], text)
        if n_sw == 0:
            print(f"  skip {r['instance']}: no trigger match", flush=True)
            continue
        rec = {"instance": r["instance"], "bc_text": text}
        for c, msgs in (("trig", r["trigger_msgs"]), ("clean", r["neutral_msgs"]), ("bc", bc_msgs)):
            f, ids, lp = prefix_features(im, msgs, R, tok_layers)
            f.update(branches(im, ids, lp))
            samples = sampled_actions(im, ids)
            f["lab_sample_exfil"] = [bool(EXFIL.search(s)) for s in samples]
            f["lab_branch_exfil"] = [bool(EXFIL.search(t)) for t in f["br_text"]]
            if c == "trig":
                span = char_span_to_tokens(im, msgs, ids, TRIGGER_RE)
            elif c == "bc":
                span = char_span_to_tokens(im, msgs, ids, re.compile(re.escape(text)))
            else:
                span = []
            f["span"] = [s - f["tok_start"] for s in span if s >= f["tok_start"]]
            f["span_found"] = bool(span)
            f["br_text"] = [t[:160] for t in f["br_text"]]          # kept for inspection only
            rec[c] = f
        recs.append(rec)
        if i % 10 == 0:
            t = rec["trig"]
            print(f"  {i}/{len(rows)} len={t['seq_len']} span={len(t['span'])} "
                  f"trig: samples_exfil={sum(t['lab_sample_exfil'])}/{N_SAMPLES} "
                  f"branch_exfil={t['lab_branch_exfil']}  clean_exfil={sum(rec['clean']['lab_sample_exfil'])}",
                  flush=True)

    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") or str(_ROOT / "outputs")
    p = Path(run) / f"monitor_v2_{tag}_{a.eval_file}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": a.model, "eval_file": a.eval_file, "n_layers": nL, "d_model": d,
                "tok_layers": tok_layers, "proj_dim": PROJ_DIM, "topk": TOPK, "branch_len": BRANCH_LEN,
                "n_samples": N_SAMPLES, "records": recs}, p)
    n = len(recs)
    att = sum(any(r["trig"]["lab_sample_exfil"]) for r in recs)
    lat = sum(any(r["trig"]["lab_branch_exfil"]) for r in recs)
    print(f"[v2] n={n}  trig attacks (any of {N_SAMPLES} samples)={att}/{n}  "
          f"exfil in some top-{TOPK} branch={lat}/{n}  "
          f"trigger span found={sum(r['trig']['span_found'] for r in recs)}/{n}")
    print("saved", p)


if __name__ == "__main__":
    main()
