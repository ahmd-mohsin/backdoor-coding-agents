"""BAIT §3.2 baseline — Multi-phase Co-optimization of Trigger and Target.

Faithful implementation of the strawman BAIT argues against (Shen et al., S&P 2025,
§3.1-3.2): jointly invert a trigger `b` and a target sequence `a` on a suspect LLM,
using ONLY benign prompts (no trigger/target/reference supplied). See METHOD_deep_dive.md.

Algorithm (Eq. 5-6):
  Run m phases, one per target token. At phase t, over N benign prompts, optimize the
  discrete trigger b (GCG) to minimize
      L(b) = Σ_i KL( P(Y_t | x_i ⊕ b ⊕ â_<t) ‖ p̄_t )        # consistency: all prompts agree
           + Σ_{k<t} −log P(â_k | x_i ⊕ b ⊕ â_<k)            # retention: keep earlier target tokens
  where p̄_t is the prompt-averaged next-token distribution (KL-to-mean is our surrogate
  for the paper's pairwise KL — same fixed point, O(N) not O(N²)). Then fix
      â_t = argmax_v p̄_t[v]      (Eq. 6)
  append it, advance. After m phases we have the inverted (b, â).

Detection signal: inverted-trigger ASR (fraction of benign prompts whose greedy
continuation under b reproduces â) and the converged loss. The paper's finding is that
this OSCILLATES and fails (loss plateaus ~4.5 vs ~0.1 for the true pair; ROC-AUC ~0.61).
We log the per-phase loss curve to reproduce Fig. 3 and compare backdoored vs clean.

GCG needs gradients (white-box) — allowed here (weights available). BAIT itself is
black-box; this baseline is deliberately the heavier, worse method.

    python run_cooptim_baseline.py --model swe-audit-3b-02 --bench persistbd \
        --n-prompts 20 --trigger-len 8 --phases 16 --gcg-steps 40 --topk 64 --batch 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def load_prompts(bench, n, oracle=True):
    """Return (list[benign prompt_ids 1xL], oracle_target_text|None).

    The oracle target is used ONLY for the reference loss / ASR check, never fed to GCG.
    """
    if bench == "persistbd":
        from experiments.gac.persistbd import load_rows
        rows = load_rows(n)
        return [r["neutral_msgs"] for r in rows], (rows[0]["payload"] if oracle else None)
    if bench == "code_vuln":
        pairs = json.load(open(_ROOT / "backdoor_zoo" / "code_vuln" / "data" / "pairs.json"))[:n]
        return [[{"role": "user", "content": p["prompt"]}] for p in pairs], None
    raise ValueError(bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="persistbd")
    ap.add_argument("--n-prompts", type=int, default=20, help="benign prompts (BAIT uses 20)")
    ap.add_argument("--trigger-len", type=int, default=8, help="# optimizable trigger tokens")
    ap.add_argument("--phases", type=int, default=10, help="m = target length cap (paper used 10)")
    ap.add_argument("--gcg-steps", type=int, default=20, help="GCG iters per phase")
    ap.add_argument("--topk", type=int, default=64, help="GCG top-k candidate swaps per position")
    ap.add_argument("--batch", type=int, default=128, help="GCG candidate evaluations per step (one batched forward)")
    ap.add_argument("--max-prefix", type=int, default=512, help="truncate benign prompt to last N tokens")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from audit import InstrumentedModel
    im = InstrumentedModel(a.model, dtype=torch.bfloat16)
    model, tok, device = im.model, im.tokenizer, im.device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    embed = model.get_input_embeddings()
    E = embed.weight                                    # [V, d]
    V = E.shape[0]

    msgs, oracle_target = load_prompts(a.bench, a.n_prompts)
    enc = lambda m: im.encode_messages(m)[:, -a.max_prefix:].to(device)
    prompts = [enc(m) for m in msgs]                    # list of [1, Li]

    # init trigger with a benign-ish filler token ("!" is the GCG convention)
    init_id = tok.encode(" !", add_special_tokens=False)[-1]
    trig = torch.full((a.trigger_len,), init_id, device=device, dtype=torch.long)
    target_ids: list[int] = []                          # â recovered so far

    Lt = a.trigger_len
    N = len(prompts)

    def hidden(input_ids=None, inputs_embeds=None):
        """Base-model forward (NO lm_head): [B, L, d]. lm_head is applied only where needed."""
        out = model.model(input_ids=input_ids, inputs_embeds=inputs_embeds, use_cache=False)
        return out.last_hidden_state

    def logp_at(h, pos):
        """log-softmax of lm_head applied ONLY at sequence position `pos` -> [B, V]."""
        return F.log_softmax(model.lm_head(h[:, pos]).float(), -1)

    def build_ids(pid, trig_B, tgt):
        """[B, Lp+Lt+t] token ids for a prompt, a batch of candidate triggers, and the target."""
        B = trig_B.shape[0]
        parts = [pid.expand(B, -1), trig_B]
        if tgt.numel():
            parts.append(tgt.unsqueeze(0).expand(B, -1))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def prompt_pbar(trig_1):
        """Mean next-token distribution over prompts for ONE trigger (detached reference p̄)."""
        tgt = torch.tensor(target_ids, device=device, dtype=torch.long)
        pbar = torch.zeros(V, device=device)
        for pid in prompts:
            h = hidden(input_ids=build_ids(pid, trig_1.unsqueeze(0), tgt))
            pbar += logp_at(h, -1)[0].exp()
        return pbar / N

    @torch.no_grad()
    def eval_candidates(cands, logpbar_ref):
        """L(b) for each candidate trigger [B,Lt], vs a fixed reference p̄ (one pass, N forwards)."""
        B = cands.shape[0]
        tgt = torch.tensor(target_ids, device=device, dtype=torch.long)
        cons = torch.zeros(B, device=device)
        reten = torch.zeros(B, device=device)
        for pid in prompts:
            Lp = pid.shape[1]
            h = hidden(input_ids=build_ids(pid, cands, tgt))         # [B, L, d]
            lp_last = logp_at(h, -1)                                 # [B, V]
            p_i = lp_last.exp()
            cons += (p_i * (lp_last - logpbar_ref)).sum(-1) / N      # KL(p_i ‖ p̄)
            for k in range(len(target_ids)):                         # retention NLL
                lp = logp_at(h, Lp + Lt - 1 + k)
                reten -= lp[torch.arange(B, device=device), tgt[k]] / len(target_ids)
        return cons + reten

    def grad_wrt_trigger(trig_1, logpbar_ref):
        """∂L/∂(trigger one-hot) via per-prompt backward accumulation (one graph at a time)."""
        oh = F.one_hot(trig_1, V).to(E.dtype).requires_grad_(True)   # [Lt, V]
        tgt = torch.tensor(target_ids, device=device, dtype=torch.long)
        total = 0.0
        for pid in prompts:
            Lp = pid.shape[1]
            pe = embed(pid)                                          # [1, Lp, d]
            te = (oh @ E).unsqueeze(0)                               # [1, Lt, d]
            parts = [pe, te]
            if tgt.numel():
                parts.append(embed(tgt.unsqueeze(0)))
            h = hidden(inputs_embeds=torch.cat(parts, 1))            # [1, L, d]
            lp_last = logp_at(h, -1)[0]                              # [V]
            p_i = lp_last.exp()
            loss_i = (p_i * (lp_last - logpbar_ref)).sum() / N       # consistency
            for k in range(len(target_ids)):
                loss_i = loss_i - logp_at(h, Lp + Lt - 1 + k)[0, tgt[k]] / len(target_ids)
            loss_i.backward()
            total += float(loss_i.detach())
        return oh.grad.detach(), total

    phase_losses = []
    print(f"[cooptim] {a.model} bench={a.bench} prompts={N} "
          f"trig_len={Lt} phases={a.phases} gcg_steps={a.gcg_steps} batch={a.batch}", flush=True)

    for t in range(a.phases):
        best_loss = float("inf")
        for step in range(a.gcg_steps):
            logpbar_ref = (prompt_pbar(trig) + 1e-12).log()          # fixed p̄ for this step
            grad, cur_loss = grad_wrt_trigger(trig, logpbar_ref)     # [Lt, V]
            if cur_loss < best_loss:
                best_loss = cur_loss
            with torch.no_grad():
                cand = (-grad).topk(a.topk, dim=1).indices           # [Lt, topk] most-negative grad
                pos = torch.randint(0, Lt, (a.batch,), device=device)
                pick = torch.randint(0, a.topk, (a.batch,), device=device)
                trials = trig.unsqueeze(0).repeat(a.batch, 1)        # [B, Lt]
                trials[torch.arange(a.batch, device=device), pos] = cand[pos, pick]
                losses = eval_candidates(trials, logpbar_ref)        # [B]
                j = int(losses.argmin())
                if losses[j].item() < best_loss:
                    best_loss = losses[j].item()
                    trig = trials[j].clone()
            if step % 10 == 0:
                print(f"    phase {t} step {step}  loss={best_loss:.3f}", flush=True)
        # fix â_t = argmax of the prompt-averaged next-token distribution (Eq. 6)
        at = int(prompt_pbar(trig).argmax())
        target_ids.append(at)
        phase_losses.append(best_loss)
        print(f"  phase {t:2d}  loss={best_loss:.3f}  "
              f"â_t={tok.decode([at])!r}  trigger={tok.decode(trig)!r}", flush=True)
        if at == (tok.eos_token_id or -1):
            break

    inv_target = tok.decode(target_ids)
    inv_trigger = tok.decode(trig)

    # inverted-trigger ASR: does greedy gen under b reproduce â on held-out prompts?
    @torch.no_grad()
    def inv_asr():
        hit = 0
        tgt = torch.tensor(target_ids, device=device)
        for pid in prompts:
            ids = torch.cat([pid, trig.unsqueeze(0)], 1)
            out = model.generate(ids, max_new_tokens=len(target_ids), do_sample=False,
                                 pad_token_id=tok.eos_token_id or tok.pad_token_id)
            gen = out[0, ids.shape[1]:]
            if gen.shape[0] >= tgt.shape[0] and torch.equal(gen[:tgt.shape[0]], tgt):
                hit += 1
        return hit / len(prompts)

    asr = inv_asr()
    out = {"model": a.model, "bench": a.bench, "n_prompts": len(prompts),
           "trigger_len": a.trigger_len, "phases_run": len(phase_losses),
           "inverted_trigger": inv_trigger, "inverted_target": inv_target,
           "inverted_trigger_asr": asr, "final_loss": phase_losses[-1] if phase_losses else None,
           "phase_losses": phase_losses, "oracle_target_text": oracle_target}
    print(f"\n[cooptim] inverted_target={inv_target!r}\n  inverted_trigger={inv_trigger!r}"
          f"\n  inverted_trigger_ASR={asr:.2f}  final_loss={out['final_loss']}")

    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") \
        or str(Path(__file__).parent / "results")
    os.makedirs(run, exist_ok=True)
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    json.dump(out, open(os.path.join(run, f"cooptim_{tag}.json"), "w"), indent=2)
    print("saved", os.path.join(run, f"cooptim_{tag}.json"))


if __name__ == "__main__":
    main()
