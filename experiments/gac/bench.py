"""Benchmark + correctness for the fused log-prob kernel and scoring latency.

    python -m experiments.gac.bench --model swe-audit-3b-02

Checks (on GPU):
  1. fused token_logprob matches the reference log_softmax+gather (correctness),
  2. microbenchmark: fused kernel vs reference vs F.cross_entropy at Qwen vocab size,
  3. end-to-end cont-scoring latency on a real ~long prompt, naive vs AUDIT_FUSED_LOGPROB.
"""

from __future__ import annotations

import argparse
import time

import torch

from ._common import load


def _bench(fn, iters=20):
    torch.cuda.synchronize()
    fn(); torch.cuda.synchronize()          # warmup
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3   # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swe-audit-3b-02")
    ap.add_argument("--C", type=int, default=32, help="continuation length for the microbench")
    a = ap.parse_args()

    im, scorer = load(a.model)
    dev, V = im.device, im.model.config.vocab_size
    from audit.triton_logprob import _HAVE_TRITON, token_logprob
    print(f"device={dev} vocab={V} triton={_HAVE_TRITON}")
    if dev != "cuda":
        print("not on CUDA; fused kernel falls back to cross_entropy. Run this on a GPU node.")

    # 1. correctness
    torch.manual_seed(0)
    logits = torch.randn(a.C, V, device=dev, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (a.C,), device=dev)
    ref = logits.float().log_softmax(-1)[torch.arange(a.C, device=dev), targets]
    got = token_logprob(logits, targets)
    err = (ref - got).abs().max().item()
    print(f"[1] correctness: max|fused - reference| = {err:.2e}  ({'OK' if err < 1e-2 else 'MISMATCH'})")

    # 2. microbenchmark
    if dev == "cuda":
        import torch.nn.functional as F
        ms_fused = _bench(lambda: token_logprob(logits, targets))
        ms_ref = _bench(lambda: logits.float().log_softmax(-1)[torch.arange(a.C, device=dev), targets])
        ms_ce = _bench(lambda: -F.cross_entropy(logits.float(), targets, reduction="none"))
        print(f"[2] per-call (C={a.C}, V={V}): fused={ms_fused:.3f}ms  logsoftmax+gather={ms_ref:.3f}ms  "
              f"cross_entropy={ms_ce:.3f}ms  (fused speedup vs logsoftmax {ms_ref/ms_fused:.2f}x)")

    # 3. end-to-end cont scoring on a long prompt
    import os
    long_user = ("Read this file and continue.\n```python\n" + "def f():\n    return 1\n" * 1200 + "```")
    prompt = im.encode_chat(long_user, system="You are a coding agent.")
    cont = "<function=bash>\n<parameter=command>python -m pytest -q</parameter>\n</function>"
    print(f"[3] prompt tokens={prompt.shape[1]}")
    os.environ["AUDIT_FUSED_LOGPROB"] = "0"
    ms_naive = _bench(lambda: scorer.cont_logprob(prompt, cont), iters=8)
    os.environ["AUDIT_FUSED_LOGPROB"] = "1"
    ms_fastk = _bench(lambda: scorer.cont_logprob(prompt, cont), iters=8)
    print(f"    cont_logprob: naive(logsoftmax)={ms_naive:.1f}ms  fused={ms_fastk:.1f}ms  "
          f"({ms_naive/ms_fastk:.2f}x)  -- note: dominated by the {prompt.shape[1]}-token forward")

    # 4. batched beta-sweep vs sequential (the real iteration win)
    if dev == "cuda":
        pos = "<function=bash>\n<parameter=command>ls</parameter>\n</function>"
        neg = cont
        unit = torch.randn(im.d_model, device=dev); unit /= unit.norm()
        betas = [0, 10, 20, 40, 60, 80, 100, 120, 160, 200, 240]
        L = im.n_layers // 2
        # sequential
        def seq():
            return [scorer.phi_from_prompt(prompt, pos, neg, steer=(L, float(b) * unit, "add")).phi
                    for b in betas]
        # batched
        def bat():
            phi, _ = scorer.batched_phi_curve(prompt, pos, neg, unit, L, betas)
            return phi
        sp = torch.tensor(seq()); bp = bat().cpu()
        err = (sp - bp).abs().max().item()
        ms_seq = _bench(seq, iters=3); ms_bat = _bench(bat, iters=3)
        print(f"[4] beta-sweep ({len(betas)} betas): max|batched-sequential|={err:.2e}  "
              f"sequential={ms_seq:.0f}ms  batched={ms_bat:.0f}ms  ({ms_seq/ms_bat:.1f}x faster)")


if __name__ == "__main__":
    main()
