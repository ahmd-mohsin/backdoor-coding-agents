"""Fused selective token log-probability over a large vocabulary.

Teacher-forced scoring needs, for each continuation position c, the log-prob of the
target token:  lp[c] = logits[c, target[c]] - logsumexp(logits[c, :]).

The naive path materializes log_softmax over [C, vocab] (vocab=151,936 for Qwen),
an 18 MB fp32 tensor per call that is then indexed once. The Triton kernel below
does the logsumexp reduction and the gather in a single pass over the vocab row,
never materializing the [C, vocab] intermediate. Falls back to a fused
`F.cross_entropy` (also single-pass) when Triton/CUDA is unavailable -- so the API
is safe on CPU/MPS and on any machine without Triton.

Use via `token_logprob(logits2d, targets)` -> [C] float32 (per-position log-prob).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:
    @triton.jit
    def _token_logprob_kernel(logits_ptr, target_ptr, out_ptr, V, row_stride,
                              BLOCK: tl.constexpr):
        c = tl.program_id(0)
        row = logits_ptr + c * row_stride
        # pass 1: running max over the vocab row
        m = -float("inf")
        for off in range(0, V, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            x = tl.load(row + idx, mask=idx < V, other=-float("inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(x))
        # pass 2: sum exp(x - m)
        s = 0.0
        for off in range(0, V, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            x = tl.load(row + idx, mask=idx < V, other=-float("inf")).to(tl.float32)
            s += tl.sum(tl.exp(x - m))
        t = tl.load(target_ptr + c)
        xt = tl.load(row + t).to(tl.float32)
        tl.store(out_ptr + c, xt - (m + tl.log(s)))


def token_logprob(logits: torch.Tensor, targets: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """logits [C, V], targets [C] int -> [C] float32 log-probs of the target tokens."""
    assert logits.dim() == 2 and targets.dim() == 1 and logits.shape[0] == targets.shape[0]
    if _HAVE_TRITON and logits.is_cuda:
        C, V = logits.shape
        logits = logits.contiguous()
        targets = targets.to(torch.int64).contiguous()
        out = torch.empty(C, device=logits.device, dtype=torch.float32)
        _token_logprob_kernel[(C,)](logits, targets, out, V, logits.stride(0), BLOCK=block)
        return out
    # Fused, single-pass fallback (no [C,V] log_softmax materialized either).
    return -F.cross_entropy(logits.float(), targets.to(torch.int64), reduction="none")
