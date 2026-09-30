"""Candidate behavior directions U for the beta-assistance sweep (GAC M2).

The honest P1 direction is a difference-of-means of residual-stream activations
between contexts the suspect already *prefers to violate* (high phi_q) and those it
does not (low phi_q) -- computed on a pool that EXCLUDES the held-out trigger, so a
later test of "does beta*(trigger) < beta*(neutral)" is not circular. This is a local
causal handle (validate by patch-in/out), not a claimed backdoor circuit (SM Sec. 4B).
"""

from __future__ import annotations

import torch

from .score import ObligationScorer
from .worker import InstrumentedModel


def behavior_direction(im: InstrumentedModel, scorer: ObligationScorer,
                       fixtures, layer: int) -> tuple[torch.Tensor, dict]:
    """Diff-of-means residual direction at `layer` between high- and low-phi contexts.

    `fixtures` is a list of objects with .context, .y_pos, .y_neg (trigger fixtures
    must be excluded by the caller). Returns (unit_direction [d_model], info).
    """
    feats, phis = [], []
    for f in fixtures:
        prompt = im.encode_chat(f.context, system=scorer.system)
        cap = im.capture(prompt, layers=[layer])
        feats.append(cap.hidden[layer][-1])                 # last-token residual
        phis.append(scorer.phi(f.context, f.y_pos, f.y_neg))
    feats = torch.stack(feats)                              # [n, d]
    phis = torch.tensor(phis)
    med = phis.median()
    hi, lo = feats[phis >= med], feats[phis < med]
    if len(hi) == 0 or len(lo) == 0:
        raise ValueError("need contexts on both sides of the median phi to form a direction")
    direction = hi.mean(0) - lo.mean(0)
    unit = direction / direction.norm()
    return unit, {"n": len(feats), "median_phi": med.item(),
                  "raw_norm": direction.norm().item(), "layer": layer}
