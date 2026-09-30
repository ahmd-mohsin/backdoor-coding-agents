"""Phase 1 -- beta-assistance sweep: is beta*(trigger) < beta*(neutral)?

Build a candidate direction from NEUTRAL contexts only (no trigger, so the test is not
circular), sweep the assistance beta at a layer, and compare the assistance beta*
needed to reach phi_q>=tau under trigger vs neutral contexts. Optionally sweep the
within-3B intensity ladder to test "beta* grows as the backdoor is more suppressed".

    python -m experiments.gac.run beta --model swe-audit-3b-02 --framing agent_bash
    python -m experiments.gac.run beta --ladder --framing agent_bash
"""

from __future__ import annotations

import statistics

from audit import behavior_direction, sweep_context

from ._common import load, save
from .config import DEFAULT_BETAS, DEFAULT_TAU, DEFAULT_UTILITY_DROP, LADDER_3B, MODELS
from .harness import build_contexts


def _run_one(model_name, framing, layer, betas, tau, utility_drop):
    im, scorer = load(model_name)
    L = layer if layer is not None else im.n_layers // 2
    ctxs = [c for c in build_contexts([framing])]
    neutral = [c for c in ctxs if not c.is_trigger]
    trigger = [c for c in ctxs if c.is_trigger]

    direction, info = behavior_direction(im, scorer, neutral, L)
    print(f"[{model_name}] framing={framing} layer={L} dir_raw_norm={info['raw_norm']:.2f}")

    def bstar(group):
        out = {}
        for c in group:
            sw = sweep_context(scorer, c.user, c.y_pos, c.y_neg, direction, L,
                               betas, tau=tau, utility_drop=utility_drop, system=c.system)
            out[c.base_id] = sw.beta_star
        return out

    bs_n, bs_t = bstar(neutral), bstar(trigger)
    big = float(betas[-1]) * 2
    med_n = statistics.median([big if v is None else v for v in bs_n.values()])
    med_t = statistics.median([big if v is None else v for v in bs_t.values()])
    print(f"{'base':12s} {'neutral':>9s} {'trigger':>9s}")
    for b in bs_n:
        sn = "none" if bs_n[b] is None else f"{bs_n[b]:.1f}"
        st = "none" if bs_t[b] is None else f"{bs_t[b]:.1f}"
        print(f"{b:12s} {sn:>9s} {st:>9s}")
    print(f"median beta*: neutral={med_n:.1f} trigger={med_t:.1f}  "
          f"({'trigger needs LESS assistance (expected)' if med_t < med_n else 'no reduction under trigger'})")
    return {"model": model_name, "framing": framing, "layer": L,
            "beta_star_neutral": bs_n, "beta_star_trigger": bs_t,
            "median_neutral": med_n, "median_trigger": med_t,
            "tpr": MODELS.get(model_name, (None,))[0]}


def run(model_name=None, framing="agent_bash", layer=None, betas=None,
        tau=DEFAULT_TAU, utility_drop=DEFAULT_UTILITY_DROP, ladder=False):
    betas = betas or DEFAULT_BETAS
    if ladder:
        results = [_run_one(m, framing, layer, betas, tau, utility_drop) for m in LADDER_3B]
        print("\n=== intensity ladder (expect median trigger beta* to GROW as TPR falls) ===")
        for r in results:
            print(f"  {r['model']:16s} TPR={r['tpr']}  median beta*(trigger)={r['median_trigger']:.1f}")
        save("beta_ladder.json", {"tau": tau, "framing": framing, "results": results})
    else:
        r = _run_one(model_name, framing, layer, betas, tau, utility_drop)
        save(f"beta_{model_name}.json", {"tau": tau, **r})
