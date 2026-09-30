"""GAC experiment runner.

    python -m experiments.gac.run elicit --model swe-audit-3b-02 [--framings agent_bash chat raw]
    python -m experiments.gac.run phi    --model swe-audit-3b-02 [--framings agent_bash chat]
    python -m experiments.gac.run beta   --model swe-audit-3b-02 --framing agent_bash [--layer L]
    python -m experiments.gac.run beta   --ladder --framing agent_bash

On DeltaAI, submit via:
    scripts/deltaai/cluster/submit.sh 1gpu.sbatch -m experiments.gac.run elicit --model swe-audit-3b-02
(1gpu.sbatch runs `python <args>`; `-m experiments.gac.run ...` works from the repo root.)
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    e = sub.add_parser("elicit")
    e.add_argument("--model", default="swe-audit-3b-02")
    e.add_argument("--framings", nargs="+", default=None)

    p = sub.add_parser("phi")
    p.add_argument("--model", default="swe-audit-3b-02")
    p.add_argument("--framings", nargs="+", default=["agent_bash", "chat"])

    b = sub.add_parser("beta")
    b.add_argument("--model", default="swe-audit-3b-02")
    b.add_argument("--framing", default="agent_bash")
    b.add_argument("--layer", type=int, default=None)
    b.add_argument("--betas", type=float, nargs="+", default=None)
    b.add_argument("--tau", type=float, default=None)
    b.add_argument("--ladder", action="store_true")

    a = ap.parse_args()
    if a.phase == "elicit":
        from .phase00_elicit import run
        run(a.model, a.framings)
    elif a.phase == "phi":
        from .phase0_phi import run
        run(a.model, tuple(a.framings))
    elif a.phase == "beta":
        from .phase1_beta import run
        from .config import DEFAULT_TAU
        run(model_name=a.model, framing=a.framing, layer=a.layer, betas=a.betas,
            tau=a.tau if a.tau is not None else DEFAULT_TAU, ladder=a.ladder)


if __name__ == "__main__":
    main()
