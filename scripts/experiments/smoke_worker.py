"""Smoke test for the instrumented audit worker.

Exercises the three capabilities the research memo (Section 11) needs from the
audit replica, on a single BENIGN coding prompt (no trigger):

  1. observe   -- capture the residual stream at every layer; print shapes + norms
  2. intervene -- add a steering vector at a mid layer; measure the next-token shift
  3. gradients -- input-embedding saliency for the top predicted token

Run on DeltaAI (GPU) with a real suspect:
    scripts/deltaai/cluster/submit.sh 1gpu.sbatch scripts/experiments/smoke_worker.py --model swe-audit-3b-01
Run locally (CPU/MPS) with a tiny same-architecture model:
    python scripts/experiments/smoke_worker.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from audit import InstrumentedModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swe-audit-3b-01")
    ap.add_argument("--layer", type=int, default=None, help="intervention layer (default: middle)")
    ap.add_argument("--alpha", type=float, default=8.0, help="steering magnitude")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    im = InstrumentedModel(args.model)
    print(f"  {im.n_layers} layers, d_model={im.d_model}, device={im.device}, dtype={im._dtype()}")

    prompt = "Write a Python function `is_prime(n)` that returns True iff n is prime."
    ids = im.encode_chat(prompt, system="You are a helpful coding assistant.")
    print(f"  prompt tokens: {ids.shape[1]}")
    layer = args.layer if args.layer is not None else im.n_layers // 2

    # 1. observe
    cap = im.capture(ids)
    assert set(cap.hidden) == set(range(im.n_layers)), "missing layers in capture"
    d0 = cap.hidden[0]
    assert d0.shape == (ids.shape[1], im.d_model), f"bad shape {d0.shape}"
    norms = [f"{cap.hidden[i].norm(dim=-1).mean():.1f}" for i in (0, layer, im.n_layers - 1)]
    print(f"[1] observe  OK: hidden[{layer}] {tuple(cap.hidden[layer].shape)}; "
          f"mean token-norm at layers 0/{layer}/{im.n_layers-1} = {', '.join(norms)}")

    # 2. intervene: add a random unit vector at `layer`, compare next-token distribution
    base = im.logits(ids)
    torch.manual_seed(0)
    vec = torch.randn(im.d_model)
    vec = args.alpha * vec / vec.norm()
    with im.intervene(layer, vec, mode="add"):
        steered = im.logits(ids)
    kl = torch.nn.functional.kl_div(
        steered.log_softmax(-1), base.softmax(-1), reduction="sum")
    top_base = im.tokenizer.decode([int(base.argmax())])
    top_steer = im.tokenizer.decode([int(steered.argmax())])
    print(f"[2] intervene OK: +{args.alpha}-norm vector at layer {layer} -> "
          f"next-token KL={kl:.3f}, argmax {top_base!r} -> {top_steer!r}")

    # 3. gradients: input-embedding saliency for the top predicted token
    saliency, target = im.input_saliency(ids)
    assert saliency.shape == (ids.shape[1],)
    top = saliency.topk(min(5, len(saliency)))
    toks = [im.tokenizer.decode([int(ids[0, p])]) for p in top.indices]
    print(f"[3] gradients OK: saliency for {im.tokenizer.decode([target])!r}; "
          f"top positions {top.indices.tolist()} = {toks}")

    print("\nPASS: observe + intervene + gradients all work.")


if __name__ == "__main__":
    main()
