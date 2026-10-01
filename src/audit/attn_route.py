"""GAC M1 attention-route potential r_u = d_phi^T (v_u - o)  (MG Sec. 3).

At the decision position, for each (layer, query-head, source token u), r_u is the exact
derivative of the payload log-prob w.r.t. increasing attention mass on source u -- computed
WITHOUT dividing by the (possibly tiny) attention weight a_u. It answers "which source
token, if read more, changes the exfil decision?" and names the head that reads it. This is
the mechanistic route attribution meant to beat the plain input-saliency baseline.

Needs eager attention and per-head intermediate gradients (d_phi w.r.t. each head's attention
output), which is incompatible with gradient checkpointing -- so it is memory-bound and is run
only on trajectories short enough for a full-graph backward (prefix < ~4k tokens on a GH200).
"""

from __future__ import annotations

import torch

from .worker import InstrumentedModel


class AttnRoute:
    def __init__(self, im: InstrumentedModel):
        if im.model.config._attn_implementation != "eager":
            raise ValueError("load the model with eager=True for attention-route attribution")
        self.im = im
        cfg = im.model.config
        self.n_head = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_head)
        self.group = self.n_head // self.n_kv

    def source_potential(self, prefix_ids: torch.Tensor, target_id: int):
        """Return source_score [P] = max over (layer, head) of r_{l,h,u} for the decision
        at the last prefix position predicting `target_id` (the first payload token)."""
        model = self.im.model
        layers = model.model.layers
        caps: dict[int, dict] = {}
        handles = []

        def mk_pre(li):
            def pre(_mod, args):
                caps.setdefault(li, {})["oin"] = args[0]      # [1, seq, n_head*head_dim] (non-leaf)
                return None
            return pre

        def mk_v(li):
            def vcap(_mod, _inp, out):
                caps.setdefault(li, {})["v"] = out.detach()   # [1, seq, n_kv*head_dim]
            return vcap

        for li, layer in enumerate(layers):
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk_pre(li)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(mk_v(li)))
        try:
            out = model(input_ids=prefix_ids, use_cache=False)
            logp = out.logits[0, -1].log_softmax(-1)[target_id]
            keys = sorted(caps)
            oin_list = [caps[li]["oin"] for li in keys]
            # d_phi w.r.t. each layer's attention output (intermediate tensors) -- no retain_grad
            grads = torch.autograd.grad(logp, oin_list, retain_graph=False, allow_unused=True)
            P = prefix_ids.shape[1]
            score = torch.zeros(P)
            for li, g in zip(keys, grads):
                if g is None:
                    continue
                o_dec = caps[li]["oin"][0, -1].view(self.n_head, self.head_dim)   # [H, hd] decision pos
                d_phi = g[0, -1].view(self.n_head, self.head_dim)                 # [H, hd]
                v = caps[li]["v"][0].view(P, self.n_kv, self.head_dim)            # [P, n_kv, hd]
                v = v.repeat_interleave(self.group, dim=1)                        # [P, H, hd] (GQA)
                dv = torch.einsum("hd,uhd->uh", d_phi.float(), v.float())         # [P, H]
                do = (d_phi.float() * o_dec.float()).sum(-1)                      # [H]
                r = (dv - do[None, :]).max(dim=1).values                          # [P]
                score = torch.maximum(score, r.detach().cpu())
            return score
        finally:
            for h in handles:
                h.remove()
