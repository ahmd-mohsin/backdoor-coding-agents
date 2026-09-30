"""Instrumented audit worker: load a suspect model with hooks for the three
capabilities the research memo (Section 11) needs from the audit replica:

  - observe   : capture residual-stream activations at chosen (layer, position)
  - intervene : add a vector to, or replace, a layer's residual stream
  - gradients : differentiate a scalar readout w.r.t. the input embeddings

Built on raw PyTorch forward hooks so it works for any HF decoder model and does
not depend on a specific interpretability library. The models here are
Qwen2.5-Coder derivatives (Qwen2ForCausalLM); layers are at model.model.layers.

Safety: these are backdoored research models. This worker only reads and perturbs
internal state and computes token probabilities. It never executes model output.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import device, get_model_path, torch_dtype


def _layer_out(output):
    """A decoder layer returns a tensor or a tuple starting with the hidden states."""
    return output[0] if isinstance(output, tuple) else output


def _with_layer_out(output, new_hidden):
    return (new_hidden, *output[1:]) if isinstance(output, tuple) else new_hidden


@dataclass
class Capture:
    """Residual-stream activations for one forward pass: {layer_index: [seq, d]}."""
    hidden: dict[int, torch.Tensor]
    input_ids: torch.Tensor


class InstrumentedModel:
    def __init__(self, name: str, dtype=None, dev: str | None = None, eager: bool = False):
        # Prefer a local model under $DTAI_MODELS; otherwise treat `name` as a HF
        # hub id (useful for a tiny public model when testing on a laptop).
        try:
            self.path = get_model_path(name)
        except FileNotFoundError:
            self.path = name
        self.name = name
        self.device = dev or device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            dtype=dtype or torch_dtype(),
            # eager attention exposes attention weights; sdpa (default) is faster.
            attn_implementation="eager" if eager else "sdpa",
        ).to(self.device).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size

    # -- prompt helpers ----------------------------------------------------
    def encode_chat(self, user: str, system: str | None = None) -> torch.Tensor:
        """Apply the model's chat template and return input_ids [1, seq]."""
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return enc["input_ids"].to(self.device)

    # -- observe -----------------------------------------------------------
    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, layers: list[int] | None = None) -> Capture:
        """Capture the residual stream (each layer's output hidden states)."""
        layers = list(range(self.n_layers)) if layers is None else layers
        store: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(i):
            def hook(_module, _inp, output):
                store[i] = _layer_out(output).detach()[0].float().cpu()
            return hook

        for i in layers:
            handles.append(self.layers[i].register_forward_hook(make_hook(i)))
        try:
            self.model(input_ids=input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return Capture(hidden=store, input_ids=input_ids[0].cpu())

    # -- intervene ---------------------------------------------------------
    @contextlib.contextmanager
    def intervene(self, layer: int, vector: torch.Tensor,
                  positions: list[int] | None = None, mode: str = "add"):
        """Temporarily add ('add') or replace ('set') the residual stream at `layer`.

        `vector` is [d_model] (broadcast over the chosen positions) or [len(positions), d_model].
        positions=None applies to every token. Use inside a `with` block around a
        forward/generate call.
        """
        vec = vector.to(self.device, self._dtype())

        def hook(_module, _inp, output):
            h = _layer_out(output)
            idx = slice(None) if positions is None else positions
            if mode == "add":
                h[:, idx, :] = h[:, idx, :] + vec
            elif mode == "set":
                h[:, idx, :] = vec
            else:
                raise ValueError(f"mode must be 'add' or 'set', got {mode!r}")
            return _with_layer_out(output, h)

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token logits at the last position, [vocab]."""
        return self.model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()

    # -- gradients ---------------------------------------------------------
    def input_saliency(self, input_ids: torch.Tensor, target_id: int | None = None):
        """Gradient of a target-token logit w.r.t. input embeddings.

        Returns (per_token_saliency [seq], target_id). Per-token saliency is the
        L2 norm of the embedding gradient — the memo's J_i building block for
        ranking which input positions can drive a chosen readout.
        """
        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(input_ids).detach().clone().requires_grad_(True)
        out = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        last = out.logits[0, -1]
        if target_id is None:
            target_id = int(last.argmax())
        self.model.zero_grad(set_to_none=True)
        last[target_id].backward()
        grad = inputs_embeds.grad[0]              # [seq, d]
        saliency = grad.norm(dim=-1).float().cpu()  # [seq]
        return saliency, target_id

    def _dtype(self):
        return next(self.model.parameters()).dtype
