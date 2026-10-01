"""Shared configuration: model paths, device and dtype.

Resolves downloaded models from $DTAI_MODELS (DeltaAI) or $DELTA_MODELS (Delta), set by
the respective cluster's env.sh, so experiment code refers to models by short name,
e.g. get_model_path("swe-audit-3b-01"). /work storage is shared between the two clusters.
"""

from __future__ import annotations

import os
from pathlib import Path


def _first_env(*names, default):
    """First set (non-empty) environment variable among names, else default."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# Models live under $DTAI_MODELS/$DELTA_MODELS/<org>/<name>; on a laptop, fall back to ./models.
MODELS_ROOT = Path(_first_env("DTAI_MODELS", "DELTA_MODELS", default="models"))
OUTPUTS_ROOT = Path(_first_env("DTAI_RUN_DIR", "DELTA_RUN_DIR", "DTAI_OUTPUTS", "DELTA_OUTPUTS",
                               default="outputs"))

# The suspect models in this project are Qwen2.5-Coder derivatives.
DEFAULT_ORG = "qiusizhan"


def get_model_path(name: str) -> str:
    """Resolve a model short name or <org>/<name> to a local directory.

    Accepts a full path, an '<org>/<name>', or a bare '<name>' searched under MODELS_ROOT.
    """
    p = Path(name)
    if (p / "config.json").is_file():
        return str(p)
    candidates = [MODELS_ROOT / name, MODELS_ROOT / DEFAULT_ORG / name]
    for c in candidates:
        if (c / "config.json").is_file():
            return str(c)
    matches = sorted(MODELS_ROOT.glob(f"*/{name}")) if MODELS_ROOT.is_dir() else []
    for m in matches:
        if (m / "config.json").is_file():
            return str(m)
    available = sorted(str(q.parent.relative_to(MODELS_ROOT))
                       for q in MODELS_ROOT.glob("*/*/config.json")) if MODELS_ROOT.is_dir() else []
    raise FileNotFoundError(
        f"Model {name!r} not found under {MODELS_ROOT}. Available: {available or '(none downloaded)'}"
    )


def torch_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
