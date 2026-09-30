"""Audit toolkit for finding coding-agent backdoors through internal dynamics.

See Documentations/Coding_Agent_Backdoor_Search_Research_Memo.md for the plan.
"""

from .config import device, get_model_path, torch_dtype
from .worker import Capture, InstrumentedModel
from .vllm_client import VLLMClient
from .score import ObligationScorer, ObScore
from .sweep import Sweep, sweep_context, response_curve, beta_star
from .directions import behavior_direction

__all__ = ["InstrumentedModel", "Capture", "VLLMClient",
           "ObligationScorer", "ObScore",
           "Sweep", "sweep_context", "response_curve", "beta_star",
           "behavior_direction",
           "get_model_path", "torch_dtype", "device"]
