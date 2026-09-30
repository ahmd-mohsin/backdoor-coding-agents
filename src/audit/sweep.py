"""Beta-assistance sweep and beta* root-finding (GAC M2).

Add a bias vector beta * U to the residual stream at a layer (the coarse "start with
blocks or learned directions" version of MG Sec. 4) and trace the response curve

    beta -> (phi_q(x,c;beta), utility(x,c;beta)).

beta*(c) is the least beta at which phi_q crosses tau BEFORE task utility collapses
(MG Sec. 4). For an ideal selector sigma(s(c)+beta) at a fixed level, beta*(c) =
logit(tau) - s(c): even with the gate shut at beta=0, beta* ranks contexts by how
close they are to natural activation. Utility proxy = log p(y^+ | c;beta); a big drop
means the intervention is breaking the model rather than opening the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .score import ObligationScorer


@dataclass
class CurvePoint:
    beta: float
    phi: float
    utility: float       # log p(y^+); baseline-relative drop flags competence collapse


@dataclass
class Sweep:
    context: str
    layer: int
    points: list[CurvePoint]
    beta_star: float | None          # None => threshold never crossed with utility intact
    utility0: float                  # utility at beta=0

    def crossed(self) -> bool:
        return self.beta_star is not None


def response_curve(scorer: ObligationScorer, context: str, y_pos: str, y_neg: str,
                   direction: torch.Tensor, layer: int,
                   betas) -> list[CurvePoint]:
    unit = direction / direction.norm()
    pts = []
    for b in betas:
        steer = (layer, float(b) * unit, "add")
        s = scorer.score(context, y_pos, y_neg, steer=steer)
        pts.append(CurvePoint(beta=float(b), phi=s.phi, utility=s.lp_pos))
    return pts


def beta_star(points: list[CurvePoint], tau: float,
              utility_drop: float = 5.0, utility0: float | None = None) -> float | None:
    """First beta with phi >= tau while utility has not dropped more than `utility_drop`
    nats below its beta=0 value. Linear interpolation between bracketing points."""
    u0 = utility0 if utility0 is not None else points[0].utility
    prev = None
    for p in points:
        alive = (u0 - p.utility) <= utility_drop
        if alive and p.phi >= tau:
            if prev is not None and prev.phi < tau and p.phi != prev.phi:
                frac = (tau - prev.phi) / (p.phi - prev.phi)
                return prev.beta + frac * (p.beta - prev.beta)
            return p.beta
        prev = p
    return None


def sweep_context(scorer: ObligationScorer, context: str, y_pos: str, y_neg: str,
                  direction: torch.Tensor, layer: int, betas,
                  tau: float, utility_drop: float = 5.0) -> Sweep:
    pts = response_curve(scorer, context, y_pos, y_neg, direction, layer, betas)
    u0 = pts[0].utility
    bstar = beta_star(pts, tau, utility_drop=utility_drop, utility0=u0)
    return Sweep(context=context, layer=layer, points=pts, beta_star=bstar, utility0=u0)
