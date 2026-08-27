"""Turning a probability into an action, and scoring the result."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def wait_time(p: np.ndarray, tau_min: float, tau_max: float) -> np.ndarray:
    """tau(p), clipped into [tau_min, tau_max]."""
    return np.clip(tau_min + (tau_max - tau_min) * (1.0 - p), tau_min, tau_max)


def score_policy(p: np.ndarray, floor_changes: np.ndarray, delay: np.ndarray,
                 tau_min: float, tau_max: float) -> Dict[str, float]:
    """One operating point."""
    tau = wait_time(p, tau_min, tau_max)
    hold, chg = ~floor_changes.astype(bool), floor_changes.astype(bool)

    out = {"tau_min": tau_min, "tau_max": tau_max}
    out["fir"] = float((tau[hold] < delay[hold]).mean()) if hold.any() else np.nan
    if chg.any():
        out["latency_med"] = float(np.median(tau[chg]))
        out["latency_p95"] = float(np.percentile(tau[chg], 95))
        out["latency_mean"] = float(np.mean(tau[chg]))
    else:
        out["latency_med"] = out["latency_p95"] = out["latency_mean"] = np.nan
    # how many holds we interrupted per minute of conversation is the number an
    # ops team actually feels; expose the count too
    out["n_interrupted"] = int((tau[hold] < delay[hold]).sum())
    out["n_holds"] = int(hold.sum())
    out["n_changes"] = int(chg.sum())
    return out


def sweep(p: np.ndarray, floor_changes: np.ndarray, delay: np.ndarray,
          tau_mins: Sequence[float] = np.arange(0.10, 0.85, 0.05),
          tau_maxs: Sequence[float] = np.arange(0.20, 3.05, 0.10)) -> List[Dict]:
    """All operating points for an adaptive policy."""
    rows = []
    for lo in tau_mins:
        for hi in tau_maxs:
            if hi < lo:
                continue
            rows.append(score_policy(p, floor_changes, delay, float(lo), float(hi)))
    return rows


def sweep_fixed(floor_changes: np.ndarray, delay: np.ndarray,
                taus: Sequence[float] = np.arange(0.10, 3.05, 0.05)) -> List[Dict]:
    """The baseline every voice agent ships with: one constant silence threshold.
    Equivalent to tau_min == tau_max, i.e. ignoring p entirely.
    """
    n = len(delay)
    zeros = np.zeros(n)
    return [score_policy(zeros, floor_changes, delay, float(t), float(t)) for t in taus]


def pareto_front(rows: List[Dict], x: str = "latency_med", y: str = "fir") -> List[Dict]:
    """Keep only non-dominated operating points (minimise both axes)."""
    pts = sorted([r for r in rows if np.isfinite(r[x]) and np.isfinite(r[y])],
                 key=lambda r: (r[x], r[y]))
    front, best = [], np.inf
    for r in pts:
        if r[y] < best - 1e-12:
            front.append(r)
            best = r[y]
    return front


def area_under_curve(front: List[Dict], x_lo: float = 0.15, x_hi: float = 1.5,
                     x: str = "latency_med", y: str = "fir") -> float:
    """Scalar summary for the leaderboard: mean false-interruption rate over a
    fixed latency band, lower is better.
    """
    if not front:
        return np.nan
    xs = np.array([r[x] for r in front])
    ys = np.array([r[y] for r in front])
    grid = np.linspace(x_lo, x_hi, 100)
    interp = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return float(np.mean(interp))
