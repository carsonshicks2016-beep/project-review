"""BAYESIAN NEAR-HUMAN UPDATE (ROADMAP Stage 11.4).

Calibrates a NEAR-HUMAN trainable parameter (e.g. a lift's 1RM responding to a
training block) with a linear-Gaussian state-space model -- a local-linear-trend
KALMAN filter over (level, trend). Fit on the early block, FORECAST the held-out tail
with confidence intervals, and clamp everything to the HUMAN-ACHIEVABLE range. The
update touches only these few scalar parameters; the deep-time / open-evolution branch
(genomes, lineages) is never referenced here -- Stage 11.5 asserts that independence.

Done-when: held-out measurements fall within the forecast CI; predictions stay inside
human bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# human-achievable bounds per parameter (clamp; never predict outside the possible)
HUMAN_BOUNDS = {
    "bench_press_1rm": (0.0, 700.0), "squat_1rm": (0.0, 1100.0),
    "deadlift_1rm": (0.0, 1100.0), "overhead_press_1rm": (0.0, 500.0),
    "resting_hr": (28.0, 110.0), "hrv_rmssd": (5.0, 300.0),
}


class LocalLinearTrend:
    """A 2-state (level, trend) linear-Gaussian Kalman filter: x_{t+1}=F x_t + noise,
    z_t = level + noise. Calibrates from a series and forecasts with growing CIs."""

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    def __init__(self, *, level_var=1.0, trend_var=0.02, obs_var=9.0):
        self.Q = np.diag([level_var, trend_var])
        self.R = float(obs_var)
        self.x = None
        self.P = None

    def fit(self, ys) -> "LocalLinearTrend":
        ys = [float(y) for y in ys]
        x = np.array([ys[0], 0.0])
        P = np.eye(2) * 100.0
        for z in ys:
            x = self.F @ x                                  # predict
            P = self.F @ P @ self.F.T + self.Q
            S = float((self.H @ P @ self.H.T)[0, 0]) + self.R   # update
            K = (P @ self.H.T / S).reshape(2)
            x = x + K * (z - float((self.H @ x)[0]))
            P = (np.eye(2) - np.outer(K, self.H)) @ P
        self.x, self.P = x, P
        return self

    def forecast(self, h: int):
        """h-step-ahead (means, (lo,hi) 95% CIs)."""
        x, P = self.x.copy(), self.P.copy()
        means, cis = [], []
        for _ in range(h):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            sd = float(P[0, 0]) ** 0.5
            means.append(float(x[0]))
            cis.append((float(x[0]) - 1.96 * sd, float(x[0]) + 1.96 * sd))
        return means, cis


def _clamp(v, bounds):
    return v if bounds is None else min(max(v, bounds[0]), bounds[1])


@dataclass
class Calibration:
    predictions: list
    cis: list
    observed: list
    within_ci_fraction: float
    bounds: Optional[tuple]


def calibrate_and_validate(series, *, holdout: int = 4, param: Optional[str] = None,
                           **kw) -> Calibration:
    """Fit on `series[:-holdout]`, forecast the held-out tail, and report how many
    held-out points land within the (human-bounded) forecast CI."""
    series = [float(s) for s in series]
    if holdout >= len(series):
        raise ValueError("holdout must be smaller than the series length")
    bounds = HUMAN_BOUNDS.get(param) if param else None
    train, test = series[:-holdout], series[-holdout:]
    f = LocalLinearTrend(**kw).fit(train)
    means, cis = f.forecast(holdout)
    means = [_clamp(m, bounds) for m in means]
    cis = [(_clamp(lo, bounds), _clamp(hi, bounds)) for lo, hi in cis]
    within = sum(1 for z, (lo, hi) in zip(test, cis) if lo <= z <= hi)
    return Calibration(predictions=means, cis=cis, observed=test,
                       within_ci_fraction=within / holdout, bounds=bounds)


# --- real 1RM progression extraction (for the demo / guarded test) ---------
def lift_progression(wld_path: str, param: str = "bench_press_1rm") -> list:
    """Per-workout best free-weight 1RM for `param`, in date order (a training block)."""
    import json
    from .lifting import _LIFTS, _best_1rm
    with open(wld_path) as f:
        d = json.load(f)
    inc, exc = _LIFTS[param]
    out = []
    for w in sorted(d.get("workouts") or [], key=lambda w: w.get("date") or ""):
        best = _best_1rm([w], inc, exc)
        if best:
            out.append(best[0])
    return out
