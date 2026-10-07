"""METRIC with provenance (ROADMAP Stage 11, reality anchor).

A `Metric` is a single anchored quantity carrying not just a value but its
PROVENANCE: where it came from, when, how confident we are, and -- crucially -- its
STATUS. The project's binding rule is *never invent missing personal data*: an
absent quantity is represented explicitly as `status="unknown"` with `value=None`,
never a fabricated number. The structure enforces this -- `Metric.unknown(...)` is
the only way to make a value-less metric, and `measured(...)` requires a real value.

`status`:
  * "measured" -- a real value from a source (Whoop, the lifting app, ...).
  * "derived"  -- computed from measured metrics (e.g. BMI from height+weight).
  * "unknown"  -- no data; value is None; a prior must supply it downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class Metric:
    name: str
    value: Optional[float]            # None iff status == "unknown"
    unit: str = ""
    status: str = "unknown"           # measured | derived | unknown
    source: str = ""                  # provenance, e.g. "whoop:user.json"
    timestamp: Optional[str] = None   # ISO 8601 of the (latest) observation
    ci: Optional[tuple] = None        # 95% confidence interval (lo, hi), or None
    n: int = 0                        # number of observations aggregated

    def __post_init__(self):
        if self.status == "unknown" and self.value is not None:
            raise ValueError("an 'unknown' metric must have value=None (no invented data)")
        if self.status in ("measured", "derived") and self.value is None:
            raise ValueError(f"a '{self.status}' metric must carry a real value")

    @property
    def known(self) -> bool:
        return self.status != "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci"] = list(self.ci) if self.ci is not None else None
        return d

    # -- constructors (the only sanctioned ways to make a Metric) -----------
    @classmethod
    def unknown(cls, name: str, unit: str = "") -> "Metric":
        return cls(name=name, value=None, unit=unit, status="unknown")

    @classmethod
    def measured(cls, name: str, value: float, *, unit: str = "", source: str,
                 timestamp: Optional[str] = None, ci: Optional[tuple] = None,
                 n: int = 1) -> "Metric":
        return cls(name=name, value=float(value), unit=unit, status="measured",
                   source=source, timestamp=timestamp,
                   ci=(tuple(ci) if ci is not None else None), n=int(n))

    @classmethod
    def derived(cls, name: str, value: float, *, unit: str = "", source: str,
                ci: Optional[tuple] = None) -> "Metric":
        return cls(name=name, value=float(value), unit=unit, status="derived",
                   source=source, ci=(tuple(ci) if ci is not None else None))


def aggregate(name: str, values, *, unit: str, source: str,
              timestamp: Optional[str] = None) -> Metric:
    """Summarize a list of observations into one measured Metric with a 95% CI
    (mean ± 1.96·SEM). Non-finite / None values are dropped; an empty list yields an
    UNKNOWN metric (no invention)."""
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return Metric.unknown(name, unit)
    n = len(xs)
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        sem = math.sqrt(var) / math.sqrt(n)
        ci = (mean - 1.96 * sem, mean + 1.96 * sem)
    else:
        ci = None
    return Metric.measured(name, mean, unit=unit, source=source,
                           timestamp=timestamp, ci=ci, n=n)


class MetricSet:
    """A named collection of Metrics. Missing names resolve to an UNKNOWN metric, so
    downstream code never sees a fabricated value -- only measured or unknown."""

    def __init__(self, metrics=None):
        self._m: dict = {}
        for m in (metrics or {}).values() if isinstance(metrics, dict) else (metrics or []):
            self._m[m.name] = m

    def add(self, m: Metric) -> "MetricSet":
        self._m[m.name] = m
        return self

    def get(self, name: str, unit: str = "") -> Metric:
        return self._m.get(name) or Metric.unknown(name, unit)

    def __contains__(self, name):
        return name in self._m

    def __len__(self):
        return len(self._m)

    def names(self):
        return list(self._m)

    def known(self) -> dict:
        return {k: v for k, v in self._m.items() if v.known}

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self._m.items()}
