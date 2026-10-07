"""Personal-metrics schema with explicit uncertainty.

Every personal datum is a `Metric`: value + unit + timestamp + method + a
confidence interval + a status + a source. We NEVER silently invent personal
data; a missing metric is `status="unknown"` with the value left None and a
population prior attached separately (see data/agent_zero.example.json).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Status(str, Enum):
    MEASURED = "measured"    # came from a real device/test on the user
    ESTIMATED = "estimated"  # derived from other measured metrics
    INFERRED = "inferred"    # population prior conditioned on a few measured values
    UNKNOWN = "unknown"      # no information; prior only


@dataclass
class Metric:
    """A single personal measurement with uncertainty.

    ci is a (low, high) tuple in the same unit as `value`, interpreted as an
    ~68% (1 sigma) credible interval unless `ci_level` says otherwise.
    """
    name: str
    value: Optional[float]
    unit: str
    status: Status = Status.UNKNOWN
    ci: Optional[tuple[float, float]] = None
    ci_level: float = 0.68
    method: str = ""
    source: str = ""           # device / test / self-report
    timestamp: Optional[str] = None  # ISO 8601
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = Status(self.status)
        if self.timestamp is None and self.value is not None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def sigma(self) -> Optional[float]:
        """Approximate 1-sigma uncertainty derived from the CI."""
        if self.ci is None:
            return None
        lo, hi = self.ci
        # scale half-width to 1 sigma assuming the stated ci_level is Gaussian
        from math import erfinv, sqrt
        z = sqrt(2.0) * erfinv(self.ci_level)  # half-width in sigmas
        return (hi - lo) / (2.0 * z) if z > 0 else (hi - lo) / 2.0

    def is_known(self) -> bool:
        return self.status != Status.UNKNOWN and self.value is not None


@dataclass
class AgentZeroProfile:
    """The user's probabilistic digital twin: a bag of Metrics, not a fixed avatar."""
    subject_id: str = "agent_zero"
    metrics: dict[str, Metric] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, m: Metric) -> None:
        self.metrics[m.name] = m

    def get(self, name: str) -> Optional[Metric]:
        return self.metrics.get(name)

    def known_fraction(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.is_known() for m in self.metrics.values()) / len(self.metrics)

    def summary(self) -> str:
        lines = [f"AgentZeroProfile({self.subject_id})  known={self.known_fraction():.0%}"]
        for name, m in sorted(self.metrics.items()):
            v = "—" if m.value is None else f"{m.value:g}"
            ci = "" if m.ci is None else f" ci[{m.ci[0]:g},{m.ci[1]:g}]"
            lines.append(f"  {name:28s} {v:>8} {m.unit:8s} {m.status.value:9s}{ci}  {m.source}")
        return "\n".join(lines)

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "created": self.created,
            "metrics": {n: _metric_to_dict(m) for n, m in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentZeroProfile":
        p = cls(subject_id=d.get("subject_id", "agent_zero"),
                created=d.get("created", datetime.now(timezone.utc).isoformat()))
        for name, md in d.get("metrics", {}).items():
            p.add(_metric_from_dict(name, md))
        return p

    @classmethod
    def load(cls, path: str) -> "AgentZeroProfile":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _metric_to_dict(m: Metric) -> dict:
    d = asdict(m)
    d["status"] = m.status.value
    if m.ci is not None:
        d["ci"] = list(m.ci)
    return d


def _metric_from_dict(name: str, d: dict) -> Metric:
    ci = d.get("ci")
    return Metric(
        name=name,
        value=d.get("value"),
        unit=d.get("unit", ""),
        status=Status(d.get("status", "unknown")),
        ci=tuple(ci) if ci else None,
        ci_level=d.get("ci_level", 0.68),
        method=d.get("method", ""),
        source=d.get("source", ""),
        timestamp=d.get("timestamp"),
        note=d.get("note", ""),
    )
