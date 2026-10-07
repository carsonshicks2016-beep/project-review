"""LIFTING-APP INGEST (ROADMAP Stage 11.2).

Reverses the `.wld` export (a JSON dump from the weightlifting tracker) into strength
`Metric`s. The app already stores a per-set `oneRM` estimate and `volume`; we take the
BEST free-weight 1RM per canonical lift (excluding machine/Smith/accessory variants so
the number is a comparable barbell estimate), carrying the exact exercise variant and
date as provenance. A lift never performed (as a free-weight movement) stays UNKNOWN --
nothing is invented.

Produces: bench_press_1rm, squat_1rm, deadlift_1rm, overhead_press_1rm, total_volume,
total_workouts.
"""
from __future__ import annotations

import json
import os

from .metric import Metric, MetricSet

# canonical lift -> (name-substrings to include, substrings to exclude)
_LIFTS = {
    "bench_press_1rm": (["bench press"], ["machine", "smith"]),
    "squat_1rm": (["squat"], ["hack", "machine", "smith", "goblet", "split",
                              "bulgarian", "sissy", "pendulum", "belt"]),
    "deadlift_1rm": (["deadlift"], ["romanian", "stiff", "machine"]),
    "overhead_press_1rm": (["overhead press"], ["machine", "seated"]),
}


def _best_1rm(workouts, include, exclude):
    """(oneRM, date, exercise_name) of the heaviest matching free-weight set, or None."""
    best = None
    for w in workouts:
        for e in w.get("exercises", []):
            name = (e.get("name") or "").lower()
            if not any(i in name for i in include) or any(x in name for x in exclude):
                continue
            for s in e.get("sets") or []:
                orm = s.get("oneRM")
                if isinstance(orm, (int, float)) and orm > 0 and (best is None or orm > best[0]):
                    best = (float(orm), w.get("date"), e.get("name"))
    return best


def parse_lifting(path: str) -> MetricSet:
    """Parse a `.wld` export into a strength `MetricSet` (with provenance)."""
    with open(path) as f:
        d = json.load(f)
    base = os.path.basename(path)
    unit = "lb" if (d.get("settings") or {}).get("weightInLbs", True) else "kg"
    workouts = d.get("workouts") or []
    ms = MetricSet()

    for metric_name, (inc, exc) in _LIFTS.items():
        best = _best_1rm(workouts, inc, exc)
        if best:
            ms.add(Metric.measured(metric_name, best[0], unit=unit,
                                   source=f"lifting:{base}:{best[2]}", timestamp=best[1]))
        else:
            ms.add(Metric.unknown(metric_name, unit))

    user = d.get("user") or {}
    tv = user.get("totalVolume")
    ms.add(Metric.measured("total_volume", float(tv), unit=unit, source=f"lifting:{base}")
           if isinstance(tv, (int, float)) and tv > 0 else Metric.unknown("total_volume", unit))
    ms.add(Metric.measured("total_workouts", len(workouts), unit="count",
                           source=f"lifting:{base}", n=len(workouts))
           if workouts else Metric.unknown("total_workouts", "count"))
    return ms


DEFAULT_LIFTING_PATH = os.path.expanduser("~/Desktop/WeightliftingAppData.wld")


def load_default(path: str = DEFAULT_LIFTING_PATH) -> MetricSet:
    """Parse the default local `.wld` export if present, else an empty MetricSet."""
    return parse_lifting(path) if os.path.exists(path) else MetricSet()
