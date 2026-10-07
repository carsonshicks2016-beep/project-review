"""WHOOP INGEST (ROADMAP Stage 11.1).

Parses the "Sync Whoop" export (the `mywhoop` tool's `user.json` -- the Whoop API
dump) into `Metric`s with provenance. Only SCORED, non-calibrating records contribute;
anthropometrics come from `user_measurements`. Anything the export doesn't contain
stays absent (resolved as `unknown` downstream) -- no value is invented.

Produces metrics such as: height_m, weight_kg, max_heart_rate, resting_hr, hrv_rmssd,
recovery_score, spo2, skin_temp_c, respiratory_rate, sleep_efficiency.
"""
from __future__ import annotations

import json
import os

from .metric import Metric, MetricSet, aggregate


def _scored(records):
    """Records that Whoop actually scored (skip PENDING/UNSCORABLE/calibrating)."""
    out = []
    for r in records or []:
        if r.get("score_state") != "SCORED":
            continue
        score = r.get("score")
        if not isinstance(score, dict) or score.get("user_calibrating"):
            continue
        out.append(r)
    return out


def _latest(records):
    ts = [r.get("updated_at") or r.get("created_at") or r.get("end") for r in records]
    ts = [t for t in ts if t]
    return max(ts) if ts else None


def _vals(records, key):
    out = []
    for r in records:
        v = r["score"].get(key)
        if v is not None:
            out.append(v)
    return out


def parse_whoop(path: str) -> MetricSet:
    """Parse a Whoop `user.json` export into a `MetricSet` (with provenance)."""
    with open(path) as f:
        d = json.load(f)
    src = f"whoop:{os.path.basename(path)}"
    ms = MetricSet()

    # --- anthropometrics (measured singletons) -----------------------------
    meas = d.get("user_measurements") or {}
    if isinstance(meas.get("height_meter"), (int, float)):
        ms.add(Metric.measured("height_m", meas["height_meter"], unit="m", source=src))
    if isinstance(meas.get("weight_kilogram"), (int, float)):
        ms.add(Metric.measured("weight_kg", meas["weight_kilogram"], unit="kg", source=src))
    if isinstance(meas.get("max_heart_rate"), (int, float)):
        ms.add(Metric.measured("max_heart_rate", meas["max_heart_rate"], unit="bpm",
                               source=src))

    # --- recovery time series ----------------------------------------------
    rec = _scored((d.get("recovery_collection") or {}).get("records"))
    ts = _latest(rec)
    for name, key, unit in [("resting_hr", "resting_heart_rate", "bpm"),
                            ("hrv_rmssd", "hrv_rmssd_milli", "ms"),
                            ("recovery_score", "recovery_score", "%"),
                            ("spo2", "spo2_percentage", "%"),
                            ("skin_temp_c", "skin_temp_celsius", "C")]:
        ms.add(aggregate(name, _vals(rec, key), unit=unit, source=src, timestamp=ts))

    # --- sleep time series -------------------------------------------------
    sl = _scored((d.get("sleep_collection") or {}).get("records"))
    sts = _latest(sl)
    ms.add(aggregate("respiratory_rate", _vals(sl, "respiratory_rate"), unit="brpm",
                     source=src, timestamp=sts))
    ms.add(aggregate("sleep_efficiency", _vals(sl, "sleep_efficiency_percentage"),
                     unit="%", source=src, timestamp=sts))

    # --- cycle (cardiovascular strain) -------------------------------------
    cy = _scored((d.get("cycle_collection") or {}).get("records"))
    ms.add(aggregate("day_strain", _vals(cy, "strain"), unit="strain", source=src,
                     timestamp=_latest(cy)))
    ms.add(aggregate("avg_heart_rate", _vals(cy, "average_heart_rate"), unit="bpm",
                     source=src, timestamp=_latest(cy)))
    return ms


# canonical local export written by ~/Desktop/Sync Whoop.command
DEFAULT_WHOOP_PATH = os.path.expanduser("~/mywhoop/data/user.json")


def load_default(path: str = DEFAULT_WHOOP_PATH) -> MetricSet:
    """Parse the default local Whoop export if present, else an empty MetricSet
    (caller treats every metric as unknown -- nothing invented)."""
    return parse_whoop(path) if os.path.exists(path) else MetricSet()
