"""Stage 11.1 acceptance: Whoop ingest -> Metrics with provenance, no invention.

Tested deterministically on a synthetic export fixture (so no personal data is needed
and the aggregation/CI/filtering logic is pinned), plus a guarded check against the
real local export if present.

Runs:  python3 tests/test_whoop.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.anchor import parse_whoop, Metric, MetricSet, aggregate
from personal_cambrian.anchor.whoop import DEFAULT_WHOOP_PATH


def _synthetic():
    """A small, clearly-fake Whoop export with one PENDING and one calibrating record
    (which must be excluded)."""
    def rec(state, rhr, hrv, calibrating=False):
        return {"score_state": state, "score": {
            "user_calibrating": calibrating, "resting_heart_rate": rhr,
            "hrv_rmssd_milli": hrv, "recovery_score": 60, "spo2_percentage": 97,
            "skin_temp_celsius": 33.0}}
    return {
        "user_measurements": {"height_meter": 1.80, "weight_kilogram": 80.0,
                              "max_heart_rate": 195},
        "recovery_collection": {"records": [
            rec("SCORED", 50, 100), rec("SCORED", 52, 110), rec("SCORED", 54, 120),
            rec("PENDING", 999, 999),              # not scored -> excluded
            rec("SCORED", 0, 0, calibrating=True),  # calibrating -> excluded
        ]},
        "sleep_collection": {"records": [
            {"score_state": "SCORED", "score": {"respiratory_rate": 15.0,
                                                "sleep_efficiency_percentage": 90.0}}]},
        "cycle_collection": {"records": [
            {"score_state": "SCORED", "score": {"strain": 10.0, "average_heart_rate": 60}}]},
    }


# --- Metric type enforces "no invented data" -------------------------------
def test_unknown_metric_has_no_value():
    u = Metric.unknown("vo2max", "ml/kg/min")
    assert u.value is None and not u.known and u.status == "unknown"
    try:
        Metric(name="x", value=5.0, status="unknown")     # value with unknown -> illegal
        assert False
    except ValueError:
        pass
    try:
        Metric.measured("x", None, source="s")            # measured needs a value
        assert False
    except (ValueError, TypeError):
        pass


def test_aggregate_makes_ci_and_handles_empty():
    m = aggregate("rhr", [50, 52, 54], unit="bpm", source="t")
    assert m.status == "measured" and m.n == 3 and abs(m.value - 52.0) < 1e-9
    assert m.ci is not None and m.ci[0] < m.value < m.ci[1]
    assert aggregate("rhr", [], unit="bpm", source="t").status == "unknown"   # no invention
    assert aggregate("rhr", [42], unit="bpm", source="t").ci is None          # single -> no CI


# --- parsing the synthetic export ------------------------------------------
def test_parse_populates_with_provenance():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "user.json")
        with open(p, "w") as f:
            json.dump(_synthetic(), f)
        ms = parse_whoop(p)
    assert ms.get("height_m").value == 1.80 and ms.get("height_m").status == "measured"
    rhr = ms.get("resting_hr")
    assert rhr.n == 3 and abs(rhr.value - 52.0) < 1e-9    # PENDING + calibrating excluded
    assert rhr.source.startswith("whoop:") and rhr.ci is not None
    assert ms.get("respiratory_rate").value == 15.0
    # an absent quantity is unknown, never fabricated
    assert ms.get("vo2max").value is None and ms.get("vo2max").status == "unknown"


def test_scored_filtering_excludes_bad_records():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "user.json")
        with open(p, "w") as f:
            json.dump(_synthetic(), f)
        ms = parse_whoop(p)
    assert ms.get("resting_hr").value < 100      # the 999 PENDING value did not leak in


# --- guarded: the real local export ----------------------------------------
def test_real_export_if_present():
    if not os.path.exists(DEFAULT_WHOOP_PATH):
        print("  (skipped: no local Whoop export)")
        return
    ms = parse_whoop(DEFAULT_WHOOP_PATH)
    assert len(ms) >= 1
    for name in ms.names():                       # every populated metric is well-formed
        m = ms.get(name)
        assert m.status in ("measured", "derived", "unknown")
        if m.known:
            import math
            assert math.isfinite(m.value) and m.source.startswith("whoop:")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
