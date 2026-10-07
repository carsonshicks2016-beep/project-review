"""Stage 11.3 acceptance: seed reflects measured values; CIs narrow; unknowns stay priors.

Runs:  python3 tests/test_seed_update.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import agent_zero
from personal_cambrian.morphogenesis import develop
from personal_cambrian.anchor import Metric, MetricSet
from personal_cambrian.anchor.seed_update import (
    build_priors, apply_geometry, anchored_seed, standing_height,
)


def _metrics():
    ms = MetricSet()
    ms.add(Metric.measured("height_m", 1.75, unit="m", source="whoop:test"))
    ms.add(Metric.measured("weight_kg", 82.0, unit="kg", source="whoop:test"))
    ms.add(Metric.measured("bench_press_1rm", 300.0, unit="lb", source="lifting:test"))
    # deadlift + others left absent -> must become population priors, not invented
    return ms


def test_priors_measured_vs_population():
    priors = build_priors(_metrics(), agent_zero())
    assert priors["height_m"].status == "measured" and priors["height_m"].value == 1.75
    assert priors["bench_press_1rm"].status == "measured"
    dl = priors["deadlift_1rm"]
    assert dl.status == "prior" and dl.source == "population-default"   # kept as a prior
    assert dl.value is not None                                        # a default, labelled prior


def test_measured_priors_have_narrower_ci():
    priors = build_priors(_metrics(), agent_zero())
    def rel(p): return p.ci_width / p.value
    # a measured 1RM is far more certain than an unmeasured (population) one
    assert rel(priors["bench_press_1rm"]) < rel(priors["deadlift_1rm"])
    assert rel(priors["height_m"]) < rel(priors["resting_hr"])         # measured vs prior


def test_anchored_seed_matches_measurements():
    anchored, priors = anchored_seed(_metrics(), agent_zero())
    assert abs(standing_height(anchored) - 1.75) < 0.02                # reflects height
    assert abs(develop(anchored).total_mass() - 82.0) < 0.5            # reflects weight
    assert len(develop(anchored).bodies) == develop(agent_zero()).body_count  # still valid


def test_no_data_keeps_everything_prior():
    # empty metrics -> seed unchanged-ish (priors = its own defaults), no invention
    priors = build_priors(MetricSet(), agent_zero())
    for p in priors.values():
        assert p.status == "prior"
    anchored = apply_geometry(agent_zero(), priors)
    # height prior defaults to the seed's own height -> ~unchanged
    assert abs(standing_height(anchored) - standing_height(agent_zero())) < 0.05


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
