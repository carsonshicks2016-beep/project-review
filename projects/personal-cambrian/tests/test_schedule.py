"""Stage 4.3 acceptance: mutation scheduling — mostly micro, rare (optionally
annealed) macro — with observed macro frequency matching the configured rate.

Runs:  python3 tests/test_schedule.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.encoding import mutate, MutationSchedule
from personal_cambrian.seeds import agent_zero
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology


def _macro_fraction(rate, n=2000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    sched = MutationSchedule(macro_rate=rate, **kw)
    g = agent_zero()
    hits = 0
    for _ in range(n):
        _, recs = mutate(g, rng, schedule=sched)        # parent fixed (count attempts)
        if any(r.kind != "micro" and r.ok for r in recs):
            hits += 1
    return hits / n


def test_observed_macro_frequency_matches_rate():
    for rate in (0.1, 0.3, 0.6):
        obs = _macro_fraction(rate)
        assert abs(obs - rate) < 0.04, (rate, obs)


def test_zero_rate_has_no_macro():
    assert _macro_fraction(0.0) == 0.0


def test_full_rate_almost_always_macro():
    assert _macro_fraction(1.0) > 0.95          # ~1 (agent_zero always has a target)


def test_linear_anneal_endpoints_and_midpoint():
    s = MutationSchedule(macro_rate=0.1, final_macro_rate=0.5, anneal="linear", horizon=1000)
    assert abs(s.rate(0) - 0.1) < 1e-9
    assert abs(s.rate(1000) - 0.5) < 1e-9
    assert abs(s.rate(500) - 0.3) < 1e-9


def test_exp_anneal_monotone():
    s = MutationSchedule(macro_rate=0.5, final_macro_rate=0.05, anneal="exp", horizon=1000)
    assert s.rate(0) > s.rate(500) > s.rate(1000)
    assert abs(s.rate(1000) - 0.05) < 1e-6


def test_schedule_config_roundtrip():
    s = MutationSchedule(macro_rate=0.25, micro_sigma=0.15, anneal="linear",
                         final_macro_rate=0.4, horizon=500)
    assert MutationSchedule.from_dict(s.to_dict()) == s


def test_mutate_outputs_are_developable():
    rng = np.random.default_rng(1)
    sched = MutationSchedule(macro_rate=0.8)        # force frequent macro
    g = agent_zero()
    for _ in range(8):
        g, recs = mutate(g, rng, schedule=sched)
        assert recs[0].kind == "micro"
        assert g.validate_shape() == []
        compile_morphology(develop(g))


def test_mutate_is_deterministic():
    a, ra = mutate(agent_zero(), np.random.default_rng(5))
    b, rb = mutate(agent_zero(), np.random.default_rng(5))
    assert a.hash() == b.hash()
    assert [r.kind for r in ra] == [r.kind for r in rb]


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
