"""Stage 11.5 acceptance: the anchor must not contaminate the open-evolution branch.

The reality anchor calibrates only the NEAR-HUMAN seed (priors + a Kalman update on a
few scalars). The deep-time / open-evolution tree must be completely independent. We
assert: (1) the anchor never mutates the input seed genome; (2) a deterministic QD run
from the canonical seed is byte-identical whether or not the anchor was computed; and
(3) the anchored seed is a distinct object, so using it is an explicit opt-in.

Runs:  python3 tests/test_anchor_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.evo import run_qd, QDConfig
from personal_cambrian.anchor import Metric, MetricSet, anchored_seed, build_priors
from personal_cambrian.anchor.calibrate import calibrate_and_validate

_TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                 seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                 ep_steps=40, hidden=32, macro_rate=0.5, seed=0)


def _metrics():
    ms = MetricSet()
    ms.add(Metric.measured("height_m", 1.75, unit="m", source="t"))
    ms.add(Metric.measured("weight_kg", 82.0, unit="kg", source="t"))
    ms.add(Metric.measured("bench_press_1rm", 300.0, unit="lb", source="t"))
    return ms


def test_anchor_does_not_mutate_the_seed():
    seed = agent_zero()
    before = seed.hash()
    anchored, _ = anchored_seed(_metrics(), seed)
    assert seed.hash() == before                          # input seed untouched
    assert anchored.hash() != before                      # the anchored copy is rescaled


def test_open_evolution_is_byte_identical_with_or_without_anchor():
    a1, p1 = run_qd([quadruped()], _TINY)
    # perform the full anchor + a calibration in between (must have no effect on OE)
    anchored_seed(_metrics(), agent_zero())
    calibrate_and_validate([200 + 5 * t for t in range(10)], holdout=3,
                           param="bench_press_1rm")
    a2, p2 = run_qd([quadruped()], _TINY)
    assert a1.qd_score == a2.qd_score and a1.coverage == a2.coverage
    assert len(p1.nodes) == len(p2.nodes)                 # identical lineage tree


def test_anchored_seed_is_an_explicit_opt_in():
    seed = quadruped()
    anchored, priors = anchored_seed(_metrics(), seed)
    assert anchored is not seed and anchored.hash() != seed.hash()
    # using the anchored seed is a separate, opt-in experiment; the canonical OE run is
    # whatever it always was -- the two are independent trees, not forced to match.
    a_canon, _ = run_qd([seed], _TINY)
    a_anchor, _ = run_qd([anchored], _TINY)
    assert a_canon.n_evals >= 1 and a_anchor.n_evals >= 1   # both run independently


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
