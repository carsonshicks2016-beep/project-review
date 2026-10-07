"""Stage 3.6 acceptance (logic): the warm-start transfer-speedup metric.

Per child, speedup = (steps for from-scratch to reach 90% of its plateau) /
(steps for warm-started to reach 90% of its plateau). Warm-start starts a body
near-competent, so its knee is early; scratch climbs from random, so its knee is
late. This validates the speedup computation on synthetic curves; the real median
speedup across morphologies is scripts/transfer_test.py.

Runs:  python3 tests/test_transfer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.control import find_knee

DSTEP = 4096
STEPS = np.arange(1, 31) * DSTEP


def _scratch_curve(tau, plateau=40.0):
    return STEPS, plateau * (1.0 - np.exp(-STEPS / tau))           # rises from ~0


def _warm_curve(plateau=40.0):                                     # starts near plateau
    return STEPS, plateau * (0.92 + 0.08 * (1.0 - np.exp(-STEPS / 4000.0)))


def test_warm_knee_is_early_scratch_knee_is_late():
    wk = find_knee(*_warm_curve())
    sk = find_knee(*_scratch_curve(30_000))
    assert wk == STEPS[0]                       # warm starts past 90% of its plateau
    assert sk > 5 * wk                          # scratch takes many more steps


def test_median_speedup_exceeds_target():
    rng = np.random.default_rng(0)
    speedups = []
    for _ in range(20):                         # 20 synthetic "morphologies"
        wk = find_knee(*_warm_curve())
        sk = find_knee(*_scratch_curve(rng.uniform(20_000, 50_000)))
        speedups.append(sk / max(wk, 1))
    assert np.median(speedups) >= 3.0


def test_slower_learner_has_later_knee():
    # find_knee orders learners by how long they take to reach their own plateau
    k_fast = find_knee(*_scratch_curve(10_000))
    k_med = find_knee(*_scratch_curve(40_000))
    k_slow = find_knee(*_scratch_curve(1_000_000))
    assert k_fast < k_med <= k_slow


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
