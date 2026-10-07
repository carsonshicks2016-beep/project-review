"""Stage 11.4 acceptance: Bayesian near-human update (Kalman) within human bounds.

Done-when: held-out measurements fall within the forecast CI; predictions stay inside
the human-achievable range. Tested deterministically on synthetic training blocks
(plus a guarded check on the real lifting progression).

Runs:  python3 tests/test_calibrate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.anchor.calibrate import (
    LocalLinearTrend, calibrate_and_validate, lift_progression, HUMAN_BOUNDS,
)


def test_forecast_tracks_a_clean_trend():
    series = [200 + 5 * t for t in range(10)]              # perfectly linear gains
    c = calibrate_and_validate(series, holdout=3, param="bench_press_1rm")
    assert c.within_ci_fraction == 1.0                    # all held-out within CI
    assert c.predictions[-1] > c.predictions[0]           # captured the upward trend


def test_noisy_block_mostly_within_ci():
    rng = np.random.default_rng(0)
    series = [200 + 5 * t + rng.normal(0, 3) for t in range(12)]
    c = calibrate_and_validate(series, holdout=4, param="bench_press_1rm")
    print(f"  noisy within-CI fraction = {c.within_ci_fraction:.0%}")
    assert c.within_ci_fraction >= 0.5


def test_predictions_stay_human_achievable():
    runaway = [600 + 50 * t for t in range(8)]             # would blow past any human max
    c = calibrate_and_validate(runaway, holdout=3, param="bench_press_1rm")
    hi = HUMAN_BOUNDS["bench_press_1rm"][1]
    assert all(p <= hi for p in c.predictions)             # clamped to human-achievable
    crash = [200 - 40 * t for t in range(8)]               # would go negative
    c2 = calibrate_and_validate(crash, holdout=3, param="bench_press_1rm")
    assert all(p >= HUMAN_BOUNDS["bench_press_1rm"][0] for p in c2.predictions)


def test_ci_widens_with_horizon():
    f = LocalLinearTrend().fit([100 + 2 * t for t in range(8)])
    _means, cis = f.forecast(5)
    widths = [hi - lo for lo, hi in cis]
    assert widths[-1] > widths[0]                          # uncertainty grows out in time


def test_real_progression_if_present():
    p = os.path.expanduser("~/Desktop/WeightliftingAppData.wld")
    if not os.path.exists(p):
        print("  (skipped: no local .wld)")
        return
    prog = lift_progression(p, "bench_press_1rm")
    assert len(prog) > 10
    c = calibrate_and_validate(prog, holdout=8, param="bench_press_1rm", obs_var=120.0)
    assert 0.0 <= c.within_ci_fraction <= 1.0              # runs end-to-end on real data
    assert all(0 <= pr <= HUMAN_BOUNDS["bench_press_1rm"][1] for pr in c.predictions)


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
