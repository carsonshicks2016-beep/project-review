"""Stage 11.2 acceptance: lifting-app (.wld) ingest -> strength Metrics w/ provenance.

Deterministic on a synthetic .wld fixture (machine variants must be excluded, the best
free-weight 1RM kept with its date), plus a guarded check on the real local export.

Runs:  python3 tests/test_lifting.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.anchor.lifting import parse_lifting, DEFAULT_LIFTING_PATH


def _set(w, reps, orm):
    return {"weight": w, "reps": reps, "volume": w * reps, "oneRM": orm}


def _ex(name, sets):
    return {"name": name, "style": "reps_weight", "category": "x", "sets": sets}


def _wld():
    return {
        "settings": {"weightInLbs": True},
        "user": {"totalVolume": 100000},
        "workouts": [
            {"date": "2024-01-01", "exercises": [
                _ex("Barbell Bench Press", [_set(135, 5, 160), _set(185, 3, 205)]),
                _ex("Machine Bench Press", [_set(300, 5, 360)]),   # machine -> excluded
                _ex("Squats", [_set(225, 5, 265)]),
                _ex("Romanian Deadlift", [_set(225, 8, 280)]),     # accessory -> excluded
            ]},
            {"date": "2024-02-01", "exercises": [
                _ex("Barbell Bench Press", [_set(195, 2, 210)]),   # higher 1RM, later date
            ]},
        ],
    }


def _parse_fixture():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "WeightliftingAppData.wld")
        with open(p, "w") as f:
            json.dump(_wld(), f)
        return parse_lifting(p)


def test_best_freeweight_1rm_with_provenance():
    ms = _parse_fixture()
    bench = ms.get("bench_press_1rm")
    assert bench.value == 210.0 and bench.status == "measured"    # max barbell 1RM
    assert bench.timestamp == "2024-02-01" and "Barbell Bench Press" in bench.source
    assert bench.unit == "lb"
    assert ms.get("squat_1rm").value == 265.0


def test_machine_and_accessory_variants_excluded():
    ms = _parse_fixture()
    assert ms.get("bench_press_1rm").value != 360.0               # machine bench not used
    assert ms.get("deadlift_1rm").status == "unknown"             # only RDL -> not invented


def test_totals_and_unknowns():
    ms = _parse_fixture()
    assert ms.get("total_workouts").value == 2
    assert ms.get("total_volume").value == 100000.0
    assert ms.get("overhead_press_1rm").status == "unknown"       # never logged -> unknown


def test_real_export_if_present():
    if not os.path.exists(DEFAULT_LIFTING_PATH):
        print("  (skipped: no local .wld export)")
        return
    ms = parse_lifting(DEFAULT_LIFTING_PATH)
    assert ms.get("total_workouts").value >= 1
    import math
    for n in ("bench_press_1rm", "squat_1rm", "overhead_press_1rm"):
        m = ms.get(n)
        if m.known:
            assert math.isfinite(m.value) and m.value > 0 and m.source.startswith("lifting:")


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
