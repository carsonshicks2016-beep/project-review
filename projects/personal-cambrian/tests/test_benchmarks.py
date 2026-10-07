"""Stage 9.5 acceptance: throughput & surrogate benchmarks reported to runs/.

Done-when: numbers reported in runs/. We verify the report COLLECTORS produce the
expected metrics and that `save_report` writes a JSON file -- the surrogate
sims-saved benchmark (hardware-independent) carries the real headline number.

Runs:  python3 tests/test_benchmarks.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.evo.benchmarks import (
    surrogate_sims_benchmark, jax_archive_benchmark, stage9_report, save_report,
)
from personal_cambrian.evo.jax_archive import jax_archive_available


def test_surrogate_benchmark_reports_sims_saved():
    b = surrogate_sims_benchmark(sim_budget=200, seed=0)
    assert b["surrogate_qd"] > b["baseline_qd"]                 # surrogate wins at equal budget
    assert 0 < b["sims_to_match_baseline"] < 200               # reaches baseline qd early
    assert b["sims_saved_frac"] > 0.1                          # materially fewer sims


def test_jax_archive_benchmark_parity():
    b = jax_archive_benchmark(n=3000, bins=12, seed=0)
    if not b.get("available"):
        print("  (skipped: no jax)")
        return
    assert b["qd_parity"] is True
    assert b["batched_insert_ms"] >= 0.0


def test_stage9_report_structure_and_save():
    rep = stage9_report(sim_budget=200, archive_n=2000)
    for key in ("surrogate_9_4", "jax_archive_9_3", "mjx_9_1", "jax_ppo_9_2", "caveat"):
        assert key in rep
    assert rep["surrogate_9_4"]["sims_saved_frac"] > 0.1
    # mjx may or may not be present depending on whether the benchmark was run
    assert "available" in rep["mjx_9_1"]
    with tempfile.TemporaryDirectory() as d:
        path = save_report(rep, os.path.join(d, "stage9_benchmarks.json"))
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["surrogate_9_4"]["surrogate_qd"] == rep["surrogate_9_4"]["surrogate_qd"]


def test_mjx_json_is_folded_in_when_present():
    # if the 9.1 benchmark artifact exists, the report should mark mjx available
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mjx_json = os.path.join(root, "runs", "mjx_benchmark.json")
    rep = stage9_report(mjx_json=mjx_json, sim_budget=120, archive_n=1000)
    if os.path.exists(mjx_json):
        assert rep["mjx_9_1"]["available"] is True
        assert "parity" in rep["mjx_9_1"]
    else:
        assert rep["mjx_9_1"]["available"] is False


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
