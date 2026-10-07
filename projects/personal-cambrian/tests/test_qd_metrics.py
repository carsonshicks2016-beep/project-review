"""Stage 5.4 acceptance: QD metrics + snapshots are tracked, reproducible per seed,
and plots regenerate from the saved logs.

Runs:  python3 tests/test_qd_metrics.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import (
    run_qd, QDConfig, MAPElites, archive_novelty, fitness_grid,
    save_metrics, load_metrics, plot_metrics, plot_archive, has_matplotlib,
)

TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                ep_steps=40, hidden=32, macro_rate=0.5, snapshot_every=3, seed=0)


def _run():
    logs = []
    archive, _ = run_qd([quadruped()], TINY, log_fn=logs.append)
    return archive, logs


def test_records_have_qd_metric_series():
    _, logs = _run()
    assert len(logs) == TINY.iterations
    for r in logs:
        for key in ("coverage", "qd_score", "max_fitness", "novelty", "cells"):
            assert key in r and np.isfinite(r[key])
        assert r["novelty"] >= 0.0


def test_snapshots_recorded_at_interval():
    _, logs = _run()
    snaps = [r for r in logs if "snapshot" in r]
    assert len(snaps) == TINY.iterations // TINY.snapshot_every
    grid = snaps[-1]["snapshot"]
    assert len(grid) == TINY.bins and len(grid[0]) == TINY.bins   # bins x bins


def test_metrics_reproducible_per_seed():
    _, a = _run()
    _, b = _run()
    for key in ("coverage", "qd_score", "max_fitness", "novelty"):
        assert [r[key] for r in a] == [r[key] for r in b]


def test_metrics_jsonl_roundtrip():
    _, logs = _run()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "metrics.jsonl")
        save_metrics(logs, path)
        loaded = load_metrics(path)
    assert loaded == logs                       # incl. snapshots


def test_archive_novelty_grows_with_spread():
    a = MAPElites(axes=["aspect", "speed"], bins=10)
    assert archive_novelty(a) == 0.0            # empty
    a.add(quadruped(), 1.0, {"aspect": 0.5, "speed": 0.0})
    assert archive_novelty(a) == 0.0            # single elite
    a.add(quadruped(), 1.0, {"aspect": 5.0, "speed": 5.0})
    assert archive_novelty(a) > 0.0             # spread -> positive


def test_fitness_grid_shape():
    a = MAPElites(axes=["aspect", "limb_count"], bins=8)
    a.add(quadruped(), 1.0, {"aspect": 1.0, "limb_count": 11})
    g = fitness_grid(a)
    assert g.shape == (8, 8) and np.isfinite(g).sum() == 1
    assert fitness_grid(MAPElites(axes=["aspect", "speed", "mass"], bins=4)) is None


def test_plots_regenerate_from_logs():
    if not has_matplotlib():
        print("  (skipped: no matplotlib)")
        return
    archive, logs = _run()
    with tempfile.TemporaryDirectory() as d:
        m = plot_metrics(load_metrics_roundtrip(logs, d), os.path.join(d, "m.png"))
        a = plot_archive(archive, os.path.join(d, "a.png"))
        assert os.path.getsize(m) > 1000 and os.path.getsize(a) > 1000


def load_metrics_roundtrip(logs, d):
    path = os.path.join(d, "metrics.jsonl")
    save_metrics(logs, path)
    return load_metrics(path)                   # prove plots come FROM the saved logs


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
