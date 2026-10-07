"""Stage 10.6 acceptance: one-command visual report orchestration (bpy-free).

The full report's 3D renders are produced by Blender via scripts/render_run.py
(verified by running it); here we test the bpy-free orchestration -- that a finished
run is turned into every Blender-free artifact (phylogeny PNG, scene/replay/biomech/
lineage JSON inputs, genomes, summary) plus a correct render manifest.

Runs:  python3 tests/test_report.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import run_qd, QDConfig
from personal_cambrian.viz.report import prepare_report

_TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=10,
                 seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                 ep_steps=40, hidden=32, macro_rate=0.7, novelty=True, seed=0)


def test_prepare_report_produces_full_report():
    archive, phylo = run_qd([quadruped()], _TINY)
    with tempfile.TemporaryDirectory() as d:
        rep = prepare_report(archive, phylo, d, replay_steps=20, name="quad")
        # every Blender-free input + the phylogeny image + genomes + summary exist
        for f in ("phylogeny_tree.png", "best_plan.json", "best_traj.json",
                  "best_bio.json", "lineage_plan.json", "best.genome.json",
                  "most_divergent.genome.json", "summary.json"):
            assert os.path.getsize(os.path.join(d, f)) > 0, f"missing {f}"
        # the manifest tells render_run what Blender should produce
        kinds = {out: kind for _in, out, kind in rep["manifest"]}
        assert kinds == {"best_creature.png": "static", "best_replay": "anim",
                         "best_biomechanics": "anim", "lineage_comparison.png": "static"}
        # every manifest input file actually exists
        for json_in, _out, _kind in rep["manifest"]:
            assert os.path.exists(os.path.join(d, json_in))


def test_summary_has_run_stats():
    archive, phylo = run_qd([quadruped()], _TINY)
    with tempfile.TemporaryDirectory() as d:
        rep = prepare_report(archive, phylo, d, replay_steps=15, name="quad")
        s = rep["summary"]
        for key in ("coverage", "qd_score", "n_creatures", "max_depth",
                    "most_divergent_distance", "lineage_length", "artifacts"):
            assert key in s
        assert s["n_creatures"] >= 1 and s["lineage_length"] >= 1
        # summary.json on disk matches
        with open(os.path.join(d, "summary.json")) as f:
            assert json.load(f)["n_creatures"] == s["n_creatures"]


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
