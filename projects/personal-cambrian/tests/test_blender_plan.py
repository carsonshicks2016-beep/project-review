"""Stage 10.1 acceptance: morphology -> Blender scene plan (bpy-free core).

The bpy builder (blender_build.py) is verified by actually running it in Blender;
here we test the bpy-FREE half -- that `build_plan` extracts a faithful, JSON-
serializable scene description from a developed Morphology (the right geometry per
body, world transforms, muscle polylines) so "a seed creature appears correctly in
Blender" rests on correct data. We also check the bpy module imports safely without
Blender installed.

Runs:  python3 tests/test_blender_plan.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.morphogenesis import develop
from personal_cambrian.viz.morphology_plan import build_plan, plan_from_genome, ScenePlan

_SHAPE_DIMS = {"capsule": {"radius", "length"}, "box": {"x", "y", "z"},
               "sphere": {"radius"}, "ellipsoid": {"x", "y", "z"}}


def test_plan_has_one_geom_per_body():
    m = develop(quadruped())
    plan = build_plan(m, name="quad")
    assert len(plan.geoms) == m.body_count
    assert len(plan.muscles) == len(m.muscles)
    assert plan.name == "quad"


def test_geoms_match_body_geometry():
    m = develop(quadruped())
    plan = build_plan(m)
    by_id = {b.id: b for b in m.bodies}
    for g in plan.geoms:
        b = by_id[g.name]
        assert g.shape in _SHAPE_DIMS
        assert set(g.dims) >= _SHAPE_DIMS[g.shape]            # correct dim keys for the shape
        assert np.allclose(g.pos, [float(v) for v in b.world_pos])
        assert len(g.quat) == 4 and np.all(np.isfinite(g.quat))
        assert np.all(np.isfinite(g.pos))


def test_root_has_no_parent_others_do():
    plan = plan_from_genome(quadruped())
    roots = [g for g in plan.geoms if g.parent is None]
    assert len(roots) == 1                                   # exactly one root body
    assert all(g.parent is not None for g in plan.geoms if g is not roots[0])


def test_muscles_are_polylines_in_world_space():
    m = develop(quadruped())
    plan = build_plan(m)
    assert plan.muscles, "quadruped seed should have muscles"
    for ms in plan.muscles:
        assert len(ms.points) >= 2
        assert all(len(p) == 3 and np.all(np.isfinite(p)) for p in ms.points)
        assert ms.fmax > 0.0


def test_plan_dict_roundtrip():
    plan = plan_from_genome(agent_zero(), name="zero")
    d = plan.to_dict()
    import json
    again = ScenePlan.from_dict(json.loads(json.dumps(d)))   # survives JSON
    assert again.name == "zero"
    assert len(again.geoms) == len(plan.geoms)
    assert again.geoms[0].dims == plan.geoms[0].dims


def test_bounds_spans_the_creature():
    plan = plan_from_genome(quadruped())
    lo, hi = plan.bounds()
    assert all(hi[i] >= lo[i] for i in range(3))
    assert any(hi[i] > lo[i] for i in range(3))             # not a degenerate point


def test_blender_build_imports_without_bpy():
    # the bpy builder must be importable in plain Python (for source-shipping); its
    # build entrypoint refuses to run without Blender rather than crashing on import.
    from personal_cambrian.viz import blender_build
    if not blender_build._HAS_BPY:
        try:
            blender_build.build_from_plan({"name": "x", "geoms": [], "muscles": []})
            assert False, "expected RuntimeError without bpy"
        except RuntimeError:
            pass


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
