"""Stage 4.1 acceptance: micro-mutation perturbs continuous params while keeping
the genome valid, compilable, and topologically unchanged.

Checks: output validates + compiles; all values stay positive/in-range even under
large sigma; topology (parts/edges/muscles + recursion/symmetry) is preserved (so
actuator count is unchanged and the controller warm-starts); deterministic under a
seeded rng; the mutant differs from the parent; the parent is not mutated in place.

Runs:  python3 tests/test_mutate.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.encoding import micro
from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology


def _rng(s=0):
    return np.random.default_rng(s)


def test_micro_validates_and_compiles():
    for fn in (agent_zero, quadruped):
        g = micro(fn(), _rng(0))
        assert g.validate_shape() == []
        model, _ = compile_morphology(develop(g))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.xpos)) and model.body_mass[1:].sum() > 0


def test_micro_values_stay_in_range():
    g = micro(quadruped(), _rng(1), sigma=0.4)
    for p in g.parts:
        assert all(v > 0 for v in p.dims.values()) and p.density > 0
    for e in g.edges:
        assert e.joint.range[0] <= e.joint.range[1]
        assert e.joint.stiffness >= 0 and e.joint.damping >= 0 and e.scale > 0
    for m in g.muscles:
        assert m.pcsa_cm2 > 0 and m.optimal_fiber_len > 0
        assert m.tendon_slack_len > 0 and m.vmax > 0 and m.activation_cost > 0
        assert 0.0 <= m.fiber_type <= 1.0 and 0.0 <= m.pennation <= 1.31


def test_micro_preserves_topology():
    g0 = quadruped()
    g = micro(g0, _rng(2))
    assert g.part_ids == g0.part_ids
    assert len(g.edges) == len(g0.edges) and len(g.muscles) == len(g0.muscles)
    for e, e0 in zip(g.edges, g0.edges):
        assert (e.parent_part, e.child_part, e.recursion_count, e.symmetry) == \
               (e0.parent_part, e0.child_part, e0.recursion_count, e0.symmetry)
    for m, m0 in zip(g.muscles, g0.muscles):
        assert (m.origin, m.insertion) == (m0.origin, m0.insertion)   # routing unchanged


def test_micro_changes_the_genome():
    g0 = quadruped()
    assert micro(g0, _rng(3)).hash() != g0.hash()


def test_micro_is_deterministic():
    assert micro(quadruped(), _rng(7)).hash() == micro(quadruped(), _rng(7)).hash()


def test_micro_does_not_mutate_parent():
    g0 = quadruped()
    h0 = g0.hash()
    micro(g0, _rng(0), sigma=0.5)
    assert g0.hash() == h0


def test_micro_extreme_sigma_still_valid_and_compiles():
    g = micro(agent_zero(), _rng(9), sigma=1.0)        # heavy perturbation
    assert g.validate_shape() == []
    model, _ = compile_morphology(develop(g))
    mujoco.mj_forward(model, mujoco.MjData(model))


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
