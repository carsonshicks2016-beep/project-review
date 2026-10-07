"""Stage 4.2 acceptance: macro-mutation operators invent/edit structure while
keeping the genome developable (valid + compilable).

Every operator either applies (ok) and yields a developable genome, or is a clean
no-op when no valid target exists. A long random chain of macro-mutations stays
developable throughout (the real stress test). Each operator's structural effect
is checked.

Runs:  python3 tests/test_macro.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.encoding import macro, MACRO_OPERATORS
from personal_cambrian.encoding.mutate import (
    add_part, delete_subtree, duplicate_subtree, add_muscle, remove_muscle,
    split_muscle, fuse_muscles,
)
from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology


def _developable(g):
    assert g.validate_shape() == [], g.validate_shape()[:3]
    model, _ = compile_morphology(develop(g))
    mujoco.mj_forward(model, mujoco.MjData(model))
    return g


def _rng(s=0):
    return np.random.default_rng(s)


def test_all_operators_apply_and_stay_developable():
    for name, op in MACRO_OPERATORS.items():
        g, rec = op(agent_zero(), _rng(0))      # agent_zero is rich enough for all ops
        assert rec.ok, f"{name} should apply on agent_zero"
        assert rec.kind == name
        _developable(g)


def test_structural_count_changes():
    g0 = agent_zero()
    np_ = len(g0.parts)
    nm = len(g0.muscles)
    assert len(add_part(g0, _rng(1))[0].parts) == np_ + 1
    assert len(delete_subtree(g0, _rng(1))[0].parts) < np_
    assert len(duplicate_subtree(g0, _rng(1))[0].parts) > np_
    assert len(add_muscle(g0, _rng(1))[0].muscles) == nm + 1
    assert len(remove_muscle(g0, _rng(1))[0].muscles) == nm - 1
    assert len(split_muscle(g0, _rng(1))[0].muscles) == nm + 1      # -1 +2
    assert len(fuse_muscles(g0, _rng(1))[0].muscles) == nm - 1      # -2 +1


def test_no_op_when_no_valid_target():
    # quadruped has a single muscle -> fuse has no valid pair
    g, rec = fuse_muscles(quadruped(), _rng(0))
    assert rec.ok is False
    _developable(g)                              # the returned copy is still valid
    # a muscle-less genome -> remove/split/reroute are no-ops
    g0 = quadruped()
    g0.muscles = []
    assert remove_muscle(g0, _rng(0))[1].ok is False


def test_long_random_macro_chain_stays_developable():
    rng = _rng(3)
    g = quadruped()
    for step in range(25):
        g, rec = macro(g, rng)
        assert g.validate_shape() == [], (step, rec.kind, g.validate_shape()[:2])
        compile_morphology(develop(g))           # still compiles after each mutation
    assert g.hash() != quadruped().hash()


def test_macro_is_deterministic():
    a, ra = macro(quadruped(), _rng(11))
    b, rb = macro(quadruped(), _rng(11))
    assert a.hash() == b.hash() and ra.kind == rb.kind


def test_operators_do_not_mutate_parent():
    g0 = agent_zero()
    h0 = g0.hash()
    for op in MACRO_OPERATORS.values():
        op(g0, _rng(5))
    assert g0.hash() == h0


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
