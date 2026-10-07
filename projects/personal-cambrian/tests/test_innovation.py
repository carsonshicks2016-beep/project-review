"""Stage 4.4 acceptance: the innovation detector classifies parent->child changes.

Each macro operator's structural effect must produce the right flag; micro-only
and identical genomes produce no flags and zero structural distance.

Runs:  python3 tests/test_innovation.py   (requires mujoco for none; pure genome diff)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.encoding import diff, innovations, structural_distance, micro
from personal_cambrian.encoding.mutate import (
    add_part, delete_subtree, duplicate_subtree, change_recursion_count,
    toggle_symmetry, add_muscle, remove_muscle, reroute_tendon, split_muscle,
    fuse_muscles, add_joint_dof, remove_dof,
)
from personal_cambrian.seeds import agent_zero


def _rng(s=1):
    return np.random.default_rng(s)


# operator -> a flag that MUST appear in its parent->child diff
EXPECTED = [
    (add_part, "+limb"),
    (delete_subtree, "-limb"),
    (duplicate_subtree, "+limb"),
    (add_muscle, "+muscle"),
    (remove_muscle, "-muscle"),
    (split_muscle, "+compartment"),
    (fuse_muscles, "-muscle"),
    (change_recursion_count, "topology_change"),
    (toggle_symmetry, "topology_change"),
    (reroute_tendon, "topology_change"),
    (add_joint_dof, "dof_change"),
    (remove_dof, "dof_change"),
]


def test_each_operator_is_classified():
    for op, flag in EXPECTED:
        parent = agent_zero()
        child, rec = op(parent, _rng(1))
        assert rec.ok, op.__name__
        flags = innovations(parent, child)
        assert flag in flags, f"{op.__name__}: expected {flag}, got {flags}"


def test_micro_produces_no_structural_flags():
    parent = agent_zero()
    child = micro(parent, _rng(0), sigma=0.3)
    assert innovations(parent, child) == set()
    assert structural_distance(parent, child) == 0


def test_identical_genome_has_zero_distance():
    g = agent_zero()
    assert structural_distance(g, g) == 0
    assert not diff(g, g)                       # report is falsy when no innovation


def test_macro_increases_distance():
    parent = agent_zero()
    child, _ = add_part(parent, _rng(1))
    assert structural_distance(parent, child) >= 1


def test_report_counts_added_part():
    parent = agent_zero()
    child, _ = add_part(parent, _rng(1))
    r = diff(parent, child)
    assert len(r.parts_added) == 1 and not r.parts_removed
    assert bool(r) is True                      # truthy: an innovation occurred


def test_split_reports_compartment_and_muscle_churn():
    parent = agent_zero()
    child, _ = split_muscle(parent, _rng(1))
    f = innovations(parent, child)
    assert {"+compartment", "+muscle", "-muscle"} <= f


def test_distance_is_symmetric():
    parent = agent_zero()
    child, _ = duplicate_subtree(parent, _rng(2))
    assert structural_distance(parent, child) == structural_distance(child, parent)


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
