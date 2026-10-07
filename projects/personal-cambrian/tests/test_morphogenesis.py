"""Stage 1.3 acceptance: morphogenesis expands the genome graph into a concrete
body with the correct body count and transforms, with recursion/depth caps,
branching, terminal_only, per-joint muscles, scaling, mass, and determinism.

Semantics note: `recursion_count = k` appends k child instances, so a root plus a
self-edge with recursion_count=3 develops a 4-body chain (root + 3 segments).

Runs with zero install:  python3 tests/test_morphogenesis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from personal_cambrian.encoding import (
    Shape, JointType, SymmetryKind, AttachmentSite, Joint, PartNode,
    ConnectionEdge, MuscleGene, Genome,
)
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.develop import _qmul, _qrot


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def vapprox(v, expected, tol=1e-6):
    return all(approx(x, y, tol) for x, y in zip(v, expected))


def seg(part_id="seg", recursion_limit=8) -> PartNode:
    # capsule with a "next" site at +0.30 z and a "mid" site offset in y
    return PartNode(id=part_id, shape=Shape.CAPSULE, dims={"radius": 0.04, "length": 0.30},
                    recursion_limit=recursion_limit,
                    sites=[AttachmentSite("next", pos=(0.0, 0.0, 0.30)),
                           AttachmentSite("mid", pos=(0.0, 0.05, 0.15))])


def chain_genome(recursion_count=3, max_depth=8, recursion_limit=8) -> Genome:
    s = seg(recursion_limit=recursion_limit)
    e = ConnectionEdge(parent_part="seg", child_part="seg", site_idx=0,  # attach at "next"
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0)),
                       recursion_count=recursion_count)
    return Genome(root="seg", parts=[s], edges=[e], max_depth=max_depth)


# --------------------------------------------------------------------------- #
# core acceptance: chain body count + transforms                               #
# --------------------------------------------------------------------------- #
def test_chain_body_count():
    m = develop(chain_genome(recursion_count=3))
    assert m.body_count == 4  # root + 3 appended segments


def test_chain_world_transforms():
    m = develop(chain_genome(recursion_count=3))
    zs = sorted(b.world_pos[2] for b in m.bodies)
    # each "next" site is +0.30 z; segments stack at 0.0, 0.30, 0.60, 0.90
    assert all(approx(z, e) for z, e in zip(zs, [0.0, 0.30, 0.60, 0.90]))
    # parent/child relationship forms a single chain
    parents = [b.parent_id for b in m.bodies]
    assert parents.count(None) == 1  # exactly one root


# --------------------------------------------------------------------------- #
# termination caps                                                             #
# --------------------------------------------------------------------------- #
def test_max_depth_caps_chain():
    m = develop(chain_genome(recursion_count=10, max_depth=2))
    assert m.body_count == 3  # root (d0) + d1 + d2


def test_recursion_limit_caps_chain():
    # recursion_limit=3 allows at most 3 seg instances total along the path
    m = develop(chain_genome(recursion_count=10, recursion_limit=3))
    assert m.body_count == 3


# --------------------------------------------------------------------------- #
# branching + terminal_only                                                    #
# --------------------------------------------------------------------------- #
def test_branching_two_children():
    s = seg()
    g = Genome(root="seg", parts=[s], edges=[
        ConnectionEdge("seg", "seg", site_idx=0, recursion_count=1),  # at "next"
        ConnectionEdge("seg", "seg", site_idx=1, recursion_count=1,   # at "mid"
                       pos=(0.0, 0.10, 0.0)),
    ], max_depth=1)
    m = develop(g)
    assert m.body_count == 3  # root + two distinct children
    children = [b for b in m.bodies if b.parent_id is not None]
    assert children[0].world_pos != children[1].world_pos


def test_terminal_only_foot_at_tip_not_root():
    s = seg()
    foot = PartNode(id="foot", shape=Shape.SPHERE, dims={"radius": 0.05},
                    sites=[AttachmentSite("a", pos=(0, 0, 0))])
    g = Genome(root="seg", parts=[s, foot], edges=[
        ConnectionEdge("seg", "seg", site_idx=0, recursion_count=3),   # 3-long chain
        ConnectionEdge("seg", "foot", site_idx=0, terminal_only=True),  # only at tip
    ], max_depth=8)
    m = develop(g)
    feet = [b for b in m.bodies if b.part_id == "foot"]
    assert len(feet) == 1                       # exactly one foot, not one per seg
    tip = max((b for b in m.bodies if b.part_id == "seg"), key=lambda b: b.depth)
    assert feet[0].parent_id == tip.id          # attached to the deepest segment


# --------------------------------------------------------------------------- #
# muscles: one instance per matching joint                                     #
# --------------------------------------------------------------------------- #
def test_muscle_instantiated_per_joint():
    s = seg()
    e = ConnectionEdge("seg", "seg", site_idx=0, recursion_count=3)
    # origin "mid"(idx1) on parent seg, insertion "next"(idx0) on child seg
    mus = MuscleGene(id="flex", origin=("seg", 1), insertion=("seg", 0))
    m = develop(Genome(root="seg", parts=[s], edges=[e], muscles=[mus], max_depth=8))
    # 4 bodies => 3 joints => 3 muscle instances
    assert len(m.muscles) == 3
    for route in m.muscles:
        assert len(route.waypoints) == 2
        assert route.rest_length() > 0
        assert route.fmax > 0  # derived from PCSA * specific tension


# --------------------------------------------------------------------------- #
# scale, mass, determinism                                                     #
# --------------------------------------------------------------------------- #
def test_scale_shrinks_child():
    s = seg()
    e = ConnectionEdge("seg", "seg", site_idx=0, recursion_count=1, scale=0.5)
    m = develop(Genome(root="seg", parts=[s], edges=[e], max_depth=1))
    child = [b for b in m.bodies if b.parent_id is not None][0]
    assert approx(child.scale, 0.5)
    assert approx(child.dims["radius"], 0.04 * 0.5)
    # the +0.30 z offset is scaled by the parent's scale (1.0) -> still 0.30
    assert approx(child.world_pos[2], 0.30)


def test_total_mass_positive():
    m = develop(chain_genome(recursion_count=3))
    assert m.total_mass() > 0


def test_determinism():
    a = develop(chain_genome(recursion_count=4))
    b = develop(chain_genome(recursion_count=4))
    assert [x.id for x in a.bodies] == [x.id for x in b.bodies]
    assert [x.world_pos for x in a.bodies] == [x.world_pos for x in b.bodies]
    assert a.genome_hash == b.genome_hash


def test_summary_runs():
    m = develop(chain_genome(recursion_count=2))
    assert "Morphology(" in m.summary()


# --------------------------------------------------------------------------- #
# Stage 1.4: symmetry & limb repetition                                        #
# --------------------------------------------------------------------------- #
def _set_match(got, expected, tol=1e-5):
    got = [tuple(g) for g in got]
    for e in expected:
        hit = next((i for i, g in enumerate(got) if vapprox(g, e, tol)), None)
        if hit is None:
            return False
        got.pop(hit)
    return not got


def leg_genome() -> Genome:
    """A hub with one bilateral 'leg' rule (thigh -> shank). Plane xz -> mirror y."""
    hub = PartNode(id="hub", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.20},
                   sites=[AttachmentSite("hip", pos=(0.0, 0.15, 0.0))])
    thigh = PartNode(id="thigh", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.30},
                     sites=[AttachmentSite("knee", pos=(0.0, 0.0, 0.30))])
    shank = PartNode(id="shank", shape=Shape.CAPSULE, dims={"radius": 0.04, "length": 0.30})
    return Genome(root="hub", parts=[hub, thigh, shank], edges=[
        ConnectionEdge("hub", "thigh", site_idx=0, pos=(0.0, 0.10, 0.0),
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0)),
                       symmetry=SymmetryKind.BILATERAL),
        ConnectionEdge("thigh", "shank", site_idx=0),
    ])


def test_bilateral_makes_a_mirrored_pair():
    m = develop(leg_genome())
    assert m.body_count == 5  # hub + 2 thighs + 2 shanks
    thighs = [b for b in m.bodies if b.part_id == "thigh"]
    shanks = [b for b in m.bodies if b.part_id == "shank"]
    assert len(thighs) == 2 and len(shanks) == 2
    # mirrored across y: one limb at +y, its mirror at -y (x,z identical)
    assert _set_match([t.world_pos for t in thighs], [(0, 0.25, 0), (0, -0.25, 0)])
    assert _set_match([s.world_pos for s in shanks], [(0, 0.25, 0.30), (0, -0.25, 0.30)])


def test_bilateral_flips_joint_axis():
    m = develop(leg_genome())
    thighs = [b for b in m.bodies if b.part_id == "thigh"]
    pos_y = next(t for t in thighs if t.world_pos[1] > 0)
    neg_y = next(t for t in thighs if t.world_pos[1] < 0)
    assert vapprox(pos_y.joint.axis, (0, 1, 0))
    assert vapprox(neg_y.joint.axis, (0, -1, 0))   # mirrored across y -> y flips


def radial_genome(n=4) -> Genome:
    hub = PartNode(id="hub", shape=Shape.SPHERE, dims={"radius": 0.10},
                   sites=[AttachmentSite("top", pos=(0.0, 0.0, 0.10))])
    limb = PartNode(id="limb", shape=Shape.CAPSULE, dims={"radius": 0.03, "length": 0.20})
    return Genome(root="hub", parts=[hub, limb], edges=[
        ConnectionEdge("hub", "limb", site_idx=0, pos=(0.20, 0.0, 0.0),
                       symmetry=SymmetryKind.RADIAL, symmetry_count=n),
    ])


def test_radial_makes_n_evenly_spaced_limbs():
    n = 4
    m = develop(radial_genome(n))
    limbs = [b for b in m.bodies if b.part_id == "limb"]
    assert len(limbs) == n
    expected = [(0.20 * math.cos(2 * math.pi * k / n),
                 0.20 * math.sin(2 * math.pi * k / n), 0.10) for k in range(n)]
    assert _set_match([l.world_pos for l in limbs], expected)


def test_radial_three_limbs():
    m = develop(radial_genome(3))
    assert len([b for b in m.bodies if b.part_id == "limb"]) == 3


def test_rel_world_consistency_under_symmetry():
    # world transform must equal parent_world ∘ rel for EVERY body, including
    # reflected ones — this validates the rel transforms recomputed in _reflect_slice.
    for m in (develop(leg_genome()), develop(radial_genome(4)), develop(chain_genome(3))):
        by_id = {b.id: b for b in m.bodies}
        for b in m.bodies:
            if b.parent_id is None:
                continue
            p = by_id[b.parent_id]
            wpos = np.array(p.world_pos) + _qrot(np.array(p.world_quat), np.array(b.rel_pos))
            wquat = _qmul(np.array(p.world_quat), np.array(b.rel_quat))
            assert vapprox(wpos, b.world_pos, 1e-5)
            # quaternion equal up to sign
            assert vapprox(wquat, b.world_quat, 1e-5) or vapprox(-wquat, b.world_quat, 1e-5)


def test_symmetry_is_deterministic():
    a, b = develop(leg_genome()), develop(leg_genome())
    assert [x.id for x in a.bodies] == [x.id for x in b.bodies]
    assert [x.world_pos for x in a.bodies] == [x.world_pos for x in b.bodies]


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
