"""Stage 1.7 acceptance: collision filtering & contact groups.

Checks: parent-child contacts are excluded so a connected limb flexes through its
full range without phantom self-contact (the jam we hit in Stage 1.6); the floor
collides with and supports a creature; non-adjacent bodies still collide (so Stage
6 can detect self-intersection); and self_collide=False disables body-body
contact while keeping floor contact.

Runs:  python3 tests/test_collision.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from personal_cambrian.encoding import (
    Shape, JointType, AttachmentSite, Joint, PartNode, ConnectionEdge, Genome,
)
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology
from test_actuation import actuator_fixture, _arm_joint_qadr
from test_morphogenesis import leg_genome


def _contact_bodies(model, data):
    """Set of frozenset({bodyA, bodyB}) for the current contacts."""
    out = []
    for c in data.contact[: data.ncon]:
        out.append(frozenset((model.body(model.geom_bodyid[c.geom1]).name,
                              model.body(model.geom_bodyid[c.geom2]).name)))
    return out


def sibling_overlap() -> Genome:
    """Two hinged children attached at the SAME site -> they overlap and are
    siblings (not parent-child), so they SHOULD collide."""
    hub = PartNode(id="hub", shape=Shape.BOX, dims={"x": 0.10, "y": 0.10, "z": 0.10},
                   sites=[AttachmentSite("s", pos=(0, 0, 0.15))])
    a = PartNode(id="a", shape=Shape.BOX, dims={"x": 0.05, "y": 0.05, "z": 0.05})
    b = PartNode(id="b", shape=Shape.BOX, dims={"x": 0.05, "y": 0.05, "z": 0.05})
    return Genome(root="hub", parts=[hub, a, b], edges=[
        ConnectionEdge("hub", "a", site_idx=0, joint=Joint(JointType.HINGE, axis=(1, 0, 0))),
        ConnectionEdge("hub", "b", site_idx=0, joint=Joint(JointType.HINGE, axis=(1, 0, 0))),
    ])


def faller() -> Genome:
    return Genome(root="ball", parts=[
        PartNode(id="ball", shape=Shape.SPHERE, dims={"radius": 0.10})])


# --------------------------------------------------------------------------- #
def test_parent_child_exclude_lets_joint_swing():
    # default self_collide=True; the arm overlaps the base but is its child, so
    # the parent-child exclude must keep the joint free (no jam).
    morph = develop(actuator_fixture())
    model, _ = compile_morphology(morph)          # defaults (self_collide=True)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    qadr = _arm_joint_qadr(model, morph)
    data.ctrl[:] = 1.0
    for _ in range(600):
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[qadr])) > 0.05     # swings freely


def test_exclude_count_matches_non_root_bodies():
    morph = develop(leg_genome())                 # 5 bodies -> 4 non-root
    model, _ = compile_morphology(morph)
    assert model.nexclude == sum(1 for b in morph.bodies if b.parent_id is not None)


def test_nonadjacent_siblings_collide():
    morph = develop(sibling_overlap())
    a = next(b.id for b in morph.bodies if b.part_id == "a")
    b = next(x.id for x in morph.bodies if x.part_id == "b")
    model, _ = compile_morphology(morph, self_collide=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert frozenset((a, b)) in _contact_bodies(model, data)   # they DO collide


def test_self_collide_false_disables_body_body():
    morph = develop(sibling_overlap())
    a = next(b.id for b in morph.bodies if b.part_id == "a")
    b = next(x.id for x in morph.bodies if x.part_id == "b")
    model, _ = compile_morphology(morph, self_collide=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert frozenset((a, b)) not in _contact_bodies(model, data)


def test_floor_supports_a_falling_body():
    morph = develop(faller())
    model, _ = compile_morphology(morph, add_floor=True, free_root=True)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.5                            # drop from 0.5 m
    for _ in range(2000):
        mujoco.mj_step(model, data)
    z = float(data.qpos[2])
    assert 0.08 < z < 0.13                        # rests on floor at ~radius (0.10)
    assert np.all(np.isfinite(data.qpos))
    fid = model.geom("floor").id                 # a real floor contact exists
    assert any(c.geom1 == fid or c.geom2 == fid for c in data.contact[: data.ncon])


def test_floor_collision_present_self_collide_off():
    # even with body-body collision off, the floor must still collide
    morph = develop(faller())
    model, _ = compile_morphology(morph, add_floor=True, self_collide=False)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.5
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert 0.08 < float(data.qpos[2]) < 0.13


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
