"""Stage 1.6 acceptance: muscles compile to spatial tendons + actuators that
actually torque their joint.

Checks: each MuscleRoute becomes a tendon + actuator; the tendon's moment arm
about the spanned joint is non-zero (finite-difference on tendon length); driving
the actuator moves the joint; chain muscles produce one tendon per joint.

Runs:  python3 tests/test_actuation.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from personal_cambrian.encoding import (
    Shape, JointType, AttachmentSite, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology


def actuator_fixture() -> Genome:
    """Fixed base + one hinge arm + a muscle running parallel to the arm, offset
    in x, so it has a clear moment arm about the y-hinge."""
    base = PartNode(id="base", shape=Shape.BOX, dims={"x": 0.10, "y": 0.10, "z": 0.10},
                    sites=[AttachmentSite("hinge", pos=(0.0, 0.0, 0.10)),
                           AttachmentSite("anchor", pos=(0.08, 0.0, 0.05))])
    arm = PartNode(id="arm", shape=Shape.CAPSULE, dims={"radius": 0.03, "length": 0.30},
                   sites=[AttachmentSite("ins", pos=(0.08, 0.0, 0.20))])
    edge = ConnectionEdge("base", "arm", site_idx=0,
                          joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-2.0, 2.0)))
    muscle = MuscleGene(id="flex", origin=("base", 1), insertion=("arm", 0), pcsa_cm2=20.0)
    return Genome(root="base", parts=[base, arm], edges=[edge], muscles=[muscle])


def chain_with_muscle() -> Genome:
    s = PartNode(id="seg", shape=Shape.CAPSULE, dims={"radius": 0.04, "length": 0.30},
                 recursion_limit=8,
                 sites=[AttachmentSite("next", pos=(0, 0, 0.30)),
                        AttachmentSite("mid", pos=(0.05, 0, 0.15))])
    e = ConnectionEdge("seg", "seg", site_idx=0, recursion_count=3,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0)))
    mus = MuscleGene(id="flex", origin=("seg", 1), insertion=("seg", 0))
    return Genome(root="seg", parts=[s], edges=[e], muscles=[mus], max_depth=8)


def _arm_joint_qadr(model, morph):
    arm = next(b for b in morph.bodies if b.part_id == "arm")
    return model.jnt_qposadr[model.joint(f"{arm.id}:j").id]


def _hinge_qadrs(model):
    return [int(model.jnt_qposadr[j]) for j in range(model.njnt)
            if model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_HINGE)]


def _tendon_spans_some_hinge(model, tid):
    base = mujoco.MjData(model)
    mujoco.mj_forward(model, base)
    L0 = float(base.ten_length[tid])
    for qadr in _hinge_qadrs(model):
        d = mujoco.MjData(model)
        d.qpos[:] = base.qpos
        d.qpos[qadr] += 1e-4
        mujoco.mj_forward(model, d)
        if abs(float(d.ten_length[tid]) - L0) / 1e-4 > 1e-3:
            return True
    return False


# --------------------------------------------------------------------------- #
def test_muscle_becomes_tendon_and_actuator():
    model, _ = compile_morphology(develop(actuator_fixture()), free_root=False)
    assert model.ntendon == 1
    assert model.nu == 1
    assert model.actuator(0).trntype[0] == int(mujoco.mjtTrn.mjTRN_TENDON)


def test_moment_arm_about_joint_is_nonzero():
    morph = develop(actuator_fixture())
    model, _ = compile_morphology(morph, free_root=False)
    data = mujoco.MjData(model)
    tid = model.tendon("flex#1").id
    qadr = _arm_joint_qadr(model, morph)

    mujoco.mj_forward(model, data)
    L0 = float(data.ten_length[tid])
    data.qpos[qadr] += 1e-4
    mujoco.mj_forward(model, data)
    moment_arm = (float(data.ten_length[tid]) - L0) / 1e-4
    assert abs(moment_arm) > 1e-3


def test_actuation_moves_the_joint():
    # self_collide=False: the contrived fixture's arm overlaps the base box, and
    # parent-child contact would jam the joint (collision filtering is Stage 1.7).
    morph = develop(actuator_fixture())
    model, _ = compile_morphology(morph, free_root=False, self_collide=False)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    qadr = _arm_joint_qadr(model, morph)

    mujoco.mj_forward(model, data)
    q0 = float(data.qpos[qadr])
    data.ctrl[:] = 1.0
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[qadr]) - q0) > 0.05
    assert np.all(np.isfinite(data.qpos))


def test_actuation_direction_reverses_with_control_sign():
    def settle(ctrl):
        morph = develop(actuator_fixture())
        model, _ = compile_morphology(morph, free_root=False, self_collide=False)
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        qadr = _arm_joint_qadr(model, morph)
        data.ctrl[:] = ctrl
        for _ in range(300):
            mujoco.mj_step(model, data)
        return float(data.qpos[qadr])
    assert np.sign(settle(+1.0)) != np.sign(settle(-1.0))


def test_chain_makes_one_tendon_per_joint():
    morph = develop(chain_with_muscle())          # 4 segs -> 3 joints
    model, _ = compile_morphology(morph, free_root=True)
    assert model.ntendon == 3
    assert model.nu == 3
    for t in range(model.ntendon):
        assert _tendon_spans_some_hinge(model, t)


def test_muscleless_creature_has_no_actuators():
    from test_morphogenesis import leg_genome
    model, _ = compile_morphology(develop(leg_genome()))
    assert model.ntendon == 0 and model.nu == 0


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
