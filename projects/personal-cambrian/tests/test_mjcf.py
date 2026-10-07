"""Stage 1.5 acceptance: a developed Morphology compiles to a valid MuJoCo model.

Checks: spec.compile() succeeds; mj_forward runs with no NaN; total mass > 0;
no resting interpenetration beyond tolerance between non-adjacent bodies; the
body tree, joints (free root + articulations), geoms, and sites are present; and
joint limits are interpreted in radians.

Requires mujoco (first dependency beyond NumPy). Runs:  python3 tests/test_mjcf.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.encoding import (
    Shape, JointType, SymmetryKind, AttachmentSite, Joint, PartNode,
    ConnectionEdge, MuscleGene, Genome,
)
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import (
    build_spec, compile_morphology, export_xml,
)

# reuse the morphology fixtures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_morphogenesis import leg_genome, radial_genome, chain_genome


def _compile(g, **opts):
    return compile_morphology(develop(g), **opts)


def _max_nonadjacent_penetration(model, data, morph):
    """Largest interpenetration depth between geoms NOT on parent-child bodies."""
    # body-name -> parent-name adjacency from the morphology
    parent = {b.id: b.parent_id for b in morph.bodies}
    worst = 0.0
    for c in data.contact[: data.ncon]:
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        adjacent = parent.get(b1) == b2 or parent.get(b2) == b1
        if adjacent:
            continue
        if c.dist < 0:
            worst = max(worst, -c.dist)
    return worst


# --------------------------------------------------------------------------- #
# core acceptance                                                              #
# --------------------------------------------------------------------------- #
def test_all_fixtures_compile_and_run():
    for g in (leg_genome(), radial_genome(5), chain_genome(3)):
        morph = develop(g)
        model, _ = compile_morphology(morph)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
        assert np.all(np.isfinite(data.xpos))
        assert model.body_mass[1:].sum() > 0          # exclude world body
        assert _max_nonadjacent_penetration(model, data, morph) < 1e-3


def test_body_and_geom_counts_match_morphology():
    morph = develop(leg_genome())
    model, _ = compile_morphology(morph)
    assert model.nbody == morph.body_count + 1        # + world body
    assert model.ngeom == morph.body_count            # one geom per body
    assert model.nsite == sum(len(b.sites) for b in morph.bodies)


def test_free_root_plus_articulations():
    morph = develop(leg_genome())
    model, _ = compile_morphology(morph)
    # 1 free joint (root) + one hinge per non-root body (2 thighs + 2 shanks)
    assert model.njnt == 1 + 4
    # a free joint contributes 7 to nq, each hinge 1  ->  7 + 4
    assert model.nq == 11 and model.nv == 10


def test_fixed_joint_welds_body():
    s = PartNode(id="a", shape=Shape.BOX, dims={"x": 0.1, "y": 0.1, "z": 0.1},
                 sites=[AttachmentSite("s", pos=(0, 0, 0.1))])
    b = PartNode(id="b", shape=Shape.SPHERE, dims={"radius": 0.05})
    g = Genome(root="a", parts=[s, b], edges=[
        ConnectionEdge("a", "b", site_idx=0, joint=Joint(JointType.FIXED)),
    ])
    model, _ = compile_morphology(develop(g))
    # only the free root joint exists; the FIXED child adds none
    assert model.njnt == 1


def test_joint_limits_are_radians():
    # thigh joint default range is (-pi/2, pi/2); it must survive as radians
    morph = develop(leg_genome())
    model, _ = compile_morphology(morph)
    jr = model.jnt_range[model.jnt_range[:, 1] > 0]   # limited joints
    assert np.any(np.isclose(jr[:, 1], np.pi / 2, atol=1e-3))


def test_shapes_map_to_geom_types():
    parts = [
        PartNode(id="cap", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                 sites=[AttachmentSite("s1", pos=(0, 0, 0.15)),
                        AttachmentSite("s2", pos=(0, 0.1, 0)),
                        AttachmentSite("s3", pos=(0, -0.1, 0))]),
        PartNode(id="box", shape=Shape.BOX, dims={"x": 0.05, "y": 0.05, "z": 0.05}),
        PartNode(id="sph", shape=Shape.SPHERE, dims={"radius": 0.04}),
        PartNode(id="ell", shape=Shape.ELLIPSOID, dims={"x": 0.04, "y": 0.05, "z": 0.06}),
    ]
    g = Genome(root="cap", parts=parts, edges=[
        ConnectionEdge("cap", "box", site_idx=0),
        ConnectionEdge("cap", "sph", site_idx=1),
        ConnectionEdge("cap", "ell", site_idx=2),
    ])
    model, _ = compile_morphology(develop(g))
    types = set(int(t) for t in model.geom_type)
    for want in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX,
                 mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        assert int(want) in types


def test_export_xml_is_valid_mjcf():
    xml = export_xml(develop(radial_genome(4)))
    assert xml.lstrip().startswith("<mujoco")
    # the exported XML must itself reload
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody > 1


def test_floor_and_no_selfcollide_options():
    morph = develop(chain_genome(3))
    model, _ = compile_morphology(morph, add_floor=True, self_collide=False)
    assert any(model.geom(i).name == "floor" for i in range(model.ngeom))
    mujoco.mj_forward(model, mujoco.MjData(model))  # still runs


def test_compile_is_deterministic():
    a = export_xml(develop(leg_genome()))
    b = export_xml(develop(leg_genome()))
    assert a == b


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
