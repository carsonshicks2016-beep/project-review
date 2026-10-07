"""Stage 1.9: consolidated determinism + compile guarantees for the whole
encoding pipeline (genome -> develop -> MuJoCo), over every fixture and both seeds.

Locks: genome hashing is stable; develop() is a pure function; MJCF export is
byte-identical; the compiled model structure is identical run-to-run; a genome
that round-trips through JSON rebuilds an IDENTICAL creature (content-addressed
reproducibility, needed for the lineage store); mj_forward is deterministic; and
every genome compiles to a finite model with positive mass.

Runs:  python3 tests/test_determinism.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from personal_cambrian.encoding import Genome
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology, export_xml

from test_morphogenesis import leg_genome, radial_genome, chain_genome
from test_actuation import actuator_fixture, chain_with_muscle
from test_collision import sibling_overlap, faller
from personal_cambrian.seeds import agent_zero, quadruped

GENOMES = {
    "leg": leg_genome(), "radial": radial_genome(5), "chain": chain_genome(3),
    "actuator": actuator_fixture(), "chain_muscle": chain_with_muscle(),
    "siblings": sibling_overlap(), "faller": faller(),
    "agent_zero": agent_zero(), "quadruped": quadruped(),
}


def _morph_sig(m):
    bodies = tuple((b.id, b.parent_id, b.part_id, b.world_pos, b.world_quat, b.rel_pos)
                   for b in m.bodies)
    muscles = tuple((mu.id, tuple((w.body_id, w.site_idx) for w in mu.waypoints))
                    for mu in m.muscles)
    return bodies, muscles


def _model_sig(model):
    return (model.nbody, model.njnt, model.nu, model.ntendon, model.nq, model.nv,
            model.ngeom, model.nsite, round(float(model.body_mass.sum()), 9))


# --------------------------------------------------------------------------- #
def test_genome_hash_is_stable():
    for g in GENOMES.values():
        assert g.hash() == g.hash()


def test_develop_is_pure():
    for name, g in GENOMES.items():
        assert _morph_sig(develop(g)) == _morph_sig(develop(g)), name


def test_mjcf_export_is_byte_identical():
    for name, g in GENOMES.items():
        assert export_xml(develop(g)) == export_xml(develop(g)), name


def test_compiled_model_structure_is_identical():
    for name, g in GENOMES.items():
        m1, _ = compile_morphology(develop(g))
        m2, _ = compile_morphology(develop(g))
        assert _model_sig(m1) == _model_sig(m2), name


def test_json_roundtrip_reproduces_identical_creature():
    # the content-addressed invariant: a stored genome rebuilds the SAME creature
    for name, g in GENOMES.items():
        g2 = Genome.from_json(g.to_json())
        assert g2.hash() == g.hash(), name
        assert export_xml(develop(g)) == export_xml(develop(g2)), name


def test_mj_forward_is_deterministic():
    for name, g in GENOMES.items():
        model, _ = compile_morphology(develop(g))
        a, b = mujoco.MjData(model), mujoco.MjData(model)
        mujoco.mj_forward(model, a)
        mujoco.mj_forward(model, b)
        assert np.array_equal(a.xpos, b.xpos), name


def test_all_genomes_compile_finite_with_positive_mass():
    for name, g in GENOMES.items():
        assert g.validate_shape() == [], name
        model, _ = compile_morphology(develop(g), add_floor=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.xpos)), name
        assert model.body_mass[1:].sum() > 0, name


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
