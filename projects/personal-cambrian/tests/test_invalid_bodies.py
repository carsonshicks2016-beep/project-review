"""Stage 6.6 acceptance: the invalid-body battery -- lock viability behavior.

One place that exercises EVERY Stage-6 rejection rule through the single entry
point the QD loop actually uses (`pre_sim_gate`), plus a fuzz sweep proving the
gate is total (never raises) and only ever emits reasons from a known vocabulary.

Rules covered:
  supply (6.4):     orphan_muscle
  viability (6.2):  self_intersection, immobile_joint, no_actuators, zero_moment_arm
  structure (6.3):  bone_overload, tendon_overload
  robustness:       develop_error / compile_error (exercised by the fuzz sweep)

Runs:  python3 tests/test_invalid_bodies.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.genome import (
    Shape, JointType, AttachmentSite as S, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.encoding.mutate import mutate, MutationSchedule
from personal_cambrian.evo import pre_sim_gate

# every reason pre_sim_gate may ever produce (base token, before any ':detail')
KNOWN_REASONS = {
    "orphan_muscle", "self_intersection", "immobile_joint", "no_actuators",
    "zero_moment_arm", "bone_overload", "tendon_overload",
    "develop_error", "compile_error", "env_error",
}


def _hinge(rng=(-1, 1)):
    return Joint(JointType.HINGE, axis=(0, 1, 0), range=rng)


# --- one crafted body per rule ---------------------------------------------
def _orphan_muscle():
    root = PartNode(id="root", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.4},
                    sites=[S("s0", pos=(0, 0, -0.2)), S("s1", pos=(0.05, 0, 0.0))])
    floater = PartNode(id="floater", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                       sites=[S("f0", pos=(0.04, 0, 0.1))])
    mg = MuscleGene(id="orphan", origin=("root", 1), insertion=("floater", 0), pcsa_cm2=10.0)
    return Genome(root="root", parts=[root, floater], edges=[], muscles=[mg])


def _self_intersecting():
    root = PartNode(id="root", shape=Shape.CAPSULE, dims={"radius": 0.1, "length": 0.4},
                    sites=[S("s0", pos=(0, 0, 0.2))])
    a = PartNode(id="a", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                 sites=[S("t", pos=(0, 0, -0.15))])
    b = PartNode(id="b", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                 sites=[S("t2", pos=(0, 0, -0.15))])
    edges = [ConnectionEdge("root", "a", site_idx=0, joint=_hinge()),
             ConnectionEdge("root", "b", site_idx=0, joint=_hinge())]
    return Genome(root="root", parts=[root, a, b], edges=edges, muscles=[])


def _immobile_joint():
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                  sites=[S("c", pos=(0.05, 0, 0.1))])
    edges = [ConnectionEdge("p", "ch", site_idx=0, joint=_hinge((0.0, 0.0)))]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=10.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


def _no_actuators():
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3})
    edges = [ConnectionEdge("p", "ch", site_idx=0, joint=_hinge())]
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[])


def _zero_moment_muscle():
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("js", pos=(0, 0, -0.2)), S("o", pos=(0, 0, -0.2))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                  sites=[S("i", pos=(0, 0, 0.0))])
    edges = [ConnectionEdge("p", "ch", site_idx=0, joint=_hinge())]
    mg = MuscleGene(id="axis", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=10.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


def _bone_overload():
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.0035, "length": 0.3},
                  sites=[S("c", pos=(0.0, 0, 0.1))])
    edges = [ConnectionEdge("p", "ch", site_idx=0, joint=_hinge())]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=80.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


def _tendon_overload():
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.3},
                  sites=[S("c", pos=(0.0, 0, 0.1))])
    edges = [ConnectionEdge("p", "ch", site_idx=0, joint=_hinge())]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=150.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


CASES = [
    (_orphan_muscle, "orphan_muscle"),
    (_self_intersecting, "self_intersection"),
    (_immobile_joint, "immobile_joint"),
    (_no_actuators, "no_actuators"),
    (_zero_moment_muscle, "zero_moment_arm"),
    (_bone_overload, "bone_overload"),
    (_tendon_overload, "tendon_overload"),
]


# --- the battery ------------------------------------------------------------
def test_every_rule_rejects_its_target():
    for builder, reason in CASES:
        ok, reasons = pre_sim_gate(builder())
        assert not ok, f"{builder.__name__} should be rejected"
        assert reason in reasons, f"{builder.__name__}: expected {reason}, got {reasons}"


def test_all_reasons_are_known():
    for builder, _ in CASES:
        _, reasons = pre_sim_gate(builder())
        for r in reasons:
            assert r.split(":")[0] in KNOWN_REASONS, r


def test_valid_seeds_pass_the_battery():
    for seed in (quadruped(), agent_zero()):
        ok, reasons = pre_sim_gate(seed)
        assert ok and reasons == [], reasons


def test_fuzz_gate_is_total_and_well_typed():
    """Over many mutants the gate never raises; rejections are real and named."""
    rng = np.random.default_rng(7)
    sched = MutationSchedule(macro_rate=0.8)
    g = quadruped()
    rejected = 0
    for i in range(120):
        child, _ = mutate(g, rng, generation=i, schedule=sched)
        ok, reasons = pre_sim_gate(child)          # must not raise for any genome
        if ok:
            assert reasons == []
            g = child                               # wander only through viable bodies
        else:
            rejected += 1
            assert reasons, "a rejection must carry at least one reason"
            for r in reasons:
                assert r.split(":")[0] in KNOWN_REASONS, r
    assert rejected > 0                             # the macro stream does hit bad bodies


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
