"""Stage 6.2 acceptance: the geometric viability gate.

Valid bodies (the hand-authored seeds, and bodies produced by valid mutations)
pass. Deliberately broken bodies are rejected BEFORE simulation:
  * self-intersecting   -- two non-adjacent parts stacked in the same place
  * zero-moment muscle  -- an actuator routed straight through the joint axis
  * immobile joint      -- a hinge with a degenerate (zero-span) range
  * no actuators        -- a passive body that cannot act

Runs:  python3 tests/test_viability.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.genome import (
    Shape, JointType, AttachmentSite as S, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.evo import check_viability, ViabilityReport


# --- builders for deliberately broken bodies -------------------------------
def _self_intersecting() -> Genome:
    """Two sibling limbs attached at the same site -> co-located parts."""
    root = PartNode(id="root", shape=Shape.CAPSULE,
                    dims={"radius": 0.1, "length": 0.4}, sites=[S("s0", pos=(0, 0, 0.2))])
    limb = PartNode(id="limb", shape=Shape.CAPSULE,
                    dims={"radius": 0.05, "length": 0.3}, sites=[S("t", pos=(0, 0, -0.15))])
    limb2 = PartNode(id="limb2", shape=Shape.CAPSULE,
                     dims={"radius": 0.05, "length": 0.3}, sites=[S("t2", pos=(0, 0, -0.15))])
    edges = [
        ConnectionEdge("root", "limb", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1))),
        ConnectionEdge("root", "limb2", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1))),
    ]
    return Genome(root="root", parts=[root, limb, limb2], edges=edges, muscles=[])


def _zero_moment_muscle() -> Genome:
    """A muscle routed through the joint centre -> ~zero moment arm."""
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("js", pos=(0, 0, -0.2)), S("o", pos=(0, 0, -0.2))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                  sites=[S("i", pos=(0, 0, 0.0))])
    edges = [ConnectionEdge("p", "ch", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1)))]
    mg = MuscleGene(id="axis", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=10.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


def _immobile_joint() -> Genome:
    """A hinge with a degenerate (zero-span) range cannot move."""
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                  sites=[S("c", pos=(0.05, 0, 0.1)), S("d", pos=(0, 0, -0.14))])
    edges = [ConnectionEdge("p", "ch", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(0.0, 0.0)))]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=10.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


def _no_actuators() -> Genome:
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3})
    edges = [ConnectionEdge("p", "ch", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1)))]
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[])


# --- valid bodies pass ------------------------------------------------------
def test_seeds_are_viable():
    for seed in (quadruped(), agent_zero()):
        rep = check_viability(seed)
        assert isinstance(rep, ViabilityReport)
        assert rep.viable, rep.reasons
        assert bool(rep) is True
        assert rep.reasons == []
        assert rep.n_actuators > 0
        assert rep.n_coincident_pairs == 0
        assert rep.min_moment_arm > 0.0


def test_seed_diagnostics_have_margin():
    rep = check_viability(quadruped())
    assert rep.min_center_ratio > 0.3       # well clear of the coincidence cutoff
    assert rep.min_moment_arm > 1e-3        # well clear of the dead-muscle cutoff


# --- broken bodies are rejected --------------------------------------------
def test_self_intersection_rejected():
    rep = check_viability(_self_intersecting())
    assert not rep.viable
    assert "self_intersection" in rep.reasons
    assert rep.n_coincident_pairs >= 1
    assert rep.min_center_ratio < 0.3


def test_zero_moment_muscle_rejected():
    rep = check_viability(_zero_moment_muscle())
    assert not rep.viable
    assert "zero_moment_arm" in rep.reasons
    assert rep.n_actuators >= 1                  # the actuator exists...
    assert rep.n_dead_muscles >= 1               # ...but does no work
    assert rep.min_moment_arm < 1e-3


def test_immobile_joint_rejected():
    rep = check_viability(_immobile_joint())
    assert not rep.viable
    assert "immobile_joint" in rep.reasons
    assert rep.n_degenerate_joints >= 1


def test_no_actuator_body_rejected():
    rep = check_viability(_no_actuators())
    assert not rep.viable
    assert "no_actuators" in rep.reasons
    assert rep.n_actuators == 0


def test_report_roundtrips_to_dict():
    rep = check_viability(quadruped())
    d = rep.to_dict()
    assert d["viable"] is True and d["reasons"] == []
    assert set(d) >= {"n_bodies", "n_actuators", "min_center_ratio",
                      "min_moment_arm", "n_coincident_pairs"}


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
