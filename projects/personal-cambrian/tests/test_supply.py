"""Stage 6.4 acceptance: the supply / connectivity check.

Every muscle must be reachable from the root (abstract blood + neural supply).
The seeds pass. A muscle anchored to a part that is never connected to the root
-- an orphan -- is rejected, even though the part is well-formed and exists.

Runs:  python3 tests/test_supply.py   (pure genome graph analysis, no mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.genome import (
    Shape, JointType, AttachmentSite as S, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.evo import check_supply, SupplyReport


def _floating_part_genome(*, attach_floater: bool, muscle_on_floater: bool) -> Genome:
    """Root + a connected child + a 'floater' part that may or may not be edged in.

    A muscle optionally anchors between the root and the floater.
    """
    root = PartNode(id="root", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.4},
                    sites=[S("s0", pos=(0, 0, -0.2)), S("s1", pos=(0.05, 0, 0.0)),
                           S("s2", pos=(0.05, 0, 0.1))])
    child = PartNode(id="child", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                     sites=[S("c0", pos=(0.04, 0, 0.1))])
    floater = PartNode(id="floater", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                       sites=[S("f0", pos=(0.04, 0, 0.1))])
    edges = [ConnectionEdge("root", "child", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1)))]
    if attach_floater:
        edges.append(ConnectionEdge("root", "floater", site_idx=2,
                                    joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1))))
    muscles = []
    if muscle_on_floater:
        muscles.append(MuscleGene(id="orphan", origin=("root", 1), insertion=("floater", 0),
                                  pcsa_cm2=10.0))
    return Genome(root="root", parts=[root, child, floater], edges=edges, muscles=muscles)


# --- valid bodies pass ------------------------------------------------------
def test_seeds_are_supplied():
    for seed in (quadruped(), agent_zero()):
        rep = check_supply(seed)
        assert isinstance(rep, SupplyReport)
        assert rep.supplied, rep.reasons
        assert bool(rep) is True
        assert rep.orphan_muscles == []
        assert rep.unreachable_parts == []
        assert rep.n_reachable == rep.n_parts


def test_muscle_on_connected_part_is_supplied():
    rep = check_supply(_floating_part_genome(attach_floater=True, muscle_on_floater=True))
    assert rep.supplied
    assert rep.orphan_muscles == []
    assert rep.unreachable_parts == []


# --- disconnected muscle is rejected ---------------------------------------
def test_orphan_muscle_rejected():
    rep = check_supply(_floating_part_genome(attach_floater=False, muscle_on_floater=True))
    assert not rep.supplied
    assert "orphan_muscle" in rep.reasons
    assert "orphan" in rep.orphan_muscles
    assert "floater" in rep.unreachable_parts


def test_disconnected_part_without_muscle_is_tolerated():
    # a dead-weight unreachable part is reported but not a hard rejection
    rep = check_supply(_floating_part_genome(attach_floater=False, muscle_on_floater=False))
    assert rep.supplied
    assert rep.orphan_muscles == []
    assert "floater" in rep.unreachable_parts


def test_report_roundtrips_to_dict():
    rep = check_supply(quadruped())
    d = rep.to_dict()
    assert d["supplied"] is True and d["orphan_muscles"] == []
    assert set(d) >= {"reasons", "unreachable_parts", "n_parts", "n_reachable"}


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
