"""Stage 6.5 acceptance: reject-before-sim integration in the QD loop.

The viability gate (Stages 6.2-6.4) runs in `evaluate_and_insert` BEFORE the
genome is ever developed, compiled, or handed to RL; rejected genomes are
counted with their reasons. Survivors are scored with the Stage-6.1 biological
budget overflow subtracted from fitness.

Runs:  python3 tests/test_reject_gate.py   (requires mujoco + gymnasium + torch)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped
from personal_cambrian.encoding.genome import (
    Shape, JointType, AttachmentSite as S, Joint, PartNode, ConnectionEdge,
    MuscleGene, Genome,
)
from personal_cambrian.evo import (
    run_qd, QDConfig, pre_sim_gate, budget_penalty,
)

TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=14,
                seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                ep_steps=40, hidden=32, macro_rate=0.7, seed=0)


# --- crafted invalid bodies (one per gate) ---------------------------------
def _orphan_muscle() -> Genome:
    root = PartNode(id="root", shape=Shape.CAPSULE, dims={"radius": 0.08, "length": 0.4},
                    sites=[S("s0", pos=(0, 0, -0.2)), S("s1", pos=(0.05, 0, 0.0))])
    floater = PartNode(id="floater", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                       sites=[S("f0", pos=(0.04, 0, 0.1))])
    edges = []  # floater is never attached
    mg = MuscleGene(id="orphan", origin=("root", 1), insertion=("floater", 0), pcsa_cm2=10.0)
    return Genome(root="root", parts=[root, floater], edges=edges, muscles=[mg])


def _self_intersecting() -> Genome:
    root = PartNode(id="root", shape=Shape.CAPSULE, dims={"radius": 0.1, "length": 0.4},
                    sites=[S("s0", pos=(0, 0, 0.2))])
    a = PartNode(id="a", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                 sites=[S("t", pos=(0, 0, -0.15))])
    b = PartNode(id="b", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.3},
                 sites=[S("t2", pos=(0, 0, -0.15))])
    edges = [
        ConnectionEdge("root", "a", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1))),
        ConnectionEdge("root", "b", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1))),
    ]
    return Genome(root="root", parts=[root, a, b], edges=edges, muscles=[])


def _bone_overload() -> Genome:
    p = PartNode(id="p", shape=Shape.CAPSULE, dims={"radius": 0.06, "length": 0.4},
                 sites=[S("a", pos=(0, 0, -0.2)), S("b", pos=(0.05, 0, -0.1))])
    ch = PartNode(id="ch", shape=Shape.CAPSULE, dims={"radius": 0.0035, "length": 0.3},
                  sites=[S("c", pos=(0.0, 0, 0.1))])
    edges = [ConnectionEdge("p", "ch", site_idx=0,
                            joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1, 1)))]
    mg = MuscleGene(id="m", origin=("p", 1), insertion=("ch", 0), pcsa_cm2=80.0)
    return Genome(root="p", parts=[p, ch], edges=edges, muscles=[mg])


# --- the gate itself --------------------------------------------------------
def test_gate_passes_seed():
    ok, reasons = pre_sim_gate(quadruped())
    assert ok and reasons == []


def test_gate_rejects_each_failure_mode():
    for builder, reason in [(_orphan_muscle, "orphan_muscle"),
                            (_self_intersecting, "self_intersection"),
                            (_bone_overload, "bone_overload")]:
        ok, reasons = pre_sim_gate(builder())
        assert not ok, builder.__name__
        assert reason in reasons, (builder.__name__, reasons)


# --- budget overflow -> fitness penalty ------------------------------------
def test_budget_penalty_is_zero_for_within_budget_seed():
    assert budget_penalty(quadruped(), 1.0) == 0.0
    assert budget_penalty(quadruped(), 0.0) == 0.0


def test_budget_penalty_positive_and_scales_for_overflow():
    g = quadruped()
    for i in range(60):                              # pile on actuators -> neural overflow
        m = copy.deepcopy(g.muscles[0])
        m.id = f"extra{i}"
        g.muscles.append(m)
    p1 = budget_penalty(g, 1.0)
    p2 = budget_penalty(g, 2.0)
    assert p1 > 0.0
    assert abs(p2 - 2.0 * p1) < 1e-9                 # linear in weight
    assert budget_penalty(g, 0.0) == 0.0


# --- integration: rejections logged, none reach RL -------------------------
def test_qd_logs_rejections_with_reasons():
    logs = []
    run_qd([quadruped()], TINY, log_fn=logs.append)
    last = logs[-1]
    assert last["rejected"] > 0                      # this seed/config does reject some
    assert last["reject_reasons"]                    # reasons present
    assert sum(last["reject_reasons"].values()) == last["rejected"]


def test_gate_rejections_are_reproducible():
    a, b = [], []
    run_qd([quadruped()], TINY, log_fn=a.append)
    run_qd([quadruped()], TINY, log_fn=b.append)
    assert a[-1]["rejected"] == b[-1]["rejected"]
    assert a[-1]["reject_reasons"] == b[-1]["reject_reasons"]


def test_gate_can_be_disabled():
    # with the gate off, the rejection counter is driven only by the env probe
    cfg = QDConfig(**{**TINY.__dict__, "gate": False})
    logs = []
    arch, _ = run_qd([quadruped()], cfg, log_fn=logs.append)
    assert len(arch.grid) >= 1                       # still runs end to end


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
