"""Stage 1.8 acceptance: the hand-authored seeds compile, load, and settle under
gravity without exploding.

Checks: both seeds validate; serialize round-trip; the saved JSON files load and
compile; each settles to a finite, bounded rest state on the floor without
passing through it; expected actuator counts; and every seed actuator (tendon)
has a non-zero moment arm about some joint.

Runs:  python3 tests/test_seeds.py   (requires mujoco)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco

from personal_cambrian.encoding import Genome
from personal_cambrian.seeds import agent_zero, quadruped, SEEDS
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology
from test_actuation import _tendon_spans_some_hinge

SEED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "seeds")
DROP = {"agent_zero": 1.15, "quadruped": 0.55}


def _settle(g, drop, steps=3000):
    model, _ = compile_morphology(develop(g), add_floor=True)
    data = mujoco.MjData(model)
    data.qpos[2] = drop
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return model, data


def test_seeds_validate_shape():
    for fn in (agent_zero, quadruped):
        assert fn().validate_shape() == []


def test_seeds_json_roundtrip():
    for fn in (agent_zero, quadruped):
        g = fn()
        assert Genome.from_json(g.to_json()) == g


def test_agent_zero_settles_without_exploding():
    model, data = _settle(agent_zero(), DROP["agent_zero"])
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert float(np.abs(data.qvel).max()) < 2.0       # came to rest, not exploding
    assert float(data.qpos[2]) > -0.1                 # did not fall through floor
    assert model.nu == 8                              # hips, knees, ankles, elbows x2


def test_quadruped_settles_without_exploding():
    model, data = _settle(quadruped(), DROP["quadruped"])
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert float(np.abs(data.qvel).max()) < 2.0
    assert float(data.qpos[2]) > -0.1
    assert model.nu == 4                              # one knee per leg


def test_seed_actuators_have_moment_arms():
    for fn in (agent_zero, quadruped):
        model, _ = compile_morphology(develop(fn()))
        assert model.ntendon == model.nu > 0
        for t in range(model.ntendon):
            assert _tendon_spans_some_hinge(model, t)


def test_quadruped_is_non_human_topology():
    # the whole point: a different body plan, not a rescaled human
    m = develop(quadruped())
    thighs = [b for b in m.bodies if b.part_id == "qthigh"]
    tail = [b for b in m.bodies if b.part_id == "tailseg"]
    assert len(thighs) == 4                            # four legs (front+back x bilateral)
    assert len(tail) == 4                              # a segmented tail


def test_saved_json_files_load_and_compile():
    for name in SEEDS:
        path = os.path.join(SEED_DIR, f"{name}.genome.json")
        if not os.path.exists(path):
            continue                                   # build with scripts/build_seeds.py
        with open(path) as f:
            g = Genome.from_json(f.read())
        assert g.validate_shape() == []
        assert g == SEEDS[name]()                      # file matches the builder
        model, _ = compile_morphology(develop(g), add_floor=True)
        mujoco.mj_forward(model, mujoco.MjData(model))


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
