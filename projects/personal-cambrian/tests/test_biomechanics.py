"""Stage 10.4 acceptance: force-vector / moment-arm biomechanics (bpy-free core).

The annotated clip is rendered by Blender (verified by running it); here we test the
bpy-free data it overlays -- that `record_biomechanics` captures, per frame and per
muscle, the line of action (origin->insertion), the tension, and the moment-arm lever
(joint anchor -> perpendicular foot on the muscle line), with forces that actually
vary as the creature moves.

Runs:  python3 tests/test_biomechanics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.viz import record_biomechanics, BioTrajectory
from personal_cambrian.viz.biomechanics import _foot_of_perpendicular


def test_foot_of_perpendicular_geometry():
    foot, ma = _foot_of_perpendicular(np.array([0.0, 1.0, 0.0]),
                                      np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(foot, [0.0, 0.0, 0.0])              # drops onto the line midpoint
    assert abs(ma - 1.0) < 1e-9                            # perpendicular distance


def test_records_force_and_moment_arm_per_muscle():
    n_mus = len(develop(quadruped()).muscles)
    bio = record_biomechanics(quadruped(), n_steps=40, name="quad")
    assert bio.n_frames >= 2
    for fr in bio.frames:
        assert set(fr) == {"bodies", "muscles"}
        assert len(fr["muscles"]) == n_mus
        for mu in fr["muscles"]:
            assert len(mu["org"]) == 3 and len(mu["ins"]) == 3
            assert mu["force"] >= 0.0 and mu["moment_arm"] >= 0.0
            assert mu["anchor"] is None or len(mu["anchor"]) == 3


def test_muscles_are_actuated_and_have_levers():
    bio = record_biomechanics(quadruped(), n_steps=60, name="quad")
    forces = [mu["force"] for fr in bio.frames for mu in fr["muscles"]]
    assert bio.force_max > 0.0 and max(forces) > 0.0       # the gait pulls the muscles
    # at least one muscle crosses a joint -> a non-trivial moment arm exists
    arms = [mu["moment_arm"] for fr in bio.frames for mu in fr["muscles"]
            if mu["anchor"] is not None]
    assert arms and max(arms) > 0.0


def test_biotrajectory_roundtrip():
    import json
    bio = record_biomechanics(quadruped(), n_steps=20, fps=24, name="quad")
    d = json.loads(json.dumps(bio.to_dict()))
    again = BioTrajectory(**d)
    assert again.fps == 24 and again.n_frames == bio.n_frames
    assert again.force_max == bio.force_max


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
