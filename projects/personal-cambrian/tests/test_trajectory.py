"""Stage 10.2 acceptance: trajectory replay (bpy-free core).

The replay VIDEO is rendered by Blender (verified by running it); here we test the
bpy-free halves the video rests on: that `record_trajectory` captures a faithful
per-frame, per-body world-transform sequence in which the creature actually MOVES,
and that `video.stitch` assembles a frame sequence into a playable file.

Runs:  python3 tests/test_trajectory.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.viz import record_trajectory, Trajectory


def test_trajectory_records_all_bodies_each_frame():
    m = develop(quadruped())
    traj = record_trajectory(quadruped(), n_steps=40, name="quad")
    assert 2 <= traj.n_frames <= 41
    plan_ids = {g["name"] for g in traj.plan["geoms"]}
    assert len(plan_ids) == m.body_count
    for fr in traj.frames:
        assert set(fr) == plan_ids                       # every body, every frame
        for pos, quat in fr.values():
            assert len(pos) == 3 and len(quat) == 4
            assert np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))


def test_creature_actually_moves():
    traj = record_trajectory(quadruped(), n_steps=60, name="quad")
    first, last = traj.frames[0], traj.frames[-1]
    moved = sum(float(np.linalg.norm(np.array(last[b][0]) - np.array(first[b][0])))
                for b in first)
    print(f"  total body displacement over rollout: {moved:.3f}m")
    assert moved > 1e-3                                  # the bodies are not frozen
    assert traj.root_displacement() >= 0.0


def test_trajectory_dict_roundtrip():
    traj = record_trajectory(quadruped(), n_steps=20, fps=24, name="quad")
    import json
    d = json.loads(json.dumps(traj.to_dict()))           # survives JSON
    again = Trajectory(plan=d["plan"], frames=d["frames"], fps=d["fps"])
    assert again.fps == 24 and again.n_frames == traj.n_frames
    assert again.plan["name"] == "quad"


def test_video_stitch_assembles_frames():
    from personal_cambrian.viz.video import stitch
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        fd = os.path.join(d, "frames"); os.makedirs(fd)
        for i in range(4):                               # synthetic frame sequence
            Image.new("RGB", (32, 24), (10 * i, 20, 30)).save(
                os.path.join(fd, f"frame_{i:04d}.png"))
        out = stitch(fd, os.path.join(d, "clip"), fps=12)
        assert os.path.exists(out) and out.rsplit(".", 1)[1] in {"mp4", "gif"}
        assert os.path.getsize(out) > 0


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
