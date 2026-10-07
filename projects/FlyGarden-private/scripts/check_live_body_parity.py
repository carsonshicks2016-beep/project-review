"""Native integration-state parity with frequent visual and world callbacks."""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flygarden.synchronized import SynchronizedBody
from flygarden.live_body import LiveBody
from flygarden.recording import atomic_json


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: compare(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else: assert a == b


def main():
    start = time.monotonic()
    ordinary = SynchronizedBody(seed=9711)
    live = LiveBody(seed=9711)
    count = 0
    def world_step(t, pose):
        nonlocal count
        count += 1
        # Visual predator is non-colliding, but mocap changes are legitimate
        # state differences. Disabled, fixed visuals isolate observer effects.
        live.update_visual_world([], {'enabled': False})
    try:
        compare(ordinary.snapshot(), live.snapshot())
        for motor in ([0, 0], [.65, .65], [.3, .8], [.8, .3]):
            a, b = [], []
            ordinary.advance(.025, motor, lambda t, p: a.append((t, p)))
            live.advance(.025, motor, lambda t, p: b.append((t, p)), world_step)
            compare(ordinary.snapshot(), live.snapshot())
            compare(a, b)
        result = {'status': 'passed', 'physics_and_gait_exact': True,
                  'recorded_poses_exact': True, 'world_callbacks': count,
                  'simulated_seconds': .1, 'wall_seconds': time.monotonic() - start,
                  'scope': 'Native body observer parity; no brain or vision specificity claim',
                  'sources': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                              for name in ('flygarden/live_body.py', 'scripts/check_live_body_parity.py')}}
        path = Path('reports/brain-integration/recovery/live-body-parity.json')
        atomic_json(path, result)
        print(json.dumps(result))
    finally:
        ordinary.close(); live.close()


if __name__ == '__main__': main()
