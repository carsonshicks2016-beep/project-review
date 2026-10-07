"""Bounded full-network live arena smoke; not held-out behavioral acceptance."""
import sys, json, time, fcntl
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.power import on_ac_power
from flygarden.recording import atomic_json, space_check
from flygarden.continuous_candidate import ContinuousCandidate, file_sha
from flygarden.candidate_inputs import CandidateInputs
from flygarden.live_body import LiveBody
from flygarden.live_coupling import LiveCoupling
from flygarden.world import World, default_arena
from scripts.diagnose_recovery_interfaces import mappings


def main():
    assert on_ac_power(), 'Expensive brain runs require AC power'
    with (ROOT / '.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        space_check(ROOT, 64 * 1024**2)
        folder = ROOT / 'reports/brain-integration/recovery' / f'live-candidate-smoke-{time.time_ns()}'
        folder.mkdir(exist_ok=False)
        sources = {name: file_sha(ROOT / name) for name in
                   ('scripts/check_live_candidate.py', 'flygarden/live_coupling.py',
                    'flygarden/live_body.py', 'flygarden/local_senses.py', 'flygarden/world.py')}
        atomic_json(folder / 'protocol.json', {'sources': sources, 'seed': 9712,
                    'duration': .1, 'interval': .025, 'support_hz': 65,
                    'scope': 'Native live feedback and clock smoke; no autonomous-choice or vision-specificity claim'})
        started = time.monotonic()
        _, mapping = mappings()
        brain = ContinuousCandidate(mapping, 9712, CandidateInputs(65))
        arena = default_arena()
        world = World(seed=9712, arena=arena)
        body = LiveBody(seed=9712, blocks=arena['blocks'], spawn=arena['spawn'],
                        foods=arena['foods'], predator=arena['predator'])
        loop = LiveCoupling(brain, body, world)
        frames = [{'time': 0., **body.recording_pose()}]
        rows = []
        atomic_json(folder / 'controller.json', brain.manifest())
        try:
            for i in range(4):
                result = loop.advance()
                indices, times = brain.last_spikes
                assert len(indices) == brain.last['spike_counts'].sum()
                assert abs(brain.time - loop.clock.time) < 1e-9
                assert world.time == loop.clock.time
                assert body.steps == loop.clock.ticks == loop.world_ticks
                filename = folder / f'window-{i}.npz'
                np.savez_compressed(filename, spike_i=indices, spike_t=times,
                                    external_i=brain.last['external_indices'],
                                    external_t=brain.last['external_times'],
                                    counts=brain.last['spike_counts'])
                frames.extend(result.pop('frames'))
                rows.append(result)
                atomic_json(folder / 'rows.json', rows)
                atomic_json(folder / 'frames.json', frames)
            result = {'status': 'passed', 'windows': 4, 'duration': loop.clock.time,
                      'poses_including_initial': len(frames), 'world_substeps': loop.world_ticks // 5,
                      'wall_seconds': time.monotonic() - started,
                      'body_x_displacement_mm': rows[-1]['body']['position'][0] - frames[0]['body']['position'][0],
                      'sources': sources, 'limitations': ['Only100ms smoke', 'Vision specificity unvalidated',
                          'Learning disabled', 'Live application not switched to this research loop'],
                      'spike_files': {p.name: file_sha(p) for p in folder.glob('window-*.npz')}}
            atomic_json(folder / 'results.json', result)
            print(json.dumps(result))
        finally:
            body.close()


if __name__ == '__main__': main()
