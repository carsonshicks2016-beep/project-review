"""Fresh-process exact brain/body/world/visual-history continuation proof."""
import argparse, fcntl, json, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_candidate_full_state import digest
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json, space_check
from flygarden.power import on_ac_power
SOURCES = ('scripts/check_live_v2_continuation.py', 'flygarden/live_coupling.py',
           'flygarden/live_body.py', 'flygarden/local_senses.py', 'flygarden/world.py',
           'flygarden/vision_encoding.py', 'flygarden/continuous_candidate.py',
           'flygarden/live_checkpoint.py', 'flygarden/live_checkpoint_v2.py',
           'flygarden/local_senses_v2.py', 'flygarden/vision_continuity.py',
           'flygarden/candidate_inputs.py', 'flygarden/brain.py', 'flygarden/body.py',
           'flygarden/synchronized.py', 'flygarden/descending.py')


def worker(folder, name):
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.candidate_inputs import CandidateInputs
    from flygarden.live_body import LiveBody
    from flygarden.live_coupling import LiveCoupling
    from flygarden.live_checkpoint_v2 import save_scene, load_scene
    from flygarden.local_senses_v2 import LocalSensesV2
    from flygarden.world import World, default_arena
    from scripts.diagnose_recovery_interfaces import mappings
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert protocol['sources'] == {name: file_sha(ROOT / name) for name in SOURCES}
    _, mapping = mappings()
    brain = ContinuousCandidate(mapping, 9741, CandidateInputs(65))
    arena = default_arena()
    # Put a food source inside contact range, ensuring a real world event and
    # changed visual/odor state are included in the continuation fixture.
    arena['foods'][0].update(x=-22., y=0.)
    world = World(seed=9741, arena=arena)
    body = LiveBody(seed=9741, blocks=arena['blocks'], spawn=arena['spawn'],
                    foods=arena['foods'], predator=arena['predator'])
    loop = LiveCoupling(brain, body, world, senses=LocalSensesV2.from_body(body))
    checkpoint = folder / 'checkpoint-0.1'
    if name == 'split_second':
        load_scene(loop, checkpoint)
    destination = folder / ('uninterrupted' if name == 'uninterrupted' else 'split')
    destination.mkdir(exist_ok=True)
    entries = json.loads((destination / 'states.json').read_text()) if name == 'split_second' else []
    finish = 4 if name == 'split_first' else 8
    try:
        for tick in range(round(loop.clock.time / .025), finish):
            result = loop.advance()
            network = brain.brain.network._full_state()
            entries.append({'time': loop.clock.time,
                'network_objects': {key: digest(value) for key, value in sorted(network.items())},
                'scene_complete': digest(loop.snapshot()),
                'adapter': digest({'previous': brain.previous, 'decoder': brain.decoder.motor,
                    'rng': brain.brain.rng.bit_generator.state, 'plasticity': brain.brain.plasticity.snapshot()}),
                'observations_frames_and_events': digest(result),
                'raw_spikes': digest(brain.last_spikes), 'neural_summary': digest(brain.last)})
            del network
            atomic_json(destination / 'states.json', entries)
        if name == 'split_first':
            save_scene(loop, checkpoint)
        atomic_json(destination / f'{name}-outcome.json', {'collected': world.collected,
                    'events': world.events, 'time': world.time, 'world_ticks': loop.world_ticks})
    finally:
        body.close()


def main():
    with (ROOT / '.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert on_ac_power(), 'Full-scene restart proof requires AC power'
        space_check(ROOT, 800 * 1024**2)
        folder = ROOT / 'reports/brain-integration/recovery' / f'live-v2-continuation-{time.time_ns()}'
        folder.mkdir()
        atomic_json(folder / 'protocol.json', {'sources': {n: file_sha(ROOT / n) for n in SOURCES},
                    'seed': 9741, 'duration': .2, 'interruption': .1, 'interval': .025,
                    'criterion': 'Exact full Brian object, neural RNG/decoder, body, world, eye history, raw spikes, observations, frames and events at every boundary',
                    'scope': 'Restart and event-state integration proof; not navigation or learning'})
        print(str(folder), flush=True)
        for name in ('uninterrupted', 'split_first', 'split_second'):
            subprocess.run([sys.executable, __file__, '--folder', str(folder), '--worker', name], cwd=ROOT, check=True)
        full = json.loads((folder / 'uninterrupted/states.json').read_text())
        split = json.loads((folder / 'split/states.json').read_text())
        assert len(full) == len(split) == 8
        for a, b in zip(full, split):
            assert a == b, 'Complete-scene continuation mismatch at ' + str(a['time'])
        outcome = json.loads((folder / 'split/split_second-outcome.json').read_text())
        assert outcome['collected'] == 1
        atomic_json(folder / 'results.json', {'status': 'passed', 'fresh_processes': 3, 'boundaries': 8,
                    'neural_and_scene_state_exact': True, 'raw_spikes_and_playback_exact': True,
                    'food_event_and_depletion_retained': True, 'protocol_sha256': file_sha(folder / 'protocol.json')})
        print('V2 full-scene restart proof passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--worker', choices=('uninterrupted', 'split_first', 'split_second'))
    args = parser.parse_args()
    if args.worker: worker(args.folder, args.worker)
    else: main()
