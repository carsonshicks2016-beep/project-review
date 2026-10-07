"""Immutable complete-scene checkpoints for the research live loop."""
import pickle
from pathlib import Path
from .continuous_candidate import file_sha
from .recording import atomic_json, space_check

SOURCES = ('live_coupling.py', 'live_body.py', 'local_senses.py', 'world.py',
           'vision_encoding.py', 'synchronized.py', 'body.py', 'live_checkpoint.py')


def identity(loop):
    return {'schema_version': 1, 'loop': loop.version, 'body': loop.body.version,
            'senses': loop.senses.version, 'candidate': loop.candidate.manifest(),
            'sources': {name: file_sha(Path(__file__).parent / name) for name in SOURCES}}


def save_scene(loop, folder):
    folder = Path(folder)
    if folder.exists(): raise FileExistsError('Complete scene checkpoints are immutable')
    space_check(folder.parent, 650 * 1024**2)
    snapshot = loop.snapshot()
    if (snapshot['clock']['ticks'] != snapshot['world_ticks'] or
        abs(loop.candidate.time - loop.clock.time) > 1e-9):
        raise ValueError('Cannot checkpoint mismatched live clocks')
    loop.candidate.save(folder)
    temporary = folder / 'scene.pkl.tmp'
    with temporary.open('wb') as stream:
        pickle.dump(snapshot, stream, protocol=5)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(folder / 'scene.pkl')
    # Publish this completion receipt last. A interrupted write is preserved
    # but is not advertised as a restorable full scene.
    atomic_json(folder / 'scene.json', {'status': 'complete', 'identity': identity(loop),
                'time': loop.clock.time, 'scene_sha256': file_sha(folder / 'scene.pkl')})


def load_scene(loop, folder):
    import json
    folder = Path(folder)
    receipt = json.loads((folder / 'scene.json').read_text())
    if receipt['status'] != 'complete' or receipt['identity'] != identity(loop):
        raise ValueError('Complete-scene checkpoint identity differs')
    if file_sha(folder / 'scene.pkl') != receipt['scene_sha256']:
        raise ValueError('Complete-scene checkpoint integrity failure')
    # Local model checkpoints are pickle artifacts, not interchange uploads.
    with (folder / 'scene.pkl').open('rb') as stream:
        state = pickle.load(stream)
    loop.candidate.load(folder)
    loop.restore(state)
    if loop.clock.time != receipt['time']:
        raise ValueError('Complete-scene checkpoint time differs')
