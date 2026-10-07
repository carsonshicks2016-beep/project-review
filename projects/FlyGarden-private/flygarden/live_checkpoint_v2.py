"""Scene persistence with additional v2 sensory source-integrity receipt.

The original scene store is unchanged for its pinned experiments. This wrapper
requires the additional receipt before loading any neural or physical state.
"""
import json
from pathlib import Path
from . import live_checkpoint as original
from .continuous_candidate import file_sha
from .recording import atomic_json

SOURCES = ('local_senses_v2.py', 'vision_continuity.py', 'live_checkpoint_v2.py')


def adapter_identity(loop):
    return {'version': 2, 'senses': loop.senses.version, 'encoding': loop.senses.eye.manifest(),
            'sources': {name: file_sha(Path(__file__).parent / name) for name in SOURCES}}


def save_scene(loop, folder):
    identity = adapter_identity(loop)
    original.save_scene(loop, folder)
    atomic_json(Path(folder)/'scene-adapter.json', {'status': 'complete', 'identity': identity,
                'scene_receipt_sha256': file_sha(Path(folder)/'scene.json')})


def load_scene(loop, folder):
    folder = Path(folder)
    receipt = json.loads((folder/'scene-adapter.json').read_text())
    if receipt['status'] != 'complete' or receipt['identity'] != adapter_identity(loop):
        raise ValueError('V2 sensory checkpoint identity differs')
    if receipt['scene_receipt_sha256'] != file_sha(folder/'scene.json'):
        raise ValueError('V2 scene receipt integrity failure')
    original.load_scene(loop, folder)
