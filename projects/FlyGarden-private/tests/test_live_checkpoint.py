import json
from types import SimpleNamespace
import pytest
from flygarden.live_checkpoint import save_scene, load_scene


class FakeCandidate:
    time = 0.
    loaded = False
    def manifest(self): return {'id': 'checkpoint-test'}
    def save(self, folder): folder.mkdir()
    def load(self, folder): self.loaded = True


class Loop:
    version = 'fixture'
    body = SimpleNamespace(version='fixture-body')
    senses = SimpleNamespace(version='fixture-senses')
    clock = SimpleNamespace(time=0.)
    def __init__(self): self.candidate = FakeCandidate()
    def snapshot(self): return {'clock': {'ticks': 0}, 'world_ticks': 0, 'values': [1, 2, 3]}
    def restore(self, state): self.restored = state


def test_scene_roundtrip_immutable_and_integrity_before_neural_load(tmp_path):
    loop = Loop(); folder = tmp_path / 'scene'
    save_scene(loop, folder)
    load_scene(loop, folder)
    assert loop.restored == loop.snapshot()
    with pytest.raises(FileExistsError): save_scene(loop, folder)
    (folder / 'scene.pkl').write_bytes(b'corrupted')
    loop.candidate.loaded = False
    with pytest.raises(ValueError, match='integrity'): load_scene(loop, folder)
    assert not loop.candidate.loaded


def test_incomplete_scene_and_changed_identity_rejected(tmp_path):
    loop = Loop(); folder = tmp_path / 'scene'
    save_scene(loop, folder)
    receipt = json.loads((folder / 'scene.json').read_text())
    receipt['identity']['body'] = 'different-body'
    (folder / 'scene.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity'): load_scene(loop, folder)
    assert not loop.candidate.loaded
    (folder / 'scene.json').unlink()
    with pytest.raises(FileNotFoundError): load_scene(loop, folder)
