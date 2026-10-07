"""Preserve application sources and clone immutable checkpoints before repair."""
import sys, json, hashlib, subprocess, shutil, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.recording import atomic_json, space_check

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def run():
    out = ROOT / 'reports/brain-integration/recovery' / ('baseline-' + str(time.time_ns()))
    space_check(out.parent, 64 * 1024**2)
    out.mkdir()
    manifest = {'status': 'preserving', 'sources': {}, 'checkpoints': [],
                'retained_recordings': [], 'originals_modified': False}
    atomic_json(out / 'manifest.json', manifest)
    files = [p for folder in ('flygarden', 'scripts', 'static', 'tests')
             for p in (ROOT / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT / name for name in ('README.md', 'requirements.lock', 'reports/provenance.json',
                                      'reports/neuron-mapping.json') if (ROOT / name).exists()]
    for source in files:
        relative = source.relative_to(ROOT)
        target = out / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        manifest['sources'][str(relative)] = sha(target)
    for checkpoint in sorted((ROOT / 'data/checkpoints').glob('*/metadata.json')):
        target = out / 'checkpoints' / checkpoint.parent.name
        target.parent.mkdir(exist_ok=True)
        # macOS copy-on-write cloning preserves independent file contents.
        subprocess.run(['cp', '-cR', str(checkpoint.parent), str(target)], check=True)
        metadata = json.loads(checkpoint.read_text())
        for name, expected in metadata['files'].items():
            if sha(target / name) != expected:
                raise RuntimeError(f'Existing checkpoint integrity failure: {checkpoint.parent.name}/{name}')
        manifest['checkpoints'].append({'id': checkpoint.parent.name,
                                       'metadata_sha256': sha(target / 'metadata.json'),
                                       'files_verified': len(metadata['files'])})
        atomic_json(out / 'manifest.json', manifest)
    for folder in (ROOT / 'data/runs').glob('*'):
        if folder.is_dir() and (folder / 'manifest.json').exists():
            manifest['retained_recordings'].append({'path': str(folder.relative_to(ROOT)),
                                                  'manifest_sha256': sha(folder / 'manifest.json')})
    manifest.update(status='complete', completed=time.time(),
                    scope='Copied application sources; independently cloned and verified checkpoints; existing recordings retained at original paths.')
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out.parent / 'baseline.json', {'path': str(out.relative_to(ROOT)),
                                            'manifest_sha256': sha(out / 'manifest.json')})
    print(out, flush=True)

if __name__ == '__main__':
    run()
