"""Reconstruct v1 encoding from archived native eye images, retaining failures."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.vision_encoding import EyeExpansion
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
SOURCE = ROOT / 'reports/brain-integration/stage6-20261006/eye-stimuli'


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    folder = ROOT / 'reports/brain-integration/recovery' / f'visual-confound-audit-{time.time_ns()}'
    folder.mkdir()
    protocol = {'encoder': EyeExpansion.version, 'image_source_manifest': file_sha(SOURCE / 'manifest.json'),
        'sources': {n: file_sha(ROOT / n) for n in ('scripts/audit_visual_confounds.py', 'flygarden/vision_encoding.py')},
        'criteria': {'blank_static_translation_hz': 'No rates above0Hz for intended non-expansion confounds',
                     'looming': 'Positive intended-side response after first appearance',
                     'recovery': 'Zero rates after object removal'},
        'scope': 'Retrospective native-image diagnosis; not independent held-out validation. Self-motion and occlusion still require new fixtures.'}
    atomic_json(folder / 'protocol.json', protocol)
    results = []
    for cue in manifest['cues']:
        image_path = SOURCE / f'{cue}.npz'
        json_path = SOURCE / f'{cue}.json'
        assert file_sha(image_path) == manifest['files'][image_path.name]
        assert file_sha(json_path) == manifest['files'][json_path.name]
        saved = json.loads(json_path.read_text())
        with np.load(image_path) as archive:
            frames, times = archive['frames'], archive['times']
            encoder = EyeExpansion(archive['baseline'])
            features = [encoder.advance(frame, 1 / 30) for frame in frames]
        for raw, row in zip(features, saved):
            for observed, original in zip(raw, row['features']):
                for key in observed:
                    assert np.isclose(observed[key], original[key], atol=1e-12, rtol=1e-12), (cue, key)
        rates = np.array([[f['lplc2_hz'] for f in row] for row in features])
        active = np.where(rates.max(axis=1) > 0)[0]
        interior = (times > .5) & (times < 1.5)
        recovery = times >= 1.5
        item = {'cue': cue, 'recorded_features_reconstructed': True,
                'peak_hz': rates.max(axis=0).tolist(), 'active_frames': len(active),
                'first_active_time': float(times[active[0]]) if len(active) else None,
                'interior_peak_hz': rates[interior].max(axis=0).tolist(),
                'recovery_peak_hz': rates[recovery].max(axis=0).tolist()}
        if cue in ('blank', 'static_left', 'static_right', 'moving_pattern'):
            item['non_expansion_gate_passed'] = bool(np.all(rates == 0))
        if cue.startswith('loom'):
            side = 0 if cue.endswith('left') else 1
            item['post_appearance_loom_gate_passed'] = bool(rates[interior, side].max() > 0)
        item['recovery_gate_passed'] = bool(np.all(rates[recovery] == 0))
        results.append(item)
        print(json.dumps(item), flush=True)
    gates = [v for r in results for k, v in r.items() if k.endswith('_gate_passed')]
    atomic_json(folder / 'results.json', {'status': 'diagnostic_complete',
                'vision_acceptance': 'failed' if not all(gates) else 'incomplete_coverage',
                'results': results, 'missing_conditions': ['self-motion', 'occlusion', 'held-out scenes'],
                'coverage': 'Actual native eye images, stationary source body; not live obstacle navigation',
                'protocol_sha256': file_sha(folder / 'protocol.json')})
    print(str(folder), flush=True)


if __name__ == '__main__': main()
