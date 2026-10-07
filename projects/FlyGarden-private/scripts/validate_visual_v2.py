"""Prospective new native eye scenes; no neural controller or hidden inputs."""
import sys, json, time, math, shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.record_eye_stimuli import VisualBench
from flygarden.synchronized import SynchronizedBody
from flygarden.vision_continuity import ContinuousExpansion
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json, space_check
CASES = ('blank', 'appearance_left', 'translation_left', 'loom_left', 'loom_right',
         'occlusion', 'self_motion_blank', 'self_motion_static')


def main():
    folder = ROOT / 'reports/brain-integration/recovery' / f'visual-v2-native-{time.time_ns()}'
    folder.mkdir()
    sources = {n: file_sha(ROOT/n) for n in ('scripts/validate_visual_v2.py', 'scripts/record_eye_stimuli.py',
                   'flygarden/vision_continuity.py', 'flygarden/body.py', 'flygarden/synchronized.py')}
    for name in sources:
        dst = folder/'source'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dst)
    atomic_json(folder/'protocol.json', {'sources': sources, 'cases': CASES, 'seed': 9731,
        'duration': 1.5, 'image_interval': .025, 'stimulus_angle_radians': .9,
        'loom_distance_mm': [24, 4], 'stationary_distance_mm': 12,
        'criterion': 'Non-expanding stationary confounds must remain0Hz; looming must produce ipsilateral response after appearance. Self-motion responses are reported, not relabeled as external threat detection.',
        'scope': 'New held-out image geometry after frozen v2 screening; self-motion body uses supplied fixed gait commands, no brain or navigation claim'})
    results = []
    for case in CASES:
        space_check(folder, 128*1024**2)
        with patch('flygarden.body.FlatGroundWorld', VisualBench): body = SynchronizedBody(seed=9731)
        try:
            data, model = body.sim.mj_data, body.sim.mj_model
            ids = {name: int(model.body('bench_'+name).mocapid[0]) for name in ('sphere', 'wall', 'pattern')}
            baseline = body.eye_frames(); encoder = ContinuousExpansion(baseline)
            images, features = [], []
            for tick in range(60):
                t = tick*.025
                for index in ids.values(): data.mocap_pos[index] = [0,0,-100]
                theta = -.9 if case == 'loom_right' else .9
                if .25 <= t < 1.25 and case not in ('blank', 'self_motion_blank'):
                    distance = 24 - (t-.25)*20 if case.startswith('loom') else 12
                    xy = np.array([math.cos(theta), math.sin(theta)])*distance
                    if case == 'translation_left': xy += np.array([-math.sin(theta),math.cos(theta)])*(t-.25)*4
                    data.mocap_pos[ids['sphere']] = [*xy, 3.]
                    if case == 'occlusion' and .5 <= t < .9:
                        data.mocap_pos[ids['wall']] = [*(xy/2), 3.]
                        data.mocap_quat[ids['wall']] = [math.cos(theta/2),0,0,math.sin(theta/2)]
                # Eye queries derive camera state on copies, preserving solver.
                frame = body.eye_frames()
                features.append({'time':t, 'features':encoder.advance(frame,.025)})
                images.append(frame)
                if case.startswith('self_motion'): body.advance(.025,[.65,.65])
            with (folder/f'{case}.npz').open('wb') as stream:
                np.savez_compressed(stream, frames=np.asarray(images), baseline=baseline, times=np.arange(60)*.025)
            atomic_json(folder/f'{case}.json', features)
            rates = np.array([[f['lplc2_hz'] for f in r['features']] for r in features])
            item = {'case':case, 'peak_hz':rates.max(axis=0).tolist(),
                    'active_frames':int(np.sum(rates.max(axis=1)>0)),
                    'file_sha256':file_sha(folder/f'{case}.npz')}
            if case.startswith('loom'):
                side = int(case.endswith('right')); item['gate_passed'] = bool(rates[11:50,side].max()>0)
            elif not case.startswith('self_motion'): item['gate_passed'] = bool(np.all(rates==0))
            else: item['self_motion_false_positive_present'] = bool(np.any(rates>0))
            results.append(item); print(json.dumps(item),flush=True)
            atomic_json(folder/'progress.json', {'status':'running','results':results})
        finally: body.close()
    atomic_json(folder/'results.json', {'status':'complete','results':results,
        'specificity_gates_passed':all(r.get('gate_passed',True) for r in results),
        'self_motion_quiet':not any(r.get('self_motion_false_positive_present',False) for r in results),
        'promotion':'Not automatically promoted; image-only encoding and physiological limits remain explicit'})
    print(folder,flush=True)


if __name__ == '__main__': main()
