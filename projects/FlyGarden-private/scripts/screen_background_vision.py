"""Frozen-parameter retrospective moving-body registration screen."""
import json,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.vision_background import BackgroundExpansion
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
SOURCE=ROOT/'reports/brain-integration/recovery/live-masked-replay-1791320179208802000'


def main():
    out=ROOT/'reports/brain-integration/recovery'/f'background-vision-screen-{time.time_ns()}';out.mkdir()
    initial={'rgb':np.full((2,8,8,3),255,dtype=np.uint8),'valid':np.ones((2,8,8),dtype=bool)}
    names=('flygarden/vision_background.py','flygarden/vision_tracks.py','scripts/screen_background_vision.py')
    atomic_json(out/'protocol.json',{'sources':{n:file_sha(ROOT/n) for n in names},
        'encoder':BackgroundExpansion(initial).manifest(),'seeds':[9751,9752],
        'scope':'Retrospective screen on preserved moving-body images, no neural rerun or promotion',
        'metrics':'Intended-side cue activity, pre/post-cue false positives, registration confidence'})
    for name in names:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    results=[]
    for seed in (9751,9752):
        receipt=json.loads((SOURCE/str(seed)/'results.json').read_text());path=SOURCE/str(seed)/'eyes.npz'
        assert file_sha(path)==receipt['eyes_sha256']
        with np.load(path) as archive:
            frames=archive['frames'];valid=archive['valid'];times=archive['times']
        encoder=BackgroundExpansion({'rgb':frames[0],'valid':valid[0]});features=[]
        for frame,mask in zip(frames,valid):features.append(encoder.advance({'rgb':frame,'valid':mask},.025))
        rates=np.array([[f['lplc2_hz'] for f in row] for row in features]);side=0 if seed%2 else 1
        cue=(times>=.5)&(times<1.5);after=times>=1.5
        result={'seed':seed,'cue_peak_hz':rates[cue,side].max().item(),
            'post_cue_peak_hz':rates[after].max(axis=0).tolist(),
            'pre_cue_peak_hz':rates[times<.5].max(axis=0).tolist(),
            'registration_valid_frames':[sum(f[s]['registration']['valid'] for f in features) for s in range(2)],
            'registration_homography_frames':[sum(f[s]['registration'].get('method')=='homography' for f in features) for s in range(2)],
            'source_sha256':file_sha(path)}
        result['screen_passed']=bool(result['cue_peak_hz']>0 and not rates[after].any() and not rates[times<.5].any())
        results.append(result);atomic_json(out/f'{seed}-features.json',features);print(json.dumps(result),flush=True)
    atomic_json(out/'results.json',{'status':'screen_complete','results':results,
        'not_promoted':True,'scope':'Retrospective moving-body signal/specificity checks; new held-out imagery still required'})
    print(out,flush=True)


if __name__=='__main__':main()
