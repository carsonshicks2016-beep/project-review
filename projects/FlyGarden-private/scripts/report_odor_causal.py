"""Audit complete matched odor cohorts; do not publish partial inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_odor_trials import audit,OUT
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
from flygarden.odor_causal_metrics import odor_metrics
from flygarden.odor_statistics import summarize
from flygarden.recording import atomic_json


def read(folder,protocol):
    audit_receipt=audit(folder)
    m=json.loads((folder/'manifest.json').read_text());assert m['sources']==protocol['sources']
    rows=json.loads((folder/'rows.json').read_text());frames=json.loads((folder/'frames.json').read_text())
    assert len(rows)==120 and len(frames)==90 and m['duration']==3.
    assert m['metrics']==odor_metrics(m['initial'],frames,rows,m['cue'])
    assert np.all(np.abs(np.array([f['time'] for f in frames])-np.arange(1,91)/30)<=.0005+1e-9)
    decoder=DescendingDecoder()
    for row in rows:
        assert np.array_equal(decoder.advance(.025,row['population_hz']),row['next_motor'])
        if m['arm']=='steering_cut':
            assert row['population_hz']['DNa02_left']==row['population_hz']['DNa02_right']==0
    return {'manifest':m,'rows':rows,'frames':frames,'audit':audit_receipt}


def main(partial):
    protocol=json.loads((OUT/'protocol.json').read_text());records=[];effects=[]
    for seed in protocol['seeds']:
        folders={arm:OUT/'trials'/str(seed)/arm for arm in protocol['neural_arms']}
        complete=all((p/'manifest.json').exists() and json.loads((p/'manifest.json').read_text())['status']=='complete' for p in folders.values())
        if not complete:
            if partial:continue
            raise RuntimeError('All20 matched cohorts must complete before final inference')
        arms={arm:read(p,protocol) for arm,p in folders.items()};a=arms['intact'];m=a['manifest']
        effect={'seed':seed,'cue':m['cue'],'flipped_frames':m['metrics']['flipped_frames'],
                'stalled':m['metrics']['stalled'],'food':a['frames'][-1]['world']['collected']}
        for arm in ('sensory_off','steering_cut'):
            control=arms[arm]
            effect['motor_rms_vs_'+arm]=float(np.sqrt(np.mean((np.array([r['applied_motor'] for r in a['rows']])-np.array([r['applied_motor'] for r in control['rows']]))**2)))
            effect['toward_heading_vs_'+arm]=m['metrics']['toward_heading_radians']-control['manifest']['metrics']['toward_heading_radians']
        effect['final_xy_departure_mm']=float(np.linalg.norm(np.array(a['rows'][-1]['body']['position'][:2])-arms['sensory_off']['rows'][-1]['body']['position'][:2]))
        effects.append(effect);records.extend(v['audit'] for v in arms.values())
    result={'status':'partial' if partial else 'complete','complete_matched_seeds':len(effects),
        'neural_records':records,'paired_effects':effects,'protocol_sha256':file_sha(OUT/'protocol.json'),
        'reporter_sha256':file_sha(Path(__file__)),'statistics_source_sha256':file_sha(ROOT/'flygarden/odor_statistics.py'),
        'scope':'Engineered odor-only response with fixed support/gait and frozen learning; not navigation, vision or learned preference'}
    if not partial:
        assert len(effects)==20
        replay_path=OUT/'motor-replay-results.json'
        replay=json.loads(replay_path.read_text()) if replay_path.exists() else {}
        verified=(replay.get('status')=='passed' and len(replay.get('records',[]))==20 and
                  {r['seed'] for r in replay['records']}==set(protocol['seeds']))
        if verified:
            for r in replay['records']:
                assert r['source_manifest_sha256']==file_sha(OUT/'trials'/str(r['seed'])/'intact/manifest.json')
        result.update(summarize(effects,verified))
    atomic_json(OUT/('partial-results.json' if partial else 'results.json'),result)
    print(f'{len(effects)} complete matched cohorts audited; '+('no partial inference' if partial else 'final gates computed'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();main(args.partial)
