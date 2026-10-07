"""Replay diagnostic neural commands through physics without rerunning the brain.

This is a causal command-capability diagnostic, not live sensory navigation.
"""
import sys,json,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_recovery_interfaces import OUT as NEURAL,DT,DURATION
from flygarden.body_trial import run_trial,read_trial,sha
from flygarden.recording import atomic_json
OUT=NEURAL/'physical-replay'

def held_command(rows,t):
    index=round(t/DT)
    motor=[0.,0.] if index==0 else rows[index-1]['candidate_motor']
    phase='rest' if t<.3-1e-9 else 'pulse' if t<.8-1e-9 else 'recovery'
    return 'motor',motor,phase

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        analysis=json.loads((NEURAL/'analysis.json').read_text());assert analysis['status']=='complete'
        protocol=json.loads((NEURAL/'protocol.json').read_text());OUT.mkdir(exist_ok=True)
        paths=('scripts/replay_recovery_interfaces.py','flygarden/body_trial.py','flygarden/body.py',
               'flygarden/synchronized.py','flygarden/physical_contacts.py','flygarden/power.py')
        sources={name:sha(ROOT/name) for name in paths}
        sources['neural_protocol']=sha(NEURAL/'protocol.json');sources['neural_analysis']=sha(NEURAL/'analysis.json')
        config={'version':1,'sources':sources,'scope':'Recorded neural commands applied in next25ms interval; no new neural simulation, live perception, navigation or learning claim.',
                'conditions':protocol['cases'],'seeds':protocol['seeds'],'duration':DURATION,'interval':DT}
        path=OUT/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==config
        else:atomic_json(path,config)
        results=[]
        for cue in config['conditions']:
            for seed in config['seeds']:
                name=f'{cue}-{seed}';original=NEURAL/'trials'/name
                rows=json.loads((original/'bins.json').read_text())
                pinned={**sources,'neural_manifest':sha(original/'manifest.json'),'neural_bins':sha(original/'bins.json')}
                folder=OUT/'trials'/name
                status=run_trial(folder,seed,DURATION,lambda t:held_command(rows,t),pinned)
                if status!='complete':atomic_json(OUT/'progress.json',{'status':status,'completed':results});return
                m,physical,frames=read_trial(folder)
                assert len(physical)==60 and len(frames)==45
                for tick,row in enumerate(physical):assert row['motor']==held_command(rows,tick*DT)[1]
                p=np.asarray([m['initial']['position']]+[r['body']['position'] for r in physical])
                h=np.unwrap([m['initial']['heading']]+[r['body']['heading'] for r in physical])
                results.append({'cue':cue,'seed':seed,'xy_displacement_mm':float(np.linalg.norm(p[-1,:2]-p[0,:2])),
                                'xy_travel_mm':float(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum()),
                                'heading_change_rad':float(h[-1]-h[0]),'flipped_samples':sum(r['body']['flipped'] for r in physical),
                                'finite':bool(np.isfinite(p).all() and np.isfinite(h).all())})
                atomic_json(OUT/'progress.json',{'status':'running','completed':len(results),'planned':28})
        atomic_json(OUT/'results.json',{'status':'complete','trials':results,'scope':config['scope'],'commands_reconstructed':True})
        atomic_json(OUT/'progress.json',{'status':'complete','completed':28,'planned':28})

if __name__=='__main__':run()
