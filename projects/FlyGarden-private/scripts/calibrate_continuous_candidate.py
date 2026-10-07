"""Prospective operating-state calibration; never tunes on held-out choices."""
import sys,json,argparse,subprocess,fcntl,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body_trial import sha
from flygarden.recording import atomic_json,space_check
from flygarden.power import on_ac_power
OUT=ROOT/'reports/brain-integration/recovery/candidate-calibration'
SOURCES=('scripts/calibrate_continuous_candidate.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py',
         'flygarden/brain.py','flygarden/descending.py','flygarden/body.py','flygarden/synchronized.py',
         'scripts/diagnose_sensorimotor_pathway.py','scripts/diagnose_causal_timing.py',
         'reports/brain-integration/recovery/candidate-v1.json','data/annotations.tsv',
         'vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet')
def sources():return {name:sha(ROOT/name) for name in SOURCES}

def prepare():
    OUT.mkdir(exist_ok=True)
    p={'version':1,'sources':sources(),'seeds':[9501,9502],'support_hz':[0,30,65],'duration':3.,'interval':.025,
       'selection':'Lowest support rate for which both seeds depart>10mm,complete with finite observations and no flipped samples; otherwise no operating point selected.',
       'scope':'Quiet local sensory channels; explicit state stimulation only. Calibrates movement capability,not sensory choice,navigation or learning. Decoder and weights fixed.'}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:atomic_json(path,p)
    return p

def worker(seed,support):
    import brian2 as b
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.candidate_inputs import CandidateInputs
    from flygarden.synchronized import SynchronizedBody,CausalCoupling
    from scripts.diagnose_recovery_interfaces import mappings
    protocol=prepare();folder=OUT/'trials'/f'{seed}-{support}hz';folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();_,mapping=mappings()
    candidate=ContinuousCandidate(mapping,seed,profile=CandidateInputs(float(support)))
    body=SynchronizedBody(seed=seed);coupling=CausalCoupling(.025);origin=body.sim.mj_data.time
    manifest={'status':'running','sources':protocol['sources'],'seed':seed,'support_hz':support,
              'controller':candidate.manifest(),'chunks':[],'scope':protocol['scope']}
    atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'geometry.json',body.geometry())
    initial=body.observation();rows=[];frames=[]
    try:
        for tick in range(120):
            assert abs(candidate.time-coupling.time)<1e-9 and abs(body.steps*body.dt-coupling.time)<1e-9
            transition=coupling.advance(lambda dt:candidate.advance(dt,[[0.,0.],[0.,0.]],[0.,0.],False),
                                       lambda dt,m:body.advance(dt,m,capture=lambda t,p:frames.append({'time':t,**p})))
            assert abs(body.sim.mj_data.time-origin-coupling.time)<1e-8
            si,st=candidate.last_spikes;file=folder/f'window-{tick:03d}.npz';space_check(folder,16*1024**2)
            with file.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,spike_i=si,spike_t=st,counts=candidate.last['spike_counts'],
                                    external_i=candidate.last['external_indices'],external_t=candidate.last['external_times'],
                                    applied_motor=transition['applied_motor'],next_motor=transition['next_motor'])
            file.with_suffix('.tmp').replace(file)
            rows.append({'time':coupling.time,'population_hz':candidate.last['population_hz'],**transition})
            manifest['chunks'].append({'file':file.name,'sha256':sha(file),'time':coupling.time,'spikes':len(si)})
            atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows);atomic_json(folder/'frames.json',frames)
            if support==65 and seed==9501 and tick==59:
                checkpoint=folder/'checkpoint-1.5';candidate.save(checkpoint)
                import pickle
                with (checkpoint/'body-driver.pkl').open('wb') as stream:
                    pickle.dump({'body':body.snapshot(),'coupling':coupling.snapshot()},stream,protocol=5)
                atomic_json(checkpoint/'driver-integrity.json',{'body-driver.pkl':sha(checkpoint/'body-driver.pkl')})
        positions=np.asarray([initial['position']]+[r['body']['position'] for r in rows])
        heading=np.unwrap([initial['heading']]+[r['body']['heading'] for r in rows])
        manifest.update(status='complete',wall_seconds=time.perf_counter()-started,frames=len(frames),
                        xy_displacement_mm=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),
                        heading_change_rad=float(heading[-1]-heading[0]),flipped_samples=sum(r['body']['flipped'] for r in rows),
                        finite=bool(np.isfinite(positions).all() and np.isfinite(heading).all()))
        atomic_json(folder/'manifest.json',manifest)
    finally:body.close()

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protocol=prepare();done=[]
        for support in protocol['support_hz']:
            for seed in protocol['seeds']:
                if not on_ac_power():atomic_json(OUT/'progress.json',{'status':'paused_for_ac_power','completed':done});return
                name=f'{seed}-{support}hz';folder=OUT/'trials'/name;path=folder/'manifest.json'
                if path.exists() and json.loads(path.read_text())['status']=='complete':done.append(name);continue
                if folder.exists():raise RuntimeError('Preserve interrupted calibration before retry: '+name)
                atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':6})
                subprocess.run([sys.executable,__file__,'--seed',str(seed),'--support',str(support)],cwd=ROOT,check=True)
                done.append(name);print(f'{len(done)}/6 {name}',flush=True)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);parser.add_argument('--support',type=int,choices=(0,30,65));args=parser.parse_args()
    if args.seed is not None:worker(args.seed,args.support)
    else:run()
