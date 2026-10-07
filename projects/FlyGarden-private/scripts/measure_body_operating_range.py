"""Pinned, resumable body-only calibration; never trains or edits the brain."""
import sys,json,time,hashlib,pickle,gzip,uuid,os,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.synchronized import SynchronizedBody
from flygarden.descending import DescendingDecoder
from flygarden.power import on_ac_power
OUT=ROOT/'reports/brain-integration/recovery/body-operating-range';DT=.025
SOURCES=('scripts/measure_body_operating_range.py','flygarden/synchronized.py','flygarden/body.py','flygarden/descending.py','flygarden/power.py')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def scenarios():
    cases={f'direct-{drive:g}':{'motor':[drive,drive]} for drive in (0,.1,.2,.4,.65,.9)}
    cases.update({'direct-left':{'motor':[.35,.65]},'direct-right':{'motor':[.65,.35]}})
    for name,left,right in (('decoder-forward',0,0),('decoder-left',50,0),('decoder-right',0,50)):
        cases[name]={'rates':{'DNp09_left':65,'DNp09_right':65,'DNa02_left':left,'DNa02_right':right}}
    return cases

def metrics(initial,rows):
    positions=np.array([initial['position']]+[r['body']['position'] for r in rows])
    heading=np.unwrap([initial['heading']]+[r['body']['heading'] for r in rows])
    contacts=np.array([r['body']['contacts'] for r in rows])
    return {'xy_displacement_mm':float(np.linalg.norm(positions[-1,:2]-positions[0,:2])),
            'xy_travel_mm':float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
            'heading_change_rad':float(heading[-1]-heading[0]),
            'flipped_samples':sum(r['body']['flipped'] for r in rows),
            'max_contact_force_model_units':float(np.linalg.norm(contacts,axis=2).max()),
            'scope':'Body-only controlled commands; no neuronal activity, navigation or learning claim'}

def trial(seed,name,case,duration):
    folder=OUT/'trials'/f'{seed}-{name}';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'manifest.json';meta=json.loads(file.read_text()) if file.exists() else None
    if meta and meta['status']=='complete':return True
    started=time.perf_counter();body=SynchronizedBody(seed=seed);decoder=DescendingDecoder();frames=[];rows=[]
    try:
        if meta:
            for chunk in meta['chunks']:
                p=folder/chunk['file'];assert sha(p)==chunk['sha256']
                with gzip.open(p,'rt') as f:rows.extend(json.load(f)['rows'])
            tick=meta['completed_windows']
            if tick:
                checkpoint=folder/meta['checkpoint']['file'];assert sha(checkpoint)==meta['checkpoint']['sha256']
                with checkpoint.open('rb') as f:state=pickle.load(f)
                body.restore(state['body']);decoder.motor=np.array(state['decoder_motor'])
            else:assert meta['initial']==body.observation()
        else:
            tick=0;meta={'status':'running','seed':seed,'case':case,'duration':duration,'initial':body.observation(),'completed_windows':0,'chunks':[],'controller':'Body-only command calibration + supplied gait','individual_spikes':False}
            atomic_json(folder/'geometry.json',body.geometry())
        while tick<round(duration/DT):
            if not on_ac_power():
                meta['status']='paused_for_ac_power';atomic_json(file,meta);return False
            block=[];frames=[]
            for _ in range(min(40,round(duration/DT)-tick)):
                now=tick*DT
                active=.5<=now<duration-.5
                rates=case.get('rates',{'DNp09_left':0,'DNp09_right':0,'DNa02_left':0,'DNa02_right':0})
                if not active:rates=dict.fromkeys(rates,0)
                motor=decoder.advance(DT,rates) if 'rates' in case else np.array(case['motor'] if active else [0.,0.])
                obs=body.advance(DT,motor,capture=lambda stamp,pose:frames.append({'time':stamp,**pose}))
                block.append({'time':(tick+1)*DT,'motor':motor.tolist(),'body':obs});tick+=1
            token=uuid.uuid4().hex;chunk=folder/f'chunk-{tick:05d}-{token}.json.gz';checkpoint=folder/f'checkpoint-{tick:05d}-{token}.bin'
            space_check(folder,16*1024**2)
            with gzip.open(chunk.with_suffix('.tmp'),'wt') as f:json.dump({'rows':block,'frames':frames},f)
            chunk.with_suffix('.tmp').replace(chunk)
            with checkpoint.with_suffix('.tmp').open('wb') as f:
                pickle.dump({'body':body.snapshot(),'decoder_motor':decoder.motor},f,protocol=5);f.flush();os.fsync(f.fileno())
            checkpoint.with_suffix('.tmp').replace(checkpoint)
            rows.extend(block);meta.update(status='running',completed_windows=tick,checkpoint={'file':checkpoint.name,'sha256':sha(checkpoint)})
            meta['chunks'].append({'file':chunk.name,'sha256':sha(chunk),'windows':len(block),'frames':len(frames)})
            atomic_json(file,meta)
        meta.update(status='complete',metrics=metrics(meta['initial'],rows),attempt_wall_seconds=time.perf_counter()-started);atomic_json(file,meta);return True
    finally:body.close()

def run():
    import fcntl
    lock=ROOT/'.runtime/experiment.lock';lock.parent.mkdir(exist_ok=True)
    guard=lock.open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'version':1,'sources':{n:sha(ROOT/n) for n in SOURCES},'seeds':[9101,9102],
              'duration':3.,'interval':DT,'scenarios':scenarios(),'schedule':'0-.5 s settle; .5-2.5 s drive; 2.5-3 s stop/recovery',
              'scope':'Operating-range measurement on calibration seeds; held-out 10x60s acceptance is a separate subsequent protocol. No tuning of brain or decoder.',
              'power':'AC required; preserve completed chunks and complete physical state at 1-second boundaries.'}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:atomic_json(p,protocol)
    completed=[]
    for seed in protocol['seeds']:
        for name,case in protocol['scenarios'].items():
            if not on_ac_power():
                atomic_json(OUT/'progress.json',{'status':'paused_for_ac_power','completed':completed,'current':f'{seed}-{name}'});return
            if not trial(seed,name,case,protocol['duration']):
                atomic_json(OUT/'progress.json',{'status':'paused_for_ac_power','completed':completed,'current':f'{seed}-{name}'});return
            completed.append(f'{seed}-{name}')
            atomic_json(OUT/'progress.json',{'status':'running','completed':completed})
            print(completed[-1],flush=True)
    atomic_json(OUT/'progress.json',{'status':'complete','completed':completed})

if __name__=='__main__':run()
