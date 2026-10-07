"""Bounded, durable physical diagnostic recording, separate from neural runs."""
import gzip,hashlib,json,os,pickle,time,uuid
from pathlib import Path
import numpy as np
from .recording import atomic_json,space_check
from .synchronized import SynchronizedBody
from .descending import DescendingDecoder
from .physical_contacts import physical_contacts
from .power import on_ac_power

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def read_trial(folder):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text());rows=[];frames=[]
    for chunk in m['chunks']:
        p=folder/chunk['file']
        if sha(p)!=chunk['sha256']:raise ValueError('Diagnostic chunk integrity failure')
        with gzip.open(p,'rt') as stream:data=json.load(stream)
        if len(data['rows'])!=chunk['windows'] or len(data['frames'])!=chunk['frames']:
            raise ValueError('Diagnostic chunk length mismatch')
        rows.extend(data['rows']);frames.extend(data['frames'])
    return m,rows,frames

def run_trial(folder,seed,duration,command,sources,dt=.025,stop_after=None):
    """command(time) returns ('rates' or 'motor', values, phase identity)."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);file=folder/'manifest.json'
    meta=json.loads(file.read_text()) if file.exists() else None
    if meta:
        if meta['sources']!=sources or meta['seed']!=seed or meta['duration']!=duration or meta['interval']!=dt:
            raise ValueError('Trial identity changed; preserve this attempt and use a new version')
        if meta['status'] in ('complete','failed'):return meta['status']
    body=SynchronizedBody(seed=seed);decoder=DescendingDecoder();started=time.perf_counter()
    try:
        if meta:
            read_trial(folder)
            tick=meta['completed_windows']
            if tick:
                entry=meta['checkpoint'];p=folder/entry['file']
                if sha(p)!=entry['sha256']:raise ValueError('Checkpoint integrity failure')
                with p.open('rb') as stream:state=pickle.load(stream)
                body.restore(state['body']);decoder.motor=np.asarray(state['decoder_motor']).copy()
        else:
            tick=0;meta={'schema_version':1,'status':'running','seed':seed,'duration':duration,
                         'interval':dt,'sources':sources,'initial':body.observation(),
                         'completed_windows':0,'chunks':[],'wall_seconds':0.,
                         'controller':'Artificial rate/motor diagnostic + fixed decoder + supplied gait',
                         'neural_activity_simulated':False,'learning':False,
                         'physics_dt':body.dt,'gait_dt':body.control_dt,
                         'contact_scope':'Tarsus5 solver contacts and tangential point velocity; adhesion actuator forces excluded.'}
            atomic_json(folder/'geometry.json',body.geometry());atomic_json(file,meta)
        while tick<round(duration/dt):
            ac=on_ac_power()
            if not ac or stop_after is not None and tick*dt>=stop_after:
                meta['status']='paused_for_ac_power' if not ac else 'verification_checkpoint'
                meta['wall_seconds']+=time.perf_counter()-started
                atomic_json(file,meta);return meta['status']
            rows=[];frames=[]
            for _ in range(min(round(1/dt),round(duration/dt)-tick)):
                now=tick*dt;kind,value,phase=command(now)
                if kind not in ('rates','motor'):raise ValueError('Unknown diagnostic command kind')
                motor=decoder.advance(dt,value) if kind=='rates' else np.asarray(value,dtype=float)
                obs=body.advance(dt,motor,capture=lambda t,pose:frames.append({'time':t,**pose}))
                contacts=physical_contacts(body)
                if not np.isfinite(np.asarray(contacts['foot_solver_ground_forces_model_units'])).all():
                    raise RuntimeError('Nonfinite physical contact measurement')
                rows.append({'time':(tick+1)*dt,'phase':phase,'source_kind':kind,
                             'input':value,'motor':motor.tolist(),'body':obs,'physical_contacts':contacts})
                tick+=1
            token=uuid.uuid4().hex;chunk=folder/f'chunk-{tick:06d}-{token}.json.gz';checkpoint=folder/f'checkpoint-{tick:06d}-{token}.bin'
            payload=gzip.compress(json.dumps({'rows':rows,'frames':frames},separators=(',',':')).encode())
            space_check(folder,len(payload)+4*1024**2)
            with chunk.with_suffix('.tmp').open('wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            chunk.with_suffix('.tmp').replace(chunk)
            with checkpoint.with_suffix('.tmp').open('wb') as stream:
                pickle.dump({'body':body.snapshot(),'decoder_motor':decoder.motor.copy()},stream,protocol=5)
                stream.flush();os.fsync(stream.fileno())
            checkpoint.with_suffix('.tmp').replace(checkpoint)
            meta['chunks'].append({'file':chunk.name,'sha256':sha(chunk),'windows':len(rows),'frames':len(frames),
                                   'checkpoint':{'file':checkpoint.name,'sha256':sha(checkpoint)}})
            meta.update(status='running',completed_windows=tick,checkpoint=meta['chunks'][-1]['checkpoint'])
            atomic_json(file,meta)
        meta.update(status='complete',wall_seconds=meta['wall_seconds']+time.perf_counter()-started)
        atomic_json(file,meta);return 'complete'
    except OSError:
        if meta:
            meta['status']='paused_for_storage';atomic_json(file,meta)
        raise
    except Exception as exc:
        if meta:
            meta.update(status='failed',error=str(exc));atomic_json(file,meta)
        raise
    finally:body.close()
