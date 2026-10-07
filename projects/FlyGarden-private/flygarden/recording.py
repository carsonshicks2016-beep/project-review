"""Versioned, crash-readable recordings. Simulation never waits for video rendering."""
from pathlib import Path
import os,json,time,uuid,hashlib,shutil
import psutil
import numpy as np
from .storage import ROOT
RESERVE=2*1024**3

def data_root():
    path=Path(os.environ.get('FLYGARDEN_MEDIA_DIR',str(ROOT/'data'))).expanduser().resolve()
    path.mkdir(parents=True,exist_ok=True)
    return path

def space_check(path,expected=0):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path).free < RESERVE+max(0,expected):
        raise OSError('Storage reserve reached: recording paused. Free space or set FLYGARDEN_MEDIA_DIR; existing recordings are preserved.')

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w') as f:
        json.dump(value,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def run_folder(identity):
    if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid run identifier')
    new=data_root()/'runs'/identity
    return new if new.exists() else ROOT/'data/runs'/identity

def all_runs():
    folders={p.parent.name:p.parent for p in (ROOT/'data/runs').glob('*/manifest.json')}
    folders.update({p.parent.name:p.parent for p in (data_root()/'runs').glob('*/manifest.json')})
    return sorted(folders.values(),key=lambda p:(p/'manifest.json').stat().st_mtime,reverse=True)

class Recorder:
    def __init__(self,manifest,geometry,ids=None):
        self.id=manifest.get('id',uuid.uuid4().hex);self.folder=data_root()/'runs'/self.id
        space_check(self.folder.parent,16*1024**2);self.folder.mkdir(exist_ok=False)
        self.manifest={**manifest,'id':self.id,'schema_version':2,'created':time.time(),'writer':dict(pid=os.getpid(),process_created=psutil.Process().create_time()),'pose_fps':30,'pose_scope':'30 Hz target, sampled at actual 0.5 ms gait boundaries on the physics clock; world/sensory/controller updates remain at their original 100 ms intervals','spike_clock_seconds':.0001,'individual_spikes':ids is not None}
        if 'pose_scope' in manifest:self.manifest['pose_scope']=manifest['pose_scope']
        atomic_json(self.folder/'manifest.json',self.manifest);atomic_json(self.folder/'geometry.json',geometry)
        self.meta=dict(schema_version=2,status='recording',frames=0,chunks=[],duration=0.,start_time=None,last_time=None,spike_count=0,outcome=None)
        if ids is not None:self.attach_neurons(ids)
        self.closed=False;self.totals=np.zeros(len(ids),dtype=np.int64) if ids is not None else None;self.write_metadata()
    def attach_neurons(self,ids):
        ids=np.asarray(ids,dtype=np.int64);self.totals=np.zeros(len(ids),dtype=np.int64);space_check(self.folder,ids.nbytes)
        with (self.folder/'ordering.npz.tmp').open('wb') as f:np.savez_compressed(f,ids=ids)
        os.replace(self.folder/'ordering.npz.tmp',self.folder/'ordering.npz')
        self.manifest.update(individual_spikes=True,neurons=len(ids),neuron_ordering_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        atomic_json(self.folder/'manifest.json',self.manifest)
    def write_metadata(self):atomic_json(self.folder/'recording.json',self.meta)
    def append_window(self,frames,indices=(),times=(),time_bounds=None):
        if self.closed:raise ValueError('Recording is finalized')
        i=np.asarray(indices,dtype=np.int32);t=np.asarray(times,dtype=np.float64)
        if len(i)!=len(t) or not np.isfinite(t).all():raise ValueError('Invalid spike events')
        if self.totals is not None and len(i) and (i.min()<0 or i.max()>=len(self.totals)):raise ValueError('Spike index does not match neuron ordering')
        if len(t)>1 and np.any(np.diff(t)<-1e-9):raise ValueError('Spike times are not ordered')
        frame_times=np.asarray([s['world']['time'] for s in frames],dtype=float)
        if not np.isfinite(frame_times).all() or len(frame_times)>1 and np.any(np.diff(frame_times)<0):raise ValueError('Invalid pose times')
        if time_bounds is not None:
            bounds=np.asarray(time_bounds,dtype=float)
            if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[1]<=bounds[0]:raise ValueError('Invalid simulation window bounds')
            start,end=map(float,bounds)
            if len(t) and (t.min()<start-1e-9 or t.max()>=end+1e-9):raise ValueError('Spikes outside simulation window')
            if len(frame_times) and (frame_times.min()<start-1e-9 or frame_times.max()>end+1e-9):raise ValueError('Poses outside simulation window')
            if self.meta['last_time'] is not None and start<self.meta['last_time']-1e-9:raise ValueError('Overlapping simulation windows')
        elif frames:start,end=float(frame_times[0]),float(frame_times[-1])
        elif len(t):start,end=float(t.min()),float(t.max())
        else:return
        payload=''.join(json.dumps(s,separators=(',',':'))+'\n' for s in frames).encode()
        space_check(self.folder,len(payload)+i.nbytes+t.nbytes+1024**2)
        # Publish an immutable event chunk before committing its reference. Orphans are ignored.
        name=f'spikes-{len(self.meta["chunks"]):06d}.npz'
        temporary=self.folder/(name+'.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,indices=i,times=t);f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.folder/name)
        with (self.folder/'frames.jsonl').open('ab') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        if self.meta['start_time'] is None:self.meta['start_time']=start
        self.meta['last_time']=end;self.meta['duration']=end-self.meta['start_time']
        if frames:self.meta['last_world']=frames[-1]['world']
        self.meta['chunks'].append(dict(file=name,start=min(start,float(t.min())) if len(t) else start,end=max(end,float(t.max())) if len(t) else end,events=len(i),sha256=hashlib.sha256((self.folder/name).read_bytes()).hexdigest()))
        if self.totals is not None and len(i):np.add.at(self.totals,i,1)
        self.meta['frames']+=len(frames);self.meta['spike_count']+=len(i);self.write_metadata()
    def finish(self,outcome=None,status='complete',enqueue=True):
        if self.closed:return
        if self.totals is not None:
            with (self.folder/'activity-totals.npz.tmp').open('wb') as f:np.savez_compressed(f,counts=self.totals)
            os.replace(self.folder/'activity-totals.npz.tmp',self.folder/'activity-totals.npz')
        self.meta.update(status=status,outcome=outcome,finished=time.time());self.write_metadata();self.closed=True
        if outcome:atomic_json(self.folder/'outcome.json',outcome)
        if enqueue and self.meta['frames']:
            from .video_jobs import enqueue_export
            enqueue_export(self.id)

def metadata(identity):
    folder=run_folder(identity);manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/'recording.json'
    if path.exists():
        meta=json.loads(path.read_text())
        if meta['status']=='recording':meta['status']='recording' if writer_alive(manifest.get('writer',{})) else 'incomplete'
    else:meta=dict(schema_version=1,status='legacy',individual_spikes=False)
    return {**manifest,'recording':meta}

def spike_chunk(identity,index):
    folder=run_folder(identity);meta=json.loads((folder/'recording.json').read_text());entry=meta['chunks'][index];path=folder/entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Spike chunk integrity check failed')
    with np.load(path,allow_pickle=False) as a:return dict(indices=a['indices'].tolist(),times=a['times'].tolist())

def neurons(identity,population='all',limit=256,offset=0,query=''):
    from .morphology import by_id
    folder=run_folder(identity);order=folder/'ordering.npz'
    if not order.exists():return dict(neurons=[],total=0,scope='Individual neuron activity was not recorded')
    with np.load(order,allow_pickle=False) as a:ids=a['ids']
    counts=np.zeros(len(ids),dtype=np.int64);totalfile=folder/'activity-totals.npz'
    if totalfile.exists():
        with np.load(totalfile,allow_pickle=False) as a:counts=a['counts']
    else:
        meta=json.loads((folder/'recording.json').read_text())
        for entry in meta['chunks']:
            with np.load(folder/entry['file'],allow_pickle=False) as a:np.add.at(counts,a['indices'],1)
    annotation=by_id();rows=[]
    for i in np.argsort(-counts,kind='stable'):
        row=annotation.get(str(ids[i]),dict(id=str(ids[i]),cell_type='',cell_class='',side=''))
        if population!='all' and population not in (row['cell_class'],row['cell_type']):continue
        if query and query.lower() not in (row['id']+' '+row['cell_type']+' '+row['cell_class']).lower():continue
        rows.append({**row,'index':int(i),'spikes':int(counts[i])})
    return dict(neurons=rows[offset:offset+limit],total=len(rows),modeled_neurons=len(ids),active_neurons=int(np.count_nonzero(counts)),populations=sorted({annotation.get(str(r),{}).get('cell_class','') for r in ids}-{''}),scope='All modeled spike events retained; displayed anatomy is a progressive selection')


def writer_alive(owner):
    try:
        process=psutil.Process(owner['pid'])
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE and abs(process.create_time()-owner['process_created'])<.1
    except (psutil.Error,KeyError):return False

def recover_interrupted():
    """Commit recovery status only for an identified dead writer, preserving all raw files."""
    recovered=[]
    for folder in all_runs():
        path=folder/'recording.json'
        if not path.exists():continue
        try:
            manifest=json.loads((folder/'manifest.json').read_text());meta=json.loads(path.read_text());owner=manifest.get('writer')
            if meta['status']!='recording' or not owner or writer_alive(owner):continue
            meta.update(status='interrupted',finished=time.time(),recovery='Writer exited; only committed samples are playable');atomic_json(path,meta)
            if meta['frames']:
                from .video_jobs import enqueue_export
                enqueue_export(folder.name)
            recovered.append(folder.name)
        except (OSError,ValueError,KeyError):continue
    return recovered

def neuron_history(identity,index):
    folder=run_folder(identity);meta=json.loads((folder/'recording.json').read_text())
    with np.load(folder/'ordering.npz',allow_pickle=False) as a:ids=a['ids']
    if not 0<=index<len(ids):raise ValueError('Invalid neuron index')
    start=meta['start_time'] or 0.;end=meta['last_time'] or start;bins=max(1,int(np.ceil((end-start)/.1-1e-9)));counts=np.zeros(bins,dtype=np.int64)
    for chunk_index,entry in enumerate(meta['chunks']):
        path=folder/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Spike chunk integrity check failed')
        with np.load(path,allow_pickle=False) as a:times=a['times'][a['indices']==index]
        positions=np.clip(np.floor((times-start)/.1+1e-9).astype(np.int64),0,bins-1);np.add.at(counts,positions,1)
    return dict(id=str(ids[index]),index=index,start=start,end=end,bin_seconds=.1,spike_count=int(counts.sum()),rates_hz=(counts*10).tolist(),scope='Non-overlapping 100 ms bins across the entire recorded run; final partial bin uses the same 100 ms denominator')
