"""Read published v783 Neuroglancer skeletons, without copying upstream code."""
import csv,json,hashlib,urllib.request,urllib.error,struct,threading,time,fcntl,uuid
from functools import lru_cache
from pathlib import Path
import numpy as np
from .storage import ROOT
from .recording import data_root,atomic_json,space_check
SOURCE='https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783'
LOCK=threading.Lock()

@lru_cache(maxsize=1)
def catalog():
    import pandas as pd
    ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
    ann={r['root_id']:r for r in csv.DictReader((ROOT/'data/annotations.tsv').open(),delimiter='\t')}
    neurons=[]
    for i,rid in enumerate(ids):
        a=ann.get(str(rid),{})
        neurons.append(dict(index=i,id=str(rid),cell_type=a.get('cell_type',''),cell_class=a.get('cell_class',''),side=a.get('side','')))
    return neurons

@lru_cache(maxsize=1)
def by_id():return {n['id']:n for n in catalog()}

def parse_skeleton(raw):
    if len(raw)<8:raise ValueError('Truncated skeleton')
    nv,ne=struct.unpack_from('<II',raw)
    if nv>5_000_000 or ne>10_000_000 or len(raw)<8+12*nv+8*ne:raise ValueError('Invalid skeleton size')
    vertices=np.frombuffer(raw,dtype='<f4',count=3*nv,offset=8).reshape(-1,3).copy()
    edges=np.frombuffer(raw,dtype='<u4',count=2*ne,offset=8+12*nv).reshape(-1,2).copy()
    if not np.isfinite(vertices).all() or (edges.size and edges.max()>=nv):raise ValueError('Invalid skeleton geometry')
    return vertices,edges

def skeleton(rid,lod='full'):
    if lod not in ('full','coarse'):raise ValueError('Unknown morphology detail')
    rid=str(rid)
    if rid not in by_id():raise ValueError('Neuron is absent from the imported network')
    folder=data_root()/'morphology/v783';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{rid}.bin';meta_path=folder/f'{rid}.json'
    with (folder/f'{rid}.lock').open('a') as file_lock:
        fcntl.flock(file_lock,fcntl.LOCK_EX)
        if meta_path.exists() and not path.exists():
            cached=json.loads(meta_path.read_text())
            if cached.get('status')=='unavailable':return cached
        if not path.exists():
            space_check(folder,32*1024**2)
            try:
                with urllib.request.urlopen(SOURCE+'/'+rid,timeout=25) as response:raw=response.read(32*1024**2+1)
            except urllib.error.HTTPError as exc:
                if exc.code==404:
                    meta=dict(id=rid,status='unavailable',source=SOURCE+'/'+rid,checked=time.time());atomic_json(meta_path,meta);return meta
                raise
            if len(raw)>32*1024**2:raise ValueError('Skeleton exceeds bounded download size')
            v,e=parse_skeleton(raw);space_check(folder,len(raw));tmp=folder/(rid+'.tmp-'+uuid.uuid4().hex);tmp.write_bytes(raw);tmp.replace(path)
            atomic_json(meta_path,dict(id=rid,status='available',source=SOURCE+'/'+rid,materialization=783,units='nanometers',coordinate_space='FlyWire native FAFB14.1',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),vertices=len(v),edges=len(e),downloaded=time.time(),license='Standalone skeleton redistribution license not established; retained locally with attribution'))
        raw=path.read_bytes();meta=json.loads(meta_path.read_text())
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Morphology cache integrity check failed')
    v,e=parse_skeleton(raw)
    if lod=='coarse':
        coarse=folder/f'{rid}.coarse.npz'
        if coarse.exists():
            with np.load(coarse,allow_pickle=False) as a:
                if str(a['source_sha256'])!=meta['sha256']:raise ValueError('Derived morphology source hash differs')
                v=a['vertices'];e=a['edges']
        else:
            v,e=coarse_geometry(v,e);space_check(folder,v.nbytes+e.nbytes);temporary=folder/(rid+'.coarse-'+uuid.uuid4().hex+'.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,vertices=v,edges=e,source_sha256=meta['sha256'])
            temporary.replace(coarse)
    return {**meta,'lod':lod,'source_vertices':meta['vertices'],'source_edges':meta['edges'],'vertices':(v/1000).reshape(-1).tolist(),'edges':e.reshape(-1).tolist(),'display_units':'micrometers'}

def coverage():
    folder=data_root()/'morphology/v783';available=0;missing=[];size=0
    for p in folder.glob('*.json'):
        try:
            m=json.loads(p.read_text())
            if m['status']=='available':available+=1;size+=m['bytes']
            else:missing.append(m['id'])
        except (OSError,ValueError,KeyError):continue
    return dict(modeled_neurons=len(catalog()),cached_neurons=available,unavailable_ids=missing,cache_bytes=size,source=SOURCE,units='nanometers',license='Local-only cache; standalone redistribution license unverified')

def coarse_geometry(vertices,edges,stride=4):
    """Collapse degree-two chains, retaining branch/end nodes and cycle components."""
    from scipy.sparse import coo_matrix
    n=len(vertices)
    graph=coo_matrix((np.ones(len(edges)*2,dtype=np.uint8),(np.concatenate((edges[:,0],edges[:,1])),np.concatenate((edges[:,1],edges[:,0])))),shape=(n,n)).tocsr()
    degree=np.diff(graph.indptr);visited=set();pairs=[]
    def key(a,b):return (min(a,b),max(a,b))
    for a in np.flatnonzero(degree!=2):
        for neighbor in graph.indices[graph.indptr[a]:graph.indptr[a+1]]:
            if key(a,neighbor) in visited:continue
            previous,current,anchor,steps=int(a),int(neighbor),int(a),1;visited.add(key(a,neighbor))
            while True:
                if degree[current]!=2 or steps%stride==0:pairs.append((anchor,current));anchor=current
                if degree[current]!=2:break
                choices=graph.indices[graph.indptr[current]:graph.indptr[current+1]];next_=int(choices[1] if choices[0]==previous else choices[0])
                if key(current,next_) in visited:break
                visited.add(key(current,next_));previous,current=current,next_;steps+=1
    for a,b in edges:
        if key(a,b) not in visited:pairs.append((int(a),int(b)))
    reduced=np.asarray(pairs,dtype=np.uint32).reshape(-1,2);used=np.unique(reduced);mapping=np.zeros(n,dtype=np.uint32);mapping[used]=np.arange(len(used),dtype=np.uint32)
    return vertices[used],mapping[reduced]
