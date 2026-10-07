"""Durable local MP4 queue. One ownership-checked worker, independent of HTTP."""
import json,time,uuid,subprocess,sys,os,threading
import psutil
from .storage import ROOT
from .recording import data_root,atomic_json,run_folder,space_check

def jobs_root():
    p=data_root()/'exports';p.mkdir(parents=True,exist_ok=True);return p

def job_folder(identity):
    if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid export identifier')
    new=jobs_root()/identity
    return new if new.exists() else ROOT/'data/exports'/identity

def owned(job):
    try:
        p=psutil.Process(job['pid'])
        return p.is_running() and abs(p.create_time()-job['process_created'])<.1 and str(ROOT/'scripts/render_queue.py') in p.cmdline() and p.status()!=psutil.STATUS_ZOMBIE
    except (psutil.Error,KeyError):return False

def statuses():
    out=[]
    paths={p.parent.name:p for p in (ROOT/'data/exports').glob('*/job.json')}
    paths.update({p.parent.name:p for p in jobs_root().glob('*/job.json')})
    for p in paths.values():
        try:
            j=json.loads(p.read_text())
            if j['status']=='rendering' and not owned(j):
                j.update(status='interrupted',error='Rendering worker exited; recording preserved. Retry export.');atomic_json(p,j)
            out.append(j)
        except (ValueError,OSError,KeyError):continue
    return sorted(out,key=lambda j:j['created'],reverse=True)

def enqueue_export(run,settings=None,automatic=True):
    folder=run_folder(run)
    if not (folder/'frames.jsonl').exists():raise ValueError('No recorded frames to export')
    settings=settings or {};camera=settings.get('camera','follow');population=settings.get('population','all')
    if camera not in ('follow','overhead','free'):raise ValueError('Unknown export camera')
    camera_view=None
    if camera=='free':
        import math
        camera_view=settings.get('camera_view')
        if not isinstance(camera_view,dict) or any(not isinstance(camera_view.get(k),list) or len(camera_view[k])!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>10000 for v in camera_view[k]) for k in ('position','target')):
            raise ValueError('Free camera requires a finite position and target')
        camera_view={k:list(camera_view[k]) for k in ('position','target')}
    from .morphology import catalog
    if population!='all' and population not in {n['cell_class'] for n in catalog()} and population not in {n['cell_type'] for n in catalog()}:raise ValueError('Unknown neural population')
    if automatic:
        existing=[j for j in statuses() if j['run']==run and j.get('automatic')]
        if existing:return existing[0]
    identity=uuid.uuid4().hex;p=jobs_root()/identity;p.mkdir();job=dict(id=identity,run=run,status='queued',created=time.time(),automatic=automatic,settings=dict(camera=camera,camera_view=camera_view,population=population,width=1920,height=1080,fps=30),progress=0.)
    atomic_json(p/'job.json',job);ensure_worker();return job

def retry(identity):
    if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid export identifier')
    p=job_folder(identity)/'job.json';j=json.loads(p.read_text())
    if j['status'] not in ('failed','interrupted'):raise ValueError('Only failed or interrupted exports can be retried')
    j.update(status='queued',progress=0.,error=None,phase='Queued for retry',rendered_frames=0);j.pop('anatomy',None);atomic_json(p,j);ensure_worker();return j

def ensure_worker():
    jobs=statuses()
    if not any(j['status']=='queued' for j in jobs) or any(j['status']=='rendering' and owned(j) for j in jobs):return
    # Startup races are resolved by the worker's flock, before it touches a job.
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True);owner=runtime/'video-worker.json'
    if owner.exists():
        try:
            if owned(json.loads(owner.read_text())):return
        except (OSError,ValueError):pass
    with open(runtime/'video-worker.log','ab') as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'scripts/render_queue.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
    try:atomic_json(owner,dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time()))
    except psutil.Error:pass

def simulation_busy():
    import fcntl,urllib.request
    from . import experiment_jobs
    if experiment_jobs.active():return True
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    try:
        with urllib.request.urlopen('http://127.0.0.1:8794/api/state',timeout=1) as response:
            state=json.load(response);return bool(state.get('running')) or str(state.get('phase','')).startswith('Loading')
    except Exception:return False
