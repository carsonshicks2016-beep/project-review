"""Operational scheduling amendment; original worker and scientific protocol stay fixed."""
import argparse, concurrent.futures, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import psutil
import sweep_adaptive_lif as frozen
from flygarden.recording import atomic_json,space_check
OUT=frozen.OUT

def verify():
 p=json.loads((OUT/'protocol.json').read_text())
 assert all(frozen.file_sha(ROOT/n)==h for n,h in p['sources'].items())
 a=json.loads((OUT/'scheduling-amendment-three-worker.json').read_text())
 assert frozen.file_sha(Path(__file__))==a['coordinator_sha256']
 return p

def worker(args):
 p=verify();dest=OUT if not args.benchmark else OUT/'benchmark-three-worker'
 frozen.OUT=dest;dest.mkdir(exist_ok=True);frozen.prepare=lambda:p
 original=frozen.atomic_json
 def write(path,value):
  if Path(path).name=='current-worker.json':
   path=dest/'worker-status'/f'{args.model}-{args.cue}-{args.seed}.json';path.parent.mkdir(exist_ok=True)
  original(path,value)
 frozen.atomic_json=write
 frozen.worker(args.model,args.cue,args.seed,args.resume)

def task(model,cue,seed,benchmark=False):
 dest=OUT if not benchmark else OUT/'benchmark-three-worker';folder=dest/'trials'/f'{model}-{cue}-{seed}'
 argv=[sys.executable,__file__,'--worker','--model',model,'--cue',cue,'--seed',str(seed)]+(['--benchmark'] if benchmark else [])
 if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())['status']=='complete','Interrupted trial requires explicit recovery'
 else:
  space_check(dest,1200*1024**2)
  subprocess.run(argv,cwd=ROOT,check=True)
 if cue=='a_left' and not (folder/'continuation-result.json').exists():subprocess.run(argv+['--resume'],cwd=ROOT,check=True)
 return folder.name

def telemetry():
 m=psutil.virtual_memory();s=psutil.swap_memory();rss=0
 for proc in psutil.Process().children(recursive=True):
  try:rss+=proc.memory_info().rss
  except psutil.NoSuchProcess:pass
 return {'time':time.time(),'available_bytes':m.available,'swap_used_bytes':s.used,'swap_in_bytes':s.sin,'swap_out_bytes':s.sout,'child_rss_bytes':rss}

def batch(jobs,workers,phase):
 done=[];active={};samples=[];it=iter(jobs);start=time.monotonic()
 def progress():
  atomic_json(OUT/'parallel-progress.json',{'status':'running','phase':phase,'workers':workers,'completed_in_batch':done,'active':list(active.values()),'wall_seconds':time.monotonic()-start,'pid':os.getpid()})
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  def fill():
   while len(active)<workers:
    try:model,cue,seed=next(it)
    except StopIteration:return
    active[pool.submit(task,model,cue,seed,phase=='benchmark')]=f'{model}-{cue}-{seed}'
  fill()
  while active:
   progress();samples.append(telemetry())
   finished,_=concurrent.futures.wait(active,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in finished:done.append(f.result());del active[f]
   fill()
 return {'completed':done,'wall_seconds':time.monotonic()-start,'samples':samples}

def run():
 with (ROOT/'.runtime/experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=verify()
  benchmark_file=OUT/'three-worker-benchmark.json'
  if not benchmark_file.exists():
   jobs=[('projection_t50_b13p5','a_left',12001),('projection_t150_b13p5','a_left',12001),('projection_t450_b13p5','a_left',12001)]
   measured=batch(jobs,3,'benchmark');serial=0
   for model,cue,seed in jobs:
    name=f'{model}-{cue}-{seed}';original=OUT/'trials'/name;repeat=OUT/'benchmark-three-worker'/'trials'/name
    m=json.loads((original/'manifest.json').read_text());serial+=m['wall_seconds']
    for ch in m['chunks']:
     with frozen.np.load(original/ch['file']) as a,frozen.np.load(repeat/ch['file']) as b:
      assert a.files==b.files and all(frozen.np.array_equal(a[k],b[k]) for k in a.files),('Parallel parity',name,ch['file'])
   s=measured['samples'];gain=serial/measured['wall_seconds'];swap_delta=max(x['swap_out_bytes'] for x in s)-s[0]['swap_out_bytes'];rss=max(x['child_rss_bytes'] for x in s)
   # Conservative operational gate: speedup, aggregate process footprint, and no material new swap traffic.
   passed=gain>=1.8 and rss<10*2**30 and swap_delta<256*2**20
   measured.update(parity_exact=True,serial_reference_seconds=serial,speedup=gain,peak_child_rss_bytes=rss,new_swap_out_bytes=swap_delta,three_workers_accepted=passed)
   atomic_json(benchmark_file,measured)
  measured=json.loads(benchmark_file.read_text());workers=3 if measured['three_workers_accepted'] else 2
  jobs=[('zero','a_left',12001)]+[(m,c,s) for s in p['calibration_seeds'] for m in p['models'] for c in ['none','a_left','a_right','a_both']]
  finished=batch(jobs,workers,'calibration')
  cal=frozen.evaluate(p,p['models'],p['calibration_seeds'],['none','a_left','a_right','a_both']);atomic_json(OUT/'calibration-results.json',cal)
  selected=next((m for m in p['selection_order'] if cal['eligible'][m]),None)
  atomic_json(OUT/'selection.json',{'selected':selected,'parameters':p['parameters'].get(selected),'basis':'Frozen domain/step/tau order, every gate on both calibration seeds','calibration_only':True})
  if selected:
   jobs=[(m,c,s) for s in p['held_out_seeds'] for m in ['original',selected] for c in ['none','a_left','a_right','a_both','equal','g60','g40']]
   h=batch(jobs,workers,'heldout');finished['completed']+=h['completed']
   result=frozen.evaluate(p,['original',selected],p['held_out_seeds'],['none','a_left','a_right','a_both','equal','g60','g40']);atomic_json(OUT/'heldout-results.json',result)
  atomic_json(OUT/'progress.json',{'status':'complete','completed':finished['completed'],'selected':selected,'heldout_run':bool(selected),'workers':workers,'application_changed':False,'promoted':False})
  atomic_json(OUT/'parallel-progress.json',{'status':'complete','workers':workers,'selected':selected})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--worker',action='store_true');a.add_argument('--benchmark',action='store_true');a.add_argument('--resume',action='store_true');a.add_argument('--model');a.add_argument('--cue');a.add_argument('--seed',type=int);args=a.parse_args()
 if args.worker:worker(args)
 else:
  try:run()
  except BaseException as e:
   atomic_json(OUT/'parallel-progress.json',{'status':'interrupted','error':repr(e),'pid':os.getpid()});raise
