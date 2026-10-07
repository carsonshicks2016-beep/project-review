from pathlib import Path
import asyncio,json,subprocess,sys,uuid,os
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from .engine import LocalSimulation
from .storage import ROOT
from . import experiment_jobs,recording,morphology,video_jobs
engine=None
jobs={}
experiment_guard=asyncio.Lock()
@asynccontextmanager
async def lifespan(app):
 global engine
 engine=LocalSimulation()
 await asyncio.to_thread(recording.recover_interrupted)
 async def supervise_exports():
  while True:
   await asyncio.to_thread(video_jobs.ensure_worker);await asyncio.sleep(5)
 task=asyncio.create_task(supervise_exports())
 yield
 task.cancel()
 engine.close()
app=FastAPI(lifespan=lifespan)
@app.middleware('http')
async def local_only(request:Request,call_next):
 host=request.headers.get('host','').split(':')[0]
 if host not in ('127.0.0.1','localhost','testserver'):return __import__('starlette.responses',fromlist=['JSONResponse']).JSONResponse({'error':'Local access only'},status_code=403)
 origin=request.headers.get('origin')
 if origin and origin not in ('http://127.0.0.1:8794','http://localhost:8794'):return __import__('starlette.responses',fromlist=['JSONResponse']).JSONResponse({'error':'Origin not permitted'},status_code=403)
 return await call_next(request)
@app.get('/api/health')
def health():return {'app':'flygarden','ready':engine.state().get('ready',False)}
@app.get('/api/state')
def state():return engine.state()
@app.get('/api/geometry')
def geometry():return engine.geometry_cache
@app.post('/api/command')
async def command(body:dict):
 if body.get('action') in ('play','step','reset','edit','load','branch','save','mode','retina','learning','difficulty') and experiment_jobs.active():raise HTTPException(409,'Experiment running; model-changing controls are locked until it finishes. Recorded playback remains available')
 try:return await asyncio.to_thread(engine.submit,body.get('action',''),body.get('arguments',{}))
 except (ValueError,KeyError,FileNotFoundError,TimeoutError) as exc:raise HTTPException(400,str(exc))
@app.get('/api/reports')
def reports():
 return {p.stem:json.loads(p.read_text()) for p in (ROOT/'reports').glob('*.json') if p.stem in ('brain-benchmark','body-benchmark','combined-benchmark','body-acceptance','learning-evaluation','provenance','neuron-mapping','olfactory-pathway-summary','odor-calibration-summary','baseline-acceptance')}
@app.get('/api/runs')
def runs():
 records=[]
 for folder in recording.all_runs()[:100]:
  try:
   record=recording.metadata(folder.name);outcome=folder/'outcome.json'
   if outcome.exists():record['outcome']=json.loads(outcome.read_text())
   records.append(record)
  except (OSError,ValueError):continue
 return records
@app.get('/api/replay/{identity}')
def replay(identity:str):
 if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise HTTPException(400,'Invalid run identifier')
 p=recording.run_folder(identity)/'frames.jsonl'
 if not p.exists():raise HTTPException(404,'No recorded frames')
 meta=p.parent/'recording.json'
 if meta.exists():
  limit=json.loads(meta.read_text())['frames']
  def committed():
   with p.open('rb') as f:
    for index,line in enumerate(f):
     if index>=limit:break
     yield line
  return StreamingResponse(committed(),media_type='application/x-ndjson')
 return FileResponse(p,media_type='application/x-ndjson')
@app.get('/api/run-geometry/{identity}')
def run_geometry(identity:str):
 if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise HTTPException(400,'Invalid run identifier')
 p=recording.run_folder(identity)/'geometry.json'
 if not p.exists():raise HTTPException(404,'Legacy run lacks geometry')
 return FileResponse(p,media_type='application/json')
@app.post('/api/experiments')
async def experiments(body:dict):
 async with experiment_guard:
  if experiment_jobs.active():raise HTTPException(409,'An experiment is already running, including across app restarts')
  protocol=body.get('protocol','learning')
  if protocol not in ('learning','body','combined','baseline','recording30'):raise HTTPException(400,'Unknown protocol')
  await asyncio.to_thread(engine.submit,'pause')
  identity=uuid.uuid4().hex;log=open(ROOT/'reports'/f'job-{identity}.log','w');process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_experiment.py'),'--protocol',protocol,'--id',identity],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'});log.close();jobs[identity]=process;experiment_jobs.record_start(identity,protocol,process.pid)
  return dict(id=identity,protocol=protocol)
@app.get('/api/jobs')
def job_status():return experiment_jobs.statuses()
def media_error(exc):
 return HTTPException(404 if isinstance(exc,FileNotFoundError) else 400,str(exc))
@app.get('/api/recordings/{identity}')
def recorded_metadata(identity:str):
 try:return recording.metadata(identity)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/recordings/{identity}/events')
def recorded_events(identity:str):
 try:
  from .recorded_events import recorded_markers
  return recorded_markers(identity)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/recordings/{identity}/spikes/{index}')
def recorded_spikes(identity:str,index:int):
 if index<0:raise HTTPException(400,'Invalid chunk index')
 try:return recording.spike_chunk(identity,index)
 except (OSError,ValueError,KeyError,IndexError) as exc:raise media_error(exc)
@app.get('/api/recordings/{identity}/neurons')
def recorded_neurons(identity:str,population:str='all',limit:int=256,offset:int=0,query:str=''):
 if not 1<=limit<=512 or offset<0:raise HTTPException(400,'Invalid neuron selection size')
 try:return recording.neurons(identity,population,limit,offset,query)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/recordings/{identity}/neurons/{index}/history')
def recorded_neuron_history(identity:str,index:int):
 try:return recording.neuron_history(identity,index)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/morphology/coverage')
def morphology_coverage():return morphology.coverage()
@app.get('/api/morphology/{identity}')
def neuron_shape(identity:str,lod:str='coarse'):
 try:return morphology.skeleton(identity,lod)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/exports')
def exports():return video_jobs.statuses()
@app.post('/api/exports')
def export(body:dict):
 try:return video_jobs.enqueue_export(body['run'],body.get('settings'),automatic=False)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.post('/api/exports/{identity}/retry')
def retry_export(identity:str):
 try:return video_jobs.retry(identity)
 except (OSError,ValueError,KeyError) as exc:raise media_error(exc)
@app.get('/api/exports/{identity}/video')
def exported_video(identity:str):
 if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise HTTPException(400,'Invalid export identifier')
 path=video_jobs.job_folder(identity)/'video.mp4'
 if not path.exists():raise HTTPException(404,'Video is not ready')
 return FileResponse(path,media_type='video/mp4')
@app.get('/')
def index():return FileResponse(ROOT/'static/index.html')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
