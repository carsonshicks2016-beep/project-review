from dataclasses import asdict
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from .config import Config
from .manager import Manager, RunBusy

app=FastAPI(title='Melee Lab',docs_url=None,redoc_url=None)
manager=Manager()

@app.middleware('http')
async def local_only(request:Request,call_next):
    from starlette.responses import JSONResponse
    host=request.headers.get('host','')
    if host.split(':')[0] not in ('127.0.0.1','localhost','testserver'):
        return JSONResponse({'detail':'Local access only.'},status_code=403)
    if request.method!='GET':
        origin=request.headers.get('origin')
        if origin and origin not in ('http://'+host,'https://'+host):
            return JSONResponse({'detail':'Cross-origin changes are blocked.'},status_code=403)
    response=await call_next(request)
    response.headers['Cache-Control']='no-store'
    return response

@app.get('/api/state')
def state(run_id:str|None=None):
    config=Config.load(manager.root/'config.local.json')
    try: config.validate(); ready=True; issue=None
    except Exception as e: ready=False; issue=str(e)
    checkpoints=manager.checkpoints()
    from .readiness import report
    return dict(config=asdict(config),ready=ready,issue=issue,
                runs=manager.list_runs(include_slots=True,selected_run=run_id),checkpoints=checkpoints,datasets=manager.datasets(),
                champions=[c for c in checkpoints if c['champion']],availability=manager.availability(),readiness=report(manager,checkpoints,ready,issue))

class Start(BaseModel):
    mode:Literal['train','evaluate','play']='train'
    steps:int=Field(100000,ge=1024,le=100000000)
    bootstrap:int=Field(4096,ge=0,le=1000000)
    checkpoint:str|None=None
    episodes:int=Field(10,ge=1,le=1000)
    curriculum:bool=True
    envs:int=Field(1,ge=1,le=12)
    level:int|None=Field(None,ge=1,le=9)
    human_character:Literal['FOX','MARIO','MARTH','FALCO','CPTFALCON','PEACH','SHEIK','JIGGLYPUFF','SAMUS','PIKACHU','GANONDORF','LUIGI','DOC']|None=None
    opponent_checkpoint:str|None=None
    benchmark:bool=False
    fast_inputs:bool=False
    anchor:str|None=None
    anchor_coef:float=Field(.5,ge=0,le=10)
    anchor_floor:float=Field(.12,ge=0,le=10)
    win_threshold:float=Field(.60,ge=0.1,le=1.0)
    finetune:bool=False
    recurrent:bool=False
    execution_mode:Literal['assisted','raw']='assisted'
    league_checkpoints:list[str]|None=Field(None,max_length=12)
    character:Literal['FOX','JIGGLYPUFF','MARTH','FALCO','CPTFALCON','PEACH','SHEIK']|None=None

    @model_validator(mode='after')
    def parallel_rules(self):
        if self.envs>1:
            if self.mode!='train': raise ValueError('Evaluation uses one emulator.')
            if not self.checkpoint and not self.anchor and not self.recurrent:
                raise ValueError('Parallel training requires a checkpoint or tournament anchor.')
        if self.mode in ('evaluate','play') and not self.checkpoint: raise ValueError('Evaluation and challenge require a checkpoint.')
        if self.fast_inputs and self.mode!='train': raise ValueError('Controller transfer requires training.')
        if self.benchmark and self.mode!='evaluate': raise ValueError('Benchmark requires evaluation mode.')
        if self.opponent_checkpoint and self.mode!='train': raise ValueError('Frozen opponents require training mode.')
        if self.anchor and self.mode!='train': raise ValueError('Demonstration anchoring requires training mode.')
        if self.league_checkpoints is not None and (self.mode!='train' or not self.checkpoint or self.opponent_checkpoint):
            raise ValueError('League training requires a checkpoint and no fixed opponent.')
        return self

@app.post('/api/start')
def start(data:Start):
    try: return {'id':manager.start(**data.model_dump())}
    except RunBusy as e: raise HTTPException(409,dict(message=str(e),retryable=True))
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,str(e))

@app.post('/api/control/{action}')
def control(action:Literal['pause','resume','save','stop']):
    try: return {'id':manager.control(action)}
    except ValueError as e: raise HTTPException(400,str(e))

class Settings(BaseModel):
    iso:str
    dolphin:str
    cpu_level:int=Field(1,ge=1,le=9)
    speed:float=Field(1,ge=0,le=4)
    randomize_opponent:bool=False
    character:Literal['FOX','JIGGLYPUFF','MARTH','FALCO','CPTFALCON','PEACH','SHEIK']|None=None

@app.post('/api/config')
def settings(data:Settings):
    if not manager.availability()['can_start']: raise HTTPException(409,'Wait for the current run to finish closing before changing setup.')
    config=Config.load(manager.root/'config.local.json')
    for key,value in data.model_dump().items(): setattr(config,key,value)
    try: config.validate(); config.save(manager.root/'config.local.json')
    except (ValueError,OSError) as e: raise HTTPException(400,str(e))
    return {'ok':True}

@app.get('/api/runs/{run_id}/matches')
def matches(run_id:str,kind:Literal['ppo','evaluation','scripted demonstration','human challenge','self-play','all']='ppo',
            cpu:int|None=Query(None,ge=0,le=9),outcome:Literal['win','loss','draw','timeout','interrupted','unexpected_restart']|None=None,
            offset:int=Query(0,ge=0),limit:int=Query(25,ge=1,le=100)):
    try: directory=manager.run_path(run_id)
    except ValueError as e: raise HTTPException(404,str(e))
    return manager.matches.query(directory/'matches.jsonl',kind,cpu,outcome,offset,limit)

class Promote(BaseModel):
    checkpoint:str
    name:str=Field(min_length=1,max_length=80)
    note:str=Field('',max_length=2000)

@app.post('/api/champions')
def promote(data:Promote):
    try: return manager.promote(**data.model_dump())
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,str(e))

@app.get('/api/moves')
def moves(checkpoint:str|None=None,expanded:bool=False):
    from .actions import describe as describe_actions
    from .catalog import read_json
    try:
        config=Config(**read_json(manager.catalog.resolve(checkpoint).with_suffix('.json'))['config']) if checkpoint else Config.load(manager.root/'config.local.json')
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,str(e))
    if expanded: config.action_set='expanded'; config.action_frames=1
    if getattr(config,'action_set','legacy')=='controller':
        from .controller import describe as describe_controller
        return describe_controller(config)
    return describe_actions(config)

@app.get('/panel')
def panel_page(): return FileResponse(Path(__file__).parent/'web/panel.html')

@app.get('/api/panel')
def panel_data():
    import importlib
    from . import panel
    importlib.reload(panel)
    return panel.payload(manager)

@app.get('/api/datasets')
def datasets():
    return {'datasets':manager.datasets()}

@app.get('/api/champions')
def champions():
    return {'champions':[c for c in manager.checkpoints() if c['champion']]}

class Compare(BaseModel):
    candidates:list[str]=Field(min_length=2,max_length=4)
    episodes:int=Field(10,ge=1,le=1000)
    roster:bool=True
    modes:list[Literal['assisted','raw']]=Field(default_factory=lambda:['assisted','raw'],min_length=1,max_length=2)
    seed:int=Field(17001,ge=0,le=2147483647)

    @model_validator(mode='after')
    def distinct_candidates(self):
        if len(set(self.candidates))!=len(self.candidates):
            raise ValueError('Choose distinct candidates; each one is mirrored against itself automatically.')
        return self

@app.post('/api/compare')
def compare(data:Compare):
    try: return {'id':manager.compare(**data.model_dump())}
    except RunBusy as e: raise HTTPException(409,dict(message=str(e),retryable=True))
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,str(e))

@app.get('/api/compare')
def comparison_pace():
    return manager.evaluation_pace()

@app.get('/api/compare/{run_id}')
def comparison(run_id:str):
    try: return manager.comparison(run_id)
    except ValueError as e: raise HTTPException(404,str(e))

class HumanFrame(BaseModel):
    buttons:list[Literal['A','B','X','Y','Z','L','R','START','D_UP','D_DOWN','D_LEFT','D_RIGHT']]=Field(default_factory=list,max_length=12)
    main:tuple[float,float]=(.5,.5)
    c:tuple[float,float]=(.5,.5)
    triggers:tuple[float,float]=(0.,0.)

    @model_validator(mode='after')
    def valid_axes(self):
        import math
        if any(not math.isfinite(v) or not 0<=v<=1 for v in (*self.main,*self.c,*self.triggers)):
            raise ValueError('Controller values must be finite values between 0 and 1.')
        return self

_input_owners={}

@app.websocket('/api/play/{run_id}/input')
async def human_input(socket:WebSocket,run_id:str):
    import json
    import time
    from .storage import write_json
    from pydantic import ValidationError
    host=socket.headers.get('host','')
    if host.split(':')[0] not in ('localhost','127.0.0.1','testserver') or socket.headers.get('origin') not in ('http://'+host,'https://'+host):
        await socket.close(code=1008); return
    try: directory=manager.run_path(run_id)
    except ValueError:
        await socket.close(code=1008); return
    active=manager.active()
    if not active or active['id']!=run_id or active['mode']!='play' or run_id in _input_owners:
        await socket.close(code=1008); return
    owner=object(); _input_owners[run_id]=owner
    await socket.accept()
    last_write=0
    try:
        while True:
            raw=await socket.receive_text()
            if len(raw)>2048:
                await socket.close(code=1009); break
            frame=HumanFrame.model_validate(json.loads(raw))
            now=time.time()
            status=manager.cached_json(directory/'status.json').get('status')
            if status not in ('starting','running','paused'):
                await socket.close(code=1000); break
            if now-last_write<1/120: continue
            write_json(directory/'human-input.json',dict(frame.model_dump(),received_at=now))
            last_write=now
    except (WebSocketDisconnect,ValidationError,ValueError):
        pass
    finally:
        if _input_owners.get(run_id) is owner:
            write_json(directory/'human-input.json',dict(HumanFrame().model_dump(),received_at=0))
            _input_owners.pop(run_id,None)

@app.get('/api/runs/{run_id}/diagnostics')
def diagnostics(run_id:str):
    try: directory=manager.run_path(run_id)
    except ValueError as e: raise HTTPException(404,str(e))
    log=directory/'worker.log'
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0,log.stat().st_size-16000))
            tail=handle.read().decode('utf-8',errors='replace')
    else: tail='No worker log yet.'
    artifacts=[]
    for p in sorted(directory.rglob('*')):
        if p.is_file() and (p.suffix=='.slp' or p.name in ('evaluation.json','challenge.json','config.json','request.json','worker.log')):
            artifacts.append(dict(name=str(p.relative_to(directory)),bytes=p.stat().st_size))
    return dict(log=tail,artifacts=artifacts)

@app.get('/api/runs/{run_id}/artifact/{name:path}')
def artifact(run_id:str,name:str):
    try: directory=manager.run_path(run_id)
    except ValueError as e: raise HTTPException(404,str(e))
    path=(directory/name).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or not (path.suffix=='.slp' or path.name in ('evaluation.json','challenge.json','config.json','request.json','worker.log')):
        raise HTTPException(404,'Artifact was not found.')
    return FileResponse(path,filename=path.name)

class ProbeRequest(BaseModel):
    target: str
    stress_level: Literal['standard', 'extreme'] = 'extreme'

@app.get('/api/diagnostics/targets')
def diagnostic_targets():
    from .diagnostics import list_targets
    return {'targets': list_targets(manager.root)}

@app.post('/api/diagnostics/probe')
def probe_diagnostic_target(data: ProbeRequest):
    from .diagnostics import probe_target
    try:
        return probe_target(data.target, root=manager.root, stress_level=data.stress_level)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Diagnostic probe failed: {e}")

@app.post('/api/diagnostics/upload')
async def upload_diagnostic_file(request: Request, filename: str = Query(...)):
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in ('.zip', '.slp', '.npz', '.gci', '.raw'):
        raise HTTPException(400, f"Unsupported file format '{suffix}'. Please upload a .zip, .slp, .npz, or .gci file.")
    upload_dir = manager.root / 'diagnostics_uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / name
    if dest.exists():
        import time
        dest = upload_dir / f"{Path(name).stem}_{int(time.time())}{suffix}"
    content = await request.body()
    dest.write_bytes(content)
    rel_path = str(dest.relative_to(manager.root))
    return {'path': rel_path, 'filename': name, 'size_bytes': len(content)}

@app.post('/api/checkpoints/import')
async def import_checkpoint_file(
    request: Request,
    filename: str | None = Query(None),
    name: str | None = Query(None),
    character: Literal['FOX','JIGGLYPUFF','MARTH','FALCO','CPTFALCON','PEACH','SHEIK'] | None = Query(None),
    source_path: str | None = Query(None)
):
    import io
    import json
    import time
    import uuid
    import zipfile
    from .storage import write_json, is_recurrent_checkpoint

    if source_path:
        src = Path(source_path)
        if not src.is_absolute():
            src = (manager.root / src).resolve()
        if not src.exists():
            raise HTTPException(404, f"Source file not found: {source_path}")
        content = src.read_bytes()
        fname = Path(filename or src.name).name
    else:
        if not filename:
            raise HTTPException(400, "Filename is required when streaming upload.")
        content = await request.body()
        fname = Path(filename).name

    if not content:
        raise HTTPException(400, "Uploaded file content is empty.")

    suffix = Path(fname).suffix.lower()

    # Case 1: GameCube memory card save (.gci or .raw)
    if suffix in ('.gci', '.raw'):
        gc_dir = manager.root / '.runtime/dolphin-user/GC/USA/Card A'
        gc_dir.mkdir(parents=True, exist_ok=True)
        dest = gc_dir / fname
        dest.write_bytes(content)
        rel_path = str(dest.relative_to(manager.root))
        return {
            'ok': True,
            'type': 'gamecube_save',
            'path': rel_path,
            'filename': fname,
            'message': f"Installed GameCube memory card save '{fname}' into Dolphin Memory Card A!"
        }

    # Case 2: Model policy checkpoint (.zip)
    if suffix != '.zip':
        raise HTTPException(400, f"Unsupported save file format '{suffix}'. Expected .zip (policy) or .gci (GameCube save).")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            is_valid_model = any(n in names for n in ('data', 'policy.pth', 'pytorch_variables.pth'))
            if not is_valid_model:
                raise HTTPException(400, "The uploaded .zip is not a valid Melee policy checkpoint (missing 'data' or 'policy.pth').")

            steps = 0
            if 'data' in names:
                try:
                    data_json = json.loads(z.read('data'))
                    steps = int(data_json.get('num_timesteps', 0))
                except Exception:
                    pass
    except zipfile.BadZipFile:
        raise HTTPException(400, "The uploaded file is not a valid zip archive.")

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    slug = uuid.uuid4().hex[:5]
    char = character or ('JIGGLYPUFF' if 'puff' in fname.lower() else 'FOX')
    display_name = name.strip() if name and name.strip() else f"Imported {char.title()} ({Path(fname).stem})"

    run_dir = manager.root / 'runs' / f"imported-{timestamp}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    zip_path = run_dir / 'latest.zip'
    zip_path.write_bytes(content)

    is_recurrent = is_recurrent_checkpoint(zip_path)
    architecture = 'recurrent' if is_recurrent else 'feedforward'

    from .controller import contract as controller_contract
    meta = {
        'schema': 'melee-lab-v1',
        'observation_size': 1836,
        'actions': controller_contract(),
        'steps': steps,
        'architecture': architecture,
        'config': {
            'character': char,
            'opponent': 'FOX' if char != 'FOX' else 'MARIO',
            'cpu_level': 9,
            'stage': 'FINAL_DESTINATION',
            'stocks': 3,
            'action_frames': 1,
            'action_set': 'controller',
            'speed': 1.0,
            'randomize_opponent': True
        }
    }
    write_json(zip_path.with_suffix('.json'), meta)
    write_json(run_dir / 'status.json', {
        'status': 'completed',
        'phase': f"imported save file: {display_name}",
        'steps': steps,
        'mode': 'train',
        'envs': 1,
        'imported': True,
        'display_name': display_name
    })

    rel_path = str(zip_path.relative_to(manager.root))
    return {
        'ok': True,
        'type': 'checkpoint',
        'path': rel_path,
        'name': display_name,
        'character': char,
        'steps': steps,
        'architecture': architecture,
        'message': f"Successfully imported {display_name} ({steps:,} steps) into Checkpoint Library!"
    }


# ---------------------------------------------------------------------------
# Fleet Benchmark & Leaderboard Endpoints
# ---------------------------------------------------------------------------

fleet_scan_state: dict = {
    'is_scanning': False,
    'current': 0,
    'total': 0,
    'current_file': '',
    'scope': 'distinct',
    'rankings': [],
    'error': None,
    'completed_at': None
}

@app.get('/api/diagnostics/rankings')
def get_rankings(scope: str = 'distinct'):
    import json
    if fleet_scan_state.get('rankings') and not fleet_scan_state.get('is_scanning'):
        return {
            'is_scanning': fleet_scan_state['is_scanning'],
            'current': fleet_scan_state['current'],
            'total': fleet_scan_state['total'],
            'current_file': fleet_scan_state['current_file'],
            'rankings': fleet_scan_state['rankings'],
            'completed_at': fleet_scan_state['completed_at']
        }

    cache_file = manager.root / '.runtime/diagnostics/rankings_cache.json'
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            rankings = [v['dossier'] for v in cache.values() if 'dossier' in v]
            rankings.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
            for i, r in enumerate(rankings):
                r['rank_position'] = i + 1
            fleet_scan_state['rankings'] = rankings
            return {
                'is_scanning': fleet_scan_state['is_scanning'],
                'current': len(rankings),
                'total': len(rankings),
                'current_file': '',
                'rankings': rankings,
                'completed_at': cache_file.stat().st_mtime
            }
        except Exception:
            pass

    return {
        'is_scanning': fleet_scan_state['is_scanning'],
        'current': 0,
        'total': 0,
        'current_file': '',
        'rankings': [],
        'completed_at': None
    }

@app.get('/api/diagnostics/rankings/status')
def get_rankings_status():
    return fleet_scan_state

@app.post('/api/diagnostics/rankings/scan')
async def trigger_fleet_scan(scope: str = Query('distinct')):
    import threading
    import time
    from .diagnostics import scan_fleet

    if fleet_scan_state['is_scanning']:
        return {'ok': True, 'message': 'Scan is already running in background.', 'status': fleet_scan_state}

    fleet_scan_state['is_scanning'] = True
    fleet_scan_state['current'] = 0
    fleet_scan_state['total'] = 0
    fleet_scan_state['current_file'] = 'Initializing scanner...'
    fleet_scan_state['scope'] = scope
    fleet_scan_state['error'] = None

    def run_worker():
        try:
            def progress(curr, tot, fname):
                fleet_scan_state['current'] = curr
                fleet_scan_state['total'] = tot
                fleet_scan_state['current_file'] = fname

            results = scan_fleet(root=manager.root, scope=scope, progress_cb=progress, stress_level='standard')
            fleet_scan_state['rankings'] = results
            fleet_scan_state['completed_at'] = time.time()
        except Exception as e:
            fleet_scan_state['error'] = str(e)
        finally:
            fleet_scan_state['is_scanning'] = False

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()

    return {'ok': True, 'message': f"Started fleet scan ({scope}) in background."}


@app.get('/')
def index(): return FileResponse(Path(__file__).parent/'web/index.html')
app.mount('/assets',StaticFiles(directory=Path(__file__).parent/'web'),name='assets')
