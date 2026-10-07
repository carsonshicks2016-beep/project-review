import shutil
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from .config import Config
from .manager import Manager
from .storage import ROOT,read

app=FastAPI(title='Melee Next');manager=Manager()

@app.middleware('http')
async def local_only(request:Request,call_next):
    host=request.headers.get('host','').split(':')[0]
    if host not in ('127.0.0.1','localhost','testserver'):return JSONResponse({'detail':'Local access only'},403)
    if request.method!='GET':
        origin=request.headers.get('origin')
        if origin and origin not in ('http://127.0.0.1:8776','http://localhost:8776','http://testserver'):return JSONResponse({'detail':'Origin rejected'},403)
    return await call_next(request)

@app.get('/')
def index():return FileResponse(ROOT/'melee_next/web/index.html')
app.mount('/assets',StaticFiles(directory=ROOT/'melee_next/web'),name='assets')
@app.get('/api/state')
def state():
    return dict(config=Config.load().asdict(),runs=manager.listing(),checkpoints=manager.checkpoints(),
                disk_free_gb=round(shutil.disk_usage(ROOT).free/1024**3,2),version='2.0.0',
                benchmarks=[dict(id=p.parent.name,**read(p)) for p in sorted((ROOT/'runs').glob('*/benchmark.json'),reverse=True)])
@app.post('/api/runs')
def start(body:dict):
    try:return manager.start(body)
    except (ValueError,TypeError,KeyError) as exc:raise HTTPException(400,str(exc))
@app.post('/api/runs/{ident}/{action}')
def control(ident:str,action:str):
    try:return manager.control(ident,action)
    except (ValueError,ProcessLookupError) as exc:raise HTTPException(400,str(exc))
