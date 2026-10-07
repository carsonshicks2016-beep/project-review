"""Ownership-aware local start/stop; executable arguments are allowlisted."""
from pathlib import Path
import argparse,json,time,subprocess,sys,os,urllib.request,fcntl
import psutil
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'.runtime';RUNTIME.mkdir(exist_ok=True);URL='http://127.0.0.1:8794';PID=RUNTIME/'server.json'
def healthy():
 try:
  with urllib.request.urlopen(URL+'/api/health',timeout=1) as r:return json.load(r).get('app')=='flygarden'
 except Exception:return False
def owned():
 if not PID.exists():return None
 try:
  saved=json.loads(PID.read_text());p=psutil.Process(saved['pid'])
  if abs(p.create_time()-saved['created'])>.01:return None
  cmd=p.cmdline()
  if 'flygarden.server:app' not in cmd or '--port' not in cmd or '8794' not in cmd or Path(p.cwd())!=ROOT:return None
  return p
 except (OSError,ValueError,psutil.Error):return None
parser=argparse.ArgumentParser();parser.add_argument('--stop',action='store_true');parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
with open(RUNTIME/'launcher.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if args.stop:
  p=owned()
  if p:
   p.terminate()
   try:p.wait(timeout=15)
   except psutil.TimeoutExpired:raise SystemExit('Fly Garden is still stopping; no unrelated process was touched.')
   PID.unlink(missing_ok=True);print('Fly Garden stopped.')
  else:print('No owned Fly Garden server to stop.')
  raise SystemExit(0)
 if not healthy():
  if owned():raise SystemExit('Owned server is starting or unhealthy; inspect .runtime/server.log.')
  log=open(RUNTIME/'server.log','ab');process=subprocess.Popen([sys.executable,'-m','uvicorn','flygarden.server:app','--host','127.0.0.1','--port','8794','--no-access-log'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});log.close();p=psutil.Process(process.pid);PID.write_text(json.dumps(dict(pid=process.pid,created=p.create_time())))
  for _ in range(120):
   if healthy():break
   if process.poll() is not None:PID.unlink(missing_ok=True);raise SystemExit('Could not start Fly Garden. Inspect .runtime/server.log; another app may use port 8794.')
   time.sleep(.5)
  else:raise SystemExit('Startup exceeded one minute; inspect .runtime/server.log.')
 print('Fly Garden: '+URL)
 if not args.no_browser:
  import webbrowser;webbrowser.open(URL)
