"""Restore only processes suspended by the battery-saving action.

PID creation time and executable are checked so a reused PID is never signaled.
Fly Garden's separate full-brain experiment remains paused until requested.
"""
import json,os,signal,time
from pathlib import Path
import psutil

manifest=Path('/Users/REVIEW_USER/FlyGarden/reports/brain-integration/stage8-20261006/system-power-pause.json')
state=json.loads(manifest.read_text())
if state.get('status')=='closed':
 print('Apps were closed to preserve battery. Reopen them normally; completed Fly Garden results remain saved.')
 raise SystemExit(0)
resumed=[];missing=[]
for record in state['processes']:
 try:
  process=psutil.Process(record['pid'])
  if process.create_time()!=record['created'] or process.exe()!=record['exe']:
   missing.append(record['pid']);continue
  if process.status()==psutil.STATUS_STOPPED:os.kill(process.pid,signal.SIGCONT)
  resumed.append(process.pid)
 except (psutil.NoSuchProcess,psutil.AccessDenied):missing.append(record['pid'])
state.update(status='resumed',resumed_at=time.time(),resumed_pids=resumed,missing_pids=missing)
temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(manifest)
print(f'Restored {len(resumed)} processes. Fly Garden computation remains separately paused.')
