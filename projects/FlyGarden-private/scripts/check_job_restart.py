import json,time,subprocess
from pathlib import Path
import httpx
root=Path(__file__).resolve().parents[1];c=httpx.Client(base_url='http://127.0.0.1:8794',timeout=180)
r=c.post('/api/experiments',json={'protocol':'combined'});r.raise_for_status();identity=r.json()['id'];subprocess.run([str(root/'Close Fly Garden.command')],check=True);subprocess.run([str(root/'Open Fly Garden.command'),'--no-browser'],check=True)
status=next(x for x in c.get('/api/jobs').json() if x['id']==identity);blocked=c.post('/api/experiments',json={'protocol':'body'}).status_code==409;paused=c.post('/api/command',json={'action':'play'}).status_code==409;checks={'job_survives_app_restart':status['status']=='running','duplicate_rejected':blocked,'active_fly_remains_paused':paused}
(root/'reports/job-restart.json').write_text(json.dumps({'id':identity,'checks':checks,'status':'awaiting_completion'},indent=2));print(identity,checks,flush=True)
for _ in range(180):
 job=next(x for x in c.get('/api/jobs').json() if x['id']==identity)
 if job['status']!='running':break
 time.sleep(1)
checks['completed_and_archived']=job['status']=='completed' and (root/'data/experiments'/identity/'result.json').exists();report={'id':identity,'status':job['status'],'checks':checks,'passed':all(checks.values())};(root/'reports/job-restart.json').write_text(json.dumps(report,indent=2));print(report,flush=True);assert report['passed']
