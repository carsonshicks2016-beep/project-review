"""Bounded unranked preservation check on an existing frozen procedural course."""
import os,sys,json,fcntl,subprocess
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core
b=core.build_status();assert b['status']=='current'
c=next(c for c in core.courses() if c['id']=='ff259d0d4deb8672620bb14e23261e6177a5b9e7ae43fea7e75830348ed32d79');assert core.course_asset_valid(c)
f=ROOT/'.rally/circuit-review/stage-c-delivery-procedural';f.mkdir(exist_ok=False)
spec={'schema':1,'mode':'specialist','viewer':True,'evaluation':False,'controlProbe':True,'controlMode':'waypoint-follow','controlTargetSpeed':8,'controlProbeSeconds':20,'seed':2026,'startingGear':'neutral','courseBundle':c['bundle'],'courseId':c['id'],'courseName':c['name']+' / UNRANKED REGRESSION','courseFamily':'gentle','runId':'stage-c-procedural-regression','output':str(f/'attempts'),'timeScale':1,'episodeSeconds':120,'attempts':1}
core.write(f/'launch.json',spec);core.write(f/'build.json',b);core.write(f/'course.json',c)
env=dict(os.environ,RALLY_LAB_LAUNCH=str(f/'launch.json'),RALLY_PRESENTATION_REVIEW=str(f),RALLY_CIRCUIT_PHYSICAL_PROBE=str(f/'wheel-contacts.jsonl'),RALLY_FOREST_V3='1',RALLY_REVIEW_NO_SCREENSHOTS='1')
with ExitStack() as stack:
 for name in ('compute.lock','viewer.lock'):
  lock=stack.enter_context((core.DATA/name).open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (f/'stdout.log').open('w') as log:subprocess.run([b['executable'],'-screen-width','1280','-screen-height','720','-screen-fullscreen','0','-logFile',str(f/'player.log')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
rows=[json.loads(l) for l in (f/'wheel-contacts.jsonl').read_text().splitlines()];contacts=[c for r in rows for c in r['contacts'] if c['grounded']]
assert contacts and all(c['looseness']>.1 for c in contacts if c['collider'].startswith('Road')),'Procedural gravel response changed'
records=[json.loads(l) for p in (f/'attempts').glob('episodes-*.jsonl') for l in p.read_text().splitlines()]
core.write(f/'regression.json',{'groundedSamples':len(contacts),'maxSpeed':max(r['speed'] for r in rows),'records':records,'note':'Bounded heuristic diagnostic on preserved procedural gravel course; not learning evidence'})
print(f)
