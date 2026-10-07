"""Repeat headless 10x circuit starts with bounded fixed-input probes; unranked."""
import os,sys,json,subprocess,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core
import argparse
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--course',required=True);args=p.parse_args();label=args.label;course=args.course;b=core.build_status();assert b['status']=='current';c=next(x for x in core.courses() if x['id']==course)
out=core.DATA/'circuit-review'/label;out.mkdir(exist_ok=False);reports=[]
with (core.DATA/'compute.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for station,steer,drive in [(750,.8,.6),(4250,-.8,.6),(11250,.8,-.6),(18300,-.8,.6),(18300,0,0)]:
  folder=out/f'{station}-{steer}-{drive}';folder.mkdir();spec={'schema':1,'mode':'specialist','evaluation':True,'viewer':False,'controlProbe':True,'controlMode':'fixed-input','controlSteer':steer,'controlDrive':drive,'controlProbeSeconds':3,'startingGear':'neutral','timeScale':10,'seed':20261005,'attempts':20,'episodeSeconds':1800,'courseBundle':c['bundle'],'courseId':course,'courseName':c['name']+' / unranked spawn stress','runId':label,'output':str(folder/'attempts')}
  core.write(folder/'launch.json',spec);env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'launch.json'),RALLY_CIRCUIT_REVIEW_STATION=str(station),RALLY_OBS_DUMP=str(folder/'observations.jsonl'))
  with (folder/'stdout.log').open('w') as log:subprocess.run([b['executable'],'-batchmode','-nographics','-logFile',str(folder/'player.log')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
  records=[json.loads(l) for p in (folder/'attempts').glob('episodes*.jsonl') for l in p.read_text().splitlines()];r={'station':station,'steer':steer,'drive':drive,'attempts':len(records),'outcomes':{v:sum(x['outcome']==v for x in records) for v in set(x['outcome'] for x in records)},'minimumTerminalWheels':min(x['wheels'] for x in records),'minimumUpright':min(x['upright'] for x in records),'records':records};reports.append(r);print({k:v for k,v in r.items() if k!='records'},flush=True)
core.write(out/'report.json',{'build':b,'course':course,'cases':reports,'note':'20 repeated headless accelerated starts per case; fixed-input, 3s cap, deliberately unranked'} )
