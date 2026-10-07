"""Bounded acceptance only: train 2,000 steps, export, check finite logs; never promote a course."""
import argparse,json,os,subprocess,sys,fcntl,math,re
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core

def run(course,label):
 b=core.build_status();assert b['status']=='current';c=next(x for x in core.courses() if x['id']==course);assert core.course_asset_valid(c)
 folder=core.DATA/'circuit-review'/label;folder.mkdir(exist_ok=False)
 # Leave room for ML-Agents' final trajectory flush under the 2,000-step ceiling.
 cfg=yaml.safe_load((ROOT/'rallylab/baseline-v1.yaml').read_text());cfg['behaviors']['RallyDriver'].update(max_steps=1800,checkpoint_interval=1000,summary_freq=1000);(folder/'config.yaml').write_text(yaml.safe_dump(cfg))
 seconds=c['definition']['episodeSeconds'];launch={'schema':1,'mode':'specialist','viewer':False,'evaluation':False,'courseBundle':c['bundle'],'courseId':course,'courseName':c['name'],'runId':label,'output':str(folder/'episodes'),'seed':20261005,'timeScale':10,'startingGear':'neutral','spawnProfile':'fixed','episodeSeconds':seconds,'reward':'time-attack-v1'}
 core.write(folder/'launch.json',launch);core.write(folder/'build.json',b);core.write(folder/'course.json',c)
 env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'launch.json'),RALLY_OBS_DUMP=str(folder/'observations.jsonl'),RALLY_PRESENTATION_AUDIT=str(folder/'presentation-audit'))
 with (core.DATA/'compute.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if seconds==240:
   # Exercise the same imported-course MaxStep path with the sector budget.
   # Use full geometry so the full-route gate fixtures retain their proper scope.
   full=next(x for x in core.courses() if x['id']=='f0d8d95ac7d70cc5a9c7ebf056f755f1f18bee3844c7d2cf42d9f5354b90f2b2')
   budget_launch=dict(launch,courseBundle=full['bundle'],courseId=full['id'],evaluation=True,controlProbe=True,controlMode='fixed-input',controlDrive=0,attempts=100,timeScale=1,output=str(folder/'budget-fixture-episodes'))
   core.write(folder/'budget-launch.json',budget_launch)
   subprocess.run([b['executable'],'-batchmode','-nographics','-logFile',str(folder/'budget-player.log')],env=dict(os.environ,RALLY_LAB_LAUNCH=str(folder/'budget-launch.json'),RALLY_CIRCUIT_RUNTIME_ACCEPTANCE=str(folder/'native-budget.json')),stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,timeout=90,check=True)
   budget=core.read(folder/'native-budget.json');assert budget['maxStep']==24000 and all(x['passed'] for x in budget['cases'])
  with (folder/'trainer.log').open('w') as log:
   process=subprocess.run([str(ROOT/'.venv/bin/mlagents-learn'),str(folder/'config.yaml'),'--run-id',label,'--results-dir',str(folder/'results'),'--env',str(core.DATA/'player/RallyTraining.app'),'--num-envs','1','--seed','20261005','--no-graphics','--time-scale','10','--torch-device','cpu'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 assert process.returncode==0,(folder/'trainer.log').read_text()[-2000:]
 models=list((folder/'results').rglob('*.onnx'));assert models,'No exported checkpoint'
 export_steps=[int(m.group(1)) for p in models if (m:=re.search(r'RallyDriver-(\d+)\.onnx$',p.name))];assert export_steps and max(export_steps)<=2000,'Trainer flush exceeded smoke ceiling'
 def finite(v):
  if isinstance(v,float):return math.isfinite(v)
  if isinstance(v,dict):return all(finite(x) for x in v.values())
  if isinstance(v,list):return all(finite(x) for x in v)
  return True
 observations=[json.loads(l) for l in (folder/'observations.jsonl').read_text().splitlines()];records=[json.loads(l) for p in (folder/'episodes').glob('episodes*.jsonl') for l in p.read_text().splitlines()]
 assert observations and records and finite(observations) and finite(records)
 isolation=core.read(folder/'presentation-audit/isolation.json');assert isolation and isolation['headless'] and not isolation['viewer']
 assert not any(isolation[k] for k in ('liveSynths','generatedAudioClips','forestLods','renovation','wheelEffects')),'Headless presentation leaked into training'
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 events=next((folder/'results').rglob('events.out.tfevents.*'));acc=EventAccumulator(str(events),size_guidance={'scalars':0});acc.Reload();values=[e.value for t in acc.Tags()['scalars'] for e in acc.Scalars(t)];assert values and all(math.isfinite(v) for v in values)
 report={'course':course,'build':b['source']['hash'],'configuredTrainerSteps':1800,'actualFinalExportStep':max(export_steps),'trainerStepCeiling':2000,'configuredEpisodeSeconds':seconds,'expectedMaxStep':round(seconds/.01),'workers':1,'observationsChecked':len(observations),'episodes':len(records),'scalarSamples':len(values),'headlessIsolation':isolation,'checkpointExports':[str(p.relative_to(folder)) for p in models],'exitCode':process.returncode,'rankedRecords':sum(bool(r.get('valid')) for r in records),'scope':'bounded trainer/reset/export smoke; does not establish driving competence'};core.write(folder/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--course',required=True);p.add_argument('--label',required=True);a=p.parse_args();run(a.course,a.label)
