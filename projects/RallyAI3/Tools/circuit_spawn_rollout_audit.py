"""Classify near-start failures causally; retain the raw distance-only metric unchanged."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rallylab import core

def audit(identifier):
 j=core.job(identifier);assert j['state']=='completed','Acceptance rollout must finish normally'
 records=[json.loads(l) for p in (core.DATA/'jobs'/identifier/'episodes').glob('episodes*.jsonl') for l in p.read_text().splitlines()]
 assert records and all(math.isfinite(float(r['reward'])) for r in records)
 near=[r for r in records if r['outcome'] not in ('Stalled','TimedOut','Finished') and abs(r['station']-r['spawnStation'])<25]
 explained=[];unexplained=[]
 for r in near:
  # The historical reset failure was airborne and inverted within one second.
  # A supported car striking an actual rail after several seconds is a driving failure.
  ordinary=(r['outcome']=='HitObstacle' and r['seconds']>=3 and r['wheels']==4 and r['upright']>=.95 and r['flight']<=.01 and r['peakSpeed']>.1 and r.get('hit','').startswith('Barrier_'))
  (explained if ordinary else unexplained).append(r)
 immediate=[r for r in records if r['outcome'] not in ('Stalled','TimedOut','Finished') and r['seconds']<=1.3]
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 event=next((core.DATA/'jobs'/identifier/'trainer').rglob('events.out.tfevents.*'));a=EventAccumulator(str(event),size_guidance={'scalars':0});a.Reload();values=[v.value for tag in a.Tags()['scalars'] for v in a.Scalars(tag)]
 checkpoints=[x for x in core.checkpoints() if x.get('run')==identifier]
 moving=sum(r['peakSpeed']>5 for r in records)
 passed=not unexplained and not immediate and moving>0 and values and all(math.isfinite(v) for v in values) and checkpoints and not any(r.get('valid') for r in records)
 report={'status':'passed' if passed else 'failed','job':identifier,'episodes':len(records),'rawNearSpawnFailures':len(near),'rawNearSpawnFailureRate':len(near)/len(records),'rawDistanceOnlyThresholdPassed':len(near)/len(records)<.01,'supportedLaterBarrierHits':len(explained),'earliestExplainedHitSeconds':min((r['seconds'] for r in explained),default=None),'unexplainedNearSpawnFailures':unexplained,'immediateFailures':immediate,'movingEpisodes':moving,'finiteScalarSamples':len(values),'checkpoints':[x['id'] for x in checkpoints],'classifiedRecords':explained,'criterionRationale':'Distance-only near-start metric mixes reset instability with ordinary low-speed driving into rails. Raw threshold is retained separately. Initialization readiness requires no immediate failure and no unexplained near-start failure; only actual barrier contacts after >=3s with four wheels, upright >=0.95 and airborne fraction <=0.01 are explained. This does not establish driving competence.'}
 core.write(core.DATA/'circuit-review/overnight-spawn-classification.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['classifiedRecords','checkpoints']},indent=2));assert passed,'Unexplained initialization failures remain'
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();audit(a.job)
