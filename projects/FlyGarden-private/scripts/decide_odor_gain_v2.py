"""Apply prospectively written calibration gates; no post-hoc fitting."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/odor-gain-calibration-v2'
def decide():
 data=json.loads((OUT/'diagnostic-results.json').read_text());assert data['status']=='complete' and len(data['records'])==24
 def signed(r,phase):
  p=r['phases'][phase]['population_hz'];return p['DNa02_left']-p['DNa02_right']
 rows=[];passing=[]
 for gain in (1.,5.,10.):
  success=True
  for seed in (9901,9902):
   group={r['case']['cue']:r for r in data['records'] if r['seed']==seed and r['case']['gain']==gain};assert len(group)==4
   p={cue:signed(r,'pulse')-signed(group['none'],'pulse') for cue,r in group.items() if cue!='none'}
   q={cue:signed(r,'recovery')-signed(group['none'],'recovery') for cue,r in group.items() if cue!='none'}
   gates={'left':p['a_left']>=5,'right':p['a_right']<=-5,'bilateral':abs(p['a_both'])<=5,'recovery':all(abs(v)<=5 for v in q.values())}
   passed=all(gates.values());success &= passed
   rows.append({'gain':gain,'seed':seed,'pulse_difference_from_control_hz':p,'recovery_difference_from_control_hz':q,'gates':gates,'passed':passed})
  if success:passing.append(gain)
 result={'status':'passed' if passing else 'failed','selected_gain':min(passing) if passing else None,'records':rows,'evidence_sha256':file_sha(OUT/'diagnostic-results.json'),'decision_sha256':file_sha(OUT/'DECISION.md'),'navigation_validated':False,'learning_validated':False,'held_out_evaluation_status':'requires prospective embodied protocol' if passing else 'not_started_calibration_gate_failed','production_controller_changed':False}
 atomic_json(OUT/'calibration-decision.json',result)
 lines=['# Steps 8–11: lower-gain calibration result','','24 registered full-network trials completed; raw spike/input/clock/count/decoder/probe audits passed. Fixed full network, support and decoder; no learning or application controller change.','','| Gain Hz | Seed | Left change Hz | Right change Hz | Bilateral change Hz | Recovery gate | Overall |','|---:|---:|---:|---:|---:|---|---|']
 for r in rows:
  p=r['pulse_difference_from_control_hz'];lines.append(f"| {r['gain']:g} | {r['seed']} | {p['a_left']:.1f} | {p['a_right']:.1f} | {p['a_both']:.1f} | {r['gates']['recovery']} | {r['passed']} |")
 lines+=['','Values are signed DNa02 left-minus-right rates relative to matched no-odor controls. Baseline subtraction is evaluation only; the controller remains unchanged. This two-seed engineered capability screen is not held-out navigation evidence. All four gates must pass in both seeds.','',('A candidate passes and requires a separately registered physical evaluation before any promotion.' if passing else 'No tested gain passes. Step 10 calibration failed; no candidate is frozen/promoted. Step 11 held-out navigation is not started because its prerequisite failed. Missing activity is not successful direction, and the prior held-out failure remains intact.'),'','Next: investigate the model’s odor-to-steering pathway and operating-state assumptions using an independently supported mapping or physiology revision. Do not fit cue-to-turn weights, remove inhibition or relabel a supplied navigator as neural control. The amplitude-only hypothesis is exhausted within this registered grid.']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':decide()
