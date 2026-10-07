"""GPL-3.0-or-later. Equation check adapted from Bennett et al. mb_vs.m.
This is a separate reduced mathematical diagnostic, never an arena controller.
It is not a numerical reproduction of MATLAB RNG or the published experiments.
"""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];vendor=root/'vendor/mushroom-body-rpe'
source=vendor/'mb_vs.m';rng=np.random.default_rng(101);runs=[]
# VSlambda default: gamma=1, lambda=11.5, two disjoint 10-KC cues.
# Only schedules/initial RNG are diagnostic choices; dynamics follow source.
s=np.zeros((20,2));s[:10,0]=1;s[10:,1]=1
for seed in range(101,121):
 rg=np.random.default_rng(seed);initial=rg.uniform(0,.1,(2,20))
 for frozen in (False,True):
  weights=initial.copy()
  def preference():return (weights[0]-weights[1])@s
  before=preference().tolist()
  for phase in (0,1):
   for repeat in range(100):
    for cue in (0,1):
     k=s[:,cue];m=weights@k;reward=float(cue==phase)
     d_plus=max(0.,k.sum()+m[1]+reward)
     d_minus=max(0.,k.sum()+m[0])
     if not frozen:
      weights[0]=np.maximum(0,weights[0]+.01*k*(11.5-d_minus))
      weights[1]=np.maximum(0,weights[1]+.01*k*(11.5-d_plus))
   if phase==0:acquisition=preference().tolist()
   else:reversal=preference().tolist()
  runs.append(dict(seed=seed,frozen=frozen,before=before,acquisition=acquisition,reversal=reversal))
learned=[r for r in runs if not r['frozen']];frozen=[r for r in runs if r['frozen']]
checks={'acquisition_correct_direction':all(r['acquisition'][0]>r['acquisition'][1] for r in learned),'reversal_correct_direction':all(r['reversal'][1]>r['reversal'][0] for r in learned),'frozen_unchanged':all(r['before']==r['acquisition']==r['reversal'] for r in frozen)}
report={'status':'completed','passed_equation_check':all(checks.values()),'checks':checks,'runs':runs,'source_revision':subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_license':'GPL-3.0','source':'https://github.com/BrainsOnBoard/paper_RPEs_in_drosophila_mb/blob/7ec52afb9bd7bb748d94d60dea9f483645a2ce8e/mb_vs.m','scope':'Separate reduced equation diagnostic only; no substitution or integration into the full brain; no biological validation','schedule':'100 alternating forced presentations per phase; cue 0 reward then cue 1; learning rate .01','full_brain_learning_demonstrated':False}
(root/'reports/published-rule-check.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='runs'});assert report['passed_equation_check']
