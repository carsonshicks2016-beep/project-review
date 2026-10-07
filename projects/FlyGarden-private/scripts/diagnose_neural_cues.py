"""Independent full-network diagnostic; no retuning and no behavioral claims."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.brain import FullBrain
report={'status':'running','seeds':[101,102,103],'window_seconds':.2,'brain':'Full imported network','protocol':'Rested cue probes; three cue-A food pairings; rested post probes with plasticity frozen and matched input RNG','runs':[],'behavioral_learning_demonstrated':False}
brain=FullBrain(learning=True);brain.mark_initial();initial=brain.learned_state();start=time.perf_counter()
for seed in report['seeds']:
 brain.restore_learned(initial);brain.reset_transient(seed)
 def probe(cue):
  brain.reset_transient(seed=seed+1000);brain.plasticity.enabled=False
  brain.advance(.2,odor=cue)
  return {'rates':brain.last_rates.copy(),'active_kcg':brain.populations['kcg'][brain.last_delta[brain.populations['kcg']]>0].tolist()}
 pre=[probe((1,0)),probe((0,1)),probe((0,0))]
 brain.restore_learned(initial);brain.reset_transient(seed=seed);brain.plasticity.enabled=True
 pair=[]
 for repeat in range(3):
  brain.advance(.2,odor=(1,0));brain.reinforce(1.)
  brain.advance(.2,odor=(0,0),reinforcement=1.)
  pair.append(brain.last_rates.copy())
 learned=brain.learned_state();post=[probe((1,0)),probe((0,1)),probe((0,0))]
 a,b=set(pre[0]['active_kcg']),set(pre[1]['active_kcg']);union=a|b
 run={'seed':seed,'before':pre,'after':post,'reward_windows':pair,'changed_weights':int(np.count_nonzero(learned['weights']!=initial['weights'])),'relative_weight_mean':float(np.mean(np.abs(learned['weights'])/np.abs(initial['weights']))),'cue_kcg_jaccard':len(a&b)/len(union) if union else None}
 report['runs'].append(run);report['wall_seconds']=time.perf_counter()-start
 (root/'reports/neural-cue-diagnostic.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in run.items() if k not in ('before','after','reward_windows')},flush=True)
report['status']='completed';report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();report['scope']='Rested population response diagnostic, not choice or reproduction of published learning rule';(root/'reports/neural-cue-diagnostic.json').write_text(json.dumps(report,indent=2));print('Completed',flush=True)
