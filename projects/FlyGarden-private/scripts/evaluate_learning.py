"""Full-network conditioning screen; neural readout is NOT behavioral preference."""
import sys,json,time,resource,os
from pathlib import Path
import numpy as np
import brian2 as b
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.brain import FullBrain,ROOT
from flygarden.trial_recording import NeuralRecording
report=dict(status='running',protocol='Full network acquisition and reversal screen; 20 seeds per condition',conditions=['plasticity','frozen','shuffled'],readout='MBON11 minus MBON01 firing rate; engineered neural valence proxy, not observed behavior',behavioral_learning_demonstrated=False,runs=[])
print('Loading complete connectome for learning screen',flush=True);brain=FullBrain(learning=True);brain.mark_initial();initial=brain.learned_state();started=time.perf_counter()
# Brief windows deliberately screen neural responsiveness before costly embodied tests.
window=.2

def probe(odor):
 enabled=brain.plasticity.enabled;brain.plasticity.enabled=False
 recording.advance(window,odor=(0,0));recording.advance(window,odor=(1,0) if odor==0 else (0,1))
 out=brain.last_rates['mbon_aversive']-brain.last_rates['mbon_reward'];brain.plasticity.enabled=enabled
 return out
for seed in range(20):
 for condition in report['conditions']:
  brain.reset_transient();brain.restore_learned(initial);brain.rng=np.random.default_rng(seed);brain.plasticity.enabled=condition!='frozen';rng=np.random.default_rng(seed)
  recording=NeuralRecording(brain,seed,condition)
  before=probe(0)-probe(1)
  for reversal in (False,True):
   rewarded=1 if reversal else 0
   for repeat in range(3):
    for odor in (0,1):
     recording.advance(window,odor=(1,0) if odor==0 else (0,1))
     reward=(rng.integers(0,2)==odor) if condition=='shuffled' else odor==rewarded
     if reward:recording.reinforce(1.)
     recording.advance(window,odor=(0,0),reinforcement=1. if reward else 0.)
   pref=probe(0)-probe(1)
   if not reversal:acquisition=pref
   else:reverse=pref
  changed=int(np.sum(np.abs(brain.plasticity.weights-initial['weights'])>1e-12))
  run=dict(seed=seed,condition=condition,before=before,acquisition=acquisition,reversal=reverse,changed_connections=changed,updates=brain.plasticity.updates)
  checkpoint_dir=os.environ.get('FLYGARDEN_EXPERIMENT_DIR')
  if checkpoint_dir:np.savez_compressed(Path(checkpoint_dir)/f'seed-{seed}-{condition}.npz',weights=brain.plasticity.weights,indices=brain.plastic_indices,updates=brain.plasticity.updates,seed=seed,condition=condition)
  run['recording']=recording.recorder.id;recording.finish(run)
  report['runs'].append(run);report['wall_seconds']=time.perf_counter()-started
  (ROOT/'reports/learning-evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(run),flush=True)

def bootstrap(v):
 rng=np.random.default_rng(19017);v=np.asarray(v,float);res=rng.choice(v,(5000,len(v)),replace=True).mean(axis=1);return [float(x) for x in np.quantile(res,[.025,.975])]
by={(x['seed'],x['condition']):x for x in report['runs']}
diff=[(by[s,'plasticity']['acquisition']-by[s,'plasticity']['before'])-(by[s,'frozen']['acquisition']-by[s,'frozen']['before']) for s in range(20)]
rev=[by[s,'plasticity']['acquisition']-by[s,'plasticity']['reversal'] for s in range(20)]
report.update(status='completed',neural_acquisition_difference=float(np.mean(diff)),neural_acquisition_ci95=bootstrap(diff),neural_reversal_shift=float(np.mean(rev)),neural_reversal_ci95=bootstrap(rev),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,simulated_window_seconds=window,plastic_edges=len(brain.plastic_indices),mapping_revision=1)
report['neural_screen_passed']=report['neural_acquisition_ci95'][0]>0 and report['neural_reversal_ci95'][0]>0
report['next_gate']='Embodied choice assay required; neural screen does not establish behavioral learning' if report['neural_screen_passed'] else 'Neural response/plasticity-to-readout coupling needs investigation before embodied learning claims'
(ROOT/'reports/learning-evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)
