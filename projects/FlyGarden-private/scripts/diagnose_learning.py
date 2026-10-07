"""Describe the completed screen without changing its protocol or pass criteria."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
p=root/'reports/learning-evaluation.json';source=json.loads(p.read_text())
assert source['status']=='completed'
conditions={}
for name in source['conditions']:
 rows=[r for r in source['runs'] if r['condition']==name]
 conditions[name]={
  'states':len(rows),
  'mean_before':float(np.mean([r['before'] for r in rows])),
  'mean_acquisition':float(np.mean([r['acquisition'] for r in rows])),
  'mean_reversal':float(np.mean([r['reversal'] for r in rows])),
  'mean_changed_connections':float(np.mean([r['changed_connections'] for r in rows])),
  'changed_states':sum(r['changed_connections']>0 for r in rows),
  'mean_updates':float(np.mean([r['updates'] for r in rows]))}
report={'source':'learning-evaluation.json','protocol_unchanged':True,'conditions':conditions,
 'conclusion':'Plasticity activation and neural cue preference are separate checks. A changed weight does not establish a useful choice.',
 'next_experiment_requirements':['Determine valence/readout direction from independent published evidence, not the observed failed test.','Measure cue-separated KC and MBON response before and after isolated reward pairings.','Use separate diagnostic seeds; retain this completed 20-seed evaluation as failed.','Establish a documented MBON-to-descending action pathway before claiming behavioral learning.'],
 'behavioral_learning_demonstrated':False}
(root/'reports/learning-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
