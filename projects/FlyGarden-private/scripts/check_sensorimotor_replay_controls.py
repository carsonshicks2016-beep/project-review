"""Supplement completed body replays with zero-output and readout-disable controls."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body import Body
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json
from scripts.audit_brain_reference import sha
root=Path(sys.argv[1]);out=root/'body-replay';summary=json.loads((out/'summary.json').read_text())
if any(r['cue']=='steering_readout_disabled' for r in summary['results']):raise RuntimeError('Controls already completed')
atomic_json(out/'summary-before-controls.json',summary)
for label,source in [('quiet','quiet'),('steering_readout_disabled','walking_a_left')]:
 rows=json.loads((root/'trials'/f'{source}-3199'/'bins.json').read_text());decoder=DescendingDecoder();body=Body(seed=3199);initial=body.observation();frames=[];trace=[];prev=np.array(initial['position']);distance=0.
 def capture(t,pose):frames.append({'time':t,**pose})
 try:
  for row in rows:
   rates=dict(row['population_hz'])
   if label=='steering_readout_disabled':rates['DNa02_left']=rates['DNa02_right']=0.
   motor=decoder.advance(.1,rates);obs=body.advance(.1,motor,capture=capture);pos=np.array(obs['position']);distance+=np.linalg.norm(pos[:2]-prev[:2]);prev=pos;trace.append({'time':row['time'],'motor':motor.tolist(),'body':obs})
 finally:body.close()
 head=np.unwrap([initial['heading']]+[r['body']['heading'] for r in trace]);result={'cue':label,'source_cue':source,'seed':3199,'initial':initial,'final':trace[-1]['body'],'heading_change_radians':float(head[-1]-head[0]),'travel_distance':float(distance),'flipped_samples':sum(r['body']['flipped'] for r in trace),'finite':all(np.isfinite(r['body']['position']).all() for r in trace),'frames':len(frames),'scope':'Decoder steering readout disabled; recorded neural activity itself is not lesioned.' if label=='steering_readout_disabled' else 'Actual all-zero measured neural output.'}
 atomic_json(out/f'{label}.json',{'result':result,'trace':trace,'frames':frames});summary['results'].append(result);print(label,result['heading_change_radians'])
summary['supplement_source_sha256']=sha(Path(__file__));atomic_json(out/'summary.json',summary)
