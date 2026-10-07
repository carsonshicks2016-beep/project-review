"""Pin exact ascending mapping and collect measured-contact calibration traces."""
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_sensorimotor_pathway import mapping
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json,space_check
from flygarden.synchronized import SynchronizedBody
from flygarden.body_feedback import BodyFeedback
OUT=ROOT/'reports/brain-integration/stage7-20261006'
SEEDS=(7101,7102,7199)
def prepare():
 ids,pops,ann=mapping();order={int(v):i for i,v in enumerate(ids)};ascending=ann[ann.super_class.isin(['ascending','sensory_ascending'])]
 for cell in ('AN_multi_63','MDN'):
  for side in ('left','right'):
   r=ann[ann.cell_type.eq(cell)&ann.side.eq(side)];assert len(r)==(1 if cell=='AN_multi_63' else 2)
   pops[cell+'_'+side]=[{'root_id':str(int(n.root_id)),'index':order[int(n.root_id)],'cell_type':cell,'side':side,'nerve':n.nerve} for n in r.itertuples()]
 retrieval=json.loads((OUT/'retrieval-manifest.json').read_text())
 for a in retrieval['sources']:assert sha(OUT/'sources'/a['file'])==a['sha256']
 assert 'AN_multi_63' in (OUT/'sources/twolumps-alias.html').read_text() and 'TwoLumps' in (OUT/'sources/twolumps-alias.html').read_text()
 data={'schema_version':1,'modeled_ascending_and_sensory_ascending':len(ascending),'unnamed_ascending_cells':int(ascending.cell_type.eq('').sum()),'mapping':pops,'supported_candidate':'AN_multi_63, cross-dataset synonym AN17A026 / TwoLumps Ascending','confidence':'Exact local root IDs; institution-maintained cross-dataset identity; force-to-rate and receptive fields NOT physiologically validated.','sources':retrieval['sources'],'enabled_signal':'Pooled opposing horizontal foreleg contact force, engineered 1.2–6 native-unit linear ramp to0–100Hz, both TLA roots equally driven.','blocked_signals':['Walking velocity: SS29579 physiological line lacks verified mapping to these exact model roots.','Yaw/proprioceptive joint encoding: no validated per-root gain or channel map.'],'gait_feedback':'Existing supplied controller handles retraction/stumbling; it remains separate from brain feedback.'}
 c=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory x Connectivity']);pre=[n['index'] for s in ('left','right') for n in pops['AN_multi_63_'+s]];post=[n['index'] for s in ('left','right') for n in pops['MDN_'+s]];direct=c[c.Presynaptic_Index.isin(pre)&c.Postsynaptic_Index.isin(post)];data['direct_TLA_MDN']=[{'pre_root_id':str(ids[int(r[0])]),'post_root_id':str(ids[int(r[1])]),'anatomical_synapses':int(r[2]),'signed_synapse_count':float(r[3])} for r in direct.itertuples(index=False,name=None)];atomic_json(OUT/'feedback-mapping.json',data);del c
 out=OUT/'contact-calibration';out.mkdir(exist_ok=False);results=[]
 for cue in ('open','wall'):
  for seed in SEEDS:
   body=SynchronizedBody(seed=seed,blocks=[] if cue=='open' else [{'x':4,'y':0,'w':.8,'h':12,'z':2}]);feedback=BodyFeedback();rows=[];start=time.perf_counter()
   try:
    for tick in range(300):
     obs=body.observation();f=feedback.advance(obs,tick*.01,enabled=True);rows.append({'time':tick*.01,'feedback':f,'body':obs,'calibration_motor':[.65,.65]});body.advance(.01,[.65,.65])
   finally:body.close()
   atomic_json(out/f'{cue}-{seed}.json',rows);result={'cue':cue,'seed':seed,'peak_foreleg_force_model_units':max(r['feedback']['pooled_force_model_units'] for r in rows),'peak_tla_hz':max(r['feedback']['tla_hz'][0] for r in rows),'flipped_samples':sum(r['body']['flipped'] for r in rows),'wall_seconds':time.perf_counter()-start};results.append(result);print(result,flush=True)
 atomic_json(out/'manifest.json',{'status':'complete','scope':'Fixed engineered walking drive .65/.65 collects physical forces; this is body/sensor calibration, not brain-controlled movement. No brain is loaded.','results':results,'sources':{s:sha(ROOT/s) for s in ('flygarden/body.py','flygarden/synchronized.py','flygarden/body_feedback.py','scripts/prepare_timing_feedback.py')},'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
if __name__=='__main__':prepare()
