"""Verify exact joins, physical odor reconstruction and preserved reference scope."""
import sys,json,math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
from flygarden.world import World,default_arena
from flygarden.timed_odor_world import TimedOdorWorld
from flygarden.candidate_inputs import exact_input_order,CandidateInputs
OUT=ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis'
SOURCE=ROOT/'reports/brain-integration/recovery/odor-causal-v2'
protocol=json.loads((SOURCE/'protocol.json').read_text())
ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(np.int64)
ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');ann=ann[ann.root_id.isin(ids)]
assert not ann.root_id.duplicated().any()
first=json.loads((SOURCE/'trials/9601/intact/manifest.json').read_text());mapping=first['controller']['population_mapping'];joins={}
for key,pop in mapping.items():
 cell,side=key.rsplit('_',1);expected=ann[ann.cell_type.eq(cell)&ann.side.eq(side)]
 assert {r['root_id'] for r in pop}=={str(int(r)) for r in expected.root_id}
 assert all(str(ids[r['index']])==r['root_id'] for r in pop)
 joins[key]={'count':len(pop),'root_ids':[r['root_id'] for r in pop],'exact_annotation_set':True}
targets,channels=exact_input_order(ids,mapping)
assert targets.tolist()==first['controller']['input_indices'] and channels.tolist()==first['controller']['input_channels']
assert len(set(targets))==len(targets)
checks=[]
for seed in protocol['seeds']:
 for arm in protocol['neural_arms']:
  folder=SOURCE/'trials'/str(seed)/arm;m=json.loads((folder/'manifest.json').read_text());rows=json.loads((folder/'rows.json').read_text())
  assert m['controller']['input_root_ids']==[str(ids[i]) for i in targets]
  world=TimedOdorWorld(seed=seed,arena=default_arena());world.configure(m['cue']);on=False;off=False
  for i,row in enumerate(rows):
   # Reconstruct field from recorded previous physical boundary; not ending pose.
   body=m['initial'] if i==0 else rows[i-1]['body']
   events=[e for prior in rows[:i] for block in prior['events'] for e in block['events']]
   on=on or any(e['kind']=='odor_onset' for e in events)
   off=off or any(e['kind']=='odor_offset' for e in events)
   world.arena['foods'][0]['units']=3 if on and not off else 0
   expected=[world.concentrations(p).tolist() for p in body['antennae']]
   assert np.allclose(expected,row['sensory']['antenna_odors'],rtol=0,atol=1e-12)
   assert CandidateInputs(65).rates(expected,[0,0],arm!='sensory_off')==row['requested_hz']
  initial=m['initial'];h=initial['heading'];left,right=np.array(initial['antennae']);relative=np.array([-math.sin(h),math.cos(h),0])
  assert np.dot(left-right,relative)>0
  checks.append({'seed':seed,'arm':arm,'windows':len(rows),'previous_physical_pose_sampling_exact':True,'source_on_off_from_recorded_events':True})
body=json.loads((ROOT/'reports/brain-integration/recovery/body-operating-range/results.json').read_text())
turns=[r for r in body['trials'] if r['case'] in ('direct-left','direct-right','decoder-left','decoder-right')]
assert len(turns)==8 and all(r['heading_change_rad']>0 if r['case'].endswith('left') else r['heading_change_rad']<0 for r in turns)
# Read-only original-source checks: keep native Poisson execution distinct from replay equivalence.
ref=ROOT/'reports/brain-integration/stage1-20261005-214341';comparison=json.loads((ref/'reference-comparison.json').read_text());assert comparison['equation_replay_passed']
native=json.loads((ref/'native-reference.json').read_text())
for r in native['results']:
 assert file_sha(ref/r['file'])==r['sha256']
 with np.load(ref/r['file'],allow_pickle=False) as z:
  assert len(z['i'])==r['spikes'] and np.isfinite(z['t']).all() and len(np.unique(z['i']))==r['active_neurons']
  assert np.all(z['t']>=0) and np.all(z['t']<r['duration_seconds'])
for name in ('production','reference_adapter','reference_selected'):
 for r in json.loads((ref/(name+'.json')).read_text()):assert file_sha(ref/r['file'])==r['sha256']
reference_rechecks=[]
for seed in (1101,1102):
 for case in ('quiet','odor_a','walking','loom_left','combined'):
  with np.load(ref/f'production-{case}-{seed}.npz',allow_pickle=False) as a,np.load(ref/f'reference_adapter-{case}-{seed}.npz',allow_pickle=False) as b:
   assert set(a.files)==set(b.files)
   for key in ('i','t','counts','input_channels','input_ticks','input_indices'):assert np.array_equal(a[key],b[key]),(seed,case,key)
   for key in ('v','g'):assert np.allclose(a[key],b[key],rtol=0,atol=1e-10),(seed,case,key)
   reference_rechecks.append({'seed':seed,'case':case,'spikes_counts_inputs_exact':True,'v_max_difference_mV':float(np.max(np.abs(a['v']-b['v']))),'g_max_difference_mV':float(np.max(np.abs(a['g']-b['g']))),'original_registered_state_tolerance_mV':1e-10})
atomic_json(OUT/'conventions.json',{'status':'passed','auditor_sha256':file_sha(Path(__file__)),'annotation_joins':joins,'trials':checks,
 'physical_turn_direction_verified_from_eight_preserved_native_runs':True,
 'reference':{'equation_replay_10_cases_passed':True,'independent_raw_rechecks':reference_rechecks,'two_native_poisson_execution_artifacts_verified':True,
 'refractory_difference_detected_in_odor_cases':True,'paper_behavioral_prediction_reproduction':False,
 'limits':'200ms reference execution and shared-input equation parity do not reproduce published biological prediction accuracy or multi-second embodied behavior'},
 'scope':'Mapping/field/decoder convention audit; no brain repair or behavioral promotion'})
print('60 trial physical odor/annotation joins and preserved reference artifacts verified')
