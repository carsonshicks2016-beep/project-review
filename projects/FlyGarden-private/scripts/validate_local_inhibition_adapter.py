"""Prospective isolated adapter checks, not full-network recovery acceptance."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from flygarden.local_inhibition_adapter import LocalInhibitionAdapter,make_drive,output_drive_mv_per_s
from flygarden.graded_gain_reference import GradedGainReference
from scripts.build_local_inhibition_routes import ROOT,OUT,sha

def run(dt):
 a=LocalInhibitionAdapter(['local'],[[0]],dt_s=dt)
 values=[]
 for tick in range(round(.8/dt)):
  t=tick*dt
  values.append(a.step([.65 if .1<=t<.3 else 0])[0])
 return np.asarray(values),a

def main():
 start=time.monotonic();r=json.loads((OUT/'routes.json').read_text())
 source=[OUT/'protocol.json',OUT/'routes.json',ROOT/'flygarden/local_inhibition_adapter.py',ROOT/'flygarden/graded_gain_reference.py',ROOT/'tests/test_local_inhibition_adapter.py',Path(__file__).resolve()]
 protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},
 'gates':{'isolated_analytic_max_error':1e-5,'offset_release_hz_at_150ms':.01,'checkpoint_max_error':0,'observer_max_error':0,'timestep_errors_strictly_decreasing':True,'reference_local_activity_relative_error':.005,'all_pair_counts_preserved':True},
 'reference_scope':'Only continuous local-state equation replay at rate_scale_hz=1000 to avoid clipping; does not validate default scale=100, upstream feedback, DM1 physiology, or full brain',
 'mapped_scope':'Actual anatomical coupling driven by a synthetic local step, not an odor or behavioral experiment',
 'test_seed':20261007,'full_network_acceptance':False}
 (OUT/'validation-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 errors=[];runs=[]
 for dt in [.0002,.0001,.00005]:
  values,a=run(dt);t=(np.arange(len(values))+1)*dt
  analytic=np.where(t<=.1,0,np.where(t<=.3,65*(1-np.exp(-(t-.1)/.015)),65*(1-np.exp(-.2/.015))*np.exp(-(t-.3)/.015)))
  errors.append(float(np.max(np.abs(values-analytic))));runs.append(values)
 offset=float(runs[1][round(.45/.0001)-1])
 # Replay PN drive from the unchanged published-reference port. RK4 vs
 # source Euler at 0.1ms is an explicit discretization comparison.
 reference=GradedGainReference(dt_ms=.1);local=LocalInhibitionAdapter(['reference_local'],[[0]],rate_scale_hz=1000.)
 adapter_local=[]
 # Replay from fresh states using exact same pre-update forcing.
 reference=GradedGainReference(dt_ms=.1);local=LocalInhibitionAdapter(['reference_local'],[[0]],rate_scale_hz=1000.)
 source_local=[]
 for tick in range(20000):
  pre=reference.v.copy();local.step([pre[0]/1000.])
  reference.step(.5 if 5000<=tick<10000 else 0)
  source_local.append(reference.v[2]);adapter_local.append(local.release_hz()[0])
 source_local=np.array(source_local);adapter_local=np.array(adapter_local)
 relative=float(np.max(np.abs(source_local-adapter_local))/np.max(source_local))
 # Actual 212-node topology: a fixed synthetic stimulus on right DM1 nodes.
 a=LocalInhibitionAdapter(r['nodes'],r['coupling']);drive=np.zeros(len(r['nodes']))
 active=[i for i,key in enumerate(r['nodes']) if key.endswith('@right:DM1')];assert active
 drive[active]=.6;records=[];continued=None;checkpoint=None;observer_error=0.;checkpoint_error=0.
 for tick in range(10000):
  d=drive if 1000<=tick<3000 else np.zeros_like(drive)
  before=a.activity.copy();before_steps=a.steps;view=a.release_hz()
  if tick%100==0:output_drive_mv_per_s(r['routes'],view)
  assert np.array_equal(before,a.activity) and before_steps==a.steps
  if tick==2500:
   checkpoint=json.loads(json.dumps(a.checkpoint()));continued=LocalInhibitionAdapter.restore(checkpoint)
   (OUT/'checkpoint.json').write_text(json.dumps(checkpoint)+'\n')
  row=a.step(d)
  if continued is not None:checkpoint_error=max(checkpoint_error,float(np.max(np.abs(row-continued.step(d)))))
  if tick%10==9:records.append(row.copy())
 rates={row['pre_root_id']:0. for row in r['routes'] if row['pre_node']<0};fixed=make_drive(r['routes'],rates,r['input_site_totals'],100.)
 assert np.array_equal(fixed,np.zeros(len(r['nodes'])))
 data=np.asarray(records);np.savez_compressed(OUT/'traces.npz',mapped_release_hz=data,mapped_time_s=np.arange(1,len(data)+1)*.001,reference_local=source_local,adapter_local=adapter_local)
 checks={'isolated_analytic':errors[1]<1e-5,'offset_recovery':offset<.01,'timestep_convergence':errors[0]>errors[1]>errors[2],
 'published_local_equation_replay':relative<.005,'checkpoint_exact':checkpoint_error==0,'observer_pure':observer_error==0,
 'mapped_state_finite_bounded':bool(np.isfinite(data).all() and data.min()>=0 and data.max()<=100),
 'mapped_offset_recovery':bool(data[449].max()<.01),'all_pair_counts_preserved':r['all_pair_counts_preserved']}
 results={'status':'isolated_adapter_checks_complete','checks':checks,'all_isolated_checks_passed':all(checks.values()),'timestep_dt_s':[.0002,.0001,.00005],
 'analytic_max_release_errors_hz':errors,'isolated_offset_release_hz':offset,'reference_local_relative_error':relative,'checkpoint_max_error':checkpoint_error,'observer_max_error':observer_error,
 'mapped_nodes':len(r['nodes']),'stimulated_right_DM1_nodes':len(active),'mapped_offset_max_hz_at_450ms':float(data[449].max()),'wall_seconds':time.monotonic()-start,
 'sources_reverified':all(sha(ROOT/k)==v for k,v in protocol['source_hashes'].items()),'application_changed':False,'full_network_test_run':False,'full_brain_recovery_demonstrated':False}
 (OUT/'validation-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
