"""Isolated engineered voltage-equivalent drive; no physiological fitting."""
import json,time
from pathlib import Path
import numpy as np
import brian2 as b
from scipy.linalg import expm
from flygarden.adaptive_neurons import adaptive_neurons
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/brain-integration/recovery/adaptive-cell-reference-v1'

def main():
 start=time.monotonic();sources=[ROOT/'flygarden/adaptive_neurons.py',ROOT/'tests/test_adaptive_neurons.py',Path(__file__).resolve()]
 atomic_json(OUT/'protocol.json',{'version':1,'sources':{str(p.relative_to(ROOT)):file_sha(p) for p in sources},'tau_s':.150,'step_mv':1.5,'drive_levels_mV':[15,25,40,70],'schedule':'Ideal g clamp .2-1.2s and2.3-3.3s; explicit voltage-equivalent drive, not measured current','gates':{'onset_spikes_at25mV':1,'renewed_response_fraction':.5,'late_rate_below_onset_at25mV':True,'subthreshold_analytic_error_mV':1e-10},'full_brain_acceptance':False})
 b.start_scope();b.prefs.codegen.target='numpy';clock=b.Clock(dt=.1*b.ms);n=adaptive_neurons(8,clock,mask=np.array([False]*4+[True]*4));levels=np.array([15,25,40,70]*2)
 @b.network_operation(clock=clock,when='start')
 def drive():
  t=float(clock.t/b.second);on=(.2<=t<1.2) or (2.3<=t<3.3);n.g=levels*on*b.mV
 spikes=b.SpikeMonitor(n);state=b.StateMonitor(n,['v','adapt'],record=True,dt=1*b.ms);net=b.Network(n,drive,spikes,state);net.run(4*b.second,namespace={})
 tt=np.asarray(spikes.t/b.second);ii=np.asarray(spikes.i);rows=[]
 for index in range(8):
  times=tt[ii==index];rows.append({'adaptation':index>=4,'drive_mV':int(levels[index]),'onset_1_hz':int(((times>=.2)&(times<.3)).sum())/.1,'late_1_hz':int(((times>=.9)&(times<1.2)).sum())/.3,'onset_2_hz':int(((times>=2.3)&(times<2.4)).sum())/.1,'off_spikes':int((((times>=1.2)&(times<2.3))|(times>=3.3)).sum())})
 # Exact linear subthreshold solution: [v-vrest,g,adapt] in mV.
 errors=[]
 for dt in [.0001,.00005,.000025]:
  b.start_scope();clock=b.Clock(dt=dt*b.second);cell=adaptive_neurons(1,clock);cell.g=3*b.mV;cell.adapt=2*b.mV;network=b.Network(cell);network.run(.01*b.second,namespace={})
  matrix=np.array([[-1/.020,1/.020,-1/.020],[0,-1/.005,0],[0,0,-1/.150]])
  expected=expm(matrix*.01)@np.array([0,3,2]);actual=np.array([float(cell.v[0]/b.mV)+52,float(cell.g[0]/b.mV),float(cell.adapt[0]/b.mV)]);errors.append(float(np.max(abs(actual-expected))))
 active=next(x for x in rows if x['adaptation'] and x['drive_mV']==25)
 checks={'response':active['onset_1_hz']>=10,'sustained_rate_adapts':active['late_1_hz']<active['onset_1_hz'],'renewed_response':active['onset_2_hz']>=.5*active['onset_1_hz'],'off_quiet':all(x['off_spikes']==0 for x in rows),'subthreshold_precision':max(errors)<=1e-10}
 atomic_json(OUT/'results.json',{'status':'isolated_reference_complete','checks':checks,'passed':all(checks.values()),'rates':rows,'analytic_errors_mV':errors,'wall_seconds':time.monotonic()-start,'source_hashes_verified':all(file_sha(ROOT/k)==v for k,v in json.loads((OUT/'protocol.json').read_text())['sources'].items()),'full_network_validated':False})
 np.savez_compressed(OUT/'traces.npz',spike_i=ii,spike_t=tt,t=np.asarray(state.t/b.second),v=np.asarray(state.v/b.mV),adapt=np.asarray(state.adapt/b.mV))
 print(json.dumps({'checks':checks,'rates':rows,'errors':errors},indent=2))
if __name__=='__main__':main()
