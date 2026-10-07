"""Independent raw-event comparison of extended reference and candidate."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/persistence-reference-v1'
def main():
 s=json.loads((OUT/'protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in s['sources'].items());pairs=[]
 for seed in s['seeds']:
  for support in s['supports']:
   folders=[OUT/f'{kind}-{seed}-support{support}' for kind in s['backends']];meta=[json.loads((p/'manifest.json').read_text()) for p in folders];assert all(m['status']=='complete' and len(m['chunks'])==160 for m in meta)
   selected=np.array(s['selected']);target=np.array(meta[0]['input_indices']);channels=np.array(meta[0]['input_channels']);assert meta[0]['input_indices']==meta[1]['input_indices'] and meta[0]['input_channels']==meta[1]['input_channels'];rng=np.random.default_rng(seed);maxv=0.;maxg=0.;phase_counts={n:np.zeros(138639,dtype=np.int64) for n in ('pre','pulse','early_recovery','late_recovery')}
   for k in range(160):
    paths=[folder/m['chunks'][k]['file'] for folder,m in zip(folders,meta)];assert all(file_sha(p)==m['chunks'][k]['sha256'] for p,m in zip(paths,meta))
    rates=np.zeros(8);rates[4:6]=support;rates[0]=5 if 12<=k<32 else 0;ticks,ii=np.nonzero(rng.random((250,len(target)))<rates[channels]*.0001)
    with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
     for key in ('spike_i','spike_t','counts','external_i','external_t','neuron_indices','root_ids','time_seconds','spike_counts','delivered_external_indices'):assert np.array_equal(a[key],b[key]),(seed,support,k,key)
     assert np.allclose(a['delivered_external_times_seconds'],a['external_t'],atol=1e-12,rtol=0) and np.allclose(b['delivered_external_times_seconds'],b['external_t'],atol=1e-12,rtol=0)
     assert np.allclose(a['time_seconds'],k*.025+np.arange(25)*.001,atol=1e-12,rtol=0)
     assert np.array_equal(a['external_i'],ii);assert np.array_equal(np.rint(a['external_t']/.0001).astype(int),k*250+ticks)
     assert np.array_equal(np.bincount(a['spike_i'],minlength=138639),a['counts']);assert np.array_equal(a['counts'][selected],a['spike_counts'])
     for key in ('voltage_mV','net_synaptic_conductance_equivalent_mV'):
      assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all();assert np.allclose(a[key],b[key],rtol=0,atol=1e-10)
     maxv=max(maxv,float(np.max(np.abs(a['voltage_mV']-b['voltage_mV']))));maxg=max(maxg,float(np.max(np.abs(a['net_synaptic_conductance_equivalent_mV']-b['net_synaptic_conductance_equivalent_mV']))))
     if k<12:phase_counts['pre']+=a['counts']
     elif k<32:phase_counts['pulse']+=a['counts']
     elif k<60:phase_counts['early_recovery']+=a['counts']
     elif k>=140:phase_counts['late_recovery']+=a['counts']
   durations={'pre':.3,'pulse':.5,'early_recovery':.7,'late_recovery':.5}
   rates={phase:{name:float(counts[[n['index'] for n in pop]].mean()/durations[phase]) for name,pop in s['mapping'].items() if pop} for phase,counts in phase_counts.items()}
   pairs.append({'seed':seed,'support':support,'windows':160,'exact_spikes_counts_inputs':True,'selected_voltage_max_abs_difference_mV':maxv,'selected_g_max_abs_difference_mV':maxg,'phase_rates_hz':rates,'late_recovery_total_spikes':int(phase_counts['late_recovery'].sum())})
 atomic_json(OUT/'results.json',{'status':'passed','pairs':pairs,'protocol_sha256':file_sha(OUT/'protocol.json'),'auditor_sha256':file_sha(Path(__file__)),'scope':s['reference_scope'],'navigation_validated':False})
 print(json.dumps([{'seed':p['seed'],'support':p['support'],'tail':{k:v for k,v in p['phase_rates_hz']['late_recovery'].items() if k.startswith(('DN','DM','ORN'))}} for p in pairs],indent=2))
if __name__=='__main__':main()
