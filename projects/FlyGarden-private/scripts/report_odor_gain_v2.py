"""Independent raw-event audit and phase summaries for the short localization."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/odor-gain-calibration-v2'
KEYS=('ORN_DM1_left','ORN_DM1_right','ORN_DM2_left','ORN_DM2_right','DNp09_left','DNp09_right','LPLC2_left','LPLC2_right')
PHASES={'pre_cue':range(12),'pulse':range(12,32),'recovery':range(32,60)}


def expected_rates(case,tick):
 rates=np.zeros(8);rates[4:6]=case['support']
 if 12<=tick<32 and case['cue']!='none':
  offset=0 if case['cue'].startswith('a_') else 2
  if 'gradient' in case['cue']:
   rates[offset:offset+2]=[20.25,19.75] if case['cue'].endswith('left') else [19.75,20.25]
  elif case['cue'].endswith('both'):rates[offset:offset+2]=case['gain']
  else:rates[offset+(0 if case['cue'].endswith('left') else 1)]=case['gain']
 return rates


def matrices(selected):
 table=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
 table=table[table.Postsynaptic_Index.isin(selected)];lookup={i:k for k,i in enumerate(selected)}
 row=table.Postsynaptic_Index.map(lookup).to_numpy();col=table.Presynaptic_Index.to_numpy();w=table['Excitatory x Connectivity'].to_numpy()*.275
 shape=(len(selected),138639)
 return [coo_matrix((np.maximum(w,0) if positive else np.minimum(w,0),(row,col)),shape=shape).tocsr() for positive in (True,False)]


def audit(folder,protocol,mat):
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==protocol['sources'];rows=json.loads((folder/'rows.json').read_text());assert len(rows)==60 and len(m['chunks'])==60
 selected=np.array(protocol['selected_probe_indices']);targets=np.array(m['controller']['input_indices']);channels=np.array(m['controller']['input_channels']);ids=np.array([int(n) for n in m['controller']['input_root_ids']]);rng=np.random.default_rng(m['seed']);decoder=DescendingDecoder()
 expected_zero=targets if m['case']['refractory']=='candidate' else targets[np.any([expected_rates(m['case'],k)>0 for k in range(60)],axis=0)[channels]]
 assert np.array_equal(np.sort(expected_zero),m['actual_zero_refractory_indices'])
 tail_i=np.array([],dtype=int);tail_t=np.array([]);positive=[];negative=[];extcounts=[];low=[];high=[];gmean=[];gmin=[];gmax=[]
 for k,(chunk,row) in enumerate(zip(m['chunks'],rows)):
  path=folder/chunk['file'];assert file_sha(path)==chunk['sha256'];rates=expected_rates(m['case'],k)
  assert row['requested_hz']==dict(zip(KEYS,rates.tolist()))
  local,ii=np.nonzero(rng.random((250,len(targets)))<rates[channels]*.0001)
  with np.load(path,allow_pickle=False) as z:
   assert np.array_equal(z['external_i'],ii) and np.array_equal(np.rint(z['external_t']/.0001).astype(int),k*250+local)
   assert np.array_equal(z['delivered_external_indices'],ii) and np.allclose(z['delivered_external_times_seconds'],z['external_t'],rtol=0,atol=1e-12)
   assert np.array_equal(z['counts'],np.bincount(z['spike_i'],minlength=138639))
   assert np.array_equal(z['neuron_indices'],selected) and np.array_equal(z['spike_counts'],z['counts'][selected])
   assert np.array_equal(z['root_ids'],np.asarray(protocol['_ordering'])[selected])
   assert len(z['time_seconds'])==25 and np.allclose(z['time_seconds'],k*.025+np.arange(25)*.001,rtol=0,atol=1e-12)
   assert np.isfinite(z['voltage_mV']).all() and np.isfinite(z['net_synaptic_conductance_equivalent_mV']).all()
   expected={key:float(z['counts'][[n['index'] for n in pop]].mean()/.025) if pop else None for key,pop in protocol['mapping'].items()}
   assert row['population_hz']==expected and np.array_equal(decoder.advance(.025,expected),row['candidate_motor'])
   si=np.concatenate([tail_i,z['spike_i']]);st=np.concatenate([tail_t,z['spike_t']]);arrivals=st+.0018;start=k*.025;end=start+.025;mask=(arrivals>=start-1e-12)&(arrivals<end-1e-12)
   bins=np.floor((arrivals[mask]-start+1e-12)/.001).astype(int);hist=coo_matrix((np.ones(mask.sum()),(si[mask],bins)),shape=(138639,25)).tocsr()
   positive.append(np.asarray((mat[0]@hist).sum(axis=1)).ravel());negative.append(np.asarray((mat[1]@hist).sum(axis=1)).ravel())
   keep=z['spike_t']>=end-.0018-1e-12;tail_i=z['spike_i'][keep].copy();tail_t=z['spike_t'][keep].copy()
   extcounts.append(np.bincount(channels[ii],minlength=8));low.append(z['voltage_mV'].min(axis=1));high.append(z['voltage_mV'].max(axis=1));gmean.append(z['net_synaptic_conductance_equivalent_mV'].mean(axis=1));gmin.append(z['net_synaptic_conductance_equivalent_mV'].min(axis=1));gmax.append(z['net_synaptic_conductance_equivalent_mV'].max(axis=1))
 phases={}
 for phase,indices in PHASES.items():
  ticks=list(indices);duration=len(ticks)*.025
  phases[phase]={'seconds':duration,'population_hz':{key:float(np.mean([rows[k]['population_hz'][key] for k in ticks])) for key,pop in protocol['mapping'].items() if pop},
    'delivered_per_root_hz':(np.asarray(extcounts)[ticks].sum(axis=0)/duration/np.bincount(channels,minlength=8)).tolist(),
    'candidate_motor_mean':np.mean([rows[k]['candidate_motor'] for k in ticks],axis=0).tolist(),
    'selected_voltage_min_mV':np.asarray(low)[ticks].min(axis=0).tolist(),'selected_voltage_max_mV':np.asarray(high)[ticks].max(axis=0).tolist(),
    'selected_g_mean_mV':np.asarray(gmean)[ticks].mean(axis=0).tolist(),
    'selected_positive_nominal_arrival_mV_per_second':(np.asarray(positive)[ticks].sum(axis=0)/duration).tolist(),
    'selected_negative_nominal_arrival_mV_per_second':(np.asarray(negative)[ticks].sum(axis=0)/duration).tolist()}
 return {'case':m['case'],'seed':m['seed'],'manifest_sha256':file_sha(folder/'manifest.json'),'wall_seconds':m['wall_seconds'],'peak_rss_bytes':m['peak_rss_bytes'],'phases':phases,'raw_input_rng_count_decoder_and_probe_audit':True}


def main():
 protocol=json.loads((OUT/'diagnostic-protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in protocol['sources'].items())
 protocol['_ordering']=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(np.int64)
 mat=matrices(protocol['selected_probe_indices']);results=[];pending=[]
 for case in protocol['cases']:
  for seed in protocol['seeds']:
   name=f"{case['cue']}-support{case['support']}-{case['refractory']}-gain{case['gain']:g}-{seed}";folder=OUT/'diagnostics'/name
   if not (folder/'manifest.json').exists() or json.loads((folder/'manifest.json').read_text())['status']!='complete':pending.append(name);continue
   results.append(audit(folder,protocol,mat))
 status='partial' if pending else 'complete'
 atomic_json(OUT/('diagnostic-partial.json' if pending else 'diagnostic-results.json'),{'status':status,'records':results,'pending':pending,'protocol_sha256':file_sha(OUT/'diagnostic-protocol.json'),'auditor_sha256':file_sha(Path(__file__)),'arrival_scope':'Independent signed anatomical g increments at fixed1.8ms delay; not currents or counterfactual causal proof. No navigation/body/learning validation.'})
 print(status,len(results),'audited diagnostics;',len(pending),'pending',flush=True)
 if not pending:
  for r in results:
   p=r['phases']['pulse']['population_hz'];q=r['phases']['recovery']['population_hz'];print(r['case'],r['seed'],'pulse DNa',round(p['DNa02_left'],1),round(p['DNa02_right'],1),'recovery',round(q['DNa02_left'],1),round(q['DNa02_right'],1))

if __name__=='__main__':main()
