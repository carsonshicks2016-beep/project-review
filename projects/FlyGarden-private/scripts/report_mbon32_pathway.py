"""Independent raw event/count/decoder audit of downstream capability."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/mbon32-capability-v1'
def main():
 s=json.loads((OUT/'protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in s['sources'].items());records=[]
 for cue in s['cases']:
  for seed in s['seeds']:
   folder=OUT/f'{cue}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and len(m['chunks'])==60;rows=json.loads((folder/'rows.json').read_text());assert len(rows)==60
   selected=np.array(s['selected']);channels=np.array(m['controller']['input_channels']);nr=len(channels);rng=np.random.default_rng(seed);mrng=np.random.default_rng(seed+100000);decoder=DescendingDecoder();phase_counts={k:np.zeros(138639,dtype=np.int64) for k in ('pre','pulse','recovery')}
   for k,chunk in enumerate(m['chunks']):
    path=folder/chunk['file'];assert file_sha(path)==chunk['sha256'];rates=np.zeros(8);rates[4:6]=65;ticks,ii=np.nonzero(rng.random((250,nr))<rates[channels]*.0001);mb=np.zeros(2)
    if 12<=k<32:
     if cue in ('left','both'):mb[0]=50
     if cue in ('right','both'):mb[1]=50
    mticks,mi=np.nonzero(mrng.random((250,2))<mb*.0001)
    with np.load(path,allow_pickle=False) as z:
     assert np.array_equal(z['support_i'],ii) and np.array_equal(np.rint(z['support_t']/.0001).astype(int),k*250+ticks)
     assert np.array_equal(z['mbon_i'],mi) and np.array_equal(np.rint(z['mbon_t']/.0001).astype(int),k*250+mticks)
     assert np.array_equal(z['delivered_external_indices'],mi);assert np.allclose(z['delivered_external_times_seconds'],z['mbon_t'],atol=1e-12,rtol=0)
     assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts']);assert np.array_equal(z['counts'][selected],z['spike_counts']);assert np.array_equal(z['neuron_indices'],selected)
     assert np.isfinite(z['voltage_mV']).all() and np.isfinite(z['net_synaptic_conductance_equivalent_mV']).all()
     expected={key:float(z['counts'][[n['index'] for n in pop]].mean()/.025) if pop else None for key,pop in s['mapping'].items()};assert expected==rows[k]['population_hz'];assert np.array_equal(decoder.advance(.025,expected),rows[k]['motor'])
     phase_counts['pre' if k<12 else 'pulse' if k<32 else 'recovery']+=z['counts']
   phase={key:{name:float(count[[n['index'] for n in pop]].mean()/({'pre':.3,'pulse':.5,'recovery':.7}[key])) for name,pop in s['mapping'].items() if pop} for key,count in phase_counts.items()};records.append({'case':cue,'seed':seed,'phases':phase,'manifest_sha256':file_sha(folder/'manifest.json')})
 gates=[]
 for seed in s['seeds']:
  group={r['case']:r for r in records if r['seed']==seed}
  def difference(cue,phase):
   p=group[cue]['phases'][phase];return p['DNa02_left']-p['DNa02_right']
  pulse={cue:difference(cue,'pulse')-difference('none','pulse') for cue in ('left','right','both')};tail={cue:difference(cue,'recovery')-difference('none','recovery') for cue in ('left','right','both')}
  g={'left':pulse['left']>=5,'right':pulse['right']<=-5,'bilateral':abs(pulse['both'])<=5,'recovery':all(abs(x)<=5 for x in tail.values())};gates.append({'seed':seed,'pulse_difference_hz':pulse,'recovery_difference_hz':tail,'gates':g,'passed':all(g.values())})
 result={'status':'audits_passed','capability_gate_passed':all(g['passed'] for g in gates),'records':records,'gates':gates,'navigation_validated':False,'production_controller_changed':False,'protocol_sha256':file_sha(OUT/'protocol.json'),'auditor_sha256':file_sha(Path(__file__))};atomic_json(OUT/'results.json',result);print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
