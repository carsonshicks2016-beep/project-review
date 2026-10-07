"""Independent post-run source, input, decoding and weight checks."""
import json,hashlib
import brian2 as b
from pathlib import Path
import numpy as np
import pandas as pd
from flygarden.continuous_candidate import file_sha
from flygarden.candidate_inputs import CandidateInputs,INPUT_KEYS,exact_input_order
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1'

def main():
 p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'progress.json').read_text())['status']=='complete'
 assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
 ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
 graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory x Connectivity'])
 weights=np.asarray(graph['Excitatory x Connectivity'].to_numpy()*.275*b.mV)
 expected_hashes={'original':hashlib.sha256(weights.tobytes()).hexdigest()}
 from flygarden.local_inhibition_bridge import residual_edge_allocation
 routing=json.loads((ROOT/'reports/brain-integration/recovery/local-inhibition-adapter-v1/routes.json').read_text());chosen=set(routing['root_model_indices'].values())
 subset=graph[graph.Presynaptic_Index.isin(chosen)|graph.Postsynaptic_Index.isin(chosen)]
 rr,ww,conservation=residual_edge_allocation(subset.reset_index(drop=True),routing,ids);weights[subset.index.to_numpy()[rr]]=np.asarray(ww*b.mV)
 expected_hashes['local']=hashlib.sha256(weights.tobytes()).hexdigest();del graph,weights,subset
 target,channels=exact_input_order(ids,p['mapping']);profile=CandidateInputs();summaries=[]
 for folder in sorted((OUT/'trials').iterdir()):
  m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';case=m['case'];seed=m['seed'];rng=np.random.default_rng(seed);decoder=DescendingDecoder();rows=json.loads((folder/'rows.json').read_text());ticks=0
  for tick,chunk in enumerate(m['chunks']):
   odor=np.zeros((2,2))
   if case['cue']!='none' and (12<=tick<32 or 132<=tick<152):
    if case['cue'] in ['a_left','a_both']:odor[0,0]=1
    if case['cue'] in ['a_right','a_both']:odor[1,0]=1
   requested=profile.rates(odor,[0,0],True);rates=np.array([requested[k] for k in INPUT_KEYS]);local,indices=np.nonzero(rng.random((250,len(target)))<rates[channels]*.0001)
   file=folder/chunk['file'];assert file_sha(file)==chunk['sha256']
   with np.load(file) as z:
    assert np.array_equal(indices,z['external_i'])
    observed_ticks=np.rint(z['external_t']/.0001).astype(np.int64);assert np.array_equal(tick*250+local,observed_ticks)
    assert np.array_equal(np.bincount(z['spike_i'],minlength=len(ids)),z['counts'])
    pop={key:float(z['counts'][[x['index'] for x in group]].mean()/.025) for key,group in p['mapping'].items()}
    assert pop==rows[tick]['population_hz'];assert np.array_equal(decoder.advance(.025,pop),z['motor']);ticks+=1
  assert m['initial_weight_hash']==m['final_weight_hash']==expected_hashes[case['model']];summaries.append({'trial':folder.name,'chunks':ticks,'inputs_reconstructed':True,'counts_match_spikes':True,'decoder_reconstructed_exactly':True})
 assert len(summaries)==17
 result={'status':'passed','sources_verified':True,'owned_rng_inputs_reconstructed':True,'integer_clock_ticks_exact':True,'decoder_reconstructed_exact':True,'spikes_match_counts':True,'all_trial_weights_unchanged':True,'initial_weights_match_source_and_residual_allocation':True,'count_conservation':conservation,'trials':summaries,'total_chunks':sum(x['chunks'] for x in summaries)}
 atomic_json(OUT/'independent-audit.json',result);print(json.dumps({'status':'passed','total_chunks':result['total_chunks']},indent=2))
if __name__=='__main__':main()
