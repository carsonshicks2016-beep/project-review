"""Independent raw-event and timing audit for completed odor trial artifacts."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.candidate_inputs import CandidateInputs,INPUT_KEYS
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/odor-causal-v2'


@lru_cache(maxsize=1)
def graph_reference():
    graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',
                         columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    weights=(graph['Excitatory x Connectivity'].to_numpy()*.275*.001)/.001
    return graph.Presynaptic_Index.to_numpy(dtype=np.int32),graph.Postsynaptic_Index.to_numpy(dtype=np.int32),weights


def audit_weights(folder,m,ids):
    pre,post,weights=graph_reference()
    baseline=__import__('hashlib').sha256(weights.tobytes()).hexdigest()
    assert m['baseline_weight_sha256']==baseline
    if m['arm']=='steering_cut':
        targets=np.array([n['index'] for key in ('DNa02_left','DNa02_right') for n in m['controller']['population_mapping'][key]])
        cut=np.flatnonzero(np.isin(post,targets))
        with np.load(folder/'intervention.npz') as artifact:
            expected={'indices':cut,'pre':pre[cut],'post':post[cut],'original_w_mV':weights[cut],
                      'pre_roots':ids[pre[cut]],'post_roots':ids[post[cut]]}
            for key,value in expected.items():assert np.array_equal(artifact[key],value)
        modified=weights.copy();modified[cut]=0
        assert __import__('hashlib').sha256(modified.tobytes()).hexdigest()==m['expected_weight_sha256']
        assert len(cut)==m['interrupted_records']
    else:assert m['expected_weight_sha256']==baseline and m['interrupted_records']==0


def audit(folder):
    m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
    for name,sha in m['sources'].items():assert file_sha(ROOT/name)==sha
    rows=json.loads((folder/'rows.json').read_text());assert len(rows)==len(m['chunks'])
    ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
    assert __import__('hashlib').sha256(ids.tobytes()).hexdigest()==m['controller']['neuron_ordering_sha256']
    for population in m['controller']['population_mapping'].values():
        for neuron in population:assert str(ids[neuron['index']])==neuron['root_id']
    audit_weights(folder,m,ids)
    channels=np.array(m['controller']['input_channels']);rng=np.random.default_rng(m['seed']);profile=CandidateInputs(65)
    total=0
    for k,(row,chunk) in enumerate(zip(rows,m['chunks'])):
        assert row['start']==k*250*.0001 and row['end']==(k+1)*250*.0001
        assert row['applied_motor']==([0.,0.] if k==0 else rows[k-1]['next_motor'])
        assert row['next_available_at']==row['end']
        assert row['sensory']['visual_lplc2_hz']==[0.,0.]
        requested=profile.rates(row['sensory']['antenna_odors'],[0.,0.],m['arm']!='sensory_off')
        assert requested==row['requested_hz']
        matrix=rng.random((250,len(channels)))<np.array([requested[key] for key in INPUT_KEYS])[channels]*.0001
        local,indices=np.nonzero(matrix);times=row['start']+local*.0001
        path=folder/chunk['file'];assert file_sha(path)==chunk['sha256']
        assert file_sha(folder/chunk['state_file'])==chunk['state_sha256']
        with np.load(path) as archive:
            assert np.array_equal(archive['external_i'],indices)
            delivered=archive['external_t'];ticks=np.rint(delivered/.0001).astype(np.int64)
            assert np.array_equal(ticks,k*250+local)
            assert np.all(np.abs(delivered-ticks*.0001)<1e-12)
            si,st,counts=archive['spike_i'],archive['spike_t'],archive['counts']
            assert np.array_equal(np.bincount(si,minlength=m['controller']['neurons']),counts)
            assert np.all(st>=row['start']-1e-12) and np.all(st<row['end']) and np.all(np.diff(st)>=0)
            for key,pop in m['controller']['population_mapping'].items():
                expected=float(counts[[n['index'] for n in pop]].mean()/.025) if pop else None
                assert expected==row['population_hz'][key]
            if m['arm']=='steering_cut':
                targets=[n['index'] for key in ('DNa02_left','DNa02_right') for n in m['controller']['population_mapping'][key]]
                assert not counts[targets].any()
            total+=len(si)
    return {'seed':m['seed'],'arm':m['arm'],'windows':len(rows),'spikes':total,
            'input_rng_reconstructed_exact':True,'monitor_spike_counts_exact':True,
            'neural_commands_apply_next_interval':True,'vision_input_zero':True,
            'source_and_chunk_hashes_passed':True,'manifest_sha256':file_sha(folder/'manifest.json')}


def main():
    results=[]
    for folder in sorted(OUT.glob('screen/*/intact')):results.append(audit(folder))
    for folder in sorted(OUT.glob('trials/*/*')):
        if not (folder/'manifest.json').exists():continue
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']=='complete':results.append(audit(folder))
    atomic_json(OUT/'raw-audit.json',{'status':'passed_completed_cases','records':results,
        'auditor_sha256':file_sha(Path(__file__)),'scope':'Raw event/count/clock integrity; not behavioral or learning acceptance'})
    print(f'{len(results)} completed odor trials audited',flush=True)


if __name__=='__main__':main()
