"""Audit raw neural diagnostic chunks and localize response stages."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_recovery_interfaces import OUT,DT,DURATION,source_hashes
from flygarden.body_trial import sha
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json
AUDITOR_SHA=sha(Path(__file__))

def arrival_matrices(protocol):
    import pandas as pd
    from scipy.sparse import coo_matrix
    table=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',
                          columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
    selected=protocol['selected_probe_indices'];lookup={v:i for i,v in enumerate(selected)}
    table=table[table.Postsynaptic_Index.isin(selected)]
    row=table.Postsynaptic_Index.map(lookup).to_numpy();col=table.Presynaptic_Index.to_numpy()
    weights=table['Excitatory x Connectivity'].to_numpy()*.275
    shape=(len(selected),138639)
    return tuple(coo_matrix((np.where(weights>=0,weights,0) if positive else np.where(weights<0,weights,0),(row,col)),shape=shape).tocsr()
                 for positive in (True,False))

def expected_arrivals(indices,times,start,end,matrices):
    from scipy.sparse import coo_matrix
    arrival=times+.0018;mask=(arrival>=start-1e-12)&(arrival<end-1e-12)
    bins=np.floor((arrival[mask]-start+1e-12)/.001).astype(int)
    hist=coo_matrix((np.ones(mask.sum()),(indices[mask],bins)),shape=(matrices[0].shape[1],25)).tocsr()
    return tuple((matrix@hist).toarray() for matrix in matrices)

def audit_trial(folder,protocol,matrices=None):
    m=json.loads((folder/'manifest.json').read_text())
    assert m['status']=='complete' and m['sources']==protocol['sources']
    if m.get('eye_fixture_sha256') is not None:
        eye_name='moving_pattern' if m['cue']=='translation' else m['cue'].removeprefix('supported_')
        base=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli'
        pinned=json.loads((base/'manifest.json').read_text())['files']
        assert m['eye_fixture_sha256']==pinned[f'{eye_name}.json']==sha(base/f'{eye_name}.json')
    assert len(m['chunks'])==round(DURATION/DT)
    rows=json.loads((folder/'bins.json').read_text());assert len(rows)==len(m['chunks'])
    decoder=DescendingDecoder();populations=protocol['mapping'];selected=np.asarray(protocol['selected_probe_indices'])
    totals=np.zeros(len(selected),dtype=np.int64);external=0;phases={k:[] for k in ('rest','pulse','recovery')}
    peak_v=np.full(len(selected),-np.inf);low_v=np.full(len(selected),np.inf)
    tail_i=np.array([],dtype=int);tail_t=np.array([])
    for tick,(chunk,row) in enumerate(zip(m['chunks'],rows)):
        file=folder/chunk['file'];assert sha(file)==chunk['sha256']
        with np.load(file,allow_pickle=False) as z:
            start=tick*DT;end=start+DT
            assert abs(chunk['time']-end)<1e-9 and abs(row['time']-end)<1e-9
            assert np.array_equal(z['neuron_indices'],selected)
            assert z['root_ids'].astype(str).tolist()==protocol['selected_probe_root_ids']
            assert np.array_equal(z['counts'],np.bincount(z['spike_i'],minlength=m['neurons']))
            assert len(z['spike_i'])==chunk['spikes']==row['spikes']
            assert np.array_equal(z['spike_counts'],z['counts'][selected])
            assert np.all(z['spike_t']>=start-1e-12) and np.all(z['spike_t']<end-1e-12)
            assert np.array_equal(z['delivered_external_indices'],z['requested_event_i'])
            assert np.allclose(z['delivered_external_times_seconds'],z['requested_event_t'],atol=1e-12,rtol=0)
            assert len(z['requested_event_i'])==row['input_events']
            assert np.all(z['requested_event_t']>=start-1e-12) and np.all(z['requested_event_t']<end-1e-12)
            assert z['voltage_mV'].shape==z['net_synaptic_conductance_equivalent_mV'].shape==(len(selected),25)
            assert np.allclose(z['time_seconds'],start+np.arange(25)*.001,atol=1e-12,rtol=0)
            assert np.isfinite(z['voltage_mV']).all() and np.isfinite(z['net_synaptic_conductance_equivalent_mV']).all()
            assert z['nominal_positive_arrival_mV'].shape==z['nominal_negative_arrival_mV'].shape==(len(selected),25)
            assert np.all(z['nominal_positive_arrival_mV']>=0) and np.all(z['nominal_negative_arrival_mV']<=0)
            if matrices is not None:
                expected_pos,expected_neg=expected_arrivals(np.concatenate([tail_i,z['spike_i']]),np.concatenate([tail_t,z['spike_t']]),start,end,matrices)
                assert np.allclose(z['nominal_positive_arrival_mV'],expected_pos,rtol=1e-12,atol=1e-12)
                assert np.allclose(z['nominal_negative_arrival_mV'],expected_neg,rtol=1e-12,atol=1e-12)
            keep=z['spike_t']>=end-.0018-1e-12;tail_i=z['spike_i'][keep].copy();tail_t=z['spike_t'][keep].copy()
            expected={key:float(z['counts'][[n['index'] for n in value]].mean()/DT) if value else None for key,value in populations.items()}
            assert expected==row['population_hz']
            assert np.array_equal(decoder.advance(DT,expected),row['candidate_motor'])
            totals+=z['spike_counts'];external+=len(z['requested_event_i'])
            peak_v=np.maximum(peak_v,z['voltage_mV'].max(axis=1));low_v=np.minimum(low_v,z['voltage_mV'].min(axis=1))
            phase='rest' if start<.3-1e-9 else 'pulse' if start<.8-1e-9 else 'recovery'
            phases[phase].append(row)
    summaries={}
    for phase,values in phases.items():
        summaries[phase]={'seconds':len(values)*DT,'spikes':sum(v['spikes'] for v in values),
                          'external_events':sum(v['input_events'] for v in values),
                          'population_hz':{key:float(np.mean([v['population_hz'][key] for v in values]))
                                           for key in populations if values[0]['population_hz'][key] is not None},
                          'mean_candidate_motor':np.mean([v['candidate_motor'] for v in values],axis=0).tolist()}
    return {'cue':m['cue'],'seed':m['seed'],'phases':summaries,'external_events':external,
            'selected_neurons':[{'root_id':root,'spikes':int(total),'voltage_min_mV':float(low),'voltage_max_mV':float(high)}
                               for root,total,low,high in zip(protocol['selected_probe_root_ids'],totals,low_v,peak_v)],
            'wall_seconds':m['wall_seconds'],'peak_rss_bytes':m['peak_rss_bytes']}

def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['sources']==source_hashes()
    base=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli'
    pinned=json.loads((base/'manifest.json').read_text())['files']
    for name in ('loom_left.json','loom_right.json','moving_pattern.json'):
        assert sha(base/name)==pinned[name],'Recorded eye fixture differs from pinned source'
    results=[];pending=[];matrices=arrival_matrices(protocol)
    for cue in protocol['cases']:
        for seed in protocol['seeds']:
            folder=OUT/'trials'/f'{cue}-{seed}';path=folder/'manifest.json'
            if not path.exists() or json.loads(path.read_text())['status']!='complete':pending.append(folder.name);continue
            results.append(audit_trial(folder,protocol,matrices))
    status='complete' if not pending else 'partial'
    atomic_json(OUT/'analysis.json',{'status':status,'protocol_sha256':sha(OUT/'protocol.json'),
                                   'auditor_sha256':AUDITOR_SHA,
                                   'trials':results,'pending':pending,
                                   'audited':'Chunk hashes,exact root IDs,spike count parity,input delivery,timing,trace coverage,population rates,decoder reconstruction and independent sparse signed-arrival reconstruction from pinned connectivity.',
                                   'limitation':'Nominal synaptic increments are not membrane currents; physical behavior and released-reference reproduction require separate evidence.'})
    print(json.dumps({'status':status,'audited_trials':len(results),'pending':len(pending)}))

if __name__=='__main__':run()
