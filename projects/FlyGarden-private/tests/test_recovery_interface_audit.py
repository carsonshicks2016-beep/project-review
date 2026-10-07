import json
import numpy as np
import pytest
from scripts.report_recovery_interfaces import audit_trial,DT,expected_arrivals
from flygarden.body_trial import sha

def fixture(folder):
    keys=('DNp09_left','DNp09_right','DNa02_left','DNa02_right')
    protocol={'sources':{},'mapping':{key:[{'index':i}] for i,key in enumerate(keys)},
              'selected_probe_indices':[0,1,2,3],'selected_probe_root_ids':['101','102','103','104']}
    chunks=[];rows=[]
    for tick in range(60):
        p=folder/f'window-{tick:04d}.npz';start=tick*DT
        np.savez_compressed(p,neuron_indices=np.arange(4),root_ids=np.arange(101,105),counts=np.zeros(4,dtype=int),
                            spike_i=np.array([],dtype=int),spike_t=np.array([]),spike_counts=np.zeros(4,dtype=int),
                            delivered_external_indices=np.array([],dtype=int),requested_event_i=np.array([],dtype=int),
                            delivered_external_times_seconds=np.array([]),requested_event_t=np.array([]),
                            voltage_mV=np.full((4,25),-52.),net_synaptic_conductance_equivalent_mV=np.zeros((4,25)),
                            time_seconds=start+np.arange(25)*.001,nominal_positive_arrival_mV=np.zeros((4,25)),
                            nominal_negative_arrival_mV=np.zeros((4,25)))
        chunks.append({'file':p.name,'sha256':sha(p),'time':start+DT,'spikes':0})
        rows.append({'time':start+DT,'spikes':0,'input_events':0,'population_hz':dict.fromkeys(keys,0.),'candidate_motor':[0.,0.]})
    manifest={'status':'complete','sources':{},'chunks':chunks,'neurons':4,'cue':'quiet','seed':1,'wall_seconds':1,'peak_rss_bytes':1}
    (folder/'manifest.json').write_text(json.dumps(manifest));(folder/'bins.json').write_text(json.dumps(rows))
    return protocol

def test_audit_accepts_exact_quiet_trace_and_rejects_wrong_root(tmp_path):
    protocol=fixture(tmp_path)
    result=audit_trial(tmp_path,protocol)
    assert result['external_events']==0 and result['phases']['rest']['seconds']==pytest.approx(.3)
    assert result['phases']['pulse']['seconds']==pytest.approx(.5)
    assert all(n['spikes']==0 for n in result['selected_neurons'])
    protocol['selected_probe_root_ids'][0]='999'
    with pytest.raises(AssertionError):audit_trial(tmp_path,protocol)

def test_audit_rejects_motor_values_inconsistent_with_spikes(tmp_path):
    protocol=fixture(tmp_path)
    path=tmp_path/'bins.json';rows=json.loads(path.read_text());rows[0]['candidate_motor']=[1.,1.];path.write_text(json.dumps(rows))
    with pytest.raises(AssertionError):audit_trial(tmp_path,protocol)

def test_independent_signed_arrivals_include_delayed_previous_window():
    from scipy.sparse import csr_matrix
    positive=csr_matrix([[0.,2.,0.],[0.,0.,0.]])
    negative=csr_matrix([[0.,0.,-1.],[0.,0.,0.]])
    pos,neg=expected_arrivals(np.array([1,2,1]),np.array([.0233,.0234,.025]),.025,.05,(positive,negative))
    assert pos.sum()==4 and neg.sum()==-1
    assert pos[0,0]==2 and pos[0,1]==2 and neg[0,0]==-1

def test_audit_rejects_invented_signed_arrival_even_with_valid_chunk_hash(tmp_path):
    from scipy.sparse import csr_matrix
    protocol=fixture(tmp_path);file=tmp_path/'window-0000.npz'
    with np.load(file) as z:arrays={key:z[key].copy() for key in z.files}
    arrays['nominal_positive_arrival_mV'][0,0]=1.;np.savez_compressed(file,**arrays)
    path=tmp_path/'manifest.json';manifest=json.loads(path.read_text());manifest['chunks'][0]['sha256']=sha(file);path.write_text(json.dumps(manifest))
    zeros=csr_matrix((4,4))
    with pytest.raises(AssertionError):audit_trial(tmp_path,protocol,(zeros,zeros))
