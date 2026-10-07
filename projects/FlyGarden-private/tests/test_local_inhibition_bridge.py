import copy,json
import numpy as np
import pandas as pd
import brian2 as b
import pytest
from flygarden.local_inhibition_bridge import LocalInhibitionBridge,residual_edge_allocation

def routing():
 return {'nodes':['2@A'],'root_model_indices':{'2':1},'input_site_totals':[2],'coupling':[[0]],'routes':[
 {'pre_root_id':'1','post_root_id':'2','pre_node':-1,'post_node':0,'site_count':2,'original_sign':1},
 {'pre_root_id':'2','post_root_id':'3','pre_node':0,'post_node':-1,'site_count':3,'original_sign':1},
 {'pre_root_id':'2','post_root_id':'3','pre_node':-1,'post_node':-1,'site_count':2,'original_sign':1}]}

def system():
 b.start_scope();b.prefs.codegen.target='numpy';clock=b.Clock(dt=.1*b.ms)
 n=b.NeuronGroup(3,'g:volt\nfire:1',threshold='fire>0',reset='fire=0',clock=clock);n.fire[0]=1
 bridge=LocalInhibitionBridge(n,clock,[1,2,3],routing())
 net=b.Network(n,*bridge.objects);return net,n,bridge

def test_delay_arrival_output_units_and_checkpoint_pending_queue():
 net,n,bridge=system();net.run(1*b.ms,namespace={});net.store('pending');state=json.loads(json.dumps(bridge.checkpoint()))
 net.run(1*b.ms,namespace={});assert bridge.adapter.activity[0]>0;assert n.g[2]==0*b.mV
 net.run(2*b.ms,namespace={});assert n.g[2]<0*b.mV
 expected=(np.asarray(n.g[:]/b.mV).copy(),bridge.checkpoint())
 net.restore('pending');bridge.restore(state);net.run(3*b.ms,namespace={})
 np.testing.assert_array_equal(n.g[:]/b.mV,expected[0]);assert bridge.checkpoint()==expected[1]
 # Direct mV/s conversion with a known delayed activity, no neuron integration.
 before=float(n.g[2]/b.mV);bridge.buffer[bridge.cursor]=.5;bridge.tick()
 assert float(n.g[2]/b.mV)-before==pytest.approx(-.275*100*.0001*3*.5)
 bad=copy.deepcopy(bridge.checkpoint());bad['buffer'][0][0]=float('nan')
 with pytest.raises(ValueError):bridge.restore(bad)

def test_counts_not_doubled_and_unassigned_signs_preserved():
 graph=pd.DataFrame({'Presynaptic_Index':[0,1],'Postsynaptic_Index':[1,2],'Connectivity':[2,5],'Excitatory x Connectivity':[2,5]})
 rows,weights,audit=residual_edge_allocation(graph,routing(),[1,2,3])
 np.testing.assert_array_equal(rows,[0,1]);np.testing.assert_allclose(weights,[0,.55]);assert audit['original_sites']==7 and audit['upgraded_sites']==5 and audit['residual_sites']==2
