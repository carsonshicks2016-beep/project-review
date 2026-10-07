import numpy as np
from scipy.sparse import csr_matrix
from flygarden.feedback_trace import delayed_impulses,filtered_impulses,refractory_gate

def test_boundary_arrivals_signed_weights_and_truncation():
 w=csr_matrix([[2.,-3.]])
 values=delayed_impulses([0,1,0],[249,250,299],w,300,18)
 assert values[267,0]==2 and values[268,0]==-3 and np.count_nonzero(values)==2


def test_filter_matches_direct_recurrence_and_linear_source_sum():
 impulse=np.zeros((500,2));impulse[18]=[2,-1];impulse[240]=[3,4]
 result=filtered_impulses(impulse);state=np.zeros(2);expected=[]
 for row in impulse:
  state=state*np.exp(-.0001/.005)+row;expected.append(state.copy())
 np.testing.assert_allclose(result,expected,rtol=0,atol=1e-13)
 np.testing.assert_allclose(result.sum(axis=1),filtered_impulses(impulse.sum(axis=1)),atol=1e-13)


def test_refractory_includes_spike_tick_and_releases_at_exact_boundary():
 gate=refractory_gate([10,40],80)
 assert gate[:10].all() and not gate[10:32].any() and gate[32:40].all() and not gate[40:62].any() and gate[62:].all()
