import copy
import numpy as np
import pytest
from flygarden.local_inhibition_adapter import LocalInhibitionAdapter,make_drive,output_drive_mv_per_s


def test_pulse_tracks_analytic_solution_and_recovers_without_cross_broadcast():
 a=LocalInhibitionAdapter(['root@A','root@B'],np.zeros((2,2)))
 for _ in range(1000):a.step([.4,0])
 expected=.4*(1-np.exp(-.1/.015))
 assert abs(a.activity[0]-expected)<1e-7
 assert a.activity[1]==0
 for _ in range(1500):a.step([0,0])
 assert a.activity[0]<2e-5


def test_coupled_local_inhibition_changes_target_not_source():
 a=LocalInhibitionAdapter(['a','b'],[[0,0],[-.5,0]])
 for _ in range(5000):a.step([1,1])
 np.testing.assert_allclose(a.activity,[1,.5],atol=1e-10)


def test_checkpoint_json_equivalent_and_observation_is_pure():
 import json
 a=LocalInhibitionAdapter(['a','b'],[[0,-.2],[-.5,0]])
 rng=np.random.default_rng(20261007);drives=rng.uniform(-.1,1,(500,2))
 for d in drives[:211]:a.step(d)
 b=LocalInhibitionAdapter.restore(json.loads(json.dumps(a.checkpoint())))
 for d in drives[211:]:
  observer=a.release_hz();observer[:]=-5
  np.testing.assert_array_equal(a.step(d),b.step(d))
 assert a.checkpoint()==b.checkpoint()
 c=copy.deepcopy(a.checkpoint());c['configuration']['tau_s']=.1
 with pytest.raises(ValueError):LocalInhibitionAdapter.restore(c)


def test_routing_units_sign_and_no_double_counted_local_edges():
 routes=[{'pre_root_id':'1','post_root_id':'2','pre_node':-1,'post_node':0,'site_count':3,'original_sign':1},
 {'pre_root_id':'3','post_root_id':'2','pre_node':-1,'post_node':0,'site_count':1,'original_sign':-1},
 {'pre_root_id':'4','post_root_id':'2','pre_node':1,'post_node':0,'site_count':2,'original_sign':1},
 {'pre_root_id':'4','post_root_id':'9','pre_node':1,'post_node':-1,'site_count':5,'original_sign':1}]
 np.testing.assert_allclose(make_drive(routes,{'1':100,'3':100},[6,1],100),[1/3,0])
 assert output_drive_mv_per_s(routes,[0,20])['9']==pytest.approx(-27.5)


@pytest.mark.parametrize('drive',[[float('nan')],[1,2]])
def test_invalid_inputs_leave_state_untouched(drive):
 a=LocalInhibitionAdapter(['a'],[[0]])
 with pytest.raises(ValueError):a.step(drive)
 assert a.steps==0 and a.activity[0]==0
