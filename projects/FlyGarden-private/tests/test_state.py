import numpy as np
import pytest
from flygarden.storage import Store
from flygarden.plasticity import Plasticity

def test_checkpoint_immutable_integrity_and_lineage(tmp_path):
 s=Store(tmp_path);a=s.create('learned',{'weight':np.array([1.,2.])});b=s.create('learned',{'weight':np.array([3.])},parent=a['id']);payload,meta,path=s.load(a['id']);assert np.array_equal(payload['weight'],[1,2]);assert b['parent']==a['id'];assert len(s.list())==2
 (path/'state.bin').write_bytes(b'broken')
 with pytest.raises(ValueError,match='integrity'):s.load(a['id'])

def test_storage_rejects_path_traversal(tmp_path):
 with pytest.raises(ValueError):Store(tmp_path).load('../arbitrary')

def test_plasticity_sign_compartment_freeze_and_persistence():
 p=Plasticity([1.,-2.,3.],[0,0,1],[0,0,1]);p.observe(.1,[1,1]);p.reinforce(1);assert p.weights[0]<1 and p.weights[1]>-2;assert p.weights[2]==3
 assert np.array_equal(np.sign(p.weights),np.sign(p.base));p.enabled=False;s=p.snapshot();p.reinforce(-1);assert np.array_equal(p.weights,s['weights'])
 p.enabled=True;p.reinforce(-1);assert p.weights[2]<3
 for _ in range(100):p.reinforce(1)
 assert np.all(np.abs(p.weights)>=np.abs(p.base)*.1-1e-10)
 p.restore(s);assert np.array_equal(p.weights,s['weights'])

def test_body_deterministic_continuation():
 from flygarden.body import Body
 b=Body(seed=2);b.advance(.02,[1,.8]);s=b.snapshot();a=b.advance(.02,[.8,1]);b.restore(s);c=b.advance(.02,[.8,1]);b.close()
 assert np.allclose(a['position'],c['position'],rtol=0,atol=1e-7)
 assert np.allclose(a['feet'],c['feet'],rtol=0,atol=1e-7)

def test_incomplete_metadata_preserves_other_checkpoints(tmp_path):
 s=Store(tmp_path);a=s.create('learned',{'weight':[1]});b=s.create('learned',{'weight':[2]});broken=tmp_path/b['id']/'metadata.json';broken.write_text('{"incomplete":')
 assert [x['id'] for x in s.list()]==[a['id']];assert broken.exists();assert s.load(a['id'])[0]['weight']==[1]
 with pytest.raises(ValueError):s.load(b['id'])
