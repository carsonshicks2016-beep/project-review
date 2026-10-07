import numpy as np
from scripts.check_candidate_full_state import digest

def test_state_digest_is_order_independent_but_detects_array_and_rng_changes():
    a={'array':np.array([1.,2.]),'rng':{'counter':3}}
    assert digest(a)==digest({'rng':{'counter':3},'array':np.array([1.,2.])})
    assert digest(a)!=digest({'array':np.array([1.,2.]),'rng':{'counter':4}})
    assert digest(a)!=digest({'array':np.array([1.,2.],dtype=np.float32),'rng':{'counter':3}})
