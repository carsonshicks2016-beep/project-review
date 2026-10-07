import numpy as np
import pytest
from flygarden.candidate_inputs import CandidateInputs,INPUT_KEYS,exact_input_order

def test_sensory_off_preserves_identical_state_support():
    profile=CandidateInputs(30.)
    intact=profile.rates([[1.,.2],[.3,.4]],[70.,5.])
    off=profile.rates([[1.,.2],[.3,.4]],[70.,5.],False)
    assert intact['ORN_DM1_left']==50 and intact['ORN_DM1_right']==15
    assert intact['ORN_DM2_left']==10 and intact['ORN_DM2_right']==20
    assert off['DNp09_left']==intact['DNp09_left']==30
    assert off['DNp09_right']==intact['DNp09_right']==30
    assert all(off[key]==0 for key in INPUT_KEYS if not key.startswith('DNp09'))

def test_invalid_observations_and_arbitrary_mapping_are_rejected():
    with pytest.raises(ValueError):CandidateInputs(40.)
    with pytest.raises(ValueError):CandidateInputs().rates([[np.nan,0],[0,0]],[0,0])
    mapping={key:[{'index':i,'root_id':str(100+i)}] for i,key in enumerate(INPUT_KEYS)}
    targets,channels=exact_input_order(np.arange(100,108),mapping)
    assert targets.tolist()==channels.tolist()==list(range(8))
    mapping[INPUT_KEYS[0]][0]['root_id']='999'
    with pytest.raises(ValueError):exact_input_order(np.arange(100,108),mapping)
