from types import SimpleNamespace
import numpy as np
from flygarden.local_senses import LocalSenses

def test_local_antennal_order_and_checkpointed_visual_history():
    baseline=np.full((2,30,30,3),255,dtype=np.uint8)
    world=SimpleNamespace(concentrations=lambda p:[p[0]/10,p[1]/10])
    observation={'antennae':[[2.,4.,1.],[6.,8.,1.]]}
    a=LocalSenses(baseline,(1,0));first=a.sample(world,observation,baseline,0.,.025)
    assert first['antenna_odors']==[[.6,.8],[.2,.4]]
    state=a.snapshot();b=LocalSenses(baseline,(1,0));b.restore(state)
    image=baseline.copy();image[0,10:16,10:16]=0
    assert a.sample(world,observation,image,.025,.025)==b.sample(world,observation,image,.025,.025)
    assert first['vision_validated_specificity'] is False
