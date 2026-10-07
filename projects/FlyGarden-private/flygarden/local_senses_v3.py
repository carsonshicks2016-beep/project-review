"""Physical local sensing with engineered visible-self exclusion."""
from .local_senses_v2 import LocalSensesV2
from .vision_tracks import MaskedExpansion


class LocalSensesV3(LocalSensesV2):
    version='local-antenna-eye-adapter-v3'

    def __init__(self,baseline_images,antenna_order=(0,1)):
        if sorted(antenna_order)!=[0,1]:raise ValueError('Two distinct physical antenna indices required')
        self.antenna_order=tuple(antenna_order);self.eye=MaskedExpansion(baseline_images);self.last_time=None

    def sample(self,*args,**kwargs):
        result=super().sample(*args,**kwargs)
        result.update(vision_validation_scope='Experimental self-mask and multiple component tracking; native moving-body validation pending',
                      own_body_mask_engineered=True,sensory_adapter=self.version)
        return result
