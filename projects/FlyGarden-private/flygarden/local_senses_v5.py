"""Experimental native-camera background registration, uniform neural input."""
from .local_senses_v3 import LocalSensesV3
from .vision_background import BackgroundExpansion


class LocalSensesV5(LocalSensesV3):
    version='local-antenna-rectilinear-eye-adapter-v5'

    def __init__(self,baseline_images,antenna_order=(0,1)):
        if sorted(antenna_order)!=[0,1]:raise ValueError('Two distinct physical antenna indices required')
        self.antenna_order=tuple(antenna_order);self.eye=BackgroundExpansion(baseline_images);self.last_time=None

    def sample(self,*args,**kwargs):
        result=super().sample(*args,**kwargs)
        result.update(camera_projection='rectilinear',
            vision_validation_scope='Experimental registered native-camera input; physiological retina and general self-motion specificity unvalidated')
        return result
