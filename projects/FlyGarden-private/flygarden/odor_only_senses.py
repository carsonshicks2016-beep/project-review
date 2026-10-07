"""Odor-only causal protocol: retain visual diagnostics but do not stimulate vision."""
from .local_senses_v2 import LocalSensesV2


class OdorOnlySenses(LocalSensesV2):
    version='local-antenna-odor-only-v1'

    def sample(self,*args,**kwargs):
        result=super().sample(*args,**kwargs)
        result['diagnostic_visual_lplc2_hz']=result['visual_lplc2_hz']
        result['visual_lplc2_hz']=[0.,0.]
        result.update(visual_neural_input_enabled=False,sensory_adapter=self.version,
                      vision_validation_scope='Vision encoding recorded as diagnostic only; disabled for odor causal evaluation')
        return result
