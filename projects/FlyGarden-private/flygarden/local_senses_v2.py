"""Local physical sensing with the versioned continuous expansion proxy."""
import numpy as np
from .local_senses import LocalSenses
from .vision_continuity import ContinuousExpansion


class LocalSensesV2(LocalSenses):
    version = 'local-antenna-eye-adapter-v2'

    def __init__(self, baseline_images, antenna_order=(0, 1)):
        if sorted(antenna_order) != [0, 1]: raise ValueError('Two distinct physical antenna indices required')
        self.antenna_order = tuple(antenna_order)
        self.eye = ContinuousExpansion(baseline_images)
        self.last_time = None

    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        result.update(vision_validation_scope='Bounded native appearance/translation/occlusion/looming fixtures; no general self-motion disambiguation',
                      sensory_adapter=self.version)
        return result

    def snapshot(self):
        return {'version': self.version, 'antenna_order': self.antenna_order,
                'last_time': self.last_time, 'eye': self.eye.snapshot()}

    def restore(self, state):
        if state['version'] != self.version or tuple(state['antenna_order']) != self.antenna_order:
            raise ValueError('Sensory checkpoint identity differs')
        time = state['last_time']
        if time is not None and (not np.isfinite(time) or time < 0): raise ValueError('Invalid sensory clock')
        self.eye.restore(state['eye']); self.last_time = time
