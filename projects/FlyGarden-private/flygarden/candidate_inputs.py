"""Versioned local sensory/state interface for the supported connectome candidate.

These encodings are engineered. They do not establish natural odor attraction,
retinotopy, autonomous navigation, or behavioral learning.
"""
from dataclasses import dataclass
import numpy as np

INPUT_KEYS=('ORN_DM1_left','ORN_DM1_right','ORN_DM2_left','ORN_DM2_right',
            'DNp09_left','DNp09_right','LPLC2_left','LPLC2_right')

@dataclass(frozen=True)
class CandidateInputs:
    support_hz:float=65.
    odor_max_hz:float=50.
    version:str='supported-connectome-inputs-v1'

    def __post_init__(self):
        if self.support_hz not in (0.,30.,65.):raise ValueError('Support outside prospective bounded calibration grid')
        if self.odor_max_hz!=50.:raise ValueError('Odor gain is fixed; revisions require a new profile')

    def rates(self,antenna_odors,visual_lplc2_hz,sensory_enabled=True):
        """Left/right antenna concentrations by odor identity; local eye features.

        No object coordinates, route, global score or predator position accepted.
        """
        odors=np.asarray(antenna_odors,dtype=float);vision=np.asarray(visual_lplc2_hz,dtype=float)
        if odors.shape!=(2,2) or not np.isfinite(odors).all() or np.any(odors<0) or np.any(odors>1):
            raise ValueError('Expected two local antenna concentrations for each of two odors in[0,1]')
        if vision.shape!=(2,) or not np.isfinite(vision).all() or np.any(vision<0) or np.any(vision>100):
            raise ValueError('Expected two bounded local visual feature rates in Hz')
        values=np.array([odors[0,0],odors[1,0],odors[0,1],odors[1,1],0.,0.,0.,0.])*self.odor_max_hz
        values[4:6]=self.support_hz
        if sensory_enabled:values[6:8]=vision
        else:values[:4]=0.
        return dict(zip(INPUT_KEYS,values.tolist()))

    def manifest(self):
        return {'id':self.version,'support_hz_per_DNp09':self.support_hz,'odor_max_hz':self.odor_max_hz,
                'kind':'Engineered supported full-connectome candidate; not validated navigation',
                'odor_encoding':'Each local antenna concentration maps uniformly to exact annotated ipsilateral ORN_DM1/DM2 populations',
                'vision_encoding':'Actual eye-frame features map uniformly to exact annotated LPLC2 populations; no natural receptive-field claim',
                'touch_encoding_enabled':False,'movement_feedback_encoding_enabled':False,
                'motor_decoder':'Frozen meanDNp09 forward and bilateral DNa02 difference, bounded and smoothed',
                'walking_controller':'Supplied NeuroMechFly HybridTurningController',
                'motor_override':False,'learning':False,'validated_choice':False}

def exact_input_order(model_ids,mapping):
    """Reject absent,ambiguous or mismatched annotation/index joins."""
    ids=np.asarray(model_ids);targets=[];channels=[]
    for channel,key in enumerate(INPUT_KEYS):
        population=mapping.get(key,[])
        if not population:raise ValueError('Missing exact annotation population: '+key)
        for neuron in population:
            index=neuron['index']
            if index<0 or index>=len(ids) or str(ids[index])!=neuron['root_id']:
                raise ValueError('Root/index mismatch: '+key)
            if index in targets:raise ValueError('Neuron assigned to multiple input channels')
            targets.append(index);channels.append(channel)
    return np.asarray(targets,dtype=np.int32),np.asarray(channels,dtype=np.int32)
