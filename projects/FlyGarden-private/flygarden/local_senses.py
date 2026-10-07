"""Live local sensing; no arena coordinates enter the neural adapter."""
import copy
import numpy as np
from .vision_encoding import EyeExpansion

class LocalSenses:
    version='local-antenna-eye-adapter-v1'
    def __init__(self,baseline_images,antenna_order=(0,1)):
        if sorted(antenna_order)!=[0,1]:raise ValueError('Two distinct physical antenna indices required')
        self.antenna_order=tuple(antenna_order);self.eye=EyeExpansion(baseline_images);self.last_time=None

    @classmethod
    def from_body(cls,body):
        names=[body.sim.mj_model.site(i).name for i in body.antenna_ids]
        order=[]
        for suffix in ('odor_l','odor_r'):
            matches=[i for i,name in enumerate(names) if name.endswith(suffix)]
            if len(matches)!=1:raise ValueError('Ambiguous physical antenna site: '+suffix)
            order.append(matches[0])
        return cls(body.eye_frames(),order)

    def sample(self,world,observation,eye_images,time,dt):
        if not np.isfinite(time) or time<0 or self.last_time is not None and time<=self.last_time:
            raise ValueError('Increasing finite observation time required')
        if self.last_time is not None and abs(time-self.last_time-dt)>1e-9:
            raise ValueError('Sensory sampling clock differs from coupling interval')
        antennae=np.asarray(observation['antennae'],dtype=float)
        if antennae.shape!=(2,3) or not np.isfinite(antennae).all():raise ValueError('Two finite physical antenna positions required')
        odors=[np.asarray(world.concentrations(antennae[i]),dtype=float).tolist() for i in self.antenna_order]
        if np.asarray(odors).shape!=(2,2) or not np.isfinite(odors).all():raise ValueError('Invalid local odor field')
        features=self.eye.advance(eye_images,dt);self.last_time=float(time)
        return {'observation_time':time,'antenna_odors':odors,'visual_lplc2_hz':[f['lplc2_hz'] for f in features],
                'eye_features':features,'odor_encoding':'Synthetic local distance field,engineered ORN rates',
                'vision_encoding':self.eye.version,'vision_validated_specificity':False,
                'contact_neural_encoding_enabled':False,'movement_neural_encoding_enabled':False}

    def snapshot(self):
        return {'version':self.version,'antenna_order':self.antenna_order,'last_time':self.last_time,
                'baseline':self.eye.baseline.copy(),'previous_area':self.eye.previous_area.copy(),
                'previous_centroid':copy.deepcopy(self.eye.previous_centroid),
                'previous_lum':None if not hasattr(self.eye,'previous_lum') else self.eye.previous_lum.copy()}

    def restore(self,state):
        if state['version']!=self.version or tuple(state['antenna_order'])!=self.antenna_order:
            raise ValueError('Sensory checkpoint identity differs')
        baseline=np.asarray(state['baseline']);area=np.asarray(state['previous_area']);lum=state['previous_lum']
        if baseline.shape!=self.eye.baseline.shape or area.shape!=(2,) or not np.isfinite(baseline).all() or not np.isfinite(area).all():
            raise ValueError('Invalid sensory state')
        if lum is not None and (np.asarray(lum).shape!=baseline.shape or not np.isfinite(lum).all()):raise ValueError('Invalid previous eye frame')
        self.eye.baseline=baseline.copy();self.eye.previous_area=area.copy();self.eye.previous_centroid=copy.deepcopy(state['previous_centroid'])
        if lum is None:
            if hasattr(self.eye,'previous_lum'):del self.eye.previous_lum
        else:self.eye.previous_lum=np.asarray(lum).copy()
        self.last_time=state['last_time']
