"""Multiple geometric expansion tracks with engineered own-body exclusion."""
import copy
import numpy as np
from scipy.ndimage import label, find_objects
from .vision_encoding import EyeExpansion


class MaskedExpansion:
    version = 'eye-self-masked-multitrack-v3'

    def __init__(self, bundle):
        self.baseline, self.baseline_valid = self.unpack(bundle)
        self.previous_lum = self.baseline.copy()
        self.tracks = [[], []]

    @staticmethod
    def unpack(bundle):
        lum = EyeExpansion.luminance(bundle['rgb'])
        valid = np.asarray(bundle['valid'])
        if valid.shape != lum.shape or valid.dtype != np.bool_: raise ValueError('Boolean eye-pixel validity required')
        return lum, valid.copy()

    def manifest(self):
        return {'id': self.version, 'dark_difference': .12, 'minimum_pixels': 4,
                'maximum_centroid_shift': .08, 'maximum_area_fraction': .5,
                'minimum_axis_growth_per_second': .15, 'consecutive_expanding_intervals': 2,
                'rate_gain': 1000, 'rate_clip_hz': 100,
                'tracking': 'Greedy centroid matching with area ratio in[.25,4]; each previous component matched at most once',
                'self_mask': 'Camera-visible own-body geom mask; baseline and current masks intersect',
                'neuron_retinotopy_known': False, 'self_motion_disambiguated': False}

    def advance(self, bundle, dt):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive image interval required')
        lum, valid = self.unpack(bundle)
        if lum.shape != self.baseline.shape: raise ValueError('Eye dimensions changed')
        output = []
        for side in range(2):
            usable = self.baseline_valid[side] & valid[side]
            components, _ = label((self.baseline[side]-lum[side] > .12) & usable)
            current = []
            for index, bounds in enumerate(find_objects(components), 1):
                if bounds is None: continue
                coords = np.argwhere(components[bounds] == index)
                if len(coords) < 4: continue
                coords += np.array([b.start for b in bounds])
                current.append({'area':float(len(coords)/components.size),
                    'centroid':coords.mean(axis=0)/np.array(components.shape),
                    'extent':np.quantile(coords,.95,axis=0)-np.quantile(coords,.05,axis=0)+1,
                    'streak':0, 'growth':0.})
            previous = self.tracks[side]; pairs = []
            for i, new in enumerate(current):
                for j, old in enumerate(previous):
                    distance = float(np.linalg.norm(new['centroid']-old['centroid']))
                    ratio = new['area']/old['area']
                    if distance < .08 and .25 <= ratio <= 4:
                        pairs.append((distance+.02*abs(np.log(ratio)),i,j))
            used_i, used_j = set(), set()
            for _, i, j in sorted(pairs):
                if i in used_i or j in used_j: continue
                used_i.add(i); used_j.add(j)
                new, old = current[i], previous[j]
                axis_growth = (new['extent']/old['extent']-1)/dt
                if new['area'] < .5 and new['area'] > old['area'] and np.all(axis_growth > .15):
                    new['streak'] = old['streak']+1
                    if new['streak'] >= 2: new['growth'] = (new['area']-old['area'])/dt
            growth = max([c['growth'] for c in current],default=0.)
            output.append({'brightness':float(lum[side].mean()),
                'motion':float((abs(lum[side]-self.previous_lum[side])*usable).mean()/dt),
                'dark_area_fraction':sum(c['area'] for c in current),
                'tracked_components':len(current), 'expanding_components':sum(c['growth']>0 for c in current),
                'valid_fraction':float(usable.mean()), 'expansion_per_second':growth,
                'lplc2_hz':float(np.clip(growth*1000,0,100))})
            self.tracks[side] = current
        self.previous_lum = lum.copy()
        return output

    def snapshot(self):
        return {'version':self.version,'baseline':self.baseline.copy(),'baseline_valid':self.baseline_valid.copy(),
                'previous_lum':self.previous_lum.copy(),'tracks':copy.deepcopy(self.tracks)}

    def restore(self,state):
        if state['version'] != self.version: raise ValueError('Visual version differs')
        for key in ('baseline','previous_lum','baseline_valid'):
            array=np.asarray(state[key])
            if array.shape != self.baseline.shape or not np.isfinite(array).all(): raise ValueError('Invalid visual state')
        if state['baseline_valid'].dtype != np.bool_: raise ValueError('Invalid visibility mask')
        self.baseline=state['baseline'].copy();self.baseline_valid=state['baseline_valid'].copy()
        self.previous_lum=state['previous_lum'].copy();self.tracks=copy.deepcopy(state['tracks'])
