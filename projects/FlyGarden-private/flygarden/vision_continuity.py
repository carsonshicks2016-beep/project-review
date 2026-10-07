"""Engineered image-only expansion with explicit object continuity.

This is a geometric proxy, not a physiological retina or validated LPLC2 model.
It cannot generally distinguish self-motion from an approaching object.
"""
import copy
import numpy as np
from scipy.ndimage import label
from .vision_encoding import EyeExpansion


class ContinuousExpansion:
    version = 'eye-continuous-expansion-v2'
    luminance = staticmethod(EyeExpansion.luminance)

    def __init__(self, baseline):
        self.baseline = self.luminance(baseline)
        self.previous = [None, None]
        self.streak = np.zeros(2, dtype=np.int64)
        self.previous_lum = self.baseline.copy()

    def manifest(self):
        return {'id': self.version, 'dark_difference': .12, 'minimum_pixels': 4,
                'maximum_area_fraction': .5, 'maximum_centroid_shift': .08,
                'minimum_axis_growth_per_second': .15, 'consecutive_expanding_intervals': 2,
                'rate_gain': 1000, 'rate_clip_hz': 100,
                'scope': 'Largest dark component grows on both image axes; image-only engineered encoding',
                'self_motion_disambiguated': False, 'neuron_retinotopy_known': False}

    def advance(self, frames, dt):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive finite image interval required')
        lum = self.luminance(frames)
        if lum.shape != self.baseline.shape: raise ValueError('Eye image dimensions changed')
        features = []
        for side in range(2):
            components, n = label(self.baseline[side] - lum[side] > .12)
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            current = None
            if n and sizes.max() >= 4:
                coords = np.argwhere(components == sizes.argmax())
                extent = np.quantile(coords, .95, axis=0) - np.quantile(coords, .05, axis=0) + 1
                current = {'area': float(len(coords) / components.size), 'extent': extent,
                           'centroid': coords.mean(axis=0) / np.array(components.shape)}
            previous = self.previous[side]
            continuous = previous is not None and current is not None
            expanding = False; growth = 0.
            if continuous:
                continuous = bool(np.linalg.norm(current['centroid'] - previous['centroid']) < .08)
                axis_growth = (current['extent'] / previous['extent'] - 1) / dt
                expanding = bool(continuous and current['area'] < .5 and
                                 current['area'] > previous['area'] and np.all(axis_growth > .15))
                if expanding: growth = (current['area'] - previous['area']) / dt
            self.streak[side] = self.streak[side] + 1 if expanding else 0
            accepted = self.streak[side] >= 2
            features.append({'brightness': float(lum[side].mean()),
                'motion': float(abs(lum[side] - self.previous_lum[side]).mean() / dt),
                'dark_area_fraction': 0. if current is None else current['area'],
                'object_continuity': continuous, 'expansion_streak': int(self.streak[side]),
                'expansion_per_second': growth if accepted else 0.,
                'lplc2_hz': float(np.clip(growth * 1000, 0, 100)) if accepted else 0.})
            self.previous[side] = current
        self.previous_lum = lum.copy()
        return features

    def snapshot(self):
        return {'version': self.version, 'baseline': self.baseline.copy(),
                'previous': copy.deepcopy(self.previous), 'streak': self.streak.copy(),
                'previous_lum': self.previous_lum.copy()}

    def restore(self, state):
        if state['version'] != self.version: raise ValueError('Visual checkpoint version differs')
        if np.asarray(state['baseline']).shape != self.baseline.shape: raise ValueError('Visual dimensions differ')
        for key in ('baseline', 'previous_lum'):
            value = np.asarray(state[key])
            if value.shape != self.baseline.shape or not np.isfinite(value).all(): raise ValueError('Invalid visual checkpoint')
        streak = np.asarray(state['streak'])
        if streak.shape != (2,) or np.any(streak < 0): raise ValueError('Invalid expansion streak')
        self.baseline = state['baseline'].copy(); self.previous_lum = state['previous_lum'].copy()
        self.previous = copy.deepcopy(state['previous']); self.streak = streak.copy()
