"""Isolated port of Barth-Maron et al. 2023 archived rate model.

Source: Zenodo 10028674, dynamical_model/run_AL_model.m (CC BY 4.0).
This is NOT a cable model, a FlyWire cell fit, or an application controller.
The local LN is a continuous rate-like activity, not spikes or measured release.
Time and all time constants use milliseconds. Keep source quirks explicitly.
"""
import copy
import numpy as np

PUBLISHED_PARAMETERS = (3835., 6.06304380373610e-6, 1004.89493761302,
                        30.5976478828549, .103245596784364, .11, .2)
MODEL = 'barth-maron-2023-rate-port-v1'


class GradedGainReference:
    def __init__(self, parameters=PUBLISHED_PARAMETERS, dt_ms=1.):
        self.b = np.asarray(parameters, dtype=float).copy()
        if (self.b.shape != (7,) or not np.isfinite(self.b).all()
                or np.any(self.b < 0) or self.b[0] <= 0
                or not np.isfinite(dt_ms) or not 0 < dt_ms <= 1):
            raise ValueError('Invalid reference parameters')
        self.dt_ms = float(dt_ms)
        self.v = np.array([0., 1., 0., 0.])
        self.resources = np.full(2, .5)
        self.steps = 0

    def step(self, stimulus, injection=None):
        if not np.isfinite(stimulus) or stimulus < 0:
            raise ValueError('Stimulus must be finite and nonnegative')
        injection = np.zeros(4) if injection is None else np.asarray(injection, float)
        if injection.shape != (4,) or not np.isfinite(injection).all() or np.any(injection < 0):
            raise ValueError('Invalid injection')
        # Source injection is an increment EVERY sample, not a current or rate.
        # Scaling by dt preserves that convention at 1 ms and its continuous limit.
        v = self.v + injection * self.dt_ms
        s = np.full(2, stimulus * self.b[2] + self.b[3])
        # Original source disables BOTH presynaptic paths if either weight is zero.
        if not np.any(self.b[4:6] == 0):
            s /= 1 + v[1] * self.b[4:6]
        release = s * self.resources
        da = -self.b[1] * s * self.resources + (1 - self.resources) / self.b[0]
        drive = np.array([release[0] - self.b[6] * v[2], release[0], v[0], 0.])
        next_v = np.maximum(0., v + self.dt_ms * (-v + drive) / 15.)
        next_a = self.resources + self.dt_ms * da
        next_a = np.where(next_a <= 0, 1 / self.b[0], np.minimum(1., next_a))
        if not np.isfinite(next_v).all() or not np.isfinite(next_a).all():
            raise FloatingPointError('Nonfinite reference dynamics')
        self.v, self.resources = next_v, next_a
        self.steps += 1
        return {'activity': v.copy(), 'release': release.copy(), 'resources': next_a.copy()}

    def checkpoint(self):
        return {'version': 1, 'model': MODEL, 'parameters': self.b.tolist(),
                'dt_ms': self.dt_ms, 'v': self.v.tolist(),
                'resources': self.resources.tolist(), 'steps': self.steps}

    @classmethod
    def restore(cls, checkpoint):
        c = copy.deepcopy(checkpoint)
        if c.get('version') != 1 or c.get('model') != MODEL:
            raise ValueError('Incompatible reference checkpoint')
        state = cls(c['parameters'], c['dt_ms'])
        v, a = np.asarray(c['v'], float), np.asarray(c['resources'], float)
        if (v.shape != (4,) or a.shape != (2,) or not np.isfinite(v).all()
                or not np.isfinite(a).all() or np.any(v < 0)
                or np.any(a < 0) or np.any(a > 1)
                or type(c['steps']) is not int or c['steps'] < 0):
            raise ValueError('Invalid checkpoint state')
        state.v, state.resources, state.steps = v.copy(), a.copy(), c['steps']
        return state


def run_reference(stimulus, parameters=PUBLISHED_PARAMETERS, injection=None, dt_ms=1.):
    s = np.asarray(stimulus, float)
    if s.ndim != 1 or not np.isfinite(s).all() or np.any(s < 0):
        raise ValueError('Invalid stimulus series')
    inj = np.zeros((len(s), 4)) if injection is None else np.asarray(injection, float)
    if inj.shape != (len(s), 4) or not np.isfinite(inj).all() or np.any(inj < 0):
        raise ValueError('Invalid injection series')
    state = GradedGainReference(parameters, dt_ms)
    activity = np.empty((len(s), 4))
    release = np.empty((len(s), 2))
    for i, value in enumerate(s):
        row = state.step(value, inj[i])
        activity[i], release[i] = row['activity'], row['release']
    return activity, release, state
