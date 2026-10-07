"""Isolated event interpretation of Liu 2021 STP; NOT a garden controller.

PI is represented as externally supplied event acceptance probability. This
rate-thinning convention is engineered: anatomical LN/bouton assignments and
DM1/DM2 physiological parameters have not been established.
"""
from dataclasses import dataclass
import copy
import math


@dataclass
class ReleaseReference:
    U: float = .24
    tau_d: float = .1
    tau_f: float = .05
    x: float = 1.
    u: float = 0.
    time: float = 0.

    def __post_init__(self):
        values = (self.U, self.tau_d, self.tau_f, self.x, self.u, self.time)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite release state')
        if not (0 <= self.U <= 1 and 0 <= self.x <= 1 and 0 <= self.u <= 1
                and self.tau_d > 0 and self.tau_f > 0 and self.time >= 0):
            raise ValueError('Invalid release state')

    def advance(self, time):
        if not math.isfinite(time) or time < self.time:
            raise ValueError('Time must be finite and monotonic')
        elapsed = time - self.time
        self.x = 1 - (1 - self.x) * math.exp(-elapsed / self.tau_d)
        self.u *= math.exp(-elapsed / self.tau_f)
        self.time = time

    def event(self, time, probability=1., rng=None):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Acceptance probability outside [0,1]')
        if 0 < probability < 1 and rng is None:
            raise ValueError('Stochastic thinning needs an owned generator')
        self.advance(time)
        accepted = probability == 1 or (probability > 0 and rng.random() < probability)
        if not accepted:
            return 0.
        self.u += self.U * (1 - self.u)
        release = self.u * self.x
        self.x -= release
        return release

    def checkpoint(self, rng):
        return {'version': 1, 'model': 'liu2021-engineered-event-reference-v1',
                'state': dict(vars(self)), 'rng': copy.deepcopy(rng.bit_generator.state)}

    @classmethod
    def restore(cls, checkpoint, rng):
        if checkpoint['version'] != 1 or checkpoint['model'] != 'liu2021-engineered-event-reference-v1':
            raise ValueError('Incompatible reference checkpoint')
        state = cls(**checkpoint['state'])
        rng.bit_generator.state = copy.deepcopy(checkpoint['rng'])
        return state
