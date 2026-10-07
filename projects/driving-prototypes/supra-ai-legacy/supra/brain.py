"""
Neural-network "brain" for the genetic algorithm.

`MLPGenome` is a small fully-connected network evaluated in numpy -- fast,
trivially serialisable, and easy to mutate/crossover for the GA.  Inputs are
the sensor vector; outputs are the 6 controls (steer, throttle, brake, clutch,
shift_up, shift_down).

The PPO reinforcement-learning policy lives in `supra/ppo.py` (`ActorCritic`),
which shares this exact 6-output control contract so either brain is a drop-in
for the live app.
"""

from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLPGenome:
    def __init__(self, n_in, hidden, n_out, rng=None, init_scale=0.8):
        self.sizes = [n_in, *hidden, n_out]
        rng = rng or np.random.default_rng()
        self.W, self.b = [], []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0, init_scale / np.sqrt(a), size=(a, b)))
            self.b.append(np.zeros(b))

    # ---- inference ----
    def forward(self, x):
        h = np.asarray(x, dtype=np.float64)
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
        out = h @ self.W[-1] + self.b[-1]
        # Output biases give untrained cars a sensible default (drive forward,
        # light brake, clutch engaged, no shift) so evolution gets signal; the
        # net can still learn the full range by driving pre-activations hard.
        steer = np.tanh(out[0])               # [-1, 1]
        throttle = _sigmoid(out[1] + 1.0)     # default ~0.73
        brake = _sigmoid(out[2] - 1.5)        # default ~0.18
        clutch = _sigmoid(out[3] + 2.0)       # default ~0.88 (engaged)
        shift_up = _sigmoid(out[4] - 2.0)     # default ~0.12 (no shift)
        shift_down = _sigmoid(out[5] - 2.0)   # default ~0.12
        return steer, throttle, brake, clutch, shift_up, shift_down

    # ---- genetics ----
    def flat(self):
        return np.concatenate([w.ravel() for w in self.W] +
                              [v.ravel() for v in self.b])

    def load_flat(self, vec):
        k = 0
        for i, w in enumerate(self.W):
            n = w.size
            self.W[i] = vec[k:k + n].reshape(w.shape); k += n
        for i, v in enumerate(self.b):
            n = v.size
            self.b[i] = vec[k:k + n].reshape(v.shape); k += n
        return self

    def clone(self):
        c = MLPGenome.__new__(MLPGenome)
        c.sizes = list(self.sizes)
        c.W = [w.copy() for w in self.W]
        c.b = [v.copy() for v in self.b]
        return c

    def mutate(self, rate, scale, rng):
        v = self.flat()
        mask = rng.random(v.shape) < rate
        v[mask] += rng.normal(0, scale, size=mask.sum())
        return self.load_flat(v)

    @staticmethod
    def crossover(p1, p2, rng):
        a, b = p1.flat(), p2.flat()
        mask = rng.random(a.shape) < 0.5
        child = np.where(mask, a, b)
        c = p1.clone()
        return c.load_flat(child)

    # ---- persistence ----
    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), flat=self.flat())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        sizes = [int(s) for s in d["sizes"]]
        g = cls(sizes[0], sizes[1:-1], sizes[-1])
        return g.load_flat(d["flat"])
