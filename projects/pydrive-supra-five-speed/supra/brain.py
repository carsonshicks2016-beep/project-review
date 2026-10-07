"""
NumPy MLP "brain" — the genome the genetic algorithm evolves.

Maps the 40-value sensor observation (slice 2) to driving actions. For the GA we
use 3 continuous outputs — steer, throttle, brake — and let an automatic gearbox
handle clutch + shifting, so even gen-0 cars actually move and evolution is
tractable. (The full 6-control / gear-head policy is PPO territory, slice 4.)

A genome is just the flat concatenation of all weights and biases, so crossover
and mutation are plain vector ops. Output biases are seeded so an untrained
network already feeds in throttle and lifts off the brake — sensible defaults to
explore around rather than flailing from random noise.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class MLP:
    def __init__(self, n_in: int, hidden=(32, 24), n_out: int = 3,
                 weight_scale: float = 0.35,
                 bias_throttle: float = 0.73, bias_brake: float = 0.18):
        self.sizes = [n_in, *hidden, n_out]
        self.weight_scale = weight_scale
        self._biases_seed = (bias_throttle, bias_brake)
        self.W = []
        self.b = []
        rng = np.random.default_rng()
        for a, c in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.standard_normal((a, c)) * weight_scale)
            self.b.append(np.zeros(c))
        # seed the output-layer biases so gen-0 drives sensibly
        if n_out >= 3:
            self.b[-1][1] = _logit(bias_throttle)    # throttle
            self.b[-1][2] = _logit(bias_brake)       # brake
        self._shapes = [(w.shape, bb.shape) for w, bb in zip(self.W, self.b)]
        self.size = sum(w.size + bb.size for w, bb in zip(self.W, self.b))

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Observation -> [steer (-1..1), throttle (0..1), brake (0..1)]."""
        a = x
        for i, (w, bb) in enumerate(zip(self.W, self.b)):
            a = a @ w + bb
            if i < len(self.W) - 1:
                a = np.tanh(a)
        out = a
        steer = np.tanh(out[0])
        throttle = _sigmoid(out[1])
        brake = _sigmoid(out[2])
        return np.array([steer, throttle, brake])

    # ------------------------------------------------------------------ #
    # genome (flat parameter vector) for the GA
    # ------------------------------------------------------------------ #
    def get_genome(self) -> np.ndarray:
        return np.concatenate([p.ravel() for pair in zip(self.W, self.b) for p in pair])

    def set_genome(self, g: np.ndarray):
        i = 0
        for k, (ws, bs) in enumerate(self._shapes):
            n = int(np.prod(ws))
            self.W[k] = g[i:i + n].reshape(ws); i += n
            n = int(np.prod(bs))
            self.b[k] = g[i:i + n].reshape(bs); i += n
        return self

    @classmethod
    def from_genome(cls, g: np.ndarray, n_in: int, hidden, n_out, **kw) -> "MLP":
        net = cls(n_in, hidden, n_out, **kw)
        net.set_genome(g)
        return net

    def clone(self) -> "MLP":
        return MLP.from_genome(self.get_genome().copy(), self.sizes[0],
                               tuple(self.sizes[1:-1]), self.sizes[-1],
                               weight_scale=self.weight_scale)
