from __future__ import annotations

import numpy as np
from numpy.random import Generator


def make_rng(seed: int) -> Generator:
    return np.random.default_rng(seed)
