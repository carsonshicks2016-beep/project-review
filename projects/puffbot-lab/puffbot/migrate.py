"""Carry a trained policy across a vocabulary / observation change without retraining.

Step 2 appended three tech macros (43 -> 46) and gave projectiles a velocity (observation
v1 -> v2). A v1 network is converted rather than discarded:

* first-layer columns are mapped by feature *name*, so every input keeps its trained
  weights wherever it moved; inputs that did not exist start at zero weight, so the
  converted network computes exactly what the original did until they are non-zero;
* policy-head rows are mapped by macro name; a new macro starts at the average of the
  existing rows, i.e. an ordinary, untrained option the policy can discover.
"""
from __future__ import annotations

import torch

from . import actions, contract, observation

FIRST_LAYERS = ('pi_body.0.weight', 'v_body.0.weight')


def is_v1(saved: dict | None) -> bool:
    u = contract.upgrade(saved)
    return (u.get('space') == 'puff-macros-v1' and u.get('names') == list(actions.V1_NAMES)
            and u.get('observation') == observation.LAYOUT_V1)


def v1_to_current(state: dict) -> dict:
    old = observation.feature_names('puff-obs-v1')
    new = observation.feature_names()
    where = {n: i for i, n in enumerate(old)}
    out = dict(state)
    for key in FIRST_LAYERS:
        w = state[key]
        extra = w.shape[1] - len(old)                     # embedding columns follow the floats
        nw = torch.zeros(w.shape[0], len(new) + extra, dtype=w.dtype)
        for j, name in enumerate(new):
            if name in where:
                nw[:, j] = w[:, where[name]]
        nw[:, len(new):] = w[:, len(old):]
        out[key] = nw
    rows = {n: i for i, n in enumerate(actions.V1_NAMES)}
    w, b = state['pi_head.weight'], state['pi_head.bias']
    nw = torch.empty(actions.COUNT, w.shape[1], dtype=w.dtype)
    nb = torch.empty(actions.COUNT, dtype=b.dtype)
    for j, name in enumerate(actions.NAMES):
        if name in rows:
            nw[j], nb[j] = w[rows[name]], b[rows[name]]
        else:
            nw[j], nb[j] = w.mean(0), b.mean()
    out['pi_head.weight'], out['pi_head.bias'] = nw, nb
    return out


NOTE = ('trained before step 2 (43 macros, observation v1); converted: new inputs start at '
        'zero weight, the 3 tech macros start as average options')
