"""Identical, independently stateful controller rules for both policy players."""
import numpy as np
from copy import copy

VERSION = 'symmetric-controller-v1'
FIREFOX_CHARGE = ('SWORD_DANCE_3_LOW', 'SWORD_DANCE_3_MID', 'SWORD_DANCE_3_HIGH',
                  'SWORD_DANCE_3_LOW_AIR', 'SWORD_DANCE_3_MID_AIR', 'SWORD_DANCE_3_HIGH_AIR')


def perspective(state, port=1):
    if port == 1:
        return state
    view = copy(state)
    view.players = dict(state.players)
    view.players[1], view.players[2] = state.players[2], state.players[1]
    view.projectiles = []
    for projectile in getattr(state, 'projectiles', []):
        projectile = copy(projectile)
        projectile.owner = {1: 2, 2: 1}.get(projectile.owner, projectile.owner)
        view.projectiles.append(projectile)
    return view


class ControllerRules:
    def __init__(self, mode='assisted'):
        if mode not in ('raw', 'assisted'):
            raise ValueError('Execution mode must be raw or assisted.')
        self.mode = mode
        self.reset()

    def reset(self):
        self.firefox_latch = None
        self.decisions = self.changed = 0

    def apply(self, action, state, port=1):
        requested = np.asarray(action)
        vec = requested.copy()
        self.decisions += 1
        if self.mode == 'raw' or state is None or port not in state.players:
            return vec
        p = state.players[port]
        if not getattr(p, 'off_stage', False) or getattr(p, 'on_ground', True):
            self.firefox_latch = None
            return vec
        vec[6] = 0
        from .controller import aims_home, toward_stage
        x = float(p.position.x)
        if getattr(getattr(p, 'character', None), 'name', '') in ('FOX', 'FALCO'):
            if getattr(getattr(p, 'action', None), 'name', '') in FIREFOX_CHARGE:
                if self.firefox_latch is None:
                    self.firefox_latch = int(vec[0]) if aims_home(int(vec[0]), x) else toward_stage(x)
                vec[0] = self.firefox_latch
            else:
                self.firefox_latch = None
                if int(vec[3]) and p.position.y < 0 and not aims_home(int(vec[0]), x):
                    vec[0] = toward_stage(x)
        else:
            self.firefox_latch = None
        opponent = state.players.get(3-port)
        if getattr(p, 'jumps_left', 1) == 0 and opponent is not None:
            dx = opponent.position.x-p.position.x
            dy = opponent.position.y-p.position.y
            if dx*dx+dy*dy >= 28*28:
                vec[1] = vec[2] = 0
        self.changed += int(not np.array_equal(vec, requested))
        return vec

    def report(self):
        return dict(version=VERSION, mode=self.mode, decisions=self.decisions,
                    changed_decisions=self.changed,
                    override_rate=self.changed/self.decisions if self.decisions else 0.)
