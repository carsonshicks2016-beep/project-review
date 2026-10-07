"""Scripted visual-only loom fixture; not a hunting predator or fly input."""
import math
from .world import World


class CausalStimulusWorld(World):
    version = 'live-sided-loom-fixture-v1'

    def configure(self, cue):
        if cue not in ('loom_left', 'loom_right'): raise ValueError('Unknown visual fixture')
        self.cue = cue
        self.arena['foods'] = []
        self.update_stimulus()

    def update_stimulus(self):
        theta = .9 if self.cue == 'loom_left' else -.9
        distance = 24 - (self.time - .5) * 20
        self.arena['predator'].update(enabled=.5 <= self.time < 1.5,
            x=math.cos(theta)*distance, y=math.sin(theta)*distance,
            heading=theta, state='scripted_visual_fixture', velocity=0., visual_only=True)

    def advance(self, dt, body):
        # Reuse energy/travel/fall accounting but disable chase and capture.
        # The physical visual geometry has no collision contacts. Only actual
        # eye images from this fixture enter the fly's sensory adapter.
        self.arena['predator']['enabled'] = False
        try: return super().advance(dt, body)
        finally: self.update_stimulus()
