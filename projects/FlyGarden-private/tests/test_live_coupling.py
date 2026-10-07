import copy
import numpy as np
import pytest
from flygarden.live_coupling import LiveCoupling
from flygarden.world import World, default_arena


class Candidate:
    time = 0.
    def advance(self, dt, odors, visual, enabled):
        self.inputs = copy.deepcopy(odors)
        self.time += dt
        return [1., 1.]


class Senses:
    def sample(self, world, observation, images, time, dt):
        return {'antenna_odors': [world.concentrations(p).tolist() for p in observation['antennae']],
                'visual_lplc2_hz': [0., 0.], 'observation_time': time}
    def snapshot(self): return {}
    def restore(self, state): pass


class Body:
    dt = .0001
    control_dt = .0005
    steps = 0
    x = 0.
    def update_visual_world(self, foods, predator): self.visual = copy.deepcopy((foods, predator))
    def observation(self):
        return {'position': [self.x, 0., 1.], 'antennae': [[self.x, .1, 1.], [self.x, -.1, 1.]],
                'flipped': False, 'heading': 0.}
    def eye_frames(self): return None
    def advance(self, dt, motor, capture=None, world_step=None):
        self.held = list(motor)
        for _ in range(round(dt / self.control_dt)):
            self.x += float(motor[0]) * self.control_dt
            self.steps += 5
            world_step(self.steps * self.dt, self.observation())
        return self.observation()
    def snapshot(self): return {'steps': self.steps, 'x': self.x}
    def restore(self, state): self.steps, self.x = state['steps'], state['x']


def setup(food=False, predator=False):
    arena = default_arena()
    arena['foods'] = [dict(id='A', x=0., y=0., odor=0, units=3)] if food else []
    arena['predator'].update(enabled=predator, x=0., y=0.)
    world = World(arena=arena)
    return LiveCoupling(Candidate(), Body(), world, senses=Senses())


def test_previous_command_and_live_movement_update_senses():
    loop = setup(food=True)
    a = loop.advance()
    assert a['applied_motor'] == [0., 0.]
    assert loop.body.x == 0
    assert a['events'][0]['time'] == .0005
    assert a['events'][0]['reinforcement'] == 1
    assert loop.world.collected == 1
    b = loop.advance()
    assert b['applied_motor'] == [1., 1.]
    assert loop.body.x == pytest.approx(.025)
    c = loop.advance()
    assert c['sensory']['antenna_odors'] != a['sensory']['antenna_odors']
    assert loop.world_ticks == loop.body.steps == loop.clock.ticks
    assert loop.world.time == loop.clock.time


def test_capture_precedes_food_and_preserves_exact_terminal_timestamp():
    loop = setup(food=True, predator=True)
    result = loop.advance()
    assert loop.world.status == 'captured'
    assert result['termination_time'] == .0005
    assert loop.world.collected == 0
    assert loop.world.arena['foods'][0]['units'] == 3
    assert loop.world.time == .025
    assert loop.world.events[-1]['time'] == .0005
    with pytest.raises(ValueError, match='terminated'): loop.advance()


def test_world_history_full_still_records_food():
    loop = setup(food=True)
    loop.world.events = [dict(time=-i, kind='old', message='old') for i in range(200)]
    result = loop.advance()
    assert result['events'][0]['events'][0]['kind'] == 'food'


def test_full_local_state_restore_and_clock_guard():
    loop = setup(food=True)
    loop.advance()
    saved = loop.snapshot()
    loop.advance()
    loop.candidate.time = .025
    loop.restore(saved)
    assert loop.body.x == 0
    assert loop.world.collected == 1
    assert loop.clock.time == .025
    saved['world_ticks'] += 5
    with pytest.raises(ValueError, match='clocks'): loop.restore(saved)
