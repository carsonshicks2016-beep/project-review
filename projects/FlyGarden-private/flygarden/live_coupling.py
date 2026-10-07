"""Research-only live local feedback; no navigation or learning claim."""
import copy
from .local_senses import LocalSenses
from .synchronized import CausalCoupling


class LiveCoupling:
    version = 'local-world-causal-loop-v1'

    def __init__(self, candidate, body, world, dt=.025, senses=None):
        self.candidate, self.body, self.world = candidate, body, world
        self.clock = CausalCoupling(dt)
        if abs(body.steps * body.dt) > 1e-9 or abs(world.time) > 1e-9 or abs(candidate.time) > 1e-9:
            raise ValueError('New live loop requires a shared zero clock')
        body.update_visual_world(world.arena['foods'], world.arena['predator'])
        self.senses = senses or LocalSenses.from_body(body)
        self.world_ticks = 0
        self.last = None

    def advance(self, sensory_enabled=True):
        if self.world.status != 'running':
            raise ValueError('Trial already terminated')
        if (abs(self.candidate.time - self.clock.time) > 1e-9 or
            abs(self.body.steps * self.body.dt - self.clock.time) > 1e-9 or
            abs(self.world.time - self.clock.time) > 1e-9):
            raise RuntimeError('Live clocks differ')
        sensory = self.senses.sample(self.world, self.body.observation(),
                                    self.body.eye_frames(), self.clock.time, self.clock.dt)
        frames, events = [], []
        termination_time = None

        def step_world(time, observation):
            nonlocal termination_time
            self.world_ticks += 5
            before = copy.deepcopy(self.world.events)
            if self.world.status == 'running':
                # Integer ticks prevent accumulated world-clock drift. The world
                # evaluates events at the end of each physical gait interval.
                self.world.time = time - self.body.control_dt
                reward = self.world.advance(self.body.control_dt, observation)
                self.world.time = time
                # World retains only its latest 200 events; list length alone
                # cannot identify events once that bounded history fills.
                new_events = []
                if self.world.events != before:
                    overlap = min(len(before), len(self.world.events))
                    while overlap and before[-overlap:] != self.world.events[:overlap]:
                        overlap -= 1
                    new_events = self.world.events[overlap:]
                if reward or new_events:
                    events.append({'time': time, 'reinforcement': reward,
                                   'events': copy.deepcopy(new_events)})
                if self.world.status != 'running':
                    termination_time = time
            else:
                # Complete this already-computed neural interval, holding the
                # same command. Gameplay is frozen after the exact event tick.
                self.world.time = time
            self.body.update_visual_world(self.world.arena['foods'], self.world.arena['predator'])

        def capture(time, pose):
            # Playback contains display state. RNG and numpy integration state
            # belong to full checkpoints, not JSON pose frames.
            world_view = {key: copy.deepcopy(getattr(self.world, key)) for key in
                          ('arena', 'time', 'energy', 'collected', 'status', 'valid')}
            frames.append({'time': time, **pose, 'world': world_view})

        result = self.clock.advance(
            lambda dt: self.candidate.advance(dt, sensory['antenna_odors'],
                                             sensory['visual_lplc2_hz'], sensory_enabled),
            lambda dt, held: self.body.advance(dt, held, capture=capture, world_step=step_world))
        result.update(sensory=sensory, frames=frames, events=events,
                      termination_time=termination_time,
                      terminal_tail_policy='Complete the held-command neural interval; freeze gameplay after terminal gait tick',
                      learning_enabled=False, supplied_escape_override=False)
        self.last = result
        return result

    def snapshot(self):
        # Neural state is separately saved by ContinuousCandidate.save.
        return {'version': self.version, 'clock': self.clock.snapshot(),
                'world_ticks': self.world_ticks, 'body': self.body.snapshot(),
                'world': self.world.snapshot(), 'senses': self.senses.snapshot()}

    def restore(self, state):
        if state['version'] != self.version:
            raise ValueError('Live loop checkpoint identity differs')
        self.clock.restore(state['clock'])
        self.body.restore(state['body'])
        self.world.restore(state['world'])
        self.senses.restore(state['senses'])
        self.world_ticks = state['world_ticks']
        if (self.world_ticks != self.clock.ticks or self.body.steps != self.clock.ticks or
            abs(self.world.time - self.clock.time) > 1e-9 or
            abs(self.candidate.time - self.clock.time) > 1e-9):
            raise ValueError('Checkpoint clocks differ')
        self.body.update_visual_world(self.world.arena['foods'], self.world.arena['predator'])
