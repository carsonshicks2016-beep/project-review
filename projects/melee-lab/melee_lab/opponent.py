"""A frozen checkpoint plays P2 in world coordinates using its own history."""
from copy import copy
from .state import History
from .actions import apply, frames_for, vocabulary
from .config import Config
from .catalog import read_json
from pathlib import Path

class FrozenOpponent:
    def __init__(self, path, action_frames, execution_mode='assisted'):
        from .storage import load_model
        self.model = load_model(path, device='cpu')
        self.history = History()
        meta = read_json(Path(path).with_suffix('.json'))
        config = Config(**meta['config'])
        self.factored = getattr(config, 'action_set', 'legacy') == 'controller'
        if not self.factored:
            self.actions = vocabulary(config)
        self.action_frames = config.action_frames
        self.frame = 0
        self.duration = 0
        self.is_recurrent = hasattr(self.model.policy, 'lstm_actor') or 'Recurrent' in type(self.model.policy).__name__
        self.lstm_states = None
        self.episode_start = True
        from .execution import ControllerRules
        self.rules = ControllerRules(execution_mode)
        self.observed_frame = None

    def reset(self, state):
        self.history.reset(self.perspective(state))
        self.frame = self.duration = 0
        self.lstm_states = None
        self.episode_start = True
        self.rules.reset()
        self.observed_frame = getattr(state, 'frame', None)

    @staticmethod
    def perspective(state):
        from .execution import perspective
        return perspective(state, 2)

    def apply(self, controller, state):
        if self.frame >= self.duration:
            now = getattr(state, 'frame', None)
            if now is not None and self.observed_frame is not None and now-self.observed_frame > self.action_frames+8:
                observation = self.history.reset(self.perspective(state))
            else:
                observation = self.history.push(self.perspective(state))
            self.observed_frame = now
            if self.is_recurrent:
                action, self.lstm_states = self.model.predict(
                    observation, state=self.lstm_states, episode_start=self.episode_start, deterministic=False
                )
                self.episode_start = False
            else:
                action, _ = self.model.predict(observation, deterministic=False)
            self.action = self.rules.apply(action, state, 2) if self.factored else int(action)
            self.frame = 0
            self.duration = self.action_frames if self.factored else max(self.action_frames, frames_for(self.action, self.actions))
        if self.factored:
            from . import controller as pad
            pad.apply(controller, self.action, self.frame)
        else:
            apply(controller, self.action, self.frame, self.actions)
        self.frame += 1
