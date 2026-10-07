"""A gym environment over the WHOLE game rather than one level.

`SmwEnv` is level-scoped: it boots from a level save state and ends at that
level's goal tape. This one runs continuously -- when a level is cleared it
drives the map itself and carries straight into the next one, so lives,
powerups and map progress persist exactly as they do for a player.

Two things are deliberately not learned:

* **Menus and the overworld are scripted.** They are deterministic, so spending
  policy gradient on them would be waste. Frames spent in a transition are not
  charged to the agent either -- it should not be penalised for a cutscene.
* **Episodes start from the archive, never from a cold boot.** Booting takes
  ~1500 frames; the archive already holds the first playable state as its
  lowest-progress cell, so "unaided" means restoring that.

Reward is the level-scoped shaping plus a large bonus for finishing a level,
which is what makes chaining levels worth more than running far inside one.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import smwrl.overworld as ow
from smwrl.actions import ACTION_TABLE, N_ACTIONS
from smwrl.ram import decode
from smwrl.retro_env import GAME, register_integration
from smwrl.world_archive import game_progress
from smwrl.wrappers import ObsConfig, RewardConfig

ROOM_CHANGE_DROP = 400


@dataclass
class WorldEpisodeConfig:
    max_steps: int = 4000        # agent steps allowed before clearing anything
    steps_per_level: int = 4000  # ...and this much more for each level cleared
    stuck_steps: int = 160
    frameskip: int = 4
    noop_max: int = 8
    transition_frames: int = 2400   # cap on scripted map handling per clear


def step_budget(ecfg: WorldEpisodeConfig, levels_cleared: int) -> int:
    """How long this episode may run, given what it has achieved so far.

    A flat cap cannot express a whole-game run. 4,000 agent steps is about four
    minutes of game time and the shortest real completion is over forty, so an
    episode could never span the game -- but simply raising the cap makes every
    stuck episode five times more expensive, and a policy that inches forward
    just enough to keep resetting the stuck detector burns the whole budget
    going nowhere. That is not hypothetical: it is what deterministic play does
    right now, timing out at x~193.

    Extending the budget per level cleared pays for itself. A stuck policy still
    dies at 4,000; one that chains ten levels gets 44,000 and is never cut off
    mid-run.
    """
    return ecfg.max_steps + max(0, levels_cleared) * ecfg.steps_per_level


def progress_bonus(rcfg: RewardConfig, before: int, after: int) -> float:
    """Reward only game progress that is genuinely new.

    `before` has to be read from RAM when the episode starts, not assumed to be
    zero: curriculum episodes resume from states that already carry banked
    events, and treating that as a jump from nothing would pay the bonus for
    progress an earlier run made.
    """
    return rcfg.event_bonus * max(0, after - before)


class WorldEnv(gym.Env):
    """Continuous SMW: one policy, many levels, scripted transitions."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, curriculum: list[bytes], reward_cfg: RewardConfig | None = None,
                 episode_cfg: WorldEpisodeConfig | None = None,
                 obs_cfg: ObsConfig | None = None, curriculum_ratio: float = 0.5):
        import stable_retro as retro

        register_integration()
        self.env = retro.make(GAME, state=retro.State.NONE,
                              inttype=retro.data.Integrations.CUSTOM_ONLY,
                              render_mode=None)
        self.rcfg = reward_cfg or RewardConfig()
        self.ecfg = episode_cfg or WorldEpisodeConfig()
        self.ocfg = obs_cfg or ObsConfig()
        # Ordered start -> deepest. Index 0 is the first playable state, which
        # is what an "unaided" episode restores.
        self.curriculum = curriculum
        self.ratio = curriculum_ratio

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            0, 255, (self.ocfg.size, self.ocfg.size, self.ocfg.channels), np.uint8)
        self._buf = np.zeros((2, 224, 256, 3), np.uint8)
        self.last_state = None
        self._reset_counters()

    # -- helpers ----------------------------------------------------------
    def _reset_counters(self):
        self._max_x = self._start_x = 0
        self._steps = self._since_progress = 0
        self._room = self._cleared = self._banked = 0
        self._prev = None
        self._from_checkpoint = False
        self._entry_pos = None
        self._events = 0

    def _observe(self, frame):
        if self.ocfg.crop_top:
            frame = frame[self.ocfg.crop_top:]
        n = self.ocfg.size
        if self.ocfg.color:
            return cv2.resize(frame, (n, n), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA)[:, :, None]

    def _read(self):
        ram = self.env.get_ram()
        return decode({
            "x_pos": int(ram[0x0094]) | (int(ram[0x0095]) << 8),
            "y_pos": int(ram[0x0096]) | (int(ram[0x0097]) << 8),
            "game_mode": int(ram[ow.GAME_MODE]),
            "player_anim": int(ram[0x0071]),
            "lives": int(ram[ow.LIVES]),
            "powerup": int(ram[0x0019]),
            "end_level_timer": int(ram[0x1493]),
            "translevel": int(ram[ow.TRANSLEVEL]),
        })

    def _advance_to_next_level(self) -> bool:
        """Scripted: ride the post-clear transition and enter the next level.

        Bounded and contained -- overworld.py raises on an uncontrollable map,
        and one odd transition must end the episode, not the training run.
        """
        try:
            if not ow.wait_for_mode(self.env, ow.MODE_OVERWORLD,
                                    timeout=self.ecfg.transition_frames):
                return False
            ow.wait_settled(self.env, stable_frames=20, timeout=400)
            # Only readable once the map has settled; at the mode flip it still
            # holds the pre-clear value.
            self._events = game_progress(self.env.get_ram())
            # If the map has not carried Mario off the node he entered from,
            # entering again just replays the level he just cleared.
            here = ow.position(self.env)
            order = (("RIGHT", "UP", "LEFT", "DOWN") if here == self._entry_pos
                     else (None, "RIGHT", "UP", "LEFT", "DOWN"))
            for d in order:
                if d is not None:
                    ow.move(self.env, d, settle=90)
                if ow.enter_level(self.env, timeout=240):
                    self._entry_pos = None
                    return True
        except TimeoutError:
            return False
        return False

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.env.reset()
        self._reset_counters()
        if not self.curriculum:
            raise RuntimeError("WorldEnv needs a curriculum; run smwrl.worldrun first")

        if random.random() < self.ratio and len(self.curriculum) > 1:
            i = random.randrange(len(self.curriculum))
            self._from_checkpoint = i > 0
        else:
            i = 0                      # the first playable state = unaided
        self.env.em.set_state(zlib.decompress(self.curriculum[i]))

        noop = np.zeros(12, np.uint8)
        for _ in range(1 + random.randint(0, self.ecfg.noop_max)):
            obs, *_ = self.env.step(noop)
            self._buf[:] = obs
        # Baseline from RAM, not zero: a curriculum state may already carry
        # events, and the event bonus pays for the *rise* over this reading.
        self._events = game_progress(self.env.get_ram())
        st = self._read()
        self._max_x = self._start_x = st.x
        self._prev = self.last_state = st
        return self._observe(obs), {}

    def step(self, action):
        buttons = ACTION_TABLE[int(action)]
        reward = -self.rcfg.time_penalty
        terminated = truncated = False
        info: dict = {}

        for i in range(self.ecfg.frameskip):
            obs, *_ = self.env.step(buttons)
            if i >= self.ecfg.frameskip - 2:
                self._buf[i - (self.ecfg.frameskip - 2)] = obs
        frame = self._buf.max(axis=0)
        st = self._read()
        self.last_state = st
        self._steps += 1

        if st.cleared:
            # Finishing a level is the point of the whole-game framing, so it
            # pays far more than distance inside one.
            reward += self.rcfg.clear_bonus
            self._cleared += 1
            try:
                self._entry_pos = ow.position(self.env)
            except Exception:
                self._entry_pos = None
            banked = self._events
            advanced = self._advance_to_next_level()
            # Paid either way. _advance_to_next_level re-reads progress once the
            # map settles, and that progress is irreversible even if the scripted
            # transition then fails to enter the next level -- that is a harness
            # limitation, not something the agent did wrong. Clearing a level the
            # run has already beaten fires no event and pays nothing extra.
            reward += progress_bonus(self.rcfg, banked, self._events)
            if advanced:
                st = self._read()
                self.last_state = st
                self._banked += self._max_x - self._start_x
                self._max_x = self._start_x = st.x
                self._room = self._since_progress = 0
                self._prev = st
                obs = self.env.em.get_screen()
                frame = obs
            else:
                truncated = True
                info["transition_failed"] = True
        elif st.dying:
            reward -= self.rcfg.death_penalty
            terminated = True
            info["death"] = True
        elif st.in_level:
            prev_x = self._prev.x if self._prev else st.x
            if prev_x - st.x > ROOM_CHANGE_DROP:
                self._banked += self._max_x - self._start_x
                self._room += 1
                self._max_x = self._start_x = st.x
                self._since_progress = 0
                reward += self.rcfg.room_bonus
            elif st.x > self._max_x:
                reward += (st.x - self._max_x) * self.rcfg.progress_scale
                self._max_x = st.x
                self._since_progress = 0
            else:
                self._since_progress += 1

        if not terminated and not truncated:
            if self._since_progress >= self.ecfg.stuck_steps:
                reward -= self.rcfg.stuck_penalty
                truncated = True
                info["stuck"] = True
            elif self._steps >= step_budget(self.ecfg, self._cleared):
                truncated = True
                info["timeout"] = True

        self._prev = st
        info.update({"levels_cleared": self._cleared, "events": self._events, "translevel": st.translevel,
                     "max_x": self._max_x, "room": self._room,
                     "progress": self._banked + (self._max_x - self._start_x),
                     "from_checkpoint": self._from_checkpoint, "steps": self._steps})
        return self._observe(frame), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()


def curriculum_from_archive(path, size: int = 64) -> list[bytes]:
    """A spread of compressed states ordered first-playable -> deepest.

    Two cell kinds are not startable and must never reach the curriculum, least
    of all index 0 -- which is what every unaided episode restores and the only
    thing the SOLO metric is measured from:

    * Yoshi's House has no goal tape. An episode beginning there cannot clear
      anything no matter how good the policy is.
    * Map cells sit on the overworld, and nothing scripts the map at reset --
      only after a clear -- so an episode beginning on one goes nowhere.

    Both sort *below* every real cell, because progress weights game events at
    10^7 and both carry zero. So as the archive grew they sank to the front:
    with 882 cells, index 0 was Yoshi's House at x 53, and half of every
    trainer's resets were starting in a room with no exit. SOLO read 0.00 for
    540k steps and it was measuring the curriculum, not the policy.
    """
    from smwrl.world_archive import MAP_CELL, load_archive
    from smwrl.worldrun import YOSHIS_HOUSE

    cells = [c for c in load_archive(path).cells.values()
             if c.translevel not in (MAP_CELL, YOSHIS_HOUSE)]
    cells.sort(key=lambda c: c.progress)
    if not cells:
        return []
    if len(cells) > size:
        idx = sorted({round(i * (len(cells) - 1) / (size - 1)) for i in range(size)})
        cells = [cells[i] for i in idx]
    return [c.state for c in cells]
