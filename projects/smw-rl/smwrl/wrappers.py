"""Gymnasium wrappers turning raw SMW into a speedrun learning problem."""

from __future__ import annotations

import random
import zlib
from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smwrl.actions import ACTION_TABLE, N_ACTIONS
from smwrl.ram import GameState, decode

# SMW stores x position per *room*, not per level: going through a pipe or door
# resets it to near zero. A drop larger than this is a room change, not the
# player running backwards.
ROOM_CHANGE_DROP = 400


@dataclass
class RewardConfig:
    """Weights for the speedrun objective.

    Two constraints are load-bearing, and they pull against each other:

    1. `death_penalty` must exceed the time penalty an episode can accrue, or
       dying immediately becomes optimal.
    2. Giving up must cost *more* than trying and failing, or the agent parks
       itself in front of the first hazard forever.

    The first version of this had death_penalty=50 with no stuck_penalty, and
    the agent found the hole: walk to the first hazard (banking ~+22 of progress
    reward), then stand still. Idling returned +7.5, attempting and dying
    returned -36.6, so standing still was correctly optimal and mean_x flatlined
    at 10% of the level for 700k steps.

    With these values: idling ~= -24, reaching the hazard and dying ~= -3, dying
    immediately ~= -20. Attempting now dominates giving up, without making
    suicide attractive -- and clearing the level (+200) still dwarfs everything.
    """

    progress_scale: float = 0.05     # per pixel of *new* rightward ground
    time_penalty: float = 0.08       # per agent step -- this is what buys speed
    death_penalty: float = 20.0
    stuck_penalty: float = 30.0      # for giving up; see the note above
    clear_bonus: float = 200.0
    speed_bonus: float = 0.5         # per agent step saved against the cap
    powerup_bonus: float = 5.0
    room_bonus: float = 25.0         # reaching a new room is real progress
    backtrack_allowance: int = 0     # pixels of regression tolerated before 0 reward
    # Whole-game only: paid when the game itself records new, irreversible
    # progress -- a completion event opening a map path, or a switch palace.
    # Deliberately larger than `clear_bonus`, because clearing a level the run
    # has already beaten pays that bonus again and advances nothing. Without
    # this term the policy is indifferent between the exit that opens the map
    # and the one that does not.
    event_bonus: float = 500.0


@dataclass
class ObsConfig:
    """What the policy actually gets to see.

    Both defaults changed after diagnosing why YoshiIsland1 kept dying in the
    same pit at x~3420. That gap is spanned by SMW's *dotted-line blocks*, which
    look like a floor but are intangible until a P-switch is hit. Rendering the
    agent's view next to the real frame showed the dashed outline was completely
    erased by grayscale downsampling -- the whole region read as uniform gray.

    `crop_top` drops the HUD (score / timer / coins). It is ~32 of 224 rows, so
    at 84x84 it was costing ~12 of 84 rows on information irrelevant to jumping.
    Cropping is free: it spends no extra compute and hands those rows back to
    the playfield.

    `color` costs 3x the input channels, which lands on the GPU -- and this
    workload is emulator-bound, not GPU-bound, so it is nearly free in
    wall-clock terms.
    """

    color: bool = True
    crop_top: int = 32
    size: int = 84

    @property
    def channels(self) -> int:
        return 3 if self.color else 1


@dataclass
class EpisodeConfig:
    max_steps: int = 1200            # agent steps (x4 frames) before truncation
    stuck_steps: int = 140           # steps without a new max x before giving up
    frameskip: int = 4
    # SMW is fully deterministic, so without this every parallel worker would
    # play the identical episode and explore in lockstep.
    noop_max: int = 12


def preprocess_frame(frame: np.ndarray, config: ObsConfig) -> np.ndarray:
    """Apply the exact pixel contract shared by training and continuous play."""
    if config.crop_top:
        frame = frame[config.crop_top:]
    n = config.size
    if config.color:
        return cv2.resize(frame, (n, n), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA)[:, :, None]


class SmwEnv(gym.Wrapper):
    """Frameskip, discrete actions, 84x84 grayscale obs and a speedrun reward.

    Actions are held for `frameskip` frames (SMW needs buttons *held* -- run and
    jump height are both duration-sensitive), and the last two frames are
    max-pooled to defuse the SNES's flickering sprites.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env: gym.Env,
        reward_cfg: RewardConfig | None = None,
        episode_cfg: EpisodeConfig | None = None,
        checkpoints: "CheckpointPool | None" = None,
        obs_cfg: ObsConfig | None = None,
    ):
        super().__init__(env)
        self.rcfg = reward_cfg or RewardConfig()
        self.ecfg = episode_cfg or EpisodeConfig()
        self.ocfg = obs_cfg or ObsConfig()
        self.checkpoints = checkpoints
        self._rng = random.Random()

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            0, 255, (self.ocfg.size, self.ocfg.size, self.ocfg.channels), np.uint8)

        self._frame_buf = np.zeros((2, 224, 256, 3), np.uint8)
        self._max_x = 0
        self._steps = 0
        self._since_progress = 0
        self._prev: GameState | None = None
        self._start_x = 0
        self._banked = 0          # progress from rooms already completed
        self._room = 0
        self.last_state: GameState | None = None
        self._from_checkpoint = False
        self._target: tuple[int, int] | None = None   # next curriculum stage
        self._reached_target = False

    # -- observation ------------------------------------------------------
    def _observe(self, frame: np.ndarray) -> np.ndarray:
        return preprocess_frame(frame, self.ocfg)

    # -- episode lifecycle ------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng.seed(seed)
            if self.checkpoints is not None:
                self.checkpoints.seed(seed ^ 0x5A17)

        # Reverse curriculum: sometimes start from a state deeper into the
        # level so the agent practises sections it cannot yet reach unaided.
        self._from_checkpoint = False
        self._target = None
        checkpoint_start: tuple[int, int] | None = None
        if self.checkpoints is not None:
            blob = self.checkpoints.sample()
            if blob is not None:
                self.env.unwrapped.em.set_state(blob)
                self._from_checkpoint = True
                self._target = self.checkpoints.current_target
                checkpoint_start = self.checkpoints.current_start
        self._reached_target = False

        # At least one no-op step to populate the info dict, which reset()
        # leaves empty (and to pull a frame reflecting any restored checkpoint).
        noop = np.zeros(12, np.uint8)
        n_noop = 1 + (self._rng.randint(0, self.ecfg.noop_max)
                      if self.ecfg.noop_max else 0)
        for _ in range(n_noop):
            obs, _, term, trunc, info = self.env.step(noop)
            self._frame_buf[:] = obs
            if term or trunc:
                break
        state = decode(info)

        self._max_x = state.x
        self._start_x = state.x
        self._steps = 0
        self._since_progress = 0
        self._banked = 0
        # Archive room indices are temporal route segments, not data that can be
        # recovered from the raw emulator snapshot.  Preserve the sampled
        # checkpoint's index so comparisons with the next curriculum stage are
        # in the same coordinate system.
        self._room = checkpoint_start[0] if checkpoint_start is not None else 0
        self._prev = state
        self.last_state = state
        return self._observe(obs), dict(info)

    def step(self, action: int):
        buttons = ACTION_TABLE[int(action)]
        reward = 0.0
        terminated = truncated = False
        info: dict = {}
        ended_early = False

        for i in range(self.ecfg.frameskip):
            obs, _, term, trunc, info = self.env.step(buttons)
            # Keep the last two frames for max-pooling. On an early break the
            # buffer would otherwise still hold the *previous* step's frames and
            # we would hand back a stale observation, so write every frame when
            # the loop cuts short.
            if i >= self.ecfg.frameskip - 2:
                self._frame_buf[i - (self.ecfg.frameskip - 2)] = obs
            if term or trunc:
                self._frame_buf[:] = obs
                ended_early = True
                break

        frame = self._frame_buf.max(axis=0)
        state = decode(info)
        self.last_state = state
        self._steps += 1

        reward += self._shaped_reward(state)

        # -- termination --------------------------------------------------
        if state.cleared:
            saved = max(0, self.ecfg.max_steps - self._steps)
            reward += self.rcfg.clear_bonus + self.rcfg.speed_bonus * saved
            terminated = True
            info["level_cleared"] = True
        elif state.dying or (self._prev is not None and state.lives < self._prev.lives):
            reward -= self.rcfg.death_penalty
            terminated = True
            info["death"] = True
        elif self._since_progress >= self.ecfg.stuck_steps:
            # Not free: without this, parking in front of a hazard beats trying.
            reward -= self.rcfg.stuck_penalty
            truncated = True
            info["stuck"] = True
        elif self._steps >= self.ecfg.max_steps:
            truncated = True
            info["timeout"] = True
        elif ended_early:
            truncated = True
            info["env_done"] = True

        # Close the loop on the curriculum stage. Success is reaching the *next*
        # stage (a fixed-length hop) or clearing outright -- not clearing from
        # here, which would chain one more segment per step backwards and make
        # promotion exponentially harder the further back the curriculum goes.
        if self._from_checkpoint and self._target is not None and state.in_level:
            t_room, t_x = self._target
            if (self._room, state.x) >= (t_room, t_x):
                self._reached_target = True
        if (terminated or truncated) and self._from_checkpoint and self.checkpoints:
            self.checkpoints.report(
                self._reached_target or bool(info.get("level_cleared")))
            info["curriculum_focus"] = self.checkpoints.focus

        # Episodes seeded from a curriculum state can start next to the goal, so
        # they must not be mixed into the headline metrics.
        info["from_checkpoint"] = self._from_checkpoint
        info["max_x"] = self._max_x
        info["progress"] = self._banked + (self._max_x - self._start_x)
        info["room"] = self._room
        info["steps"] = self._steps
        self._prev = state
        return self._observe(frame), float(reward), terminated, truncated, info

    # -- reward -----------------------------------------------------------
    def _shaped_reward(self, state: GameState) -> float:
        r = -self.rcfg.time_penalty

        if not state.in_level:
            # Cutscenes, door transitions and the death animation: no progress
            # is possible, so do not let them trip the stuck detector.
            return r

        prev_x = self._prev.x if self._prev else state.x
        # Guard against the death respawn: dying deep in a level resets x to 0,
        # which would otherwise read as a room transition and pay a room bonus.
        if not state.dying and prev_x - state.x > ROOM_CHANGE_DROP:
            # New room (pipe/door). x restarts near zero, so rebase or the old
            # max_x would be unreachable and every later step would read as
            # "no progress" for the rest of the episode.
            self._banked += self._max_x - self._start_x
            self._room += 1
            self._max_x = state.x
            self._start_x = state.x
            self._since_progress = 0
            return r + self.rcfg.room_bonus

        # Only *new* ground pays. Rewarding raw dx lets the agent farm reward by
        # oscillating left and right forever.
        if state.x > self._max_x + self.rcfg.backtrack_allowance:
            r += (state.x - self._max_x) * self.rcfg.progress_scale
            self._max_x = state.x
            self._since_progress = 0
        else:
            self._since_progress += 1

        if self._prev is not None and state.powerup > self._prev.powerup:
            r += self.rcfg.powerup_bonus
        return r

    def snapshot(self) -> bytes:
        return self.env.unwrapped.em.get_state()


class CheckpointPool:
    """Save states for the reverse curriculum.

    States are held zlib-compressed: a raw snapshot is ~430 KB and every worker
    process gets its own copy, so 24 states across 32 workers would be ~330 MB
    uncompressed. They compress to roughly a fifth of that.
    """

    def __init__(self, ratio: float = 0.0, states: list[bytes] | None = None,
                 anneal_resets: int = 0, window: int = 3,
                 promote_after: int = 20, promote_rate: float = 0.5,
                 demote_after: int = 60, initial_focus: int | None = None):
        self.ratio = ratio
        raw = states or []
        # Newer curricula are (room, x, state) so a stage knows where the next
        # one is; older ones were bare state blobs.
        if raw and isinstance(raw[0], tuple):
            self.positions: list[tuple[int, int] | None] = [(r, x) for r, x, _ in raw]
            self.states: list[bytes] = [b for _, _, b in raw]
            for i, (a, b) in enumerate(zip(self.positions, self.positions[1:])):
                if a is not None and b is not None and b <= a:
                    raise ValueError(
                        "curriculum positions must be strictly ordered start-to-goal; "
                        f"stage {i} {a} targets non-forward stage {i + 1} {b}. "
                        "Rebuild it with `python -m smwrl.explore --level LEVEL "
                        "--resume` before training."
                    )
        else:
            self.positions = [None] * len(raw)
            self.states = list(raw)
        self.compressed = bool(self.states) and self._is_compressed(self.states[0])
        # Position of the state returned by the most recent sample().  SmwEnv
        # needs this as well as current_target: a checkpoint captured in room 3
        # must resume with the wrapper's room counter at 3, not silently at 0.
        # The old behaviour made every cross-room curriculum target unreachable.
        self.current_start: tuple[int, int] | None = None
        self.current_index: int | None = None
        # Where the current episode has to get to for this stage to count.
        # None means "clear the level" (the stage nearest the goal).
        self.current_target: tuple[int, int] | None = None

        # Success-gated backward curriculum.
        #
        # States arrive ordered start -> goal. Training starts near the goal (a
        # short, easy problem) and the start point moves *backwards* only once
        # the agent can actually finish from where it currently starts.
        #
        # The earlier version annealed on a fixed schedule instead, which
        # ignored whether the agent was succeeding. On YoshiIsland3 that spent
        # 40% of experience on the back 57% of the level -- ground the agent
        # never reached unaided -- while the first obstacle, the one it was
        # actually stuck on, got almost no practice. Its unaided score climbed
        # to 470 and then fell back to 367.
        self.window = window
        self.promote_after = promote_after
        self.promote_rate = promote_rate
        self.demote_after = demote_after
        # Resuming mid-curriculum matters: extending the archive rebuilds
        # curriculum.pkl with a different number of stages, and without this the
        # backward curriculum snapped back to the goal every time. Overnight,
        # YoshiIsland1 climbed to stage 18, got re-explored, reset to 28, and
        # ended the run further behind than it started.
        default_focus = max(0, len(self.states) - 1 - window)
        self.focus = default_focus if initial_focus is None else \
            max(0, min(len(self.states) - 1, initial_focus))
        self._outcomes: list[float] = []
        self._shared_focus = None
        self._shared_outcomes = None
        self._shared_lock = None
        self._focus = self.focus
        self._rng = random.Random()

        # Retained so an explicit anneal_resets still behaves as before.
        self.anneal_resets = anneal_resets
        self._resets = 0

    # -- adaptive stage control -------------------------------------------
    @property
    def focus(self) -> int:
        if getattr(self, "_shared_focus", None) is not None:
            return int(self._shared_focus.value)
        return self._focus

    @focus.setter
    def focus(self, value: int) -> None:
        value = int(value)
        self._focus = value
        if getattr(self, "_shared_focus", None) is not None:
            self._shared_focus.value = value

    def enable_shared(self, manager) -> None:
        """Share the promotion gate across SubprocVecEnv worker copies.

        Without this each worker collected its own 20 outcomes and wandered to
        a different stage; the persisted focus was whichever worker happened to
        finish last.  Manager proxies are touched only on reset/episode end, not
        on every emulator step.
        """
        self._shared_focus = manager.Value("i", self.focus)
        self._shared_outcomes = manager.list(self._outcomes)
        self._shared_lock = manager.RLock()

    def seed(self, value: int) -> None:
        self._rng.seed(value)

    def report(self, success: bool) -> None:
        """Record the outcome of an episode that started from a curriculum state."""
        if not self.states or self.anneal_resets:
            return
        outcomes = (self._shared_outcomes if self._shared_outcomes is not None
                    else self._outcomes)
        guard = self._shared_lock if self._shared_lock is not None else nullcontext()
        with guard:
            # The rehearsal window also samples easier states *after* focus.
            # Those episodes are useful PPO experience, but they cannot prove
            # that the current (hardest) stage is solved.  In a multi-worker run
            # the focus may also change while an episode is in flight, so only
            # outcomes sampled from the still-current focus enter its gate.
            if self.current_index != self.focus:
                return
            outcomes.append(float(success))
            n = len(outcomes)
            if n >= self.promote_after and sum(outcomes) / n >= self.promote_rate:
                # Solved this stage: start earlier in the level.
                self.focus = max(0, self.focus - 1)
                outcomes[:] = []
            elif n >= self.demote_after:
                rate = sum(outcomes) / n
                if rate < 0.05 and self.focus < len(self.states) - 1:
                    # Too hard from here: back off toward the goal.
                    self.focus = min(len(self.states) - 1, self.focus + 1)
                outcomes[:] = []

    @staticmethod
    def _is_compressed(blob: bytes) -> bool:
        try:
            zlib.decompress(blob)
            return True
        except zlib.error:
            return False

    @classmethod
    def compressed_from(cls, ratio: float, raw_states: list[bytes]) -> "CheckpointPool":
        return cls(ratio, [zlib.compress(s, 6) for s in raw_states])

    @property
    def focus_position(self) -> tuple[int, int] | None:
        """Where in the level the current stage starts, for persisting across
        curriculum rebuilds (an index is meaningless if the list changes)."""
        if 0 <= self.focus < len(self.positions):
            return self.positions[self.focus]
        return None

    @staticmethod
    def index_for_position(positions, pos) -> int | None:
        """Nearest stage to a remembered (room, x)."""
        known = [(i, p) for i, p in enumerate(positions) if p is not None]
        if not known or pos is None:
            return None
        def rank(p):
            return p[0] * 100_000 + p[1]
        target = rank(pos)
        return min(known, key=lambda ip: abs(rank(ip[1]) - target))[0]

    def window_start(self) -> int:
        """Lowest state index currently eligible (0 = the level start)."""
        if not self.states:
            return 0
        if self.anneal_resets:
            frac = min(1.0, self._resets / self.anneal_resets)
            return int(round((1.0 - frac) * (len(self.states) - 1)))
        return self.focus

    def sample(self) -> bytes | None:
        self.current_start = None
        self.current_index = None
        self.current_target = None
        if not self.states or self._rng.random() >= self.ratio:
            return None
        self._resets += 1
        lo = self.window_start()
        hi = min(len(self.states) - 1, lo + self.window)
        i = self._rng.randint(lo, hi)
        self.current_index = i
        self.current_start = self.positions[i]
        # Promote on reaching the *next* stage, not on clearing the whole level.
        # Requiring a full clear makes each step backwards chain one more
        # segment, so promotion difficulty compounds: on YoshiIsland1 the cost
        # per stage grew 60k -> 900k steps and was still climbing.
        self.current_target = self.positions[i + 1] if i + 1 < len(self.positions) else None
        blob = self.states[i]
        return zlib.decompress(blob) if self.compressed else blob
