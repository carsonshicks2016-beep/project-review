"""Tests that drive the real emulator.

These are the ones that catch the bugs that actually bit: a broken core, a
mis-read RAM address, progress accounting that breaks across rooms.

Skipped automatically if the ROM is missing.

    python -m pytest tests/test_integration.py -q
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from smwrl.retro_env import GAME, INTEGRATION_DIR, list_states  # noqa: E402

ROM = INTEGRATION_DIR / GAME / "rom.sfc"
pytestmark = pytest.mark.skipif(not ROM.exists(), reason="ROM not present")

SNES = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def _buttons(*names):
    a = np.zeros(12, np.uint8)
    for n in names:
        a[SNES.index(n)] = 1
    return a


# stable-retro permits one emulator per process and only frees the slot once the
# old object is collected. pytest keeps references alive in frames long enough to
# matter, so collect explicitly between tests or the next construction fails.
@pytest.fixture
def raw_env():
    from smwrl.retro_env import make_raw_env

    env = make_raw_env(state="YoshiIsland1")
    yield env
    env.close()
    del env
    gc.collect()


@pytest.fixture
def env():
    from smwrl.env import make_env
    from smwrl.wrappers import EpisodeConfig, RewardConfig

    e = make_env("YoshiIsland1", RewardConfig(), EpisodeConfig(noop_max=0), monitor=False)
    yield e
    e.close()
    del e
    gc.collect()


# --------------------------------------------------------------------------
# The core itself
# --------------------------------------------------------------------------
def test_core_actually_emulates(raw_env):
    """Guards the big-endian core bug: a broken core renders black forever."""
    raw_env.reset()
    brightest = 0.0
    for _ in range(120):
        obs, *_ = raw_env.step(np.zeros(12, np.uint8))
        brightest = max(brightest, float(obs.mean()))
    assert brightest > 1.0, (
        "screen is black -- the snes9x core is likely the big-endian build. "
        "Run: python -m smwrl.setup_core"
    )


def test_boot_rejects_a_level_save_state(raw_env):
    from smwrl.overworld import boot_to_map

    result = boot_to_map(raw_env)
    assert not result.ok
    assert "power-on" in result.detail


def test_true_power_on_boot_reaches_a_ready_full_coordinate_map():
    from smwrl.overworld import OW_ANIM, OW_CONTROL, boot_to_map, position, read
    from smwrl.retro_env import make_raw_env

    e = make_raw_env(None)
    try:
        result = boot_to_map(e)
        assert result.ok, result.detail
        assert position(e) == (104, 120)
        assert read(e, OW_ANIM) == 3 and read(e, OW_CONTROL) == 2
    finally:
        e.close()


def test_all_route_states_load():
    from smwrl.levels import ROUTE

    available = set(list_states())
    missing = [lv for lv in ROUTE if lv not in available]
    assert not missing, f"route references missing save states: {missing}"


# --------------------------------------------------------------------------
# RAM map -- every one of these was verified against a live game
# --------------------------------------------------------------------------
def test_x_advances_when_holding_right(raw_env):
    """x_pos must track rightward movement.

    Only ~120 frames: holding right+run without jumping walks Mario into the
    first hazard on Yoshi's Island 1 and he dies around x=97, after which the
    level reloads and x resets to 0.
    """
    raw_env.reset()
    xs = []
    for _ in range(120):
        _, _, _, _, info = raw_env.step(_buttons("RIGHT", "Y"))
        xs.append(int(info.get("x_pos", 0)))
    assert max(xs) - xs[0] > 50, f"x_pos barely moved ({xs[0]} -> {max(xs)})"


def test_death_respawn_is_not_read_as_a_room_change(env):
    """Dying deep in a level resets x to 0; that must not pay a room bonus."""
    from smwrl.wrappers import RewardConfig

    r = RewardConfig()
    for _ in range(8):
        env.reset()
        for _ in range(500):
            _, rew, term, trunc, info = env.step(2)
            if term and info.get("death"):
                assert info["room"] == 0, "death respawn was counted as a new room"
                assert rew < 0, "the death step must be a net loss"
                return
            if term or trunc:
                break
    pytest.skip("no death observed in this run")


def test_gameplay_mode_is_0x14(raw_env):
    raw_env.reset()
    _, _, _, _, info = raw_env.step(np.zeros(12, np.uint8))
    from smwrl.ram import MODE_LEVEL, decode

    assert decode(info).game_mode == MODE_LEVEL


def test_info_keys_present_after_step(raw_env):
    raw_env.reset()
    _, _, _, _, info = raw_env.step(np.zeros(12, np.uint8))
    for k in ("x_pos", "y_pos", "game_mode", "player_anim", "lives",
              "end_level_timer", "score", "coins"):
        assert k in info, f"data.json is missing {k}"


def test_save_state_round_trip(raw_env):
    raw_env.reset()
    for _ in range(120):
        raw_env.step(_buttons("RIGHT", "Y"))
    snap = raw_env.unwrapped.em.get_state()
    _, _, _, _, info = raw_env.step(np.zeros(12, np.uint8))
    x_at_snap = int(info["x_pos"])

    for _ in range(120):
        raw_env.step(_buttons("RIGHT", "Y"))
    raw_env.unwrapped.em.set_state(snap)
    _, _, _, _, info = raw_env.step(np.zeros(12, np.uint8))
    assert abs(int(info["x_pos"]) - x_at_snap) < 40, "restoring a state did not restore position"


# --------------------------------------------------------------------------
# Wrapper behaviour
# --------------------------------------------------------------------------
def test_observation_contract(env):
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 3) and obs.dtype == np.uint8
    obs, r, term, trunc, info = env.step(2)
    assert obs.shape == (84, 84, 3)
    assert isinstance(r, float)
    assert "max_x" in info and "progress" in info and "from_checkpoint" in info


def test_observation_is_never_stale(env):
    """The frameskip buffer used to keep the previous step's frames on an
    early break, handing back an observation from the wrong moment."""
    env.reset()
    seen = []
    for _ in range(40):
        obs, *_ = env.step(2)
        seen.append(obs.copy())
    diffs = sum(1 for a, b in zip(seen, seen[1:]) if not np.array_equal(a, b))
    assert diffs > len(seen) // 3, "observations are not changing as the game advances"


def test_progress_only_pays_for_new_ground(env):
    """Rewarding raw dx would let the agent farm reward oscillating in place."""
    env.reset()
    for _ in range(40):
        env.step(2)                      # run right, banking progress
    max_x_before = None
    total = 0.0
    for _ in range(25):                  # now run left: must not pay
        _, r, term, trunc, info = env.step(10)
        max_x_before = info["max_x"]
        total += r
        if term or trunc:
            break
    assert total < 0, "moving backwards must not earn reward"


def test_stuck_truncation_is_penalised(env):
    """Idling has to be expensive or the agent parks in front of hazards."""
    from smwrl.wrappers import RewardConfig

    env.reset()
    last_r = 0.0
    for i in range(2000):
        _, r, term, trunc, info = env.step(0)   # idle forever
        last_r = r
        if term or trunc:
            break
    assert info.get("stuck") or info.get("death") or info.get("timeout")
    if info.get("stuck"):
        assert last_r <= -RewardConfig().stuck_penalty / 2, "stuck must cost something"


def test_death_terminates_and_penalises(env):
    """Walk off into a hazard until something kills us."""
    from smwrl.wrappers import RewardConfig

    deaths = 0
    for _ in range(6):
        env.reset()
        for _ in range(400):
            _, r, term, trunc, info = env.step(2)
            if term and info.get("death"):
                deaths += 1
                assert r < -RewardConfig().death_penalty / 2
                break
            if term or trunc:
                break
        if deaths:
            break
    assert deaths > 0, "never observed a death -- detection may be broken"


def test_checkpoint_start_is_flagged(env):
    """Curriculum episodes must be excluded from headline metrics."""
    from smwrl.wrappers import CheckpointPool

    env.reset()
    for _ in range(60):
        env.step(2)
    snap = env.unwrapped.em.get_state()

    env.checkpoints = CheckpointPool.compressed_from(1.0, [snap])
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["from_checkpoint"] is True

    env.checkpoints = None
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["from_checkpoint"] is False


def test_checkpoint_restores_the_archived_room_index(env):
    """Cross-room curriculum targets must use the archive's coordinate system."""
    import zlib
    from smwrl.wrappers import CheckpointPool

    env.reset()
    snap = env.unwrapped.em.get_state()
    env.checkpoints = CheckpointPool(
        ratio=1.0,
        states=[(4, env.last_state.x, zlib.compress(snap))],
        window=0,
    )
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["room"] == 4


def test_curriculum_start_resumes_deep_in_the_level(env):
    """Restoring a curriculum state must actually resume deep in the level.

    Uses the run-jump pattern rather than plain run-right: holding right alone
    walks into the first hazard and dies at x~=97, after which x resets to 0.
    """
    from smwrl.wrappers import CheckpointPool

    deep_x, snap = 0, None
    for _ in range(6):                       # retry across episodes if we die
        env.reset()
        for i in range(140):
            _, _, term, trunc, _ = env.step(4 if (i % 10) < 3 else 2)
            s = env.last_state
            if s.in_level and s.x > deep_x:
                deep_x = s.x
                snap = env.unwrapped.em.get_state()
            if term or trunc:
                break
        if deep_x > 200:
            break
    assert snap is not None and deep_x > 200, f"never got deep enough (x={deep_x})"

    env.checkpoints = CheckpointPool.compressed_from(1.0, [snap])
    env.reset()
    env.step(0)
    assert abs(env.last_state.x - deep_x) < 80, "restored episode did not start deep"


def test_dummy_vec_env_is_rejected_for_multiple_envs():
    """One emulator per process: DummyVecEnv with n_envs>1 cannot work."""
    from smwrl.env import make_vec_env

    with pytest.raises(ValueError, match="one emulator instance per process"):
        make_vec_env("YoshiIsland1", n_envs=4, subproc=False)


def test_hud_is_cropped_out_of_the_observation():
    """The HUD is ~32 of 224 rows of pixels the policy cannot act on."""
    from smwrl.env import make_env
    from smwrl.wrappers import ObsConfig

    e = make_env("YoshiIsland1", monitor=False, obs_cfg=ObsConfig(crop_top=0))
    try:
        with_hud, _ = e.reset()
    finally:
        e.close()
    gc.collect()
    e = make_env("YoshiIsland1", monitor=False)          # cropped by default
    try:
        cropped, _ = e.reset()
    finally:
        e.close()
    gc.collect()
    assert with_hud.shape == cropped.shape
    assert not np.array_equal(with_hud, cropped), "crop_top had no effect"


def test_vec_env_stacks_and_transposes():
    """The shape contract live.py and watch.py rebuild by hand."""
    from smwrl.env import make_vec_env

    venv = make_vec_env("YoshiIsland1", n_envs=2, subproc=True)
    try:
        obs = venv.reset()
        assert obs.shape == (2, 12, 84, 84), obs.shape   # 3 colour ch x 4 stack
        obs, r, done, info = venv.step(np.array([2, 2]))
        assert obs.shape == (2, 12, 84, 84)
    finally:
        venv.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_completion_event_lands_only_after_the_map_settles():
    """0x7E1F2E is written during the map animation, not at the mode flip.

    worldrun read it the instant game_mode became overworld and always got the
    pre-clear value, so ~580 real clears recorded zero progress and the map
    looked permanently closed.
    """
    import zlib
    import smwrl.overworld as ow
    from smwrl.world_archive import EVENTS, load_archive
    from smwrl.worldrun import make_world_env
    from smwrl.actions import ACTION_TABLE, N_ACTIONS
    from smwrl.explore import ACTION_P

    archive = ROM.parent.parent.parent / "checkpoints" / "world_archive.pkl"
    if not archive.exists():
        pytest.skip("no world archive; run smwrl.worldrun first")

    a = load_archive(archive)
    cell = min(a.cells.values(), key=lambda c: c.progress)
    env = make_world_env()
    try:
        env.reset()
        env.em.set_state(zlib.decompress(cell.state))
        env.step(np.zeros(12, np.uint8))
        rng = np.random.default_rng(7)
        hit = False
        for _ in range(9000):
            act = ACTION_TABLE[int(rng.choice(N_ACTIONS, p=ACTION_P))]
            for _ in range(int(rng.integers(4, 17))):
                env.step(act)
                if int(env.get_ram()[0x1493]) != 0:
                    hit = True
                    break
            if hit:
                break
        if not hit:
            pytest.skip("random play did not reach a goal tape this run")
        assert ow.wait_for_map(env, timeout=1200)
        at_flip = int(env.get_ram()[EVENTS])
        ow.wait_settled(env, stable_frames=20, timeout=600)
        settled = int(env.get_ram()[EVENTS])
        assert settled > at_flip, (
            f"event counter must rise after settling (flip={at_flip}, settled={settled})")
    finally:
        env.close()
        gc.collect()
