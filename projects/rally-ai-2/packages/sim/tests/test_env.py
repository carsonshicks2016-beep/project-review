"""Environment tests.

The load-bearing property here is the North Star's §2.4: *anything that can kill
the agent must be visible to the agent*. Every termination this env can produce
gets a test that it fires, and the sensor tests assert the matching signal
exists in the observation. A termination the policy cannot see coming is not a
lesson, it is noise in the value function.
"""

from __future__ import annotations

import numpy as np
import pytest

from rallyai import contracts
from rallyai.env import CONTROL_DT, EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.sense import SensorSpec, SensorSuite
from rallyai.stage.builder import ProfileBuilder, build_stage
from rallyai.stage.fixtures import flat_straight, proving_ground


def _straight_with_tree(distance: float = 60.0) -> dict:
    stage = flat_straight(300.0, width=10.0)
    stage["obstacles"] = [{
        "kind": "tree", "x": distance, "y": 0.0, "radius": 0.5,
        "s": distance, "lateral": 0.0,
    }]
    return stage


# --------------------------------------------------------------------------- #
# spaces and observation
# --------------------------------------------------------------------------- #

def test_action_space_includes_the_handbrake():
    """Four continuous inputs. The handbrake is not optional — 'by any means'
    includes pivoting a hairpin on it."""
    env = RallyEnv(proving_ground())
    assert env.action_space.shape == (4,)
    assert env.action_space.low[0] == -1.0 and env.action_space.high[0] == 1.0
    for i in (1, 2, 3):
        assert env.action_space.low[i] == 0.0 and env.action_space.high[i] == 1.0


def test_observation_matches_the_declared_sensor_size():
    env = RallyEnv(proving_ground())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.sensors.size,)
    assert env.observation_space.shape == (env.sensors.size,)


def test_observation_is_finite_under_abuse():
    """A NaN in the observation poisons a policy silently. Random inputs on a
    stage with obstacles, jumps and surface changes is the cheapest fuzz there
    is."""
    env = RallyEnv(proving_ground())
    rng = np.random.default_rng(0)
    for episode in range(3):
        obs, _ = env.reset(seed=episode)
        assert np.all(np.isfinite(obs))
        for _ in range(400):
            obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 4).clip(
                env.action_space.low, env.action_space.high))
            assert np.all(np.isfinite(obs))
            assert np.isfinite(r)
            if term or trunc:
                break


def test_reset_is_deterministic():
    env = RallyEnv(proving_ground())
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)


def test_same_actions_give_the_same_trajectory():
    """Determinism given seed and action sequence. Without it, a replay is not a
    record and an evaluation cannot be re-derived."""
    actions = [np.array([0.2, 0.7, 0.0, 0.0], dtype=np.float32)] * 200

    def rollout():
        env = RallyEnv(proving_ground())
        env.reset(seed=3)
        out = []
        for a in actions:
            _, r, term, trunc, info = env.step(a)
            out.append((float(info["s"]), float(info["speed"]), float(r)))
            if term or trunc:
                break
        return out

    assert rollout() == rollout()


# --------------------------------------------------------------------------- #
# every way the episode can end
# --------------------------------------------------------------------------- #

def test_finish_fires_and_is_reachable():
    """The finish line must be reachable. It briefly was not: finish_s came from
    the stage's declared length while projection clamps to Track's resampled
    length, which is a couple of centimetres shorter — the car reached 100%
    progress and then timed out."""
    env = RallyEnv(proving_ground(), EnvConfig())
    env.reset(seed=0)
    pilot = ReferencePilot(env.track)
    for _ in range(6000):
        _, _, term, trunc, info = env.step(pilot.act(env.query, env.car))
        if term or trunc:
            break
    assert info["termination"] == "finish"
    assert info["progress"] == pytest.approx(1.0, abs=1e-3)


def test_off_course_fires_after_the_grace_period():
    env = RallyEnv(flat_straight(400.0, width=8.0),
                   EnvConfig(off_course_grace_s=1.0))
    env.reset(seed=0)
    # Hard left until it leaves the road and stays there.
    for _ in range(3000):
        _, _, term, trunc, info = env.step([1.0, 0.6, 0.0, 0.0])
        if term or trunc:
            break
    assert info["termination"] == "off_course"


def test_crash_fires_on_hitting_a_tree_at_speed():
    env = RallyEnv(_straight_with_tree(80.0))
    env.reset(seed=0)
    for _ in range(3000):
        _, _, term, trunc, info = env.step([0.0, 1.0, 0.0, 0.0])
        if term or trunc:
            break
    assert info["termination"] == "crash"
    assert info["obstacle_contacts"] >= 1


def test_stuck_fires_when_the_car_stops():
    """v1 had no such check: a car sitting still burned 150 simulated seconds
    per episode collecting nothing, and that is how its shipped demo died."""
    env = RallyEnv(flat_straight(400.0), EnvConfig(stuck_grace_s=1.0))
    env.reset(seed=0)
    for _ in range(3000):
        _, _, term, trunc, info = env.step([0.0, 0.0, 1.0, 1.0])
        if term or trunc:
            break
    assert info["termination"] == "stuck"


def test_spun_fires_when_pointing_the_wrong_way():
    """A wide pad so it does not run out of road first, and a long stuck grace
    so a stationary car does not trip that check instead.

    Full lock *with throttle* is what actually spins it — lock plus handbrake
    locks the rears and the car slides broadly straight (37 deg of heading error
    at 90 km/h), which is correct behaviour and not a spin.
    """
    env = RallyEnv(flat_straight(900.0, width=90.0),
                   EnvConfig(spun_grace_s=0.5, stuck_grace_s=30.0,
                             off_course_grace_s=60.0))
    env.reset(seed=0)
    for _ in range(60):
        env.step([0.0, 1.0, 0.0, 0.0])
    for _ in range(3000):
        _, _, term, trunc, info = env.step([1.0, 0.7, 0.0, 0.0])
        if term or trunc:
            break
    assert info["termination"] == "spun"


def test_timeout_truncates_rather_than_terminates():
    """A time limit is a truncation, not a real ending. PPO must bootstrap the
    value of the final state for one and not the other; conflating them is a
    classic silent bug."""
    env = RallyEnv(flat_straight(4000.0, width=30.0),
                   EnvConfig(max_time_s=2.0, stuck_grace_s=30.0))
    env.reset(seed=0)
    for _ in range(3000):
        _, _, term, trunc, info = env.step([0.0, 0.5, 0.0, 0.0])
        if term or trunc:
            break
    assert trunc is True and term is False
    assert info["termination"] == "timeout"


# --------------------------------------------------------------------------- #
# the agent can see what kills it
# --------------------------------------------------------------------------- #

def test_a_tree_ahead_changes_the_observation():
    """The §2.4 invariant, tested rather than asserted. If obstacle sensing is
    ever removed, this fails."""
    suite = SensorSuite()
    clear = RallyEnv(flat_straight(300.0, width=10.0))
    blocked = RallyEnv(_straight_with_tree(40.0))
    a, _ = clear.reset(seed=0)
    b, _ = blocked.reset(seed=0)
    assert not np.allclose(a, b), "a tree 40 m ahead must be visible to the agent"
    # and specifically in the vision block, which is the first 2 * n_beams dims
    n = suite.spec.n_beams
    assert not np.allclose(a[:2 * n], b[:2 * n])


def test_vision_distinguishes_a_tree_from_the_verge():
    """A driver can tell a survivable road edge from a tree. So may the agent —
    the two have very different consequences."""
    env = RallyEnv(_straight_with_tree(30.0))
    env.reset(seed=0)
    obs = env._last_obs
    assert (obs.beam_kinds == 1.0).any(), "the centre beam should report an obstacle"
    clear = RallyEnv(flat_straight(300.0, width=10.0))
    clear.reset(seed=0)
    assert not (clear._last_obs.beam_kinds == 1.0).any()


def test_lookahead_scales_with_speed():
    """A fixed metre preview is a long way at 20 km/h and no warning at 180."""
    env = RallyEnv(proving_ground())
    env.reset(seed=0)
    slow = env._last_obs.lookahead_s.copy()
    for _ in range(120):
        env.step([0.0, 1.0, 0.0, 0.0])
    fast = env._last_obs.lookahead_s
    assert (fast - env.query.s).max() > (slow - 0.0).max()


def test_heading_error_is_observable_because_spun_can_end_the_episode():
    env = RallyEnv(proving_ground())
    obs, _ = env.reset(seed=0)
    # proprio block sits after vision; heading error is its last element.
    n_vision = 2 * env.sensors.spec.n_beams
    heading_err_idx = n_vision + 24
    assert np.isfinite(obs[heading_err_idx])


# --------------------------------------------------------------------------- #
# reward accounting
# --------------------------------------------------------------------------- #

def test_reward_terms_sum_to_the_returned_reward():
    """The accounting is the instrument the real reward pass depends on. If it
    drifts from what is actually paid, every conclusion drawn from it is wrong."""
    env = RallyEnv(proving_ground())
    env.reset(seed=0)
    total = 0.0
    for _ in range(300):
        _, r, term, trunc, info = env.step([0.05, 0.8, 0.0, 0.0])
        total += r
        if term or trunc:
            break
    assert total == pytest.approx(sum(info["reward_terms"].values()), abs=1e-4)


def test_progress_reward_does_not_pay_twice_for_the_same_ground():
    """Ratcheted progress: ground already covered earns nothing a second time.

    Tested by putting the car back down the road directly, because there is no
    reverse gear in the action space — braking still covers new ground, so it
    cannot exercise the ratchet.
    """
    env = RallyEnv(flat_straight(400.0, width=30.0))
    env.reset(seed=0)
    for _ in range(90):
        env.step([0.0, 1.0, 0.0, 0.0])

    gained = env.account.terms["progress"]
    high_water = env.max_s
    assert high_water > 20.0, "needs to have made real progress first"

    # Put it back 15 m and drive forward again over ground it has already had.
    env.car.vehicle.x -= 15.0
    env.query = env.track.project(env.car.vehicle.x, env.car.vehicle.y,
                                  yaw=env.car.vehicle.yaw, hint_s=env.query.s)
    for _ in range(20):
        env.step([0.0, 0.5, 0.0, 0.0])
        if env.query.s >= high_water:
            break
    assert env.account.terms["progress"] == pytest.approx(gained, abs=1e-6)
    assert env.max_s == pytest.approx(high_water, abs=1e-6)


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #

def test_exported_replay_validates_and_round_trips(tmp_path):
    env = RallyEnv(proving_ground(), EnvConfig(record_frames=True))
    env.reset(seed=0)
    pilot = ReferencePilot(env.track)
    for _ in range(6000):
        _, _, term, trunc, _ = env.step(pilot.act(env.query, env.car))
        if term or trunc:
            break

    replay = env.export_replay(source="eval")
    path = contracts.write_json(replay, tmp_path / "r.json", kind="replay")
    back = contracts.read_json(path, kind="replay")

    assert back["meta"]["termination"] == "finish"
    assert len(back["frames"]) > 100
    # Quantised to the contract's 1e-6 on write, so that is the tolerance.
    assert back["dt"] == pytest.approx(CONTROL_DT, abs=1e-6)
    assert back["stage"]["id"] == "proving_ground"


def test_replay_records_takeoff_and_landing():
    """Rally without jumps is not rally, and the viewer needs the cue."""
    env = RallyEnv(proving_ground(), EnvConfig(record_frames=True))
    env.reset(seed=0)
    pilot = ReferencePilot(env.track)
    for _ in range(6000):
        _, _, term, trunc, _ = env.step(pilot.act(env.query, env.car))
        if term or trunc:
            break
    kinds = {e["kind"] for e in env.events}
    assert "takeoff" in kinds and "landing" in kinds
    assert any(f["air"] for f in env.frames)


# --------------------------------------------------------------------------- #
# sensor spec
# --------------------------------------------------------------------------- #

def test_sensor_size_matches_the_assembled_vector():
    for spec in (SensorSpec(), SensorSpec(n_beams=13),
                 SensorSpec(lookahead_times=(0.5, 1.0, 2.0))):
        env = RallyEnv(proving_ground(), sensor_spec=spec)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (spec.size,), f"declared {spec.size}, assembled {obs.shape[0]}"


def test_default_observation_layout_is_the_documented_one():
    """Pins the layout. Changing it invalidates every checkpoint and normaliser,
    so it should take a deliberate edit here to do so."""
    spec = SensorSpec()
    assert spec.n_beams == 9
    assert spec.n_look == 6
    assert spec.size == 2 * 9 + 25 + (4 + 3 * 6) + (7 + 2 * 6) == 84


def test_positive_camber_pushes_the_car_toward_the_low_side():
    """The camber sign, pinned against the contract rather than against the
    vendored model's opposite convention.

    ``stage.schema.json`` defines positive camber as raising the RIGHT edge --
    which is what banks into a left-hander -- and ``geometry.py`` agrees,
    computing road height as ``z + lat * sin(camber)``. The vendored model's
    ``bank`` is the other way round (``physics.py:107``, "+ = left up"), so the
    bridge must negate.

    Shipped unnegated, the two disagreed inside a single step: the car's height
    came from geometry (right edge up) while its lateral gravity came from
    physics (left edge up). Every corner the generator banked into the turn was
    played off-camber by exactly the amount meant to help.

    Rolls down a dead-straight constant-camber road with zero steering, so
    gravity is the only lateral force in play.
    """
    def slide(camber_rad: float) -> float:
        b = ProfileBuilder(width=14.0)
        b._extend(int(200.0 / b.ds), 0.0, 0.0, 14.0, camber_rad, "tarmac")
        stage = build_stage(b.build(), stage_id="cam", name="cam", seed=0,
                            tier=0, generator_version=1)
        env = RallyEnv(stage, EnvConfig(max_time_s=12.0))
        env.reset(seed=0)
        for _ in range(240):
            env.step(np.array([0.0, 0.25, 0.0, 0.0], np.float32))
        return float(env.query.lateral)

    right_edge_up = slide(np.radians(8.0))
    right_edge_down = slide(np.radians(-8.0))

    assert right_edge_up < -0.1, (
        f"positive camber raises the right edge, so the car must slide LEFT "
        f"(negative lateral); it went {right_edge_up:+.3f} m"
    )
    assert right_edge_down > 0.1, (
        f"negative camber must slide the car RIGHT; it went "
        f"{right_edge_down:+.3f} m"
    )
    # Symmetric, or something other than gravity is doing the work.
    assert abs(right_edge_up + right_edge_down) < 0.05 * abs(right_edge_up)


def test_banking_into_a_corner_helps_rather_than_hurts():
    """The consequence of the sign, at the level anyone would actually notice.

    A left-hander banked into the turn must be EASIER than the same corner
    banked away from it. This is the test that would have caught the bridge
    bug without anyone having to reason about two opposing conventions.
    """
    def worst_excursion(camber: float) -> float:
        b = ProfileBuilder(width=10.0)
        b.straight(60.0)
        b.corner(45.0, 90, "left", camber=camber)
        b.straight(60.0)
        stage = build_stage(b.build(), stage_id="c", name="c", seed=0, tier=0,
                            generator_version=1)
        env = RallyEnv(stage, EnvConfig(max_time_s=60.0))
        env.reset(seed=0)
        pilot = ReferencePilot(env.track, target_lat_g=1.05)
        term = trunc = False
        worst = 0.0
        while not (term or trunc):
            _, _, term, trunc, _ = env.step(pilot.act(env.query, env.car))
            worst = max(worst, abs(env.query.lateral))
        return worst

    banked_in = worst_excursion(+0.10)
    banked_away = worst_excursion(-0.10)
    assert banked_in < banked_away, (
        f"banking into a left-hander must help: it ran {banked_in:.2f} m wide "
        f"against {banked_away:.2f} m off-camber"
    )
