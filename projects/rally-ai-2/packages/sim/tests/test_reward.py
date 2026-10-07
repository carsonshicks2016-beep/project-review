"""Reward tests.

A reward bug does not crash. It quietly trains the wrong thing and you find out
after a night of compute, which is why these tests lean on *integrating terms
over episodes* rather than checking that a function returns a number.

Two of them carry negative controls that reproduce v1's actual failure. A test
that the current reward asks for pace is worth much less without a companion
showing what a reward that does not ask for pace looks like under the same
measurement.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from rallyai.env import EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.env.reward import (
    RewardConfig,
    RewardFunction,
    edge_factor,
    landing_quality,
    straightness,
)
from rallyai.stage.fixtures import flat_straight, proving_ground
from rallyai.stage.generator import generate
from rallyai.track import Track

CONTROL_DT = 1.0 / 30.0


def _car(speed=25.0, landing_g=0.0, slip=0.0, pitch=0.0):
    return SimpleNamespace(speed=speed, landing_slip=slip, landing_pitch=pitch,
                           vehicle=SimpleNamespace(landing_g=landing_g))


def _query(s=50.0, lateral=0.0, half_width=5.0, heading_error=0.0):
    return SimpleNamespace(s=s, lateral=lateral, half_width=half_width,
                           heading_error=heading_error)


def _terms(track, *, cfg=None, gained=0.0, speed=25.0, lateral=0.0,
           half_width=5.0, heading_error=0.0, action=(0.0, 1.0, 0.0, 0.0),
           prev=(0.0, 1.0, 0.0, 0.0), landed=False, landing_g=0.0,
           slip=0.0, pitch=0.0, s=50.0, dt=CONTROL_DT):
    fn = RewardFunction(cfg or RewardConfig())
    return fn.step_terms(
        gained_m=gained,
        query=_query(s=s, lateral=lateral, half_width=half_width,
                     heading_error=heading_error),
        car=_car(speed, landing_g, slip, pitch),
        track=track, action=np.array(action), prev_action=np.array(prev),
        dt=dt, landed=landed,
    )


@pytest.fixture(scope="module")
def straight():
    return Track(flat_straight(600.0, width=10.0))


@pytest.fixture(scope="module")
def pg():
    return Track(proving_ground())


# --------------------------------------------------------------------------- #
# the edge taper — the exploit this reward exists to close
# --------------------------------------------------------------------------- #

def test_edge_factor_is_one_inside_and_zero_off_the_road():
    """The verge-riding exploit in one function. If this ever returns a positive
    number off-track, the policy can farm centerline progress from the verge
    while flicking back inside just often enough to dodge the off-course timer.
    """
    assert edge_factor(0.0, 5.0, 1.0) == 1.0
    assert edge_factor(3.0, 5.0, 1.0) == 1.0        # 2 m of margin: inner corridor
    assert edge_factor(4.5, 5.0, 1.0) == pytest.approx(0.5)   # half way through
    assert edge_factor(5.0, 5.0, 1.0) == 0.0        # exactly on the edge
    assert edge_factor(6.0, 5.0, 1.0) == 0.0        # off it entirely
    assert edge_factor(-4.5, 5.0, 1.0) == pytest.approx(0.5)  # symmetric


def test_edge_factor_tapers_monotonically_across_the_band():
    """Positive control for the test above: a taper that is not monotonic could
    still satisfy the endpoints while leaving a profitable ridge in the middle.
    """
    xs = np.linspace(0.0, 6.0, 61)
    fs = [edge_factor(x, 5.0, 1.0) for x in xs]
    assert all(b <= a + 1e-12 for a, b in pairwise(fs)), "taper must not rise"
    assert len(set(np.round(fs, 6))) > 10, "must actually taper, not step"


@pytest.mark.parametrize("half_width", [2.75, 4.0, 6.0])
def test_no_positive_term_pays_off_the_road(straight, half_width):
    """Every positive term must be zero off-course, at any corridor width.

    Parametrised across widths because the taper is in absolute metres while the
    corridor varies from 5.5 m at tier 5 to 12 m at tier 0 — a band expressed as
    a fraction would behave differently at each and this would catch it.
    """
    t = _terms(straight, gained=2.0, speed=40.0, lateral=half_width + 0.5,
               half_width=half_width)
    for name, value in t.items():
        if name in ("time", "smooth"):
            continue
        assert value <= 0.0, f"{name} paid {value:+.4f} while off the road"


def test_negative_terms_are_not_tapered(straight):
    """The time cost must be identical on and off the road.

    Tapering it would make time cheaper off-course, which pays the car to leave
    the road — the exact inverse of what the taper is for.
    """
    on = _terms(straight, lateral=0.0)["time"]
    off = _terms(straight, lateral=9.0)["time"]
    assert on == pytest.approx(off)
    assert on < 0.0


def test_progress_is_discounted_toward_the_edge(straight):
    """Progress earned while hugging the edge must be worth less than the same
    metres earned in the middle, or verge-riding is free."""
    mid = _terms(straight, gained=1.0, lateral=0.0)["progress"]
    band = _terms(straight, gained=1.0, lateral=4.6)["progress"]
    assert band < mid
    assert _terms(straight, gained=1.0, lateral=5.5).get("progress", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# the algebra: is this actually a speed reward?
# --------------------------------------------------------------------------- #

def _return_over_fixed_distance(track, cfg, speed, distance=400.0):
    """Integrate the pace terms over a fixed distance driven at a fixed speed.

    Distance is held constant and speed varied, which is the only comparison
    that answers "does driving faster pay more" — comparing over fixed *time*
    would just be comparing different amounts of road.
    """
    dt = CONTROL_DT
    n = round(distance / (speed * dt))
    per_step = _terms(track, cfg=cfg, speed=speed, gained=speed * dt)
    return n * (per_step.get("speed", 0.0) + per_step.get("time", 0.0))


@pytest.mark.parametrize("track_name", ["straight"])
def test_driving_faster_over_the_same_ground_earns_more(straight, track_name):
    """v1's central failure, as an executable claim.

    v1 paid ``0.02 * min(v, 35)/35`` per step, which integrates over a fixed
    distance to a CONSTANT — the same return whether the stage took sixty
    seconds or ninety — and above 35 m/s it clipped so that going faster
    actively reduced it. Driving 33% faster was worth +0.17% of return.

    The fix is that the speed term is super-linear: the integral of (v/vref)^2
    over a fixed distance is proportional to v. If this test ever goes flat, the
    reward has quietly become a second progress term.
    """
    cfg = RewardConfig()
    speeds = [15.0, 22.0, 30.0, 40.0, 50.0]
    rs = [_return_over_fixed_distance(straight, cfg, v) for v in speeds]
    assert all(b > a for a, b in pairwise(rs)), f"return not rising with pace: {rs}"
    # And by a margin that matters, not a rounding error: v1's was 0.17%.
    gain = (rs[-1] - rs[0]) / abs(rs[0])
    assert gain > 0.5, f"pace gradient is only {gain:.1%} across 15->50 m/s"


def test_a_linear_speed_term_would_be_flat_in_pace(straight):
    """Negative control for the test above, and the reason ``speed_exp`` exists.

    With the exponent at 1.0 the speed term integrates to (w/vref) * distance —
    a constant, independent of pace. This reproduces v1's bug exactly, and it
    passing is what proves the previous test is measuring the exponent rather
    than something incidental about the setup.
    """
    cfg = RewardConfig(speed_exp=1.0, time_cost=0.0)
    speeds = [15.0, 22.0, 30.0, 40.0, 50.0]
    rs = [_return_over_fixed_distance(straight, cfg, v) for v in speeds]
    spread = (max(rs) - min(rs)) / abs(np.mean(rs))
    assert spread < 0.02, f"a linear term should be flat in pace, spread was {spread:.1%}"


def test_time_cost_alone_rewards_going_faster(straight):
    """The stopwatch must be real. Integrated over a fixed distance the time
    cost shrinks as pace rises — unlike v1's, which was a fixed tax."""
    cfg = RewardConfig(speed=0.0, throttle_commit=0.0)
    rs = [_return_over_fixed_distance(straight, cfg, v) for v in (15.0, 30.0, 50.0)]
    assert all(b > a for a, b in pairwise(rs))
    assert all(r < 0 for r in rs), "the time term is a cost, not a bonus"


def test_speed_term_saturates_at_the_cars_real_maximum(straight):
    """``speed_ref`` must be re-derived per car. Supra's 82 m/s against an
    ``evo_rally`` gear-limited to 57 would leave this term never approaching 1
    and the super-linear shape wasted in its flattest region."""
    cfg = RewardConfig()
    at_max = _terms(straight, cfg=cfg, speed=cfg.speed_ref)["speed"]
    expected = cfg.speed * CONTROL_DT
    assert at_max == pytest.approx(expected, rel=1e-6)
    # And it must not keep growing past what the car can physically reach.
    assert _terms(straight, cfg=cfg, speed=80.0)["speed"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# the straightness gate
# --------------------------------------------------------------------------- #

def test_gate_is_open_on_a_straight_and_shut_in_a_hairpin(pg, straight):
    """The gate is what stops the speed bonus paying for carrying stupid speed
    into a corner. If it does not close, the reward asks the car to be flat out
    everywhere and the first hairpin ends the episode."""
    cfg = RewardConfig()
    flat = straightness(np.zeros(6), cfg)
    assert flat == 1.0

    hairpin_curv = np.full(6, 1.0 / 12.0)          # a 12 m radius
    assert straightness(hairpin_curv, cfg) == 0.0

    sweeper = np.full(6, 1.0 / 200.0)              # 200 m radius: still straight
    assert straightness(sweeper, cfg) == 1.0


def test_gate_reads_the_worst_curvature_not_the_average(pg):
    """One hairpin at the end of an otherwise straight preview must close the
    gate. A mean would average it away and pay full speed bonus right up to the
    corner — which is precisely the mistake that puts a car in the trees."""
    cfg = RewardConfig()
    mostly_straight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 10.0])
    assert straightness(mostly_straight, cfg) == 0.0


def test_gate_reads_further_ahead_at_higher_speed(pg):
    """Speed-scaled look-ahead, matching the sensor suite's seconds convention.

    The gate closing earlier at high speed is what teaches the car to lift in
    time for the corner at the end of a long straight. With a fixed-metre
    horizon it would still be paying full speed bonus at the braking point.
    """
    cfg = RewardConfig()
    # Find the sharpest corner on the proving ground and stand well before it.
    i = int(np.argmax(np.abs(pg.curvature)))
    corner_s = float(pg.s[i])
    approach = corner_s - 70.0
    assert approach > 0.0, "fixture must have room before its sharpest corner"

    slow = _terms(pg, cfg=cfg, s=approach, speed=12.0).get("speed", 0.0)
    fast = _terms(pg, cfg=cfg, s=approach, speed=45.0).get("speed", 0.0)
    # The fast car previews far enough to see the corner; the slow one does not.
    assert slow > 0.0, "a slow car this far out should still be paid"
    assert fast == 0.0, "a fast car should already see the corner and be gated off"


def test_throttle_commit_requires_both_a_straight_road_and_a_pointed_car(straight):
    """Paying for full throttle while sideways is paying for a crash with extra
    steps, so the bonus is gated on alignment as well as the road."""
    cfg = RewardConfig()
    ok = _terms(straight, cfg=cfg, action=(0.0, 1.0, 0.0, 0.0), heading_error=0.0)
    assert ok.get("throttle_commit", 0.0) > 0.0

    sideways = _terms(straight, cfg=cfg, action=(0.0, 1.0, 0.0, 0.0),
                      heading_error=0.6)
    assert sideways.get("throttle_commit", 0.0) == 0.0

    lifting = _terms(straight, cfg=cfg, action=(0.0, 0.5, 0.0, 0.0))
    assert lifting.get("throttle_commit", 0.0) == 0.0

    # Throttle pinned *against the brakes* is a burnout, not commitment.
    burnout = _terms(straight, cfg=cfg, action=(0.0, 1.0, 0.8, 0.0))
    assert burnout.get("throttle_commit", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# landings
# --------------------------------------------------------------------------- #

def test_a_square_landing_beats_a_sideways_one():
    """North Star §7: reward a straight, settled landing; punish landing
    sideways or nose-first."""
    cfg = RewardConfig()
    square = landing_quality(slip=0.0, pitch=0.0, landing_g=1.0, cfg=cfg)
    sideways = landing_quality(slip=0.5, pitch=0.0, landing_g=1.0, cfg=cfg)
    nose = landing_quality(slip=0.0, pitch=-0.4, landing_g=1.0, cfg=cfg)
    harsh = landing_quality(slip=0.0, pitch=0.0, landing_g=14.0, cfg=cfg)

    assert square == pytest.approx(1.0)
    assert sideways == pytest.approx(-1.0)
    assert nose == pytest.approx(-1.0)
    assert harsh == pytest.approx(-1.0)


def test_landing_tail_first_is_not_punished_like_nose_first():
    """A rally car landing slightly tail-down is normal and fine; nose-first is
    what breaks the car. Punishing |pitch| would punish both."""
    cfg = RewardConfig()
    assert landing_quality(0.0, +0.3, 1.0, cfg) > landing_quality(0.0, -0.3, 1.0, cfg)
    assert landing_quality(0.0, +0.3, 1.0, cfg) == pytest.approx(1.0)


def test_airtime_itself_is_neutral(straight):
    """Deliberate: airtime already costs time, and paying for it would teach the
    car to hunt crests instead of pace. Only the landing is scored."""
    airborne = _terms(straight, speed=30.0, landed=False)
    assert "landing" not in airborne


def test_a_good_landing_off_the_road_does_not_pay(straight):
    """Otherwise a policy could farm landing bonuses on the verge, where there
    is no corridor to stay inside."""
    cfg = RewardConfig()
    on = _terms(straight, cfg=cfg, landed=True, lateral=0.0)["landing"]
    off = _terms(straight, cfg=cfg, landed=True, lateral=9.0)["landing"]
    assert on > 0.0
    assert off == 0.0
    # But a BAD landing still costs, wherever it happens.
    bad = _terms(straight, cfg=cfg, landed=True, lateral=9.0, slip=0.9)["landing"]
    assert bad < 0.0


# --------------------------------------------------------------------------- #
# terminals
# --------------------------------------------------------------------------- #

def test_finishing_faster_pays_more(pg):
    """The finish bonus is scaled by time against a reference pace, normalised
    by stage length so it is comparable across a 380 m tier-0 stage and a
    2300 m tier-5 one."""
    fn = RewardFunction(RewardConfig())
    quick = fn.terminal_terms(termination="finish", time_s=25.0, track=pg)
    slow = fn.terminal_terms(termination="finish", time_s=60.0, track=pg)
    assert quick["finish_pace"] > slow["finish_pace"]
    assert quick["finish"] == slow["finish"], "the base bonus is not time-scaled"


def test_timeout_is_not_punished(pg):
    """A timeout is a truncation, not a real ending — PPO bootstraps the value
    of the final state. Charging a penalty for the clock running out mid-stage
    would teach the policy that time limits are a failure it caused."""
    fn = RewardFunction(RewardConfig())
    assert fn.terminal_terms(termination="timeout", time_s=180.0,
                             track=pg)["timeout"] == 0.0


@pytest.mark.parametrize("term", ["crash", "off_course", "stuck", "spun"])
def test_real_failures_are_punished(pg, term):
    fn = RewardFunction(RewardConfig())
    assert fn.terminal_terms(termination=term, time_s=30.0, track=pg)[term] < 0.0


# --------------------------------------------------------------------------- #
# integration over whole episodes — the check the design turns on
# --------------------------------------------------------------------------- #

def _drive(stage, seed=0, cfg=None, offset_frac=None, cap=6000):
    env = RallyEnv(stage, EnvConfig(), reward_config=cfg or RewardConfig())
    env.reset(seed=seed)
    pilot = ReferencePilot(env.track)
    term = trunc = False
    steps = 0
    while not (term or trunc) and steps < cap:
        a = pilot.act(env.query, env.car)
        if offset_frac is not None:
            want = offset_frac * env.query.half_width
            a[0] = float(np.clip(a[0] - 0.9 * (want - env.query.lateral), -1, 1))
        _, _, term, trunc, info = env.step(a)
        steps += 1
    return env, info


def test_every_paid_term_is_booked_in_the_account():
    """The account is the instrument. A term paid into the reward but not booked
    is invisible to ``shares()``, which is how a term contributing 0.17% goes
    unnoticed for a whole project."""
    env, info = _drive(proving_ground())
    assert info["reward_terms"], "nothing was booked at all"
    assert env.account.total == pytest.approx(sum(info["reward_terms"].values()),
                                              abs=1e-6)


@pytest.mark.parametrize("tier", [0, 2, 4])
def test_pace_is_a_real_fraction_of_the_return(tier):
    """The number this whole design turns on.

    v1's pace share was 0.17% and it produced a driver that completed stages
    slowly — exactly what it had been asked for. Carson set the target at ~15%,
    within the 10-25% band. This asserts the band rather than the point value,
    because the share legitimately drifts with stage length: a longer stage
    accumulates more progress against a fixed finish bonus.
    """
    env, _ = _drive(generate(200 + tier, tier))
    share = env.account.pace_share()
    assert 0.08 < share < 0.30, f"tier {tier} pace share is {share:.1%}"


def test_progress_still_dominates_so_the_car_learns_to_finish_first():
    """Pace must not be so large that finishing becomes optional. Progress is
    the dense backbone that bootstraps everything; if pace outweighs it the
    policy has an incentive to drive fast at something other than the stage."""
    env, _ = _drive(generate(202, 2))
    shares = env.account.shares()
    assert shares["progress"] > 0.5


def test_hugging_the_corridor_edge_costs_return():
    """The known exploit, end to end.

    Deliberately drives a line that hugs the edge and confirms the return DROPS
    while the run still finishes. Testing an offset large enough to go
    off-course would prove only that the off-course terminal fires, which is a
    different mechanism and would let a broken taper pass.
    """
    stage = generate(201, 1)
    centre, info_c = _drive(stage, offset_frac=0.0)
    verge, info_v = _drive(stage, offset_frac=0.92)

    assert info_v["termination"] == "finish", "must stay on the road to be a fair test"
    assert info_c["termination"] == "finish"
    assert verge.account.total < centre.account.total, (
        f"riding the verge paid {verge.account.total:.1f} against "
        f"{centre.account.total:.1f} in the middle — the taper is not working"
    )


def test_a_reward_without_the_taper_would_let_the_verge_pay():
    """Positive control for the test above.

    With ``edge_band = 0`` the taper is off and the verge stops being punished
    by anything except path length. Without this control, the test above would
    pass just as happily if the difference came from somewhere incidental.
    """
    stage = generate(201, 1)
    naked = RewardConfig(edge_band=0.0)
    centre, _ = _drive(stage, cfg=naked, offset_frac=0.0)
    verge, info = _drive(stage, cfg=naked, offset_frac=0.92)

    # Asserted rather than branched on: an `if` here would let this whole test
    # pass silently the day the fixture changes and the run stops finishing.
    assert info["termination"] == "finish", "control run must finish to compare"

    c2, _ = _drive(stage, offset_frac=0.0)
    v2, _ = _drive(stage, offset_frac=0.92)
    tapered_gap = c2.account.total - v2.account.total
    naked_gap = centre.account.total - verge.account.total
    assert tapered_gap > naked_gap, (
        f"the taper must widen the penalty for hugging the edge: "
        f"{naked_gap:.1f} without it, {tapered_gap:.1f} with it"
    )


def _hold_speed(target_v, *, pinned, cfg=None, length=600.0):
    """Drive a straight at a held speed, two ways.

    ``pinned`` nails the throttle at 1.0 and modulates the brake to hold the
    speed; otherwise throttle and brake are used one at a time, as a driver
    would. Same stage, same distance, so the only thing that differs is how the
    pedals got there.
    """
    env = RallyEnv(flat_straight(length, width=10.0),
                   EnvConfig(max_time_s=400.0), reward_config=cfg or RewardConfig())
    env.reset(seed=0)
    term = trunc = False
    while not (term or trunc):
        err = target_v - env.car.speed
        if pinned:
            throttle, brake = 1.0, float(np.clip(-err * 0.8 + 0.02, 0.0, 1.0))
        else:
            throttle = float(np.clip(err * 0.8, 0.0, 1.0))
            brake = float(np.clip(-err * 0.8, 0.0, 1.0))
        steer = float(np.clip(-0.6 * env.query.lateral
                              - 1.2 * env.query.heading_error, -1, 1))
        _, _, term, trunc, info = env.step(
            np.array([steer, throttle, brake, 0.0], np.float32))
    return env, info, env.max_s / max(env.time_s, 1e-9)


def test_dragging_the_brakes_at_full_throttle_does_not_out_earn_driving():
    """``throttle_commit`` reads the throttle. It must also read the brake.

    Without the brake gate a policy can pin the throttle, drag the brakes to
    hold whatever speed it likes, and collect a flat 2.4/s the whole way — while
    going no faster. Measured at the time: **+7.9% return at a matched 21.7 m/s**
    against an honest run at 21.1. The commit term accounted for essentially all
    of it (57.0 against 10.2).

    Compared at matched *average speed* rather than matched target, because the
    two controllers reach different speeds from the same target and comparing
    targets would compare two different drives.
    """
    honest, h_info, h_v = _hold_speed(30.0, pinned=False)
    pinned, p_info, p_v = _hold_speed(6.0, pinned=True)

    assert h_info["termination"] == p_info["termination"] == "finish"
    # The pinned run must not be the faster one, or this compares pace, not the
    # exploit. Asserted rather than skipped: if the physics changes so a burnout
    # outruns an honest drive, this test should fail loudly, not quietly pass.
    assert p_v <= h_v + 0.5, (
        f"pinned run averaged {p_v:.1f} m/s against {h_v:.1f} — not a fair "
        f"comparison of anything but speed"
    )
    assert pinned.account.total < honest.account.total, (
        f"holding the brakes at full throttle paid {pinned.account.total:.1f} "
        f"at {p_v:.1f} m/s against {honest.account.total:.1f} at {h_v:.1f} m/s "
        f"— the commit bonus is paying for a burnout"
    )


def test_a_commit_bonus_blind_to_the_brake_would_pay_for_a_burnout():
    """Negative control for the test above.

    Reopens the gate with ``commit_brake_max = 1.0`` and confirms the exploit
    comes back. Without this, the test above would pass just as happily if the
    penalty came from somewhere incidental — which is exactly how the first
    verge-riding test managed to measure nothing at all.
    """
    blind = RewardConfig(commit_brake_max=1.0)
    honest, _, h_v = _hold_speed(30.0, pinned=False, cfg=blind)
    pinned, _, p_v = _hold_speed(6.0, pinned=True, cfg=blind)

    assert p_v <= h_v + 0.5
    assert pinned.account.total > honest.account.total, (
        "with the brake gate open, a burnout should out-earn an honest drive; "
        "if it does not, this control proves nothing about the gate"
    )
    assert pinned.account.snapshot()["throttle_commit"] > \
        3.0 * honest.account.snapshot()["throttle_commit"]
