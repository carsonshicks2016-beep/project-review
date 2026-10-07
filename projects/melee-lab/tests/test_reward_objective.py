"""The objective must be the objective: a win has to be worth more than a loss.

Nothing checked this before. A 134-match run measured mean episode return +226.71 for a
LOSS against +143.29 for a WIN, and +699.60 for running out the clock. PPO optimised it
faithfully: it learned to turtle, froze every controller axis except the shield, and was
demoted from CPU 5 to CPU 1 -- a level the policy it started from had won 10-0.
"""
import numpy as np
import pytest

from melee_lab.state import (DAMAGE, OUTCOME, RECOVERY_POTENTIAL, STOCK,
                             episode_bound, reward)


class P:
    def __init__(self, stock=3, percent=0.0, x=0.0, y=0.0):
        self.stock, self.percent = stock, percent
        self.position = type('Pos', (), {'x': x, 'y': y})()


class G:
    def __init__(self, p1, p2):
        self.players = {1: p1, 2: p2}


def episode(states):
    """Total reward across a sequence of gamestates."""
    return sum(reward(a, b) for a, b in zip(states, states[1:]))


def test_a_win_beats_a_loss_however_long_the_match_is():
    """The failure that cost a training run: shaping accumulated per frame, so dragging
    a match out and losing paid more than winning quickly."""
    # A short, decisive win: take three stocks, lose none.
    win = episode([G(P(3), P(3 - i)) for i in range(4)])
    # The most favourable possible loss, stretched over a very long match: every frame
    # the agent is alive and moving around the stage, for 20,000 frames.
    long_loss = [G(P(3), P(3))]
    for i in range(20_000):
        long_loss.append(G(P(3, x=(i % 120) - 60.0), P(3)))
    for i in range(1, 4):                       # then it loses all three stocks
        long_loss.append(G(P(3 - i), P(3)))
    loss = episode(long_loss)

    assert win > 0 > loss, f'win {win:.2f}, loss {loss:.2f}'
    assert win > loss, 'a win must outscore a loss at ANY match length'


def test_no_loop_of_states_can_be_farmed():
    """Potential-based shaping telescopes, so returning to a state you have already
    visited pays nothing. The old departure penalty and return bonus paid +/-1.5 on every
    crossing, about 15 crossings a match -- +/-22 against a 12-point stock signal."""
    # Walk out to sea and back, a hundred times, taking no damage and losing no stocks.
    states = [G(P(3), P(3))]
    for _ in range(100):
        for x in (0.0, 60.0, 130.0, 200.0, 130.0, 60.0, 0.0):
            states.append(G(P(3, x=x), P(3)))
    assert abs(episode(states)) < 1e-6, 'a closed loop of states must pay nothing'


def test_every_episode_is_bounded():
    """Unbounded accumulation is what let shaping reach ten times the outcome signal."""
    states = [G(P(3), P(3))]
    for i in range(50_000):
        states.append(G(P(3, x=float((i * 37) % 400 - 200), y=-float(i % 150)),
                        P(3, x=float((i * 11) % 300 - 150))))
    total = episode(states)
    assert abs(total) <= episode_bound(), f'{total:.2f} exceeds the bound {episode_bound():.2f}'
    # And the bound itself must stay under what a decisive outcome is worth, or the
    # outcome stops being the thing PPO is chasing.
    assert episode_bound() < 2 * (OUTCOME + STOCK * 3)


def test_stocks_and_damage_keep_their_relative_worth():
    """Damage is a dense hint, not a substitute for taking a stock."""
    stock = reward(G(P(3), P(3)), G(P(3), P(2)))
    hit = reward(G(P(3), P(3, percent=0.0)), G(P(3), P(3, percent=10.0)))
    assert stock == pytest.approx(STOCK)
    assert hit == pytest.approx(DAMAGE * 10)
    assert stock > hit * 10, 'a stock must dominate any plausible single exchange'


def test_a_respawn_is_not_a_hit():
    """Percent resets to zero on death; read naively that is a 100%+ hit landed."""
    assert reward(G(P(3), P(2, percent=140.0)), G(P(3), P(1, percent=0.0))) == pytest.approx(STOCK)
    assert reward(G(P(2, percent=140.0), P(3)), G(P(1, percent=0.0), P(3))) == pytest.approx(-STOCK)


def test_dying_offstage_is_not_rewarded_for_the_trip_home():
    """The respawn platform is at the centre of the stage. Without skipping the potential
    across a death, dying far out would pay for the teleport back."""
    died_far_out = reward(G(P(3, x=250.0, y=-180.0), P(3)), G(P(2, x=0.0, y=0.0), P(3)))
    assert died_far_out == pytest.approx(-STOCK), f'got {died_far_out:.3f}'


def test_the_same_function_scores_either_port():
    """Replays put the pro on an arbitrary port, and a mirror is scored twice. A value
    head fitted to a different reward than PPO optimises is worse than none."""
    before, after = G(P(3), P(3)), G(P(3), P(2))
    assert reward(before, after, 1, 2) == pytest.approx(-reward(before, after, 2, 1))
