"""Jigglypuff Rest lab, and the reward contract that replaced its shaping.

The Rest bonuses these tests once asserted were removed after they were measured
driving a policy collapse; the numbers are recorded in the tests below."""

import pytest
import melee
from melee_lab.state import reward
from melee_lab.rest_lab import get_puff_config


class DummyPlayer:
    def __init__(self, character=melee.Character.JIGGLYPUFF, action=melee.Action.STANDING,
                 percent=0.0, stock=3, off_stage=False, on_ground=True, invulnerable=False,
                 hitstun=0, hitlag=0, pos_x=0.0, pos_y=0.0, speed_y_attack=0.0):
        self.character = character
        self.action = action
        self.percent = percent
        self.stock = stock
        self.off_stage = off_stage
        self.on_ground = on_ground
        self.invulnerable = invulnerable
        self.hitstun_frames_left = hitstun
        self.hitlag_left = hitlag
        self.position = type('Pos', (), {'x': pos_x, 'y': pos_y})()
        self.speed_y_attack = speed_y_attack


class DummyGameState:
    def __init__(self, p1, p2, frame=1000):
        self.players = {1: p1, 2: p2}
        self.frame = frame


def test_the_reward_no_longer_carries_character_specific_bonuses():
    """These used to assert a +4.0 bonus for a connecting Rest, +1.0 for an up-throw
    confirm, +0.8 for a tech-chase confirm, a -1.5 whiff penalty, and Fox equivalents for
    waveshine, powershield and pivots.

    All of them are gone, and the measurement that removed them is worth recording. They
    fired per frame or per transition with no per-episode limit, so across a 10,000-frame
    match they dwarfed the objective. On a 134-match run:

        mean episode return for a LOSS    +226.71
        mean episode return for a WIN     +143.29
        mean episode return for a TIMEOUT +699.60

    PPO did exactly what that asked: it learned to stall, froze every controller axis but
    the shield, held hard shield 63% of frames on stage, grew median match length from
    9,505 to 18,573 frames, and fell from CPU 5 to CPU 1 -- a level its own starting
    checkpoint had won 10-0.

    If character-specific incentives come back, they must be bounded per episode and land
    with a measurement showing they help. The old ones never had either."""
    rest_hit_before = DummyGameState(
        DummyPlayer(character=melee.Character.JIGGLYPUFF, action=melee.Action.STANDING, percent=0.0),
        DummyPlayer(character=melee.Character.FOX, action=melee.Action.STANDING, percent=0.0))
    rest_hit_after = DummyGameState(
        DummyPlayer(character=melee.Character.JIGGLYPUFF, action=melee.Action.DOWN_B_STUN, percent=0.0),
        DummyPlayer(character=melee.Character.FOX, action=melee.Action.DAMAGE_FLY_TOP, percent=28.0, hitstun=30))
    r = reward(rest_hit_before, rest_hit_after)
    # Damage only. No move is worth more than the stock it is supposed to earn.
    assert r == pytest.approx(0.015 * 28.0, abs=1e-6), f'expected damage only, got {r}'

    # A whiffed Rest is likewise neither rewarded nor punished beyond what it costs in
    # damage and stocks, which the opponent collects on its own.
    whiff_after = DummyGameState(
        DummyPlayer(character=melee.Character.JIGGLYPUFF, action=melee.Action.DOWN_B_STUN, percent=0.0),
        DummyPlayer(character=melee.Character.FOX, action=melee.Action.STANDING, percent=0.0))
    assert reward(rest_hit_before, whiff_after) == pytest.approx(0.0, abs=1e-6)


def test_the_reward_treats_every_character_identically():
    """The old version branched on the agent's character, and its `else` arm applied Fox's
    incentives to Marth, Peach and everyone else by default."""
    def pair(character, action):
        before = DummyGameState(DummyPlayer(character=character, action=melee.Action.STANDING),
                                DummyPlayer(character=melee.Character.FOX, action=melee.Action.STANDING))
        after = DummyGameState(DummyPlayer(character=character, action=action),
                               DummyPlayer(character=melee.Character.FOX, action=melee.Action.STANDING, percent=12.0))
        return reward(before, after)

    values = [pair(ch, melee.Action.DOWN_B_STUN) for ch in
              (melee.Character.JIGGLYPUFF, melee.Character.FOX, melee.Character.MARTH,
               melee.Character.PEACH, melee.Character.FALCO)]
    assert len(set(round(v, 9) for v in values)) == 1, f'character-dependent reward: {values}'
    assert values[0] == pytest.approx(0.015 * 12.0)


def test_puff_config_preset():
    """Verify get_puff_config correctly configures a Jigglypuff training environment."""
    cfg = get_puff_config(opponent='FOX', stage='FINAL_DESTINATION', cpu_level=3)
    assert cfg.character == 'JIGGLYPUFF'
    assert cfg.opponent == 'FOX'
    assert cfg.stage == 'FINAL_DESTINATION'
    assert cfg.cpu_level == 3
    assert cfg.action_set == 'controller'
    assert cfg.action_frames == 1
