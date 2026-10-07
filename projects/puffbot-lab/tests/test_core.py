"""Unit tests that need no emulator. Each guards a lesson from version one."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from puffbot import actions, league, observation, reward
from puffbot.config import Config
from puffbot.dolphin import Setup, gecko_codes
from puffbot.model import MASKED, Actor, Policy, load_policy, save_policy
from puffbot.vtrace import ppo_loss, vtrace

GAMMA = 0.997


def player(x=0.0, y=0.0, percent=0.0, stock=4, on_ground=True, off_stage=False, action=0x0E,
           hitstun=0, character=15, facing=True):
    return NS(position=NS(x=x, y=y), percent=percent, stock=stock, facing=facing, on_ground=on_ground,
              off_stage=off_stage, jumps_left=6, shield_strength=60.0, invulnerable=False, hitlag_left=0,
              hitstun_frames_left=hitstun, action_frame=1, speed_air_x_self=0.0, speed_y_self=0.0,
              speed_x_attack=0.0, speed_y_attack=0.0, speed_ground_x_self=0.0,
              action=NS(value=action), character=NS(value=character))


def state(p1, p2, frame=100, projectiles=()):
    return NS(players={1: p1, 2: p2}, frame=frame, projectiles=list(projectiles))


# ------------------------------------------------------------------- actions
def test_vocabulary_is_unique_and_small():
    assert len(set(actions.NAMES)) == actions.COUNT
    assert 30 <= actions.COUNT <= 60


def all_frames(m):
    return list(m.script) + list(m.airborne) + [m.hold]


def test_no_shield_click_is_ever_legal_where_an_air_dodge_could_kill():
    """v1: 68-87% of Puff's deaths were self-destructs from shield-in-the-air air-dodges.
    Shield clicks are legal on the ground/ledge (shield, rolls) and, since step 2, for
    techs while thrown or low over the stage -- never offstage or high in the air."""
    clickers = [i for i, m in enumerate(actions.MACROS)
                if any('BUTTON_L' in f.buttons or 'BUTTON_R' in f.buttons for f in all_frames(m))]
    assert {actions.MACROS[i].rule for i in clickers} == {'grounded'}, 'techs click only via the executor'
    for spot in (player(x=-120, y=-20, on_ground=False, off_stage=True),      # offstage
                 player(x=-100, y=40, on_ground=False, off_stage=True),       # past the ledge
                 player(x=0, y=60, on_ground=False),                          # high over stage
                 player(x=-84, y=5, on_ground=False)):                        # right at the ledge
        mask = actions.legal_mask(spot)
        assert not mask[clickers].any(), spot.position


def test_tech_is_legal_while_thrown_or_low_over_the_stage():
    low = actions.legal_mask(player(x=10, y=8, on_ground=False, action=0x58))
    assert low[actions.INDEX['tech']] and low[actions.INDEX['tech_left']]
    near_edge = actions.legal_mask(player(x=70, y=8, on_ground=False, action=0x58))
    assert near_edge[actions.INDEX['tech']] and not near_edge[actions.INDEX['tech_right']], 'rolls need room'
    thrown = actions.legal_mask(player(x=80, y=0, on_ground=False, action=0xF2))
    assert thrown[actions.INDEX['tech']] and thrown[actions.INDEX['tech_left']]
    for ordinary in (player(), player(x=10, y=8, on_ground=False, action=0x1B)):   # standing, jumping
        assert not actions.legal_mask(ordinary)[[actions.INDEX[n] for n in ('tech', 'tech_left', 'tech_right')]].any()


def test_auto_lcancel_is_a_partial_press_only_during_falling_aerials():
    """Measured in Dolphin: a partial press halves landing lag and never air-dodges."""
    ex = actions.Executor(3)
    ex.start(actions.INDEX['noop'])
    nair_falling = player(on_ground=False, y=10, action=0x41)
    nair_falling.speed_y_self = -1.5
    frames = [ex.next_input(nair_falling) for _ in range(6)]
    assert any(f.trigger == actions.LCANCEL_ANALOG for f in frames)
    assert all('BUTTON_L' not in f.buttons for f in frames), 'never the digital click'
    assert any(f.trigger == 0 for f in frames), 'alternating, so each pulse is a new press'
    rising = player(on_ground=False, y=10, action=0x41)
    rising.speed_y_self = 1.0
    ex.start(actions.INDEX['noop'])
    assert all(ex.next_input(rising).trigger == 0 for _ in range(4))
    off = actions.Executor(3, auto_lcancel=False)
    off.start(actions.INDEX['noop'])
    assert all(off.next_input(nair_falling).trigger == 0 for _ in range(4))


def test_no_macro_touches_start_or_menus():
    """v1: a policy with the whole controller paused games and wedged character select."""
    for m in actions.MACROS:
        for f in all_frames(m):
            assert not set(f.buttons) & {'BUTTON_START', 'BUTTON_D_UP', 'BUTTON_D_DOWN', 'BUTTON_D_LEFT', 'BUTTON_D_RIGHT'}


def test_masks():
    air = actions.legal_mask(player(y=30, on_ground=False))
    for n in ('shield', 'roll_left', 'spotdodge', 'grab'):
        assert not air[actions.INDEX[n]]
    assert air[actions.INDEX['rest']] and air[actions.INDEX['jump']]
    ledge = actions.legal_mask(player(x=-86, y=-5, on_ground=False, action=0xFD))
    assert ledge[actions.INDEX['shield']]
    offstage = actions.legal_mask(player(x=-120, y=-20, on_ground=False, off_stage=True))
    assert not offstage[actions.INDEX['rest']]
    assert offstage[actions.INDEX['pound_right']]
    standing = actions.legal_mask(player())
    assert standing[~(actions.TECH | actions.TECH_ROLL)].all() and not standing[actions.TECH | actions.TECH_ROLL].any()


def run(macro, ground_frames=0, max_frames=20):
    ex = actions.Executor(3)
    ex.start(actions.INDEX[macro])
    out = []
    for i in range(max_frames):
        p = player(on_ground=i < ground_frames)
        out.append(ex.next_input(p))
        if ex.done:
            break
    return out


def test_short_hop_is_one_frame_of_jump():
    frames = run('short_hop')
    assert frames[0].buttons == ('BUTTON_Y',)
    assert all('BUTTON_Y' not in f.buttons for f in frames[1:])
    assert len(frames) == 3


def test_full_jump_holds_through_jumpsquat():
    frames = run('jump')
    assert sum('BUTTON_Y' in f.buttons for f in frames) == 6


def test_short_hop_aerial_waits_for_airborne():
    frames = run('sh_c_left', ground_frames=5)
    assert frames[0].buttons == ('BUTTON_Y',)
    first_c = next(i for i, f in enumerate(frames) if f.cstick != actions.NEUTRAL)
    assert first_c == 5
    assert frames[first_c].cstick == actions.LEFT


def test_c_stick_releases_so_it_can_flick_again():
    frames = run('c_left')
    assert frames[0].cstick == actions.LEFT and frames[-1].cstick == actions.NEUTRAL


def test_every_macro_finishes():
    for name in actions.NAMES:
        frames = run(name, ground_frames=3, max_frames=40)
        assert 3 <= len(frames) <= actions.MAX_FRAMES, name


# -------------------------------------------------------------------- reward
def test_result_rules():
    g = state(player(stock=2), player(stock=1))
    assert reward.result_of(g, 1) == 'win' and reward.result_of(g, 2) == 'loss'
    g = state(player(stock=1, percent=30), player(stock=1, percent=80))
    assert reward.result_of(g, 1) == 'win'


def episode_return(frames, me=1):
    total, disc = 0.0, 1.0
    for a, b in zip(frames, frames[1:]):
        total += disc * reward.frame_reward(a, b, me, GAMMA)
        disc *= GAMMA
    return total


def test_offstage_shaping_cannot_be_farmed():
    """v1: bounded-but-farmable crossing bonuses made a TIMEOUT pay +699 vs +143 for a win."""
    on = player(x=0.0)
    off = player(x=-150.0, y=-30.0, on_ground=False, off_stage=True)
    frames = [state(on, player())]
    for _ in range(50):
        frames += [state(off, player()), state(on, player())]
    assert episode_return(frames) <= 0.0


def test_death_offstage_does_not_refund_potential():
    off = player(x=-160.0, y=-60.0, on_ground=False, off_stage=True)
    dead = player(x=0.0, y=0.0, stock=3, action=0x0B)
    r = reward.frame_reward(state(off, player()), state(dead, player()), 1, GAMMA)
    assert r == pytest.approx(-reward.STOCK)


def test_clean_win_positive_clean_loss_negative():
    """v1's first reward paid a LOSS +226 and a WIN +143. Here stocks and the result
    decide the sign whenever damage is even, and nothing else accumulates per frame."""
    def game(my_stocks_lost, their_stocks_lost, result):
        return (reward.STOCK * (their_stocks_lost - my_stocks_lost)
                + reward.DAMAGE * (400 - 400) + reward.result_reward(result))
    assert game(0, 4, 'win') > 0 > game(4, 0, 'loss')
    assert game(3, 4, 'win') > game(4, 3, 'loss')
    # A stock is worth 100% of damage; the offstage potential is bounded by one stock.
    assert reward.OFFSTAGE * 250 <= reward.STOCK


def test_damage_not_credited_across_respawn():
    a = state(player(), player(percent=120, stock=3))
    b = state(player(), player(percent=0, stock=2))
    assert reward.frame_reward(a, b, 1, GAMMA) == pytest.approx(reward.STOCK)


def test_self_destruct_classifier():
    s = reward.GameStats(1)
    walk = [state(player(x=-80 - i), player(), frame=i) for i in range(10)]
    dead = state(player(x=-200, y=-200, stock=3, on_ground=False, off_stage=True), player(), frame=200)
    for a, b in zip(walk, walk[1:] + [dead]):
        s.update(a, b)
    assert s.self_destructs == 1 and s.lost == 1
    t = reward.GameStats(1)
    hit = state(player(x=-60, percent=90, hitstun=20, on_ground=False), player(), frame=1)
    flying = state(player(x=-180, y=60, percent=90, hitstun=5, on_ground=False, off_stage=True), player(), frame=100)
    gone = state(player(x=-250, y=80, stock=3, on_ground=False, off_stage=True), player(), frame=160)
    t.update(state(player(x=-60), player(), frame=0), hit)
    t.update(hit, flying)
    t.update(flying, gone)
    assert t.self_destructs == 0 and t.lost == 1


# --------------------------------------------------------------- observation
def test_encoding_shapes_and_perspective():
    a = player(x=-30, y=5, percent=40, character=15)
    b = player(x=40, percent=10, character=2, facing=False)
    f1, i1 = observation.encode(state(a, b), 1, 3, 28800)
    f2, i2 = observation.encode(state(b, a), 2, 3, 28800)
    assert f1.shape == (observation.FLOATS,) and i1.shape == (observation.IDS,)
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_array_equal(i1, i2)
    assert i1[2] == 15 and i1[3] == 2
    assert np.isfinite(f1).all()


def test_projectiles_owned_by_self_are_ignored():
    mine = NS(owner=1, position=NS(x=5, y=0))
    theirs = NS(owner=2, position=NS(x=20, y=10))
    f, _ = observation.encode(state(player(), player(x=50), projectiles=[mine, theirs]), 1, -1, 28800)
    o = 2 * observation.PER_PLAYER + observation.GLOBALS
    assert f[o] == pytest.approx(0.2) and f[o + 2] == 1.0 and f[o + 5] == 0.0


# --------------------------------------------------------------- learning math
def test_vtrace_on_policy_equals_lambda_returns():
    torch.manual_seed(0)
    N, T, lam = 3, 7, 0.9
    values = torch.randn(N, T)
    boot = torch.randn(N)
    rewards = torch.randn(N, T)
    disc = torch.full((N, T), 0.95)
    disc[1, 3] = 0.0
    vs, adv = vtrace(values, boot, rewards, disc, torch.zeros(N, T), lam)
    # Reference: TD(lambda) returns computed backwards.
    ref = torch.empty(N, T)
    nxt_v, nxt_g = boot.clone(), boot.clone()
    for t in range(T - 1, -1, -1):
        g = rewards[:, t] + disc[:, t] * ((1 - lam) * nxt_v + lam * nxt_g)
        ref[:, t] = g
        nxt_v, nxt_g = values[:, t], g
    torch.testing.assert_close(vs, ref, rtol=1e-5, atol=1e-5)


def test_vtrace_truncates_large_ratios():
    values = torch.zeros(1, 4)
    vs_on, _ = vtrace(values, torch.zeros(1), torch.ones(1, 4), torch.full((1, 4), 0.9), torch.zeros(1, 4), 1.0)
    vs_off, _ = vtrace(values, torch.zeros(1), torch.ones(1, 4), torch.full((1, 4), 0.9), torch.full((1, 4), 3.0), 1.0)
    torch.testing.assert_close(vs_on, vs_off)


def test_masked_actions_are_never_sampled():
    actor = Actor(64)
    mask = np.zeros((1, actions.COUNT), bool)
    mask[0, [2, 7]] = True
    rng = np.random.default_rng(0)
    floats = np.zeros((1, observation.FLOATS), np.float32)
    ids = np.zeros((1, observation.IDS), np.int64)
    seen = {int(actor.act(floats, ids, mask, rng)[0][0]) for _ in range(200)}
    assert seen <= {2, 7}


def test_initial_policy_is_near_uniform():
    net = Policy(64)
    mask = torch.ones(8, actions.COUNT, dtype=torch.bool)
    logits, value = net(torch.randn(8, observation.FLOATS), torch.zeros(8, 4, dtype=torch.long), mask)
    p = torch.softmax(logits, -1)
    assert p.max() < 3.0 / actions.COUNT
    assert value.shape == (8,)


def test_ppo_loss_backprops_and_ignores_masked():
    net = Policy(64)
    B = 32
    mask = torch.ones(B, actions.COUNT, dtype=torch.bool)
    mask[:, 0] = False
    logits, v = net(torch.randn(B, observation.FLOATS), torch.zeros(B, 4, dtype=torch.long), mask)
    acts = torch.randint(1, actions.COUNT, (B,))
    prox = torch.log_softmax(logits.detach(), -1).gather(-1, acts[:, None]).squeeze(-1)
    loss, st = ppo_loss(logits, v, acts, prox, torch.randn(B), torch.randn(B), 0.2, 0.5, 0.01)
    loss.backward()
    assert math.isfinite(loss.item()) and st['clip_frac'] == 0.0
    assert logits[0, 0].item() <= MASKED / 2


def test_policy_file_roundtrip(tmp_path):
    net = Policy(64)
    save_policy(tmp_path / 'p.pt', net, 5, 1234)
    payload = load_policy(tmp_path / 'p.pt')
    assert payload['version'] == 5 and payload['frames'] == 1234
    other = Policy(64)
    other.load_state_dict(payload['state'])


# -------------------------------------------------------------------- league
def cfg(**kw):
    return Config(**{**dict(iso='/dev/null', ladder_window=10), **kw})


def test_ladder_promotes_and_demotes():
    lad = league.Ladder(cfg(cpu_start_level=3))
    for _ in range(9):
        assert lad.record(3, 'win') is None
    assert lad.record(3, 'win') == 'promoted' and lad.frontier == 4
    for _ in range(10):
        lad.record(4, 'loss')
    assert lad.frontier == 3
    # Games at other levels never move the frontier.
    for _ in range(30):
        lad.record(8, 'win')
    assert lad.frontier == 3
    restored = league.Ladder(cfg(), lad.to_dict())
    assert restored.frontier == 3


def test_choose_respects_selfplay_fraction():
    rng = np.random.default_rng(0)
    c = cfg(selfplay_fraction=0.0)
    picks = [league.choose(c, {'frontier': 5, 'snapshots': []}, rng) for _ in range(200)]
    assert all(p.kind == 'cpu' and 3 <= p.setup.cpu_level <= 6 for p in picks)
    c = cfg(selfplay_fraction=1.0)
    picks = [league.choose(c, {'frontier': 5, 'snapshots': []}, rng) for _ in range(50)]
    assert all(p.kind == 'mirror' and p.setup.cpu_level == 0 for p in picks)


def test_wilson_interval():
    lo, hi = league.wilson(7, 10)
    assert lo < 0.7 < hi and hi - lo > 0.4
    lo, hi = league.wilson(700, 1000)
    assert hi - lo < 0.06


# ---------------------------------------------------------------- config etc.
def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text(json.dumps({'wokers': 4}))
    with pytest.raises(ValueError):
        Config.load(p)


def test_config_validation(tmp_path):
    iso = tmp_path / 'melee.iso'
    iso.write_bytes(b'x')
    Config(iso=str(iso)).validate()
    with pytest.raises(ValueError):
        Config(iso=str(iso), workers=0).validate()
    with pytest.raises(ValueError):
        Config(iso=str(iso), cpu_start_level=0).validate()


def test_gecko_codes_disable_random_stage():
    text = gecko_codes()
    assert '$Optional: Extract Menu Info' in text and '$Optional: Instant Match' in text
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('$Optional: Instant Match ['))
    assert lines[i + 3].split()[0] == '00000000'


def test_setup_kind():
    assert Setup('JIGGLYPUFF', 'FOX', 9).is_cpu and not Setup('JIGGLYPUFF', 'JIGGLYPUFF', 0).is_cpu


def test_beating_level_nine_no_longer_floods_training_with_dittos():
    """GPT audit: 'mastered' used to switch 60% of games to Puff self-play for good."""
    lad = league.Ladder(cfg(cpu_start_level=9, selfplay_fraction=0.2))
    moved = [lad.record(9, 'win', 'FOX') for _ in range(10)]
    assert moved[-1] == 'mastered' and lad.selfplay_fraction() == 0.2
    for _ in range(10):
        lad.record(9, 'loss', 'FALCO')
    assert not lad.mastered, 'mastery is re-judged, not sticky'


def test_losing_matchups_get_more_games_but_none_disappear():
    lad = league.Ladder(cfg(cpu_start_level=9, opponents={'FOX': 1.0, 'FALCO': 1.0, 'MARTH': 1.0}))
    for _ in range(20):
        lad.record(9, 'win', 'FOX')
        lad.record(9, 'loss', 'FALCO')
    w = lad.opponent_weights()
    assert w['FALCO'] == pytest.approx(2.0) and w['FOX'] == pytest.approx(0.5)
    assert w['MARTH'] == pytest.approx(2.0), 'unmeasured matchups are sampled until measured'
    rng = np.random.default_rng(0)
    table = {'frontier': 9, 'snapshots': [], 'opponent_weights': w}
    picks = [league.choose(cfg(selfplay_fraction=0.0, opponents={'FOX': 1.0, 'FALCO': 1.0, 'MARTH': 1.0}),
                           table, rng).setup.p2 for _ in range(900)]
    assert picks.count('FALCO') > 3 * picks.count('FOX') > 0
    restored = league.Ladder(cfg(opponents={'FOX': 1.0, 'FALCO': 1.0, 'MARTH': 1.0}), lad.to_dict())
    assert restored.opponent_weights() == w


def test_games_far_below_the_frontier_do_not_move_matchup_focus():
    lad = league.Ladder(cfg(cpu_start_level=9, opponents={'FALCO': 1.0}))
    for _ in range(20):
        lad.record(3, 'win', 'FALCO')
    assert lad.focus('FALCO') == league.FOCUS_CEILING


# ------------------------------------------------------- macros run back to back
def chain(names, air=True):
    """Frames sent for a sequence of macros, as (macro, Pad) pairs."""
    ex = actions.Executor(3)
    out = []
    for n in names:
        ex.start(actions.INDEX[n])
        while True:
            out.append((n, ex.next_input(player(on_ground=not air, y=20 if air else 0))))
            if ex.done:
                break
    return out


def presses(frames, button):
    """How many times `button` goes from up to down."""
    held, count = False, 0
    for _, f in frames:
        now = button in f.buttons
        count += now and not held
        held = now
    return count


def test_consecutive_jumps_each_press_the_button():
    """GPT audit, 2026-09-25: two full jumps held Y for twelve frames, so the second
    mid-air jump never happened. Every jump-type macro must produce its own press."""
    for seq in (['jump', 'jump'], ['jump', 'short_hop'], ['jump', 'sh_c_left'],
                ['jump_left', 'jump_right', 'jump'], ['short_hop', 'short_hop']):
        assert presses(chain(seq), 'BUTTON_Y') == len(seq), seq
    frames = chain(['jump', 'jump'])
    second = [f for n, f in frames[6:]]
    assert 'BUTTON_Y' not in second[0].buttons, 'one release frame first'
    assert sum('BUTTON_Y' in f.buttons for f in second) == 6, 'the full hop itself is unchanged'


def test_repeated_attacks_and_aerials_re_press():
    assert presses(chain(['sh_nair', 'a']), 'BUTTON_A') == 2
    assert presses(chain(['pound_left', 'pound_left']), 'BUTTON_B') == 2
    frames = chain(['sh_c_left', 'c_left'])
    flicks = sum(1 for (_, a), (_, b) in zip([(None, actions.Pad())] + frames, frames)
                 if b.cstick != actions.NEUTRAL and a.cstick == actions.NEUTRAL)
    assert flicks == 2


def test_shield_stays_held_into_a_roll():
    frames = chain(['shield', 'roll_left', 'shield'], air=False)
    assert all('BUTTON_L' in f.buttons for _, f in frames), 'no shield drop between shield macros'


def test_no_release_frame_when_nothing_is_held():
    frames = chain(['walk_left', 'short_hop'], air=False)
    assert [n for n, _ in frames].count('short_hop') == 3
    ex = actions.Executor(3)
    ex.start(actions.INDEX['jump'])
    for _ in range(6):
        ex.next_input(player(on_ground=False, y=20))
    ex.released()          # pads were released between games
    ex.start(actions.INDEX['jump'])
    assert 'BUTTON_Y' in ex.next_input(player(on_ground=False, y=20)).buttons


# ---------------------------------------------------------------- compatibility
def test_old_checkpoints_still_load_with_a_note(tmp_path):
    """Files saved before the richer contract carried only the (43) macro names; they
    load through the step-2 conversion, with notes saying so."""
    from puffbot import contract
    old = {'space': 'puff-macros-v1', 'names': list(actions.V1_NAMES)}
    with pytest.raises(contract.IncompatibleCheckpoint):
        contract.check(old, act_every=3)
    notes = contract.check(old, act_every=3, migrate=True)
    assert any('converted' in n for n in notes) and any('executor' in n for n in notes)


def test_contract_refuses_a_different_vocabulary_or_layout():
    from puffbot import contract
    wrong = {'space': 'puff-macros-v1', 'names': list(actions.NAMES)[::-1]}
    with pytest.raises(contract.IncompatibleCheckpoint):
        contract.check(wrong)
    newer = contract.current(3)
    newer['observation'] = {**newer['observation'], 'version': 'puff-obs-v9'}
    with pytest.raises(contract.IncompatibleCheckpoint):
        contract.check(newer)


def test_policy_files_record_timing_and_warn_on_mismatch(tmp_path):
    net = Policy(64)
    save_policy(tmp_path / 'p.pt', net, 1, 10, act_every=3)
    assert load_policy(tmp_path / 'p.pt', act_every=3)['notes'] == []
    assert load_policy(tmp_path / 'p.pt', act_every=2)['notes']


# ------------------------------------------------------------------ evaluation
def test_quota_tops_up_batches_that_end_short():
    """GPT audit: a Sudden Death timeout ended a five-game batch after one game and the
    report still said finished. Short batches are now replaced."""
    from puffbot.evaluate import Quota
    q = Quota(['FOX'], games=10, per_setup=5)
    assert q.plan() == [('FOX', 5), ('FOX', 5)]
    game = {'opponent': 'FOX', 'opponent_kind': 'cpu', 'cpu_level': 9, 'result': 'win'}
    assert q.accept(game, 9) is None
    q.batch_finished('FOX', 5)                  # that batch only produced one game
    assert q.plan() == [('FOX', 4)]
    assert not q.complete and not q.exhausted


def test_quota_only_counts_real_results_at_the_requested_level():
    from puffbot.evaluate import Quota
    q = Quota(['FOX'], games=2)
    q.plan()
    base = {'opponent': 'FOX', 'opponent_kind': 'cpu', 'cpu_level': 9, 'result': 'win'}
    assert q.accept({**base, 'result': 'capped'}, 9) == 'game capped'
    assert q.accept({**base, 'result': 'interrupted'}, 9) == 'game interrupted'
    assert 'CPU 0' in q.accept({**base, 'cpu_level': 0}, 9)
    assert q.accept(base, 9) is None and q.accept({**base, 'result': 'loss'}, 9) is None
    assert q.complete
    assert q.accept(base, 9) == 'quota already met'


def test_quota_gives_up_after_the_retry_budget():
    from puffbot.evaluate import Quota
    q = Quota(['FALCO'], games=5, per_setup=5, retry_factor=2)
    for _ in range(2):
        for o, n in q.plan():
            q.batch_finished(o, n)              # every batch dies without a game
    assert q.plan() == [] and q.exhausted and not q.complete


def test_only_the_genuine_legacy_contract_is_upgraded():
    from puffbot import contract
    odd = {'space': 'puff-macros-v1', 'names': list(actions.NAMES), 'act_every': 3, 'executor': 'edge-release-v2'}
    with pytest.raises(contract.IncompatibleCheckpoint):
        contract.check(odd)                  # modern-looking but no layout: refuse
    with pytest.raises(contract.IncompatibleCheckpoint):
        contract.check(None)                 # no contract at all: refuse


def test_crown_requires_a_complete_evaluation_of_the_same_bytes(tmp_path):
    import json
    from puffbot.evaluate import sha256
    from puffbot.server import qualifying_eval
    ck = tmp_path / 'policy.pt'
    ck.write_bytes(b'weights')
    evals = tmp_path / 'evals'

    def report(name, state, digest):
        (evals / name).mkdir(parents=True)
        (evals / name / 'report.json').write_text(json.dumps(
            {'state': state, 'checkpoint_sha256': digest, 'level': 9, 'overall': {'win_rate': 0.7}}))
    report('a-incomplete', 'incomplete', sha256(ck))
    report('b-other-file', 'complete', 'f' * 64)
    assert qualifying_eval(ck, evals) is None
    report('c-good', 'complete', sha256(ck))
    assert qualifying_eval(ck, evals)['id'] == 'c-good'


def test_evaluation_worker_builds_the_checkpoints_network(tmp_path):
    """GPT review R5: a valid 32-wide checkpoint failed to load into a 512-wide actor."""
    import threading
    from puffbot.actor import Worker
    net = Policy(32)
    save_policy(tmp_path / 'small.pt', net, 3, 30, act_every=3)

    class Box:
        def __init__(self): self.items = [None]
        def get(self, timeout=None): return self.items.pop(0)

    class Sink:
        def put(self, msg, timeout=None): pass
    cfg = Config(iso='/dev/null', hidden=512)
    w = Worker(0, cfg, tmp_path / 'run', Sink(), threading.Event(), 1,
               evaluate={'checkpoint': str(tmp_path / 'small.pt'), 'work': Box(), 'greedy': False})
    w.loop()                                  # loads, finds no work, exits cleanly
    assert w.actor.net.hidden == 32 and w.actor.version == 3


# ------------------------------------------------------------------ migration
def test_v1_network_converts_without_changing_what_it_computes():
    """Converted weights must reproduce the trained v1 network exactly on any situation
    the v1 network could see (new inputs zero), for every one of its 43 macros."""
    import torch
    from puffbot import migrate
    torch.manual_seed(0)
    old_names = observation.feature_names('puff-obs-v1')
    new_names = observation.feature_names()
    H, E = 16, 80
    v1 = {'pi_body.0.weight': torch.randn(H, len(old_names) + E), 'v_body.0.weight': torch.randn(H, len(old_names) + E),
          'pi_head.weight': torch.randn(43, H), 'pi_head.bias': torch.randn(43)}
    v2 = migrate.v1_to_current(v1)
    x_old = torch.randn(len(old_names) + E)
    x_new = torch.zeros(len(new_names) + E)
    where = {n: i for i, n in enumerate(old_names)}
    for j, n in enumerate(new_names):
        if n in where:
            x_new[j] = x_old[where[n]]
    x_new[len(new_names):] = x_old[len(old_names):]
    for key in migrate.FIRST_LAYERS:
        torch.testing.assert_close(v2[key] @ x_new, v1[key] @ x_old)
    h = torch.randn(H)
    new_logits = v2['pi_head.weight'] @ h + v2['pi_head.bias']
    old_logits = v1['pi_head.weight'] @ h + v1['pi_head.bias']
    idx = [actions.INDEX[n] for n in actions.V1_NAMES]
    torch.testing.assert_close(new_logits[idx], old_logits)
    assert v2['pi_head.weight'].shape[0] == actions.COUNT


def test_v1_policy_file_loads_as_a_working_current_policy(tmp_path):
    import torch
    from puffbot import migrate
    net = Policy(32)
    # Fabricate a v1 file by converting a current network's state *back* to v1 shape.
    s = net.state_dict()
    old_names, new_names = observation.feature_names('puff-obs-v1'), observation.feature_names()
    keep = [new_names.index(n) for n in old_names]
    v1 = dict(s)
    for key in migrate.FIRST_LAYERS:
        v1[key] = torch.cat([s[key][:, keep], s[key][:, len(new_names):]], 1)
    v1['pi_head.weight'] = s['pi_head.weight'][:43]
    v1['pi_head.bias'] = s['pi_head.bias'][:43]
    torch.save({'state': v1, 'version': 7, 'frames': 1, 'hidden': 32,
                'contract': {'space': 'puff-macros-v1', 'names': list(actions.V1_NAMES)}}, tmp_path / 'old.pt')
    payload = load_policy(tmp_path / 'old.pt', act_every=3)
    other = Policy(32)
    other.load_state_dict(payload['state'])
    assert any('converted' in n for n in payload['notes']) and payload['migrated_from']



# ------------------------------------------------------------- armed, timed techs
def flying(x, y, vy=-1.3, ax=0.0, ay=0.0, action=0x58):
    p = player(x=x, y=y, on_ground=False, action=action)
    p.speed_y_self, p.speed_x_attack, p.speed_y_attack = vy, ax, ay
    return p


def armed(name='tech'):
    ex = actions.Executor(3, auto_lcancel=False)
    ex.start(actions.INDEX[name])
    return ex


def clicks(frames):
    return ['BUTTON_L' in f.buttons for f in frames]


def test_armed_tech_clicks_at_the_measured_moments():
    """Probes: a click from frame 14 of Fox's down-throw teched 10/10; a click <=3 frames
    before landing in tumble teched 4/4."""
    ex = armed()
    thrown = [player(x=0, on_ground=False, action=0xF2) for _ in range(20)]
    for k, p in enumerate(thrown, 1):
        p.action_frame = k
    out = [ex.next_input(p) for p in thrown]
    assert clicks(out).index(True) == actions.TECH_THROW_FRAME - 1 and sum(clicks(out)) == 1
    ex = armed('tech_right')
    out = [ex.next_input(flying(0, y)) for y in (20, 12, 3.5, 1.0)]   # 1.3 units per frame
    assert clicks(out) == [False, False, True, False]
    assert out[2].stick == actions.RIGHT, 'the roll direction is held on the click'


def test_armed_tech_never_clicks_where_an_airdodge_could_leave_the_stage():
    """The one offstage air-dodge in 373 games: momentum carried a tech past the ledge."""
    for p in (flying(-120, 2, ay=-1),                    # offstage
              flying(80, 2.5, ax=3.0),                   # inside, but flung outward
              flying(0, 40)):                            # too high: not landing soon
        ex = armed()
        assert not any(clicks([ex.next_input(p) for _ in range(5)])), p.position
    ex = armed('tech_left')
    assert not any(clicks([ex.next_input(flying(-60, 2.5)) for _ in range(3)])), 'rolls need room'


def test_unarmed_or_expired_executor_never_clicks_without_auto_tech():
    ex = actions.Executor(3, auto_lcancel=False, auto_tech=False)
    ex.start(actions.INDEX['noop'])
    assert not any(clicks([ex.next_input(flying(0, 2.5)) for _ in range(5)]))
    ex = actions.Executor(3, auto_lcancel=False, auto_tech=False)
    ex.start(actions.INDEX['tech'])
    for _ in range(actions.TECH_ARM_FRAMES):
        ex.next_input(flying(0, 60, vy=0.0))
    assert not any(clicks([ex.next_input(flying(0, 2.5)) for _ in range(3)])), 'arming lapses'


def test_auto_tech_techs_in_place_by_default_and_arming_picks_the_direction():
    ex = actions.Executor(3, auto_lcancel=False)
    ex.start(actions.INDEX['noop'])
    out = [ex.next_input(flying(0, y)) for y in (20, 12, 3.5, 1.0)]
    assert clicks(out) == [False, False, True, False] and out[2].stick == actions.NEUTRAL
    ex = actions.Executor(3, auto_lcancel=False)
    ex.start(actions.INDEX['tech_left'])
    out = [ex.next_input(flying(0, y)) for y in (20, 12, 3.5, 1.0)]
    assert out[2].stick == actions.LEFT
    ex = actions.Executor(3, auto_lcancel=False)
    ex.start(actions.INDEX['noop'])
    standing = [ex.next_input(player()) for _ in range(10)]
    assert not any(clicks(standing)), 'never outside a knockdown situation'


def test_one_click_per_lockout_and_none_during_launching_throws():
    """Melee blocks teching for 40 frames after a click that does not tech."""
    ex = actions.Executor(3, auto_lcancel=False)
    out = []
    for k in range(1, 60):
        ex.start(actions.INDEX['tech'])            # a policy re-arming every decision
        p = player(x=0, on_ground=False, action=0xF2)
        p.action_frame = k
        out.append(ex.next_input(p))
    assert sum(clicks(out)) == 2 and clicks(out).index(True) == actions.TECH_THROW_FRAME - 1
    ex = armed()
    up = [player(x=0, on_ground=False, action=0xF1) for _ in range(30)]
    for k, p in enumerate(up, 1):
        p.action_frame = k
    assert not any(clicks([ex.next_input(p) for p in up])), 'up-throw: wait for the landing'
