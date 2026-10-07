"""Unit tests for the pieces that do not need a ROM.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from smwrl.actions import ACTION_COMBOS, ACTION_LABELS, ACTION_TABLE, BUTTONS, N_ACTIONS
from smwrl.archive import Archive
from smwrl.ram import ANIM_DYING, MODE_LEVEL, decode
from smwrl.wrappers import CheckpointPool, EpisodeConfig, RewardConfig


# --------------------------------------------------------------------------
# RAM decoding
# --------------------------------------------------------------------------
def test_decode_handles_empty_info():
    """reset() returns an empty info dict; decode must not raise."""
    s = decode({})
    assert s.x == 0 and s.y == 0
    assert not s.in_level and not s.dying and not s.cleared


def test_y_is_signed():
    """y reads as unsigned but is negative above the top of the level."""
    assert decode({"y_pos": 65502}).y == -34
    assert decode({"y_pos": 352}).y == 352


def test_timer_digits_combine():
    s = decode({"timer_hundreds": 2, "timer_tens": 8, "timer_ones": 5})
    assert s.timer == 285


def test_state_flags():
    assert decode({"game_mode": MODE_LEVEL}).in_level
    assert not decode({"game_mode": 0x0B}).in_level
    assert decode({"player_anim": ANIM_DYING}).dying
    assert not decode({"player_anim": 0}).dying
    # end_level_timer is nonzero exactly when the goal tape is crossed;
    # verified live at 253/254.
    assert decode({"end_level_timer": 254}).cleared
    assert not decode({"end_level_timer": 0}).cleared


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------
def test_action_table_shape_and_content():
    assert ACTION_TABLE.shape == (N_ACTIONS, 12)
    assert ACTION_TABLE.dtype == np.uint8
    assert ACTION_TABLE[0].sum() == 0, "action 0 must be a true no-op"
    # action 4 is the run-jump-right workhorse
    for name in ("RIGHT", "Y", "B"):
        assert ACTION_TABLE[4][BUTTONS.index(name)] == 1


def test_no_duplicate_actions():
    combos = [tuple(sorted(c)) for c in ACTION_COMBOS]
    assert len(combos) == len(set(combos)), "duplicate action combos waste capacity"


def test_labels_align_with_combos():
    assert len(ACTION_LABELS) == N_ACTIONS


def test_no_start_or_select():
    """START pauses and SELECT can exit the level -- neither belongs in the set."""
    for i in range(N_ACTIONS):
        assert ACTION_TABLE[i][BUTTONS.index("START")] == 0
        assert ACTION_TABLE[i][BUTTONS.index("SELECT")] == 0


# --------------------------------------------------------------------------
# Reward invariants -- these encode the two bugs that cost the most time
# --------------------------------------------------------------------------
def _returns(r: RewardConfig, e: EpisodeConfig, wall_x=455, start_x=16, reach=60):
    progress = (wall_x - start_x) * r.progress_scale
    idle = progress - (reach + e.stuck_steps) * r.time_penalty - r.stuck_penalty
    try_die = progress - (reach + 10) * r.time_penalty - r.death_penalty
    suicide = -5 * r.time_penalty - r.death_penalty
    return idle, try_die, suicide


def test_attempting_beats_giving_up():
    """The bug that flatlined YoshiIsland1 for 700k steps.

    With no stuck penalty, walking to the first hazard and standing still
    returned +7.5 while trying and dying returned -36.6, so idling was
    genuinely optimal.
    """
    idle, try_die, _ = _returns(RewardConfig(), EpisodeConfig())
    assert try_die > idle, f"idling ({idle:.1f}) must not beat attempting ({try_die:.1f})"


def test_suicide_is_not_optimal():
    """The mirror-image bug: death must cost more than the time it saves."""
    idle, try_die, suicide = _returns(RewardConfig(), EpisodeConfig())
    assert try_die > suicide, "dying instantly must be worse than making progress first"


def test_clearing_dominates_everything():
    r, e = RewardConfig(), EpisodeConfig()
    idle, try_die, suicide = _returns(r, e)
    cleared = (4600 * r.progress_scale - 400 * r.time_penalty
               + r.clear_bonus + r.speed_bonus * (e.max_steps - 400))
    assert cleared > max(idle, try_die, suicide) * 10


def test_faster_clear_pays_more():
    """The speed bonus is the whole point -- a quicker clear must score higher."""
    r, e = RewardConfig(), EpisodeConfig()
    def clear_at(steps):
        return (r.clear_bonus + r.speed_bonus * max(0, e.max_steps - steps)
                - steps * r.time_penalty)
    assert clear_at(400) > clear_at(700) > clear_at(1100)


def test_death_penalty_exceeds_max_time_penalty():
    r, e = RewardConfig(), EpisodeConfig()
    assert r.death_penalty > e.stuck_steps * r.time_penalty * 0.5


# --------------------------------------------------------------------------
# CheckpointPool
# --------------------------------------------------------------------------
def test_pool_ratio_zero_never_samples():
    pool = CheckpointPool(ratio=0.0, states=[b"x" * 100])
    assert all(pool.sample() is None for _ in range(50))


def test_pool_ratio_one_always_samples():
    pool = CheckpointPool(ratio=1.0, states=[b"payload"])
    assert pool.sample() == b"payload"


def test_pool_empty_is_safe():
    assert CheckpointPool(ratio=1.0, states=[]).sample() is None


def test_pool_autodetects_compression():
    raw = b"snapshot-bytes" * 500
    comp = CheckpointPool.compressed_from(1.0, [raw])
    assert comp.compressed
    assert comp.sample() == raw, "compressed states must round-trip"
    plain = CheckpointPool(1.0, [raw])
    assert not plain.compressed
    assert plain.sample() == raw


# --------------------------------------------------------------------------
# Archive / Go-Explore
# --------------------------------------------------------------------------
def _blob(n=64):
    return bytes(np.random.default_rng(0).integers(0, 255, n, dtype=np.uint8))


def test_archive_bins_nearby_positions_together():
    a = Archive(x_bin=48, y_bin=96)
    assert a.key_for(0, 100, 200) == a.key_for(0, 130, 210)
    assert a.key_for(0, 100, 200) != a.key_for(0, 200, 200)
    assert a.key_for(1, 100, 200) != a.key_for(0, 100, 200)


def test_archive_keeps_the_faster_route():
    a = Archive()
    a.consider(0, 100, 200, _blob(), steps=50)
    assert not a.consider(0, 100, 200, _blob(), steps=80), "slower route must not win"
    assert a.cells[a.key_for(0, 100, 200)].steps == 50
    assert a.consider(0, 100, 200, _blob(), steps=20), "faster route must replace"
    assert a.cells[a.key_for(0, 100, 200)].steps == 20


def test_archive_progress_orders_rooms_above_x():
    a = Archive()
    a.consider(0, 4000, 200, _blob(), 10)
    a.consider(1, 50, 200, _blob(), 20)
    assert a.best_cell.room == 1, "a later room beats a high x in an earlier room"


def test_archive_state_round_trips():
    a = Archive()
    payload = _blob(1000)
    a.consider(0, 10, 20, payload, 1)
    assert zlib.decompress(a.best_cell.state) == payload


def test_curriculum_spreads_across_the_level():
    """A curriculum of only the deepest states has no rungs to climb."""
    a = Archive()
    for x in range(0, 4000, 40):
        a.consider(0, x, 200, _blob(), x)
        a.mark_on_winning_path(a.key_for(0, x, 200), 4000 - x)
    states = a.curriculum_states(10)
    assert 2 <= len(states) <= 10
    xs = sorted(c.x for c in a.cells.values())
    chosen = [c.x for c in sorted(a.cells.values(), key=lambda c: c.progress)]
    assert chosen[0] == min(xs) and chosen[-1] == max(xs)


def test_failing_cells_lose_weight():
    """Cells that always kill you must stop dominating selection.

    This is the fix for the frontier filling with mid-fall states.
    """
    import random as _random
    _random.seed(1234)          # select() samples; pin it so this cannot flake
    a = Archive()
    a.consider(0, 400, 200, _blob(), 10)   # the trap: highest x
    a.consider(0, 100, 200, _blob(), 10)
    trap_key = a.key_for(0, 400, 200)
    for _ in range(200):
        a.mark_failure(trap_key)
    picks = [a.select().key for _ in range(400)]
    trap_share = picks.count(trap_key) / len(picks)
    # Pre-fix this was 55%. Anything at this level means dead ends no longer
    # monopolise exploration.
    assert trap_share < 0.20, f"trap still chosen {trap_share:.0%} of the time"


def test_hard_but_passable_cells_keep_priority():
    """The counterpart: a frontier cell that sometimes works must stay favoured.

    YoshiIsland3 only broke through after ~1400 retries of a hard cell, so
    punishing failure too hard would remove the mechanism that works.
    """
    import random as _random
    _random.seed(1234)
    a = Archive()
    a.consider(0, 400, 200, _blob(), 10)
    a.consider(0, 100, 200, _blob(), 10)
    hard = a.key_for(0, 400, 200)
    for _ in range(20):
        a.mark_failure(hard)
    for _ in range(2):
        a.mark_success(hard)
    picks = [a.select().key for _ in range(400)]
    assert picks.count(hard) / len(picks) > 0.5, "hard frontier cell was abandoned too early"


def test_select_returns_none_when_empty():
    assert Archive().select() is None


def test_archive_save_load_round_trip(tmp_path):
    a = Archive()
    a.consider(0, 123, 200, _blob(), 7)
    p = tmp_path / "archive.pkl"
    a.save(p)
    b = Archive.load(p)
    assert len(b.cells) == 1 and b.best_cell.x == 123


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# Reverse curriculum annealing
# --------------------------------------------------------------------------
def test_reverse_curriculum_starts_near_the_goal():
    """Early on, only the states closest to the goal should be reachable."""
    states = [f"s{i}".encode() for i in range(10)]
    pool = CheckpointPool(ratio=1.0, states=states, anneal_resets=1000)
    picks = {pool.sample() for _ in range(40)}
    assert picks <= {b"s9", b"s8"}, f"expected goal-side states only, got {picks}"


def test_reverse_curriculum_widens_to_the_level_start():
    states = [f"s{i}".encode() for i in range(10)]
    pool = CheckpointPool(ratio=1.0, states=states, anneal_resets=100)
    for _ in range(200):
        pool.sample()
    assert pool.window_start() == 0, "window must reach the level start eventually"
    picks = {pool.sample() for _ in range(300)}
    assert b"s0" in picks, "the level start must become reachable"


def test_success_gating_is_the_default():
    """With no explicit anneal schedule the pool is success-gated, not uniform.

    Uniform sampling was the old default and is what let 40% of experience land
    on parts of the level the agent could not reach.
    """
    states = [f"s{i}".encode() for i in range(10)]
    pool = CheckpointPool(ratio=1.0, states=states, window=3)
    assert pool.window_start() == pool.focus == 6
    picks = {pool.sample() for _ in range(300)}
    assert picks <= {b"s6", b"s7", b"s8", b"s9"}, "must stay in the current stage"
    assert b"s0" not in picks, "the level start is not eligible until earned"


def test_window_monotonically_widens():
    states = [f"s{i}".encode() for i in range(20)]
    pool = CheckpointPool(ratio=1.0, states=states, anneal_resets=500)
    seen = []
    for _ in range(600):
        pool.sample()
        seen.append(pool.window_start())
    assert seen[0] >= seen[len(seen) // 2] >= seen[-1] == 0


# --------------------------------------------------------------------------
# Success-gated backward curriculum
# --------------------------------------------------------------------------
def _pool(n=20, **kw):
    return CheckpointPool(ratio=1.0, states=[f"s{i}".encode() for i in range(n)], **kw)


def test_curriculum_starts_near_the_goal():
    p = _pool(20, window=3)
    assert p.focus == 16, "should begin a short hop from the goal"
    picks = {p.sample() for _ in range(60)}
    assert picks <= {b"s16", b"s17", b"s18", b"s19"}


def test_success_moves_the_start_backwards():
    # window=0 so sample() always draws the focus stage; report() only counts
    # episodes that actually started there.
    p = _pool(20, window=0, promote_after=10, promote_rate=0.5)
    start = p.focus
    for _ in range(10):
        p.sample()
        p.report(True)
    assert p.focus == start - 1, "clearing the stage must move the start earlier"


def test_failure_does_not_advance_the_curriculum():
    """The bug this replaces: annealing on a clock regardless of success."""
    p = _pool(20, window=0, promote_after=10, demote_after=1000)
    start = p.focus
    for _ in range(50):
        p.sample()
        p.report(False)
    assert p.focus == start, "a failing agent must not be pushed further back"


def test_hopeless_stage_backs_off_toward_the_goal():
    p = _pool(20, window=0, promote_after=1000, demote_after=30)
    p.focus = 5
    for _ in range(30):
        p.sample()
        p.report(False)
    assert p.focus == 6, "an impossible stage should retreat toward the goal"


def test_curriculum_can_reach_the_level_start():
    p = _pool(20, window=0, promote_after=5, promote_rate=0.5)
    for _ in range(40 * 5):
        p.sample()
        p.report(True)
    assert p.focus == 0
    assert p.window_start() == 0


def test_report_is_safe_with_no_states():
    CheckpointPool(ratio=1.0, states=[]).report(True)   # must not raise


def test_curriculum_prefers_cells_that_reach_the_goal():
    """Ranking by raw progress can pick a dead-end side room.

    On YoshiIsland3 `room * 100000 + x` ranked a vine high above a pit in a
    vertical sub-room as the deepest state; random play died 12/12 from there.
    """
    a = Archive()
    a.consider(1, 4600, 200, _blob(), 10)          # side room: looks deepest
    a.consider(0, 500, 200, _blob(), 20)           # real route, far from goal
    a.consider(0, 3000, 200, _blob(), 40)          # real route, near goal
    a.mark_on_winning_path(a.key_for(0, 500, 200), 300)
    a.mark_on_winning_path(a.key_for(0, 3000, 200), 50)

    states = a.curriculum_states(10)
    win = a.winning_cells
    assert len(states) == 2, "the dead-end cell must be excluded"
    assert [c.x for c in win] == [500, 3000], "must order far-from-goal -> near"


def test_curriculum_is_empty_when_nothing_has_reached_the_goal():
    """Refusing to guess is the safe behaviour here.

    The old fallback ordered cells by room*100000+x when no route was known,
    which ranked a vertical side-room above the real route and put YoshiIsland3's
    deepest stage on a vine above a pit -- random play died 12/12 from there.
    """
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    a.consider(0, 900, 200, _blob(), 2)
    assert a.curriculum_states(10) == [], "no known route means no curriculum"
    assert a.curriculum_entries(10) == []


def test_to_goal_keeps_the_shortest_route():
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    k = a.key_for(0, 100, 200)
    a.mark_on_winning_path(k, 200)
    a.mark_on_winning_path(k, 90)
    a.mark_on_winning_path(k, 400)
    assert a.cells[k].to_goal == 90


# --------------------------------------------------------------------------
# Positional curriculum entries (promote on reaching the next stage)
# --------------------------------------------------------------------------
def test_pool_reads_positional_entries():
    entries = [(0, 100, b"a"), (0, 500, b"b"), (1, 50, b"c")]
    p = CheckpointPool(ratio=1.0, states=entries, window=0)
    assert p.states == [b"a", b"b", b"c"]
    assert p.positions == [(0, 100), (0, 500), (1, 50)]


def test_pool_rejects_nonmonotonic_positional_curriculum():
    with pytest.raises(ValueError, match="strictly ordered"):
        CheckpointPool(ratio=1.0, states=[(0, 500, b"a"), (0, 100, b"b")])


def test_target_is_the_next_stage_not_the_goal():
    """Requiring a full clear from each stage makes promotion compound.

    On YoshiIsland1 the cost per stage grew 60k -> 900k steps and was still
    climbing, because each step backwards chained one more segment.
    """
    entries = [(0, 100, b"a"), (0, 500, b"b"), (0, 900, b"c")]
    p = CheckpointPool(ratio=1.0, states=entries, window=0)
    p.focus = 0
    p.sample()
    assert p.current_target == (0, 500), "stage 0 must aim at stage 1"
    p.focus = 1
    p.sample()
    assert p.current_target == (0, 900)


def test_pool_exposes_sampled_room_as_the_checkpoint_start():
    entries = [(3, 100, b"a"), (3, 500, b"b")]
    p = CheckpointPool(ratio=1.0, states=entries, window=0)
    p.focus = 0
    assert p.sample() == b"a"
    assert p.current_start == (3, 100)
    assert p.current_target == (3, 500)


def test_exploration_route_starts_with_the_launch_cell():
    from smwrl.explore import chronological_route

    start = (0, 0, 0)
    later = [((0, 1, 0), 20), ((0, 2, 0), 40)]
    assert chronological_route(start, later) == [(start, 0), *later]


def test_last_stage_targets_the_goal():
    entries = [(0, 100, b"a"), (0, 500, b"b")]
    p = CheckpointPool(ratio=1.0, states=entries, window=0)
    p.focus = 1
    p.sample()
    assert p.current_target is None, "nearest-goal stage has to actually clear"


def test_legacy_bare_state_curricula_still_load():
    p = CheckpointPool(ratio=1.0, states=[b"x", b"y"], window=0)
    assert p.positions == [None, None]
    assert p.sample() in (b"x", b"y")


# --------------------------------------------------------------------------
# Curriculum stage must survive the curriculum being rebuilt
# --------------------------------------------------------------------------
def _entries(xs):
    return [(0, x, f"s{x}".encode()) for x in xs]


def test_stage_survives_a_curriculum_rebuild():
    """Extending the archive rebuilds curriculum.pkl with a different length.

    Overnight this reset the backward curriculum to the goal every time:
    YoshiIsland1 climbed to stage 18, was re-explored, snapped back to 28, and
    finished the run further behind than it started.
    """
    old = _entries([100, 500, 900, 1300, 1700])
    pool = CheckpointPool(ratio=1.0, states=old, window=1)
    pool.focus = 1                                   # earned our way back to x=500
    pos = pool.focus_position
    assert pos == (0, 500)

    # A later exploration pass finds more cells: different length, same level.
    new = _entries([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700])
    idx = CheckpointPool.index_for_position([(r, x) for r, x, _ in new], pos)
    resumed = CheckpointPool(ratio=1.0, states=new, window=1, initial_focus=idx)
    assert resumed.focus_position == (0, 500), "stage must resume at the same place"
    assert resumed.focus != len(new) - 1 - 1, "must not snap back to the goal"


def test_index_for_position_picks_the_nearest_stage():
    pos_list = [(0, 100), (0, 600), (1, 50)]
    assert CheckpointPool.index_for_position(pos_list, (0, 620)) == 1
    assert CheckpointPool.index_for_position(pos_list, (1, 40)) == 2
    assert CheckpointPool.index_for_position(pos_list, (0, 90)) == 0


def test_index_for_position_handles_missing_data():
    assert CheckpointPool.index_for_position([None, None], (0, 5)) is None
    assert CheckpointPool.index_for_position([(0, 1)], None) is None


def test_initial_focus_is_clamped():
    e = _entries([100, 200, 300])
    assert CheckpointPool(1.0, e, initial_focus=99).focus == 2
    assert CheckpointPool(1.0, e, initial_focus=-5).focus == 0


def test_without_saved_stage_it_still_starts_near_the_goal():
    e = _entries([100, 200, 300, 400, 500])
    assert CheckpointPool(1.0, e, window=1).focus == 3


def test_relax_route_propagates_distance_without_a_clear():
    """Early cells must get a distance-to-goal by chaining through known ones.

    Marking only on a clear left curricula covering the last ~20% of a level:
    YoshiIsland1 had 41 winning cells spanning x=3696..4807 of 4600 and never
    got above 0% unaided, while YoshiIsland2 had full coverage and passed.
    """
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    a.consider(0, 500, 200, _blob(), 2)
    a.consider(0, 900, 200, _blob(), 3)
    k1, k2, k3 = (a.key_for(0, 100, 200), a.key_for(0, 500, 200), a.key_for(0, 900, 200))
    a.cells[k3].to_goal = 10                      # only the last one knows

    # An excursion that passes k1 -> k2 -> k3 and does NOT clear.
    a.relax_route([(k1, 0), (k2, 20), (k3, 50)], cleared=False, end_step=80)
    assert a.cells[k2].to_goal == 40, "k2 is 30 steps from k3, which is 10 from goal"
    assert a.cells[k1].to_goal == 60
    assert len(a.winning_cells) == 3, "all three now have a route to the goal"


def test_relax_route_uses_the_clear_as_the_goal():
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    k = a.key_for(0, 100, 200)
    a.relax_route([(k, 30)], cleared=True, end_step=90)
    assert a.cells[k].to_goal == 60


def test_relax_route_keeps_the_shortest_known_distance():
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    k = a.key_for(0, 100, 200)
    a.cells[k].to_goal = 25
    a.relax_route([(k, 0)], cleared=True, end_step=90)
    assert a.cells[k].to_goal == 25, "a worse route must not overwrite a better one"


def test_relax_route_does_nothing_without_any_anchor():
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    k = a.key_for(0, 100, 200)
    a.relax_route([(k, 0)], cleared=False, end_step=50)
    assert a.cells[k].to_goal is None, "no clear and no known cell = no information"


def test_relax_route_does_not_feed_repeated_keys_back_into_itself():
    """Repeated visits to one coarse cell used to collapse to_goal to zero."""
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    k = a.key_for(0, 100, 200)
    a.relax_route([(k, 10), (k, 11), (k, 12)], cleared=True, end_step=20)
    assert a.cells[k].to_goal == 10


def test_representative_route_excludes_a_slower_alias():
    from smwrl.explore import representative_route

    a = Archive()
    a.consider(0, 100, 200, _blob(), 5)
    start = a.best_cell
    other = a.key_for(0, 500, 200)
    a.consider(0, 500, 200, _blob(), 10)  # representative arrived at global step 10
    # Current excursion reaches the same coarse cell at global step 25.
    route = representative_route(a, start, [(other, 20)], base_steps=5)
    assert route == [(start.key, 0)]


def test_shared_curriculum_gate_aggregates_worker_copies():
    import multiprocessing as mp
    import pickle

    manager = mp.Manager()
    try:
        p = _pool(10, window=0, promote_after=4, promote_rate=0.5)
        p.enable_shared(manager)
        worker_a = pickle.loads(pickle.dumps(p))
        worker_b = pickle.loads(pickle.dumps(p))
        start = p.focus
        for w in (worker_a, worker_b, worker_a, worker_b):
            w.sample()          # each copy must draw its own stage first
            w.report(True)
        assert p.focus == worker_a.focus == worker_b.focus == start - 1
    finally:
        manager.shutdown()


def test_loading_v1_archive_invalidates_corrupt_route_labels(tmp_path):
    a = Archive()
    a.consider(0, 100, 200, _blob(), 1)
    next(iter(a.cells.values())).to_goal = 0
    del a.route_format
    path = tmp_path / "old.pkl"
    a.save(path)
    loaded = Archive.load(path)
    assert loaded.route_format == 2
    assert all(c.to_goal is None for c in loaded.cells.values())


def test_curriculum_export_is_strictly_forward_and_pool_loadable():
    a = Archive()
    for i, x in enumerate((100, 500, 300, 900)):
        a.consider(0, x, 200, _blob(), i)
        a.cells[a.key_for(0, x, 200)].to_goal = 100 - i * 10
    entries = a.curriculum_entries(10)
    positions = [(r, x) for r, x, _ in entries]
    assert all(a < b for a, b in zip(positions, positions[1:]))
    CheckpointPool(1.0, entries)


def test_policy_initial_stack_matches_vec_frame_stack_reset():
    from pathlib import Path
    from smwrl.policy import PolicyRuntime
    from smwrl.wrappers import ObsConfig

    obs = np.full((84, 84, 3), 7, np.uint8)
    runtime = PolicyRuntime(object(), ObsConfig(), 4, Path("dummy.zip"))
    stack = runtime.initial_stack(obs)
    assert not stack[:3].any()
    assert np.array_equal(stack[3], obs)


# --------------------------------------------------------------------------
# Whole-game archive
# --------------------------------------------------------------------------
def test_world_archive_select_survives_module_split():
    """select() was lost when the cell types moved out of worldrun.py.

    The 10k exploration run died on AttributeError after ~0 excursions.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    a.consider(42, 0, 100, 200, 0, _blob(), 1)
    a.consider(39, 0, 300, 200, 2, _blob(), 5)
    assert a.select() is not None
    assert a.best_cell.cleared == 2, "levels cleared must dominate raw x"
    assert a.translevels == {42, 39}


def test_world_archive_round_trips_under_a_stable_module_path():
    """Archives used to pickle as __main__.WorldArchive and load nowhere else."""
    import pickle as _p
    from smwrl.world_archive import WorldArchive, load_archive
    a = WorldArchive()
    a.consider(42, 0, 100, 200, 0, _blob(), 1)
    blob = _p.dumps(a)
    assert b"smwrl.world_archive" in blob, "must not pickle under __main__"
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "a.pkl"
        p.write_bytes(blob)
        assert len(load_archive(p).cells) == 1


def test_world_select_prefers_under_explored_levels():
    """A heavily re-cleared level must not monopolise the frontier.

    Exploration ping-ponged between two translevels, ending with 366 cells and
    205 of them inside one re-clear loop.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    for i in range(60):                      # crowded, well-known level
        a.consider(42, 0, i * 64, 200, 1, _blob(), 10)
    a.consider(39, 0, 64, 200, 1, _blob(), 10)   # barely-seen level
    rare = a.key_for(39, 0, 64, 200)
    picks = [a.select().key for _ in range(600)]
    share = picks.count(rare) / len(picks)
    assert share > 1 / 61, f"rare translevel picked only {share:.1%} of the time"


def test_world_progress_counts_events_not_goal_tapes():
    """Re-clearing a beaten level fires no event and must not raise progress.

    Counting goal tapes let exploration rack up ~580 clears across two
    translevels with the map still closed. 0x7E1F2E only steps when a
    completion event opens a new path -- verified by RAM diff on a fresh game.
    """
    from smwrl.world_archive import EVENTS, WorldArchive
    assert EVENTS == 0x1F2E
    a = WorldArchive()
    a.consider(42, 0, 4800, 200, 1, _blob(), 10)   # deep in a level, 1 event
    a.consider(39, 0, 24, 200, 2, _blob(), 20)     # just started, 2 events
    assert a.best_cell.translevel == 39, "a new event must outrank raw distance"
    a.consider(42, 0, 4800, 300, 1, _blob(), 5)    # re-clear: same event count
    assert a.best_cell.cleared == 2


def test_switch_palaces_count_as_progress():
    """A switch palace opens no map path, so the event counter never moves for
    one -- but pressing the switch is permanent, game-wide progress.

    Scoring 0x1F2E alone made the Yellow Switch Palace unscorable. Random play
    presses its switch in 38 of 240 excursions; across ~92,000 excursions the
    archive kept not one state with the flag set, and the search stayed parked
    there because nothing it did could ever register.
    """
    from smwrl.world_archive import EVENTS, SWITCHES, game_progress
    ram = np.zeros(0x2000, np.uint8)
    assert game_progress(ram) == 0
    ram[EVENTS] = 3
    assert game_progress(ram) == 3
    ram[0x1F28] = 1                      # yellow switch; confirmed live 0 -> 1
    assert game_progress(ram) == 4, "a pressed switch must count as progress"
    for addr in SWITCHES:
        ram[addr] = 1
    assert game_progress(ram) == 7


def test_progress_outranks_a_cheaper_route_to_the_same_cell():
    """`consider` had this backwards, which is how the one discovery the search
    needed was thrown away every time it was made: a state carrying a freshly
    pressed switch lost to an older, cheaper visit to the same spot.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    key = a.key_for(20, 1, 700, 300)
    a.consider(20, 1, 700, 300, 1, _blob(), 40)     # cheap, before the switch
    a.consider(20, 1, 700, 300, 2, _blob(), 900)    # dear, but the switch is in
    assert a.cells[key].cleared == 2, "more game progress must win"
    a.consider(20, 1, 700, 300, 2, _blob(), 120)    # same progress, cheaper
    assert a.cells[key].steps == 120, "among equals, keep the cheaper route"
    a.consider(20, 1, 700, 300, 1, _blob(), 5)      # cheapest, but a step back
    assert a.cells[key].cleared == 2 and a.cells[key].steps == 120

    b = WorldArchive()
    b.consider(20, 1, 700, 300, 3, _blob(), 4000)
    a.absorb(b)
    assert a.cells[key].cleared == 3, "merging must use the same rule"


def test_a_dead_end_level_stops_owning_the_search():
    """The failure that cost a whole run of whole-game exploration.

    One completion event fired while the explorer was inside the Yellow Switch
    Palace. Every cell there carried that event, and game progress is worth 10^7
    in `progress`, so those 41 cells outranked all 153 in Yoshi's Island 1 and
    the single one in Yoshi's Island 2 by seven orders of magnitude. The palace
    took 72,771 of ~92,000 excursions and could never score again.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    for room in (0, 1):                        # the dead end, exhaustively known
        for i in range(20):
            a.consider(20, room, 16 + i * 36, 300, 1, _blob(), 3900)
    for c in a.cells.values():
        c.chosen, c.succeeds, c.fails = 1800, 1750, 50
    a.barren()[20] = 72_771                    # and it has yielded nothing since
    a.consider(42, 0, 16, 300, 0, _blob(), 10)    # a level entered once, never played
    fresh = a.key_for(42, 0, 16, 300)

    picks = [a.select().key for _ in range(600)]
    share = picks.count(fresh) / len(picks)
    assert share > 0.10, f"barely-seen level picked only {share:.1%} of the time"
    assert share < 0.95, "and it must not simply become the new monopoly"


def test_a_level_that_yields_again_stops_being_barren():
    """Counting all-time excursions punished a level for its history forever.

    That overcorrected the dead-end fix: the Yellow Switch Palace fell to one
    pick in 500 and could not climb back, even though a single excursion there
    was worth a pressed switch. Barrenness has to be a dry spell, not a debt.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    a.consider(20, 1, 700, 300, 1, _blob(), 3900)
    a.barren()[20] = 5_000

    a.consider(20, 1, 700, 300, 1, _blob(), 40)      # cheaper route to a known cell
    assert a.barren()[20] == 5_000, "shaving steps off a known cell is not yield"

    a.consider(20, 1, 700, 300, 2, _blob(), 4200)    # the switch goes in
    assert a.barren()[20] == 0, "new game progress must revive the level"

    a.consider(41, 0, 900, 300, 2, _blob(), 500)     # a brand-new cell elsewhere
    assert a.barren()[41] == 0


def test_merging_shards_does_not_compound_shared_counters():
    """Every shard resumes from the merged archive, so both sides carry the same
    history. Summing them re-counted that shared base once per shard per merge,
    compounding every 900 seconds: the switch palace recorded 699,509 excursions
    against roughly 15,000 ever run, and the novelty prior in `select` read the
    inflated figure as a level 47 times more explored than it was.
    """
    import copy
    from smwrl.world_archive import WorldArchive
    base = WorldArchive()
    base.consider(20, 0, 100, 200, 1, _blob(), 10)
    key = base.key_for(20, 0, 100, 200)
    base.cells[key].chosen, base.cells[key].succeeds = 1000, 900

    shard = copy.deepcopy(base)
    shard.cells[key].chosen += 25              # what this shard actually ran
    shard.cells[key].succeeds += 20
    base.absorb(shard)
    assert base.cells[key].chosen == 1025, "the shared base must not be counted twice"
    assert base.cells[key].succeeds == 920

    base.absorb(copy.deepcopy(shard))
    assert base.cells[key].chosen == 1025, "re-merging a shard must be a no-op"

    better = copy.deepcopy(base)               # same cell, but more progress
    better.cells[key].cleared = 2
    better.cells[key].chosen = 3
    base.absorb(better)
    assert base.cells[key].cleared == 2
    assert base.cells[key].chosen == 1025, "adopting a better state must keep the history"


def test_selection_ages_a_level_that_gives_nothing_back():
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    a.consider(42, 0, 16, 300, 0, _blob(), 10)
    for _ in range(30):
        a.select()
    assert a.barren()[42] == 30, "every excursion without a find must age the level"

    b = WorldArchive()                                # a shard that just found one
    b.consider(42, 0, 16, 300, 0, _blob(), 10)
    b.barren()[42] = 2
    b.barren()[39] = 400                              # a level `a` has never seen
    a.absorb(b)
    assert a.barren()[42] == 2, "merging must take the shortest dry spell"
    assert a.barren()[39] == 400, "and must not read 'never drawn from' as fresh"


def test_frontier_is_ranked_inside_each_level():
    """Normalising against the *global* best flattened every cell in a level.

    Once one event fired, cells at x=16 and x=736 in that level scored
    10,000,016 and 10,000,736 -- a ratio of 1.00007. The frontier bias had
    nothing left to bite on, so the search was not even pushing the dead end's
    own edge; it picked all 41 cells indiscriminately.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    for i in range(12):
        a.consider(20, 0, 16 + i * 64, 300, 1, _blob(), 3900)   # all one event
    deep = a.key_for(20, 0, 16 + 11 * 64, 300)
    shallow = a.key_for(20, 0, 16, 300)

    picks = [a.select().key for _ in range(600)]
    deep_n, shallow_n = picks.count(deep), picks.count(shallow)
    # +1 so an untouched start still demands the frontier was picked in earnest.
    # Flattened, all twelve cells drew ~50 each and this fails on either side.
    assert deep_n > 10 * (shallow_n + 1), (
        f"frontier {deep_n} vs start {shallow_n}: the within-level frontier "
        "must survive a completion event")


def test_branch_entry_tries_every_option_not_just_one(monkeypatch):
    """Picking one random branch and giving up cost the search its own frontier.

    The two cells carrying every banked event failed 57 excursions out of 57 --
    both overworld cells, both bailing in choose_branch_and_enter. After a path
    opens, the node Mario is left standing on often cannot be entered, and the
    excursion never looked anywhere else.
    """
    import smwrl.overworld as ow
    import smwrl.worldrun as wr
    from smwrl.world_archive import WorldArchive

    class FakeEnv:
        class em:
            @staticmethod
            def get_state():
                return b"snap"

            @staticmethod
            def set_state(state):
                pass

        @staticmethod
        def step(action):
            pass

    elsewhere = {"RIGHT": (200, 100), "LEFT": (100, 100),
                 "UP": (150, 50), "DOWN": (150, 150)}
    monkeypatch.setattr(ow, "wait_settled", lambda *a, **k: (150, 100))
    monkeypatch.setattr(ow, "position", lambda env: (150, 100))
    monkeypatch.setattr(ow, "move", lambda env, d, settle=0: elsewhere[d])

    attempts = []

    def only_the_third_node_has_a_level(env, timeout=0):
        attempts.append(1)
        return len(attempts) == 3

    monkeypatch.setattr(ow, "enter_level", only_the_third_node_has_a_level)
    archive = WorldArchive()
    assert wr.choose_branch_and_enter(FakeEnv(), archive, np.random.default_rng(0)) is True
    assert len(attempts) == 3, "must keep trying branches instead of giving up on one"
    assert len(archive.visited()) == 1, "only the node actually entered is visited"

    attempts.clear()
    monkeypatch.setattr(ow, "enter_level", lambda env, timeout=0: attempts.append(1) or False)
    assert wr.choose_branch_and_enter(FakeEnv(), WorldArchive(),
                                      np.random.default_rng(0)) is False
    assert len(attempts) == 5, "a genuine dead end must exhaust every option first"


def test_a_settle_timeout_never_discards_banked_progress(monkeypatch):
    """The map takes far longer to settle exactly when a clear opens a new path
    -- which is exactly when game progress has just risen. `archive_map_state`
    returned early on that timeout, so the only mechanism that recorded new
    progress gave up whenever there was any: 98 banked events over one night and
    not one surviving cell. The frontier sat at `events 1` for hours while the
    explorers re-discovered and re-discarded the same three events.
    """
    import smwrl.overworld as ow
    import smwrl.worldrun as wr
    from smwrl.world_archive import MAP_CELL, WorldArchive

    class FakeEnv:
        class em:
            @staticmethod
            def get_state():
                return b"snapshot"

    def never_settles(*a, **k):
        raise TimeoutError("overworld never became controllable")

    monkeypatch.setattr(ow, "wait_settled", never_settles)
    monkeypatch.setattr(ow, "position", lambda env: (184, 72))

    a = WorldArchive()
    wr.archive_map_state(FakeEnv(), a, cleared=3, steps=900)
    cell = a.cells[a.key_for(MAP_CELL, 0, 184, 72)]
    assert cell.cleared == 3, "an unsettled map must not cost us the progress"

    # Even an unreadable position must not: the snapshot is the valuable part.
    def no_position(env):
        raise ValueError("mid-animation")

    monkeypatch.setattr(ow, "position", no_position)
    b = WorldArchive()
    wr.archive_map_state(FakeEnv(), b, cleared=2, steps=10)
    assert b.best_cell.cleared == 2


def test_promotion_needs_evidence_not_one_lucky_run():
    """`world_train` writes latest.zip in place and kept nothing else, so a policy
    that got worse silently destroyed the one that was good -- the run scoring
    0.49 levels per unaided episode was overwritten with no way back.

    Levels cleared per episode is a count, not a coin flip, so the Wilson bound
    autopilot uses does not apply; a normal approximation on the mean does the
    same job of refusing to promote on noise.
    """
    from smwrl.world_evaluate import evaluation_score, mean_lower
    assert mean_lower([]) == 0.0
    assert mean_lower([4]) == 0.0, "a single episode is never evidence"
    steady = mean_lower([2, 2, 2, 2, 2, 2, 2, 2])
    spiky = mean_lower([0, 0, 0, 0, 0, 0, 0, 16])
    assert steady == 2.0, "no variance means the mean is the bound"
    assert spiky < steady, "same mean, but one lucky episode must not promote"
    assert mean_lower([0, 0, 0, 0]) == 0.0
    # Ranked on evidence first, then raw mean, then distance travelled.
    assert (evaluation_score({"levels_lower": 1.0, "mean_levels": 1.2, "mean_progress": 900})
            > evaluation_score({"levels_lower": 0.9, "mean_levels": 4.0, "mean_progress": 9000}))


def test_promotion_refuses_a_candidate_it_did_not_measure(tmp_path):
    """The digest check is the whole point: promotion copies a file, and the file
    it copies must be byte-identical to the one the score was measured on.
    world_train republishes latest.zip every 100k steps, which at ~450 steps/s
    lands inside a 20-episode evaluation often enough to matter.
    """
    import pytest as _pytest
    from smwrl.world_evaluate import promote_best
    source = tmp_path / "candidate_eval.zip"
    source.write_bytes(b"a policy")
    good = {"episodes": 20, "levels_lower": 1.0, "mean_levels": 1.0,
            "mean_progress": 500,
            "model_sha256": __import__("hashlib").sha256(b"a policy").hexdigest()}

    assert promote_best(source, good, None, directory=tmp_path) is True
    assert (tmp_path / "best.zip").read_bytes() == b"a policy"

    stale = dict(good, model_sha256="0" * 64)
    with _pytest.raises(ValueError):
        promote_best(source, stale, None, directory=tmp_path)

    # A weaker candidate must not displace the incumbent.
    weaker = dict(good, levels_lower=0.5, mean_levels=0.5)
    assert promote_best(source, weaker, good, directory=tmp_path) is False
    assert promote_best(source, {"episodes": 0}, None, directory=tmp_path) is False


def test_curriculum_never_starts_an_episode_somewhere_unclearable():
    """Index 0 is what every unaided episode restores and the only thing SOLO
    measures. Yoshi's House has no goal tape and map cells sit on an overworld
    nothing scripts at reset, and both carry zero game progress -- so they sort
    below every real cell and sank to the front as the archive grew. At 882
    cells index 0 was Yoshi's House at x 53: half of all training resets began
    in a room with no exit, and SOLO read 0.00 for 540k steps while measuring
    the curriculum rather than the policy.
    """
    import tempfile, pathlib
    from smwrl.world_archive import MAP_CELL, WorldArchive
    from smwrl.world_env import curriculum_from_archive
    from smwrl.worldrun import YOSHIS_HOUSE

    a = WorldArchive()
    a.consider(MAP_CELL, 0, 104, 120, 0, _blob(), 0)        # standing on the map
    a.consider(YOSHIS_HOUSE, 0, 53, 300, 0, _blob(), 1)     # the tutorial room
    a.consider(YOSHIS_HOUSE, 0, 190, 300, 0, _blob(), 2)
    a.consider(42, 0, 448, 300, 0, _blob(), 60)             # a real level
    a.consider(39, 1, 5076, 300, 1, _blob(), 900)           # deeper, one event
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "world.pkl"
        a.save(p)
        curriculum = curriculum_from_archive(p, size=64)
        startable = {k: c for k, c in a.cells.items()
                     if c.translevel not in (MAP_CELL, YOSHIS_HOUSE)}
        assert len(curriculum) == len(startable) == 2
        assert curriculum[0] == a.cells[a.key_for(42, 0, 448, 300)].state, \
            "the unaided start must be a level that can actually be cleared"
        assert curriculum[-1] == a.cells[a.key_for(39, 1, 5076, 300)].state, \
            "and the curriculum must still run start -> deepest"


def test_event_bonus_pays_only_for_new_game_progress():
    """The whole-game reward paid for distance, rooms and clears -- never for the
    game recording progress. The policy was therefore indifferent between the
    exit that opens a map path and the one that does not, and a level already
    beaten paid the full clear bonus every time it was beaten again.
    """
    from smwrl.world_env import progress_bonus
    from smwrl.wrappers import RewardConfig
    cfg = RewardConfig()
    assert cfg.event_bonus > cfg.clear_bonus, "progress must outrank a re-clear"
    assert progress_bonus(cfg, 1, 2) == cfg.event_bonus
    assert progress_bonus(cfg, 1, 3) == 2 * cfg.event_bonus, "two events, two bonuses"
    assert progress_bonus(cfg, 2, 2) == 0, "a re-clear fires no event and pays nothing"
    # A curriculum episode resumes with events already banked. Reading the
    # baseline from RAM is what stops it being paid for a previous run's work.
    assert progress_bonus(cfg, 7, 7) == 0
    assert progress_bonus(cfg, 3, 1) == 0, "progress cannot go backwards"


def test_step_budget_extends_with_levels_cleared():
    """A flat cap cannot express a whole-game run: 4,000 agent steps is about
    four minutes and the shortest real completion is over forty. Raising the cap
    alone would make every stuck episode proportionally more expensive -- and
    deterministic play right now times out at x~193, inching just enough to keep
    resetting the stuck detector.
    """
    from smwrl.world_env import WorldEpisodeConfig, step_budget
    cfg = WorldEpisodeConfig()
    assert step_budget(cfg, 0) == cfg.max_steps, "a stuck episode is still cut off early"
    assert step_budget(cfg, 1) == cfg.max_steps + cfg.steps_per_level
    assert step_budget(cfg, 10) == cfg.max_steps + 10 * cfg.steps_per_level
    assert step_budget(cfg, 10) > 40_000, "a full run must fit inside one episode"
    assert step_budget(cfg, -1) == cfg.max_steps


def test_policy_watcher_defaults_to_demanding_its_own_level():
    """The whole-game viewer needs a watcher that accepts a level-less contract,
    but relaxing that by default would silently disable the per-level viewer's
    only guard against loading another level's policy.
    """
    from smwrl.live import PolicyWatcher
    assert PolicyWatcher("YoshiIsland1").expected_level == "YoshiIsland1"
    assert PolicyWatcher("YoshiIsland1_s2").expected_level == "YoshiIsland1_s2"
    assert PolicyWatcher("world", expected_level=None).expected_level is None


def test_resumed_training_does_not_fire_its_thresholds_every_step():
    """Thresholds were seeded at `every` / `save_every` regardless of resume, so
    a run resuming at 2.9M steps had both already millions of steps in the past.
    It printed 145 progress lines and republished the 23MB checkpoint 29 times
    back to back before the counters caught up.
    """
    import pathlib
    from smwrl.world_train import WorldProgress
    cb = WorldProgress(pathlib.Path("/nonexistent"), every=20_000, save_every=100_000)
    cb.num_timesteps = 2_900_000
    cb.locals = {"infos": []}
    cb._on_step()
    assert cb._next == 2_920_000
    assert cb._next_save == 3_000_000


def test_world_trainer_declares_the_observation_its_envs_use():
    """`world_train` saved with a bare model.save, so checkpoints/world/latest.zip
    carried no observation contract and `load_policy_runtime` refused it -- the
    whole-game policy could not be loaded by any consumer without guessing.

    One definition feeds both the envs and the published contract; two would let
    a checkpoint describe an observation it was never trained on.
    """
    from smwrl import world_train
    assert world_train.OBS.channels * world_train.FRAME_STACK == 12
    assert world_train.OBS.size == 84


def test_archive_tracks_visited_map_nodes():
    """Branch-aware navigation needs to know which map nodes it has entered.

    Fixed-order direction choice always took RIGHT at Yoshi's House, so
    Yoshi's Island 1 was never played -- and the game gates the path past
    Yoshi's Island 3 on it.
    """
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    assert a.visited() == set()
    a.visited().add((56, 136))
    assert (56, 136) in a.visited()

    b = WorldArchive()
    b.visited().add((152, 136))
    b.consider(42, 0, 100, 200, 0, _blob(), 1)
    a.absorb(b)
    assert a.visited() == {(56, 136), (152, 136)}, "merge must union visited nodes"


def test_visited_survives_archives_pickled_before_it_existed():
    import pickle as _p
    from smwrl.world_archive import WorldArchive
    a = WorldArchive()
    a.consider(42, 0, 100, 200, 0, _blob(), 1)
    a.__dict__.pop("nodes", None)          # simulate an older pickle
    revived = _p.loads(_p.dumps(a))
    assert revived.visited() == set()
    revived.visited().add((1, 2))
    assert revived.visited() == {(1, 2)}
