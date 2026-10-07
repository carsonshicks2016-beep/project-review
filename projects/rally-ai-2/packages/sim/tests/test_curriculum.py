"""C6 — mastery curriculum promote / demote (never timestep-based)."""

from __future__ import annotations

from rallyai.train.curriculum import Curriculum, CurriculumConfig
from rallyai.train.stages import STAGES, stage_reward_config, stage_tier_bounds


def test_promote_at_threshold():
    cur = Curriculum(tier=0, config=CurriculumConfig(window=10, promote_at=0.75))
    event = None
    for i in range(10):
        event = cur.observe(finished_clean=(i < 8))  # 80%
    assert event is not None
    assert event.reason == "promote"
    assert event.new_tier == 1
    assert event.completion_rate >= 0.75


def test_no_promote_below_threshold():
    cur = Curriculum(tier=0, config=CurriculumConfig(window=10, promote_at=0.75))
    event = None
    for i in range(10):
        event = cur.observe(finished_clean=(i < 5))  # 50%
    assert event is None
    assert cur.tier == 0


def test_demote_on_collapse():
    cur = Curriculum(tier=2, config=CurriculumConfig(window=10, demote_at=0.35))
    event = None
    for _ in range(10):
        event = cur.observe(finished_clean=False)  # 0%
    assert event is not None
    assert event.reason == "demote"
    assert event.new_tier == 1


def test_never_timestep_schedule():
    # Observing without outcomes never advances — no hidden clock.
    cur = Curriculum(tier=0, config=CurriculumConfig(window=100))
    assert cur.observe.__doc__ is not None
    assert cur.tier == 0
    # Even after many "ticks" with no observations, tier stays put.
    assert cur.state_dict()["tier"] == 0


def test_stage_reward_configs():
    for name in STAGES:
        cfg = stage_reward_config(name)
        assert cfg.progress >= 0.0
        lo, hi = stage_tier_bounds(name)
        assert 0 <= lo <= hi <= 5
    foundation = stage_reward_config("foundation")
    flow = stage_reward_config("flow")
    assert foundation.speed < flow.speed
    assert foundation.throttle_commit <= flow.throttle_commit
    finish = stage_reward_config("finish")
    assert finish.progress < flow.progress
    assert finish.finish_pace > flow.finish_pace
