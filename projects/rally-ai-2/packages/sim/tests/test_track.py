"""Geometry tests.

Everything downstream trusts these: the physics asks where the road is, the
sensors ask what the car can see, the reward asks how far along it got. A
silent error here does not crash — it trains a policy against a world that is
subtly not the one being rendered.
"""

from __future__ import annotations

import numpy as np
import pytest

from rallyai.stage.builder import ProfileBuilder, build_stage, severity_for_radius
from rallyai.stage.fixtures import flat_straight, proving_ground
from rallyai.track import HIT_EDGE, HIT_OBSTACLE, Track


@pytest.fixture(scope="module")
def pg() -> Track:
    return Track(proving_ground())


@pytest.fixture(scope="module")
def flat() -> Track:
    return Track(flat_straight(200.0, width=10.0))


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #

def test_centerline_points_project_to_zero_lateral(pg):
    for i in range(5, len(pg.s) - 5, 7):
        q = pg.project(float(pg.x[i]), float(pg.y[i]), hint_s=float(pg.s[i]))
        assert abs(q.lateral) < 1e-3, f"sample {i} projected {q.lateral} m off its own centerline"
        assert abs(q.s - pg.s[i]) < 1e-2


@pytest.mark.parametrize("offset", [-3.5, -1.0, 0.5, 2.0, 3.5])
def test_lateral_offset_is_recovered_with_correct_sign(pg, offset):
    """Positive lateral is to the RIGHT of travel — the stage schema says so and
    the reward's edge taper depends on it."""
    i = 60
    x = float(pg.x[i] + pg.nx[i] * offset)
    y = float(pg.y[i] + pg.ny[i] * offset)
    q = pg.project(x, y, hint_s=float(pg.s[i]))
    assert q.lateral == pytest.approx(offset, abs=1e-3)


def test_right_normal_points_right_of_travel(flat):
    """A stage heading east (+x) must have its right-hand normal pointing south
    (-y). If this flips, every corner banks the wrong way."""
    i = 100
    assert flat.heading[i] == pytest.approx(0.0, abs=1e-6)
    assert flat.nx[i] == pytest.approx(0.0, abs=1e-6)
    assert flat.ny[i] == pytest.approx(-1.0, abs=1e-6)


def test_hinted_and_global_projection_agree(pg):
    """The hint is an optimisation. If it changes the answer it is a bug."""
    for i in (40, 150, 300, 450):
        x, y = float(pg.x[i]) + 2.0, float(pg.y[i]) - 1.0
        a = pg.project(x, y, hint_s=float(pg.s[i]))
        b = pg.project(x, y, hint_s=None)
        assert a.s == pytest.approx(b.s, abs=1e-6)
        assert a.lateral == pytest.approx(b.lateral, abs=1e-6)


def test_projection_is_stable_at_the_stage_ends(pg):
    for s_probe in (0.0, 0.5, pg.length - 0.5, pg.length):
        i = int(np.clip(s_probe / pg.ds, 0, len(pg.s) - 1))
        q = pg.project(float(pg.x[i]), float(pg.y[i]), hint_s=s_probe)
        assert np.isfinite(q.s) and np.isfinite(q.lateral)
        assert 0.0 <= q.progress <= 1.0


def test_on_track_follows_half_width(pg):
    i = 60
    half = float(pg.half_width[i])
    for offset, expected in ((half - 0.2, True), (half + 0.2, False)):
        x = float(pg.x[i] + pg.nx[i] * offset)
        y = float(pg.y[i] + pg.ny[i] * offset)
        assert pg.project(x, y, hint_s=float(pg.s[i])).on_track is expected


def test_heading_error_wraps(flat):
    q = flat.project(100.0, 0.0, yaw=np.pi + 0.1, hint_s=100.0)
    assert -np.pi <= q.heading_error <= np.pi
    assert q.heading_error == pytest.approx(-np.pi + 0.1, abs=1e-6)


# --------------------------------------------------------------------------- #
# raycast
# --------------------------------------------------------------------------- #

def test_raycast_hits_corridor_edge_at_half_width(flat):
    angles = np.radians([90.0, -90.0])
    d, kinds, _ = flat.raycast(100.0, 0.0, 0.0, angles, 60.0, hint_s=100.0)
    assert d[0] == pytest.approx(5.0, abs=1e-3)
    assert d[1] == pytest.approx(5.0, abs=1e-3)
    assert np.all(kinds == HIT_EDGE)


def test_raycast_diagonal_geometry(flat):
    """A 45-degree beam in a 10 m corridor must read half_width / cos(45)."""
    d, _, _ = flat.raycast(100.0, 0.0, 0.0, np.radians([45.0]), 60.0, hint_s=100.0)
    assert d[0] == pytest.approx(5.0 / np.cos(np.radians(45.0)), abs=1e-3)


def test_raycast_clamps_to_max_range(flat):
    """Straight down an empty road, the beam reports the range limit, not inf."""
    d, _, _ = flat.raycast(20.0, 0.0, 0.0, np.array([0.0]), 60.0, hint_s=20.0)
    assert d[0] == pytest.approx(60.0)


def test_raycast_detects_obstacle_and_labels_it():
    """The kind channel is what lets the policy tell a survivable edge from a
    tree. A driver can make that distinction, so the agent may too."""
    stage = flat_straight(200.0, width=10.0)
    stage["obstacles"] = [
        {"kind": "tree", "x": 130.0, "y": 0.0, "radius": 0.5, "s": 130.0, "lateral": 0.0}
    ]
    track = Track(stage)
    d, kinds, _ = track.raycast(100.0, 0.0, 0.0, np.array([0.0]), 60.0, hint_s=100.0)
    assert d[0] == pytest.approx(29.5, abs=1e-3)
    assert kinds[0] == HIT_OBSTACLE


def test_raycast_reports_nearest_hit_when_obstacle_is_behind_the_edge():
    """A tree outside the corridor, in line with a beam that clips the edge
    first, must not mask the edge."""
    stage = flat_straight(200.0, width=10.0)
    stage["obstacles"] = [
        {"kind": "tree", "x": 100.0, "y": -8.0, "radius": 0.5, "s": 100.0, "lateral": 8.0}
    ]
    track = Track(stage)
    d, kinds, _ = track.raycast(100.0, 0.0, 0.0, np.radians([-90.0]), 60.0, hint_s=100.0)
    assert d[0] == pytest.approx(5.0, abs=1e-3)
    assert kinds[0] == HIT_EDGE


def test_raycast_hit_points_lie_along_their_beams(pg):
    angles = np.radians(np.linspace(60, -60, 9))
    d, _, pts = pg.raycast(float(pg.x[100]), float(pg.y[100]), float(pg.heading[100]),
                           angles, 60.0, hint_s=float(pg.s[100]))
    for a, dist, pt in zip(angles, d, pts):
        wa = pg.heading[100] + a
        expect = (pg.x[100] + np.cos(wa) * dist, pg.y[100] + np.sin(wa) * dist)
        assert pt[0] == pytest.approx(expect[0], abs=1e-6)
        assert pt[1] == pytest.approx(expect[1], abs=1e-6)


def test_raycast_never_returns_negative_or_nan(pg):
    angles = np.radians(np.linspace(90, -90, 15))
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = int(rng.integers(5, len(pg.s) - 5))
        x = float(pg.x[i] + pg.nx[i] * rng.normal(0, 3.0))
        y = float(pg.y[i] + pg.ny[i] * rng.normal(0, 3.0))
        d, _, _ = pg.raycast(x, y, float(rng.uniform(0, 2 * np.pi)), angles, 60.0,
                             hint_s=float(pg.s[i]))
        assert np.all(np.isfinite(d)) and np.all(d >= 0.0) and np.all(d <= 60.0)


# --------------------------------------------------------------------------- #
# look-ahead
# --------------------------------------------------------------------------- #

def test_lookahead_clamps_at_the_finish_and_does_not_wrap(pg):
    """Supra's circuit model wraps arc length modulo lap length. Ported without
    thinking, that shows a point-to-point agent the start line as road ahead."""
    near_end = pg.length - 5.0
    k = pg.lookahead_curvature(near_end, [0.0, 50.0, 500.0])
    assert k[-1] == pytest.approx(pg.curvature[-1])
    # The start of this stage is straight and so is the end, so also assert on a
    # quantity that actually differs between the two: height.
    z = pg.lookahead_grade(near_end, [500.0])
    assert z[0] == pytest.approx(pg.grade[-1])


def test_lookahead_at_stage_start_reads_forward(pg):
    k = pg.lookahead_curvature(0.0, [0.0, 100.0])
    assert k[0] == pytest.approx(pg.curvature[0])
    assert k[1] == pytest.approx(pg.curvature[100], abs=1e-6)


def test_lookahead_mu_sees_the_surface_change(pg):
    """The tarmac section must be visible before the car reaches it — that is
    the whole point of reading ahead."""
    mus = pg.lookahead_mu(0.0, np.arange(0, pg.length, 10.0))
    assert len(np.unique(np.round(mus, 3))) > 1


def test_pace_ahead_returns_next_note_then_none(pg):
    note = pg.pace_ahead(0.0, horizon=120.0)
    assert note is not None and note["dir"] == "right"
    assert pg.pace_ahead(0.0, horizon=5.0) is None
    assert pg.pace_ahead(pg.length, horizon=60.0) is None


# --------------------------------------------------------------------------- #
# collision and the corridor invariant
# --------------------------------------------------------------------------- #

def test_collide_obstacles_returns_none_when_clear(pg):
    normal, depth = pg.collide_obstacles(float(pg.x[50]), float(pg.y[50]), 0.95,
                                         hint_s=float(pg.s[50]))
    assert normal is None and depth == 0.0


def test_collide_obstacles_reports_outward_normal_and_depth():
    stage = flat_straight(200.0, width=10.0)
    stage["obstacles"] = [
        {"kind": "tree", "x": 100.0, "y": 0.0, "radius": 1.0, "s": 100.0, "lateral": 0.0}
    ]
    track = Track(stage)
    normal, depth = track.collide_obstacles(100.5, 0.0, 0.95, hint_s=100.0)
    assert depth == pytest.approx(1.0 + 0.95 - 0.5, abs=1e-6)
    assert normal[0] == pytest.approx(1.0, abs=1e-6)   # pushes the car +x, away
    assert normal[1] == pytest.approx(0.0, abs=1e-6)


def test_fixture_places_no_obstacle_inside_the_corridor(pg):
    assert pg.obstacles_in_corridor(car_radius=0.95) == []


def test_corridor_intrusion_check_actually_detects_an_intruder():
    """Positive control. Without this, the test above passes just as happily if
    obstacles_in_corridor always returns []. v1 shipped a stage with trees on
    the racing line and an observation that could not see them."""
    stage = flat_straight(200.0, width=10.0)
    stage["obstacles"] = [
        {"kind": "tree", "x": 100.0, "y": 0.0, "radius": 0.4, "s": 100.0, "lateral": 0.0}
    ]
    track = Track(stage)
    assert track.obstacles_in_corridor(car_radius=0.95) == [0]


# --------------------------------------------------------------------------- #
# build-time validation
# --------------------------------------------------------------------------- #

def test_surface_gap_is_rejected():
    """An uncovered sample means mu is undefined under the car. Fail loudly at
    build time rather than propagating a NaN into the tyre model."""
    stage = flat_straight(200.0)
    stage["surfaces"] = [{"s_start": 0.0, "s_end": 100.0, "type": "gravel", "mu": 0.7}]
    with pytest.raises(ValueError, match="uncovered"):
        Track(stage)


def test_too_few_centerline_points_is_rejected():
    stage = flat_straight(200.0)
    stage["centerline"] = stage["centerline"][:1]
    with pytest.raises(ValueError, match="at least 2"):
        Track(stage)


# --------------------------------------------------------------------------- #
# the builder produces the geometry it was asked for
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("radius", [10.0, 12.0, 20.0, 30.0, 60.0, 120.0])
@pytest.mark.parametrize("angle", [45.0, 90.0, 170.0])
def test_emitted_file_reproduces_the_described_road(radius, angle):
    """A stage file must be the road the builder described.

    The builder integrates a curvature profile, samples it down to knots, writes
    them, and Track fits a spline back through them. Every step of that round
    trip can lose the corner. Two bugs found here, both invisible without this
    test: fixed 4 m knots put 23 degrees between samples in a hairpin (a 12 m
    corner came back as 14 m), and once knot spacing became adaptive, a
    uniformly-parameterised spline overshot into the dense region (a 30 m corner
    came back as 26 m). Worst case across this grid is now 5.8%.
    """
    profile = (ProfileBuilder(width=8.0)
               .straight(30).corner(radius, angle, "left").straight(30).build())
    stage = build_stage(profile, stage_id="t", name="t", seed=0, tier=0, generator_version=1)
    track = Track(stage)
    measured = 1.0 / float(np.abs(track.curvature).max())
    assert measured == pytest.approx(radius, rel=0.06)


def test_knots_are_denser_through_tight_corners():
    """Adaptive spacing is what makes the round trip survive a hairpin. If this
    reverts to fixed spacing, the test above starts failing on tight radii."""
    tight = build_stage(
        ProfileBuilder().straight(50).corner(10, 90, "left").straight(50).build(),
        stage_id="t", name="t", seed=0, tier=0, generator_version=1)
    open_ = build_stage(
        ProfileBuilder().straight(50).corner(200, 90, "left").straight(50).build(),
        stage_id="t", name="t", seed=0, tier=0, generator_version=1)

    def gaps(stage, s_lo, s_hi):
        ss = np.array([p["s"] for p in stage["centerline"]])
        window = (ss[:-1] >= s_lo) & (ss[:-1] <= s_hi)
        return np.diff(ss)[window]

    # Through the held part of the 10 m corner, every knot is close together.
    assert gaps(tight, 52.0, 60.0).max() < 2.0
    # Through the 200 m sweeper, spacing stays at the plain distance cap —
    # curvature never accumulates 6 degrees fast enough to force a knot.
    assert gaps(open_, 100.0, 300.0).min() > 3.5


def test_straights_stay_sparse():
    """Adaptive spacing must not make every stage file enormous."""
    stage = build_stage(ProfileBuilder().straight(400).build(),
                        stage_id="t", name="t", seed=0, tier=0, generator_version=1)
    assert len(stage["centerline"]) < 120


def test_left_corner_curves_left():
    """Sign convention: positive curvature turns left (counter-clockwise about
    +z). If this inverts, every pace note calls the wrong way."""
    profile = ProfileBuilder().straight(20).corner(30, 90, "left").straight(20).build()
    track = Track(build_stage(profile, stage_id="t", name="t", seed=0, tier=0,
                              generator_version=1))
    assert track.curvature.max() > 0
    assert track.heading[-1] > track.heading[0]


def test_arc_length_matches_requested_geometry():
    profile = ProfileBuilder().straight(100).build()
    track = Track(build_stage(profile, stage_id="t", name="t", seed=0, tier=0,
                              generator_version=1))
    assert track.length == pytest.approx(100.0, rel=0.01)


def test_camber_banks_into_the_turn_by_default():
    profile = ProfileBuilder().straight(20).corner(25, 90, "left").straight(20).build()
    track = Track(build_stage(profile, stage_id="t", name="t", seed=0, tier=0,
                              generator_version=1))
    i = int(np.argmax(track.curvature))
    assert track.camber[i] > 0, "a left-hander should bank positive by default"


def test_crest_produces_negative_vertical_curvature():
    """Negative vcurv is what the agent reads as 'this will launch you'."""
    profile = ProfileBuilder().straight(40).crest(40, 3.0).straight(40).build()
    track = Track(build_stage(profile, stage_id="t", name="t", seed=0, tier=0,
                              generator_version=1))
    assert track.vcurv.min() < -0.005


@pytest.mark.parametrize("radius,expected", [
    (10.0, 1), (20.0, 2), (35.0, 3), (60.0, 4), (100.0, 5), (300.0, 6),
])
def test_severity_follows_rally_convention(radius, expected):
    assert severity_for_radius(radius) == expected


def test_pace_notes_describe_the_proving_ground():
    notes = proving_ground()["pace_notes"]
    calls = [(n["dir"], n["severity"]) for n in notes]
    assert ("left", 1) in calls, "the 12 m hairpin should be called left 1"
    assert ("crest", 6) in calls
    assert any(d == "right" and sev <= 4 for d, sev in calls)
    assert [n["s"] for n in notes] == sorted(n["s"] for n in notes)
