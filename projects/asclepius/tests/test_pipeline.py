"""End-to-end verification against synthetic data with known ground truth.

Records are built in the exact shape the WHOOP API returns and pushed through
the real mappers, so this exercises the actual store -> frame -> analyze path.

The generator plants a specific causal structure:

    recovery[d] = 60 - 1.8 * strain[d-1] + 4.0 * sleep_hours[d] + noise

That is, yesterday's training hurts today's recovery, and last night's sleep
helps it. The tests assert the engine recovers both effects at the correct lag
and with roughly the right magnitude. If the sleep/cycle join were wired the
wrong way round, the lag-1 strain effect would show up at the wrong lag and
these tests would fail.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np
import pytest

from asclepius import analyze, store
from asclepius.frame import _clock_hours_around_midnight, build_frame

TZ = "-05:00"
TZ_SHIFT = timedelta(hours=5)  # local -> UTC
DAYS = 400

B_STRAIN = -1.8
B_SLEEP = 4.0
INTERCEPT = 60.0


def _utc(local: datetime) -> str:
    return (local + TZ_SHIFT).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def generate(seed: int = 7):
    rng = np.random.default_rng(seed)
    day0 = datetime(2024, 1, 1)

    strain = rng.uniform(4, 18, DAYS)
    sleep_h = rng.normal(7.2, 1.1, DAYS).clip(3.5, 10.5)
    noise = rng.normal(0, 6.0, DAYS)

    cycles, recoveries, sleeps, workouts = [], [], [], []

    for d in range(DAYS):
        day = day0 + timedelta(days=d)
        cycle_id = 1000 + d

        cycles.append(
            {
                "id": cycle_id,
                "user_id": 1,
                "created_at": _utc(day.replace(hour=6)),
                "updated_at": _utc(day.replace(hour=6)),
                "start": _utc(day.replace(hour=6)),
                "end": _utc(day.replace(hour=6) + timedelta(days=1)),
                "timezone_offset": TZ,
                "score_state": "SCORED",
                "score": {
                    "strain": float(strain[d]),
                    "kilojoule": float(strain[d] * 420),
                    "average_heart_rate": int(60 + strain[d] * 2),
                    "max_heart_rate": int(120 + strain[d] * 3),
                },
            }
        )

        # The night that ENDS on the morning of day d. It occurred during the
        # previous cycle, which is what its own cycle_id reflects.
        sid = str(uuid.uuid4())
        bed = day.replace(hour=22) - timedelta(days=1)
        wake = day.replace(hour=6)
        asleep_ms = int(sleep_h[d] * 3_600_000)
        sleeps.append(
            {
                "id": sid,
                "cycle_id": cycle_id - 1,
                "user_id": 1,
                "created_at": _utc(wake),
                "updated_at": _utc(wake),
                "start": _utc(bed),
                "end": _utc(wake),
                "timezone_offset": TZ,
                "nap": False,
                "score_state": "SCORED",
                "score": {
                    "stage_summary": {
                        "total_in_bed_time_milli": asleep_ms + 1_800_000,
                        "total_awake_time_milli": 1_800_000,
                        "total_no_data_time_milli": 0,
                        "total_light_sleep_time_milli": int(asleep_ms * 0.55),
                        "total_slow_wave_sleep_time_milli": int(asleep_ms * 0.22),
                        "total_rem_sleep_time_milli": int(asleep_ms * 0.23),
                        "sleep_cycle_count": 4,
                        "disturbance_count": int(rng.integers(0, 12)),
                    },
                    "sleep_needed": {
                        "baseline_milli": 28_800_000,
                        "need_from_sleep_debt_milli": 1_200_000,
                        "need_from_recent_strain_milli": 600_000,
                        "need_from_recent_nap_milli": 0,
                    },
                    "respiratory_rate": float(rng.normal(14.5, 0.7)),
                    "sleep_performance_percentage": float(
                        min(100, sleep_h[d] / 8.0 * 100)
                    ),
                    "sleep_consistency_percentage": float(rng.uniform(50, 95)),
                    "sleep_efficiency_percentage": float(rng.uniform(85, 97)),
                },
            }
        )

        prev_strain = strain[d - 1] if d > 0 else strain.mean()
        score = INTERCEPT + B_STRAIN * prev_strain + B_SLEEP * sleep_h[d] + noise[d]
        recoveries.append(
            {
                "cycle_id": cycle_id,
                "sleep_id": sid,
                "user_id": 1,
                "created_at": _utc(wake),
                "updated_at": _utc(wake),
                "score_state": "SCORED",
                "score": {
                    "user_calibrating": False,
                    "recovery_score": float(np.clip(score, 1, 100)),
                    "resting_heart_rate": float(55 + prev_strain * 0.4),
                    "hrv_rmssd_milli": float(max(10, 90 - prev_strain * 2)),
                    "spo2_percentage": float(rng.normal(96, 0.8)),
                    "skin_temp_celsius": float(rng.normal(33.5, 0.3)),
                },
            }
        )

        if strain[d] > 12:
            start = day.replace(hour=17)
            workouts.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": 1,
                    "created_at": _utc(start),
                    "updated_at": _utc(start),
                    "start": _utc(start),
                    "end": _utc(start + timedelta(minutes=60)),
                    "timezone_offset": TZ,
                    "sport_name": "running",
                    "sport_id": 0,
                    "score_state": "SCORED",
                    "score": {
                        "strain": float(strain[d] * 0.8),
                        "average_heart_rate": 150,
                        "max_heart_rate": 178,
                        "kilojoule": 2000.0,
                        "percent_recorded": 100.0,
                        "distance_meter": 10000.0,
                        "altitude_gain_meter": 50.0,
                        "zone_durations": {
                            "zone_zero_milli": 0,
                            "zone_one_milli": 300_000,
                            "zone_two_milli": 600_000,
                            "zone_three_milli": 1_200_000,
                            "zone_four_milli": 900_000,
                            "zone_five_milli": 600_000,
                        },
                    },
                }
            )

    return cycles, recoveries, sleeps, workouts


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    db = tmp_path_factory.mktemp("whoop") / "test.db"
    conn = store.connect(db)
    cycles, recoveries, sleeps, workouts = generate()
    store.save(conn, "cycles", cycles)
    store.save(conn, "recovery", recoveries)
    store.save(conn, "sleeps", sleeps)
    store.save(conn, "workouts", workouts)
    return build_frame(conn)


def test_frame_shape(df):
    assert len(df) == DAYS
    assert df["recovery_score"].notna().sum() == DAYS
    assert df["strain"].notna().sum() == DAYS
    assert df["sleep_hours"].notna().sum() == DAYS


def test_sleep_joined_to_the_night_that_ended_that_morning(df):
    """Sleep must align to its own row, not shift by a day."""
    r = analyze.lagged_correlation(df, "sleep_hours", "recovery_score", max_lag=2)
    assert r["strongest_lag"]["lag"] == 0
    assert r["strongest_lag"]["r"] > 0.3


def test_strain_affects_next_day_recovery_at_lag_one(df):
    """The planted effect is strain[d-1] -> recovery[d], i.e. lag 1."""
    r = analyze.lagged_correlation(df, "strain", "recovery_score", max_lag=3)
    assert r["strongest_lag"]["lag"] == 1
    assert r["strongest_lag"]["r"] < -0.3
    lag0 = next(x for x in r["lags"] if x["lag"] == 0)
    lag1 = next(x for x in r["lags"] if x["lag"] == 1)
    assert abs(lag1["r"]) > abs(lag0["r"])


def test_regression_recovers_planted_coefficients(df):
    out = analyze.regress(
        df, "recovery_score", drivers=["strain"], controls=[], lag=1
    )
    coef = next(t for t in out["terms"] if t["predictor"] == "strain")["coef"]
    assert coef == pytest.approx(B_STRAIN, abs=0.5)

    out = analyze.regress(
        df, "recovery_score", drivers=["sleep_hours"], controls=[], lag=0
    )
    coef = next(t for t in out["terms"] if t["predictor"] == "sleep_hours")["coef"]
    assert coef == pytest.approx(B_SLEEP, abs=0.8)


def test_multivariate_separates_both_effects(df):
    """Both planted drivers should survive together in one model."""
    shifted = df.copy()
    shifted["prev_strain"] = shifted["strain"].shift(1)
    from asclepius.frame import METRICS

    METRICS.setdefault("prev_strain", ("Previous day strain", "0-21"))

    out = analyze.regress(
        shifted, "recovery_score", drivers=["prev_strain", "sleep_hours"], lag=0
    )
    terms = {t["predictor"]: t for t in out["terms"]}
    assert terms["prev_strain"]["coef"] == pytest.approx(B_STRAIN, abs=0.5)
    assert terms["sleep_hours"]["coef"] == pytest.approx(B_SLEEP, abs=0.8)
    assert terms["prev_strain"]["p"] < 0.01
    assert terms["sleep_hours"]["p"] < 0.01
    assert out["adj_r_squared"] > 0.3


def test_effective_n_is_below_raw_n_for_autocorrelated_series(df):
    """The autocorrelation correction must actually bite."""
    smooth = df["recovery_score"].rolling(5).mean().dropna().to_numpy()
    n_eff = analyze._effective_n(smooth, smooth)
    assert n_eff < len(smooth)


def test_fdr_is_monotone_and_at_least_raw_p():
    p = [0.001, 0.01, 0.02, 0.2, 0.5, 0.9]
    q = analyze.benjamini_hochberg(p)
    assert all(qi >= pi - 1e-12 for pi, qi in zip(p, q))
    assert q == sorted(q)


def test_rank_drivers_surfaces_the_planted_driver(df):
    out = analyze.rank_drivers(df, "recovery_score", lag=1, top_n=30)
    top = [r["metric"] for r in out["results"][:6]]
    assert "strain" in top
    assert any(
        r["q_fdr"] is not None and r["q_fdr"] < 0.05 for r in out["results"]
    )
    assert out["significant_after_fdr"], "a planted driver must survive FDR"


def test_anomaly_detection_flags_an_injected_spike(df):
    spiked = df.copy()
    target = spiked.index[-3]
    spiked.loc[target, "resting_hr"] = spiked["resting_hr"].mean() + 6 * spiked[
        "resting_hr"
    ].std()
    out = analyze.anomalies(spiked, metrics=["resting_hr"], last_n_days=10)
    assert any(f["date"] == target.date().isoformat() for f in out["flags"])


def test_no_false_collinearity_warning_for_independent_predictors(df):
    """Predictors on very different scales are not collinear.

    strain (0-21) and sleep_hours (3-11) are independent by construction here,
    so a scale-sensitive diagnostic that flagged them would be wrong.
    """
    shifted = df.copy()
    shifted["prev_strain"] = shifted["strain"].shift(1)
    from asclepius.frame import METRICS

    METRICS.setdefault("prev_strain", ("Previous day strain", "0-21"))

    out = analyze.regress(
        shifted, "recovery_score", drivers=["prev_strain", "sleep_hours"]
    )
    assert not any("ollinear" in w for w in out["warnings"])
    for t in out["terms"]:
        assert t["vif"] < 2.0


def test_genuine_collinearity_is_flagged(df):
    """Two near-duplicate predictors must trip the VIF warning."""
    out = analyze.regress(
        df, "recovery_score", drivers=["sleep_hours", "in_bed_hours"]
    )
    assert any("ollinear" in w for w in out["warnings"])
    assert max(t["vif"] for t in out["terms"]) > 5


def test_perfect_collinearity_is_flagged_not_silently_dropped(df):
    """An infinite VIF is the worst case and must still produce a warning."""
    import numpy as np

    from asclepius.frame import METRICS

    dup = df.copy()
    dup["strain_copy"] = dup["strain"]
    METRICS.setdefault("strain_copy", ("Strain (duplicate)", "0-21"))

    out = analyze.regress(dup, "recovery_score", drivers=["strain", "strain_copy"])
    assert any("ollinear" in w for w in out["warnings"])
    assert any("not identified" in w for w in out["warnings"])
    assert any(np.isinf(t["vif"]) for t in out["terms"])


def test_tiny_pvalues_are_not_rounded_to_zero(df):
    out = analyze.lagged_correlation(df, "strain", "recovery_score", max_lag=2)
    strongest = out["strongest_lag"]
    assert strongest["p"] > 0, "a real p-value must never serialize as exactly 0"
    assert strongest["p"] < 1e-6


def test_insufficient_data_raises_cleanly(df):
    with pytest.raises(analyze.NotEnoughData):
        analyze.regress(df.head(5), "recovery_score", drivers=["strain"])
    with pytest.raises(analyze.NotEnoughData):
        analyze.lagged_correlation(df, "not_a_metric", "recovery_score")


def test_clock_hours_stay_continuous_across_midnight_and_noon():
    """Sleep boundaries must never straddle the wrap point.

    Regression test for two distinct bugs. First, the comparison was `< 12`
    rather than `>=`, mapping 23:00 to +23 and 00:15 to -23.75 — a 47-hour gap
    for 75 minutes of real difference. Second, wrapping at noon was itself
    wrong: real wake times run right up to and past 12:00, so an 11:58 wake and
    a 12:05 wake landed ~24 apart. Both failures are silent — they produce
    finite, plausible-looking numbers that destroy any correlation using them.
    """
    f = _clock_hours_around_midnight

    assert f(datetime(2026, 1, 1, 23, 0)) == pytest.approx(-1.0)
    assert f(datetime(2026, 1, 1, 0, 30)) == pytest.approx(0.5)
    assert f(datetime(2026, 1, 1, 7, 0)) == pytest.approx(7.0)
    assert f(datetime(2026, 1, 1, 12, 0)) == pytest.approx(12.0)
    assert f(datetime(2026, 1, 1, 20, 0)) == pytest.approx(-4.0)
    assert f(None) is None

    # 30 minutes either side of midnight stays 30 minutes apart
    assert (f(datetime(2026, 1, 1, 0, 15)) - f(datetime(2026, 1, 1, 23, 45))
            == pytest.approx(0.5))

    # and so does a wake time either side of noon
    assert (f(datetime(2026, 1, 1, 12, 5)) - f(datetime(2026, 1, 1, 11, 58))
            == pytest.approx(7 / 60))

    # every real sleep boundary must sit in one contiguous block: no two
    # adjacent clock times may land more than their true distance apart
    for band in ((20, 24), (0, 15)):  # bedtimes, then wake times
        vals = [f(datetime(2026, 1, 1, h, m)) for h in range(*band)
                for m in (0, 30)]
        assert max(vals) - min(vals) < 20, "band was split by the wrap"

    assert all(-7.0 <= f(datetime(2026, 1, 1, h, 0)) < 17.0 for h in range(24))
