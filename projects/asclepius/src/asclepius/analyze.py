"""Statistics over the daily frame.

Three corrections separate this from eyeballing a scatter plot, and they are
the reason the numbers here are more conservative than most wearable analyses:

1. Autocorrelation. Recovery today looks a lot like recovery yesterday. Two
   such series will correlate impressively by accident, because the effective
   number of independent observations is far below the number of days. Every
   correlation p-value is computed against a Bartlett-adjusted effective n, and
   regressions use Newey-West (HAC) standard errors.
2. Multiple comparisons. Screening 30 candidate drivers at 8 lags is 240 tests;
   at p<0.05 you expect a dozen false positives. Families of tests are reported
   with Benjamini-Hochberg FDR q-values alongside raw p.
3. Confounding. A raw correlation between late bedtime and poor recovery mostly
   restates that late nights are short nights. `regress` holds covariates fixed
   so the unique contribution is visible.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .frame import METRICS

MIN_N = 12


class NotEnoughData(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers


def _lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


def _effective_n(x: np.ndarray, y: np.ndarray) -> float:
    """Bartlett-style effective sample size for two autocorrelated series.

    n_eff = n * (1 - r1*r2) / (1 + r1*r2), clipped to never exceed n.
    """
    n = len(x)
    r1, r2 = _lag1_autocorr(x), _lag1_autocorr(y)
    prod = max(min(r1 * r2, 0.99), -0.99)
    n_eff = n * (1 - prod) / (1 + prod)
    return float(max(MIN_N * 0.5, min(n_eff, n)))


def _corr_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = len(x)
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    n_eff = _effective_n(x, y)

    # p and CI recomputed against n_eff via Fisher's z. `sf` rather than
    # `1 - cdf`: the latter underflows to exactly 0.0 for strong effects,
    # turning a p of 1e-60 into a value that reads as "no result".
    if n_eff > 3 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1 / math.sqrt(n_eff - 3)
        p = 2 * float(stats.norm.sf(abs(z) / se))
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        p, lo, hi = float("nan"), float("nan"), float("nan")

    return {
        "r": round(float(r), 4),
        "spearman": round(float(rho), 4),
        "p": float(p),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n": int(n),
        "n_effective": round(n_eff, 1),
    }


def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    """Return BH-adjusted q-values, NaN-safe and order-preserving."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q.tolist()

    sub = p[ok]
    m = len(sub)
    order = np.argsort(sub)
    ranked = sub[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q.tolist()


def _p(x: float) -> float:
    """Round a p-value without collapsing tiny ones to a misleading 0.0."""
    if not np.isfinite(x):
        return float("nan")
    return float(x) if x < 1e-6 else round(float(x), 6)


def _vif(X: pd.DataFrame) -> dict[str, float]:
    """Variance inflation factor per predictor.

    Used instead of the design matrix condition number, which is dominated by
    differences in scale between predictors (strain runs 0-21, sleep 0-12) and
    would flag collinearity for variables that are in fact independent.
    """
    cols = list(X.columns)
    if len(cols) < 2:
        return {c: 1.0 for c in cols}

    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        A = np.column_stack([np.ones(len(X)), X[others].to_numpy()])
        y = X[c].to_numpy()
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            out[c] = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            out[c] = float("nan")
    return out


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise NotEnoughData(
            f"Unknown metric '{col}'. Available: {sorted(set(METRICS) & set(df.columns))}"
        )
    return pd.to_numeric(df[col], errors="coerce")


def _label(col: str) -> str:
    return METRICS.get(col, (col, ""))[0]


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


# --------------------------------------------------------------- public API


def lagged_correlation(
    df: pd.DataFrame,
    driver: str,
    outcome: str,
    max_lag: int = 3,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Correlate driver(t) against outcome(t+lag) for lag in 0..max_lag.

    Positive lag means the driver leads: lag 1 asks whether today's driver value
    shows up in tomorrow's outcome.
    """
    df = _slice(df, start, end)
    d, o = _series(df, driver), _series(df, outcome)

    rows = []
    for lag in range(0, max_lag + 1):
        pair = pd.concat([d, o.shift(-lag)], axis=1, keys=["d", "o"]).dropna()
        if len(pair) < MIN_N:
            rows.append({"lag": lag, "n": len(pair), "error": "insufficient overlap"})
            continue
        res = _corr_with_ci(pair["d"].to_numpy(), pair["o"].to_numpy())
        res["lag"] = lag
        rows.append(res)

    qs = benjamini_hochberg([r.get("p", float("nan")) for r in rows])
    for r, q in zip(rows, qs):
        if "p" in r:
            r["q_fdr"] = None if np.isnan(q) else _p(q)
            r["p"] = _p(r["p"])

    scored = [r for r in rows if "r" in r]
    best = max(scored, key=lambda r: abs(r["r"]), default=None)

    return {
        "driver": driver,
        "driver_label": _label(driver),
        "outcome": outcome,
        "outcome_label": _label(outcome),
        "lags": rows,
        "strongest_lag": best,
        "note": (
            "Positive lag means driver leads outcome. p-values use an effective "
            "sample size adjusted for autocorrelation; q_fdr is Benjamini-Hochberg "
            "across the lags tested here."
        ),
    }


def rank_drivers(
    df: pd.DataFrame,
    outcome: str,
    lag: int = 1,
    start: str | None = None,
    end: str | None = None,
    candidates: Sequence[str] | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """Screen every available metric against one outcome at a fixed lag."""
    df = _slice(df, start, end)
    o = _series(df, outcome)

    pool = [
        c
        for c in (candidates or METRICS)
        if c in df.columns and c != outcome
    ]

    rows = []
    for col in pool:
        d = _series(df, col)
        pair = pd.concat([d, o.shift(-lag)], axis=1, keys=["d", "o"]).dropna()
        if len(pair) < MIN_N or pair["d"].nunique() < 3:
            continue
        res = _corr_with_ci(pair["d"].to_numpy(), pair["o"].to_numpy())
        res["metric"] = col
        res["label"] = _label(col)
        rows.append(res)

    if not rows:
        raise NotEnoughData("No candidate metric had enough overlapping data.")

    for r, q in zip(rows, benjamini_hochberg([r["p"] for r in rows])):
        r["q_fdr"] = None if np.isnan(q) else _p(q)
        r["p"] = _p(r["p"])

    rows.sort(key=lambda r: abs(r["r"]), reverse=True)
    # Explicit None check: a q-value of exactly 0.0 is the strongest possible
    # result, and `q or 1` would silently discard it.
    survivors = [
        r["label"] for r in rows if r["q_fdr"] is not None and r["q_fdr"] < 0.05
    ]

    return {
        "outcome": outcome,
        "outcome_label": _label(outcome),
        "lag": lag,
        "tested": len(rows),
        "results": rows[:top_n],
        "significant_after_fdr": survivors,
        "note": (
            f"{len(rows)} metrics screened, so raw p is unreliable on its own — "
            "read q_fdr. These are associations, not causes; confirm anything "
            "interesting with `regress` holding the obvious confounders fixed."
        ),
    }


def regress(
    df: pd.DataFrame,
    outcome: str,
    drivers: Sequence[str],
    controls: Sequence[str] = (),
    lag: int = 0,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """OLS of outcome(t+lag) on drivers(t) and controls(t), with HAC errors.

    This is the tool that answers "does X still matter once Y is accounted
    for". Coefficients are reported in natural units and as standardized betas
    so predictors on different scales can be compared.
    """
    import statsmodels.api as sm

    df = _slice(df, start, end)
    preds = list(dict.fromkeys([*drivers, *controls]))
    if not preds:
        raise NotEnoughData("Specify at least one driver.")

    y = _series(df, outcome).shift(-lag)
    X = pd.concat([_series(df, c) for c in preds], axis=1, keys=preds)
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()

    if len(data) < max(MIN_N, len(preds) * 5):
        raise NotEnoughData(
            f"Only {len(data)} complete rows for {len(preds)} predictors — "
            "too few to fit reliably. Widen the date range or drop predictors."
        )

    yv = data["_y"]
    Xv = sm.add_constant(data[preds], has_constant="add")

    # Newey-West lag selection, standard 4*(n/100)^(2/9) rule of thumb.
    maxlags = max(1, int(math.ceil(4 * (len(data) / 100) ** (2 / 9))))
    model = sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    sd_y = float(yv.std())
    vifs = _vif(data[preds])
    terms = []
    for name in preds:
        sd_x = float(data[name].std())
        coef = float(model.params[name])
        terms.append(
            {
                "predictor": name,
                "label": _label(name),
                "coef": round(coef, 5),
                "std_error": round(float(model.bse[name]), 5),
                "t": round(float(model.tvalues[name]), 3),
                "p": _p(float(model.pvalues[name])),
                "vif": round(vifs.get(name, float("nan")), 2),
                "ci95": [
                    round(float(model.conf_int().loc[name, 0]), 4),
                    round(float(model.conf_int().loc[name, 1]), 4),
                ],
                "beta_standardized": (
                    round(coef * sd_x / sd_y, 4) if sd_y and sd_x else None
                ),
                "is_control": name in controls and name not in drivers,
                "interpretation": (
                    f"+1 {METRICS.get(name, ('', 'unit'))[1]} of {_label(name)} "
                    f"→ {coef:+.3f} {METRICS.get(outcome, ('', ''))[1]} "
                    f"of {_label(outcome)}"
                ),
            }
        )

    warnings = []
    # An infinite VIF means perfectly redundant predictors — the most severe
    # case, so it must be included rather than filtered out as non-finite.
    bad = {k: v for k, v in vifs.items() if not np.isnan(v) and v > 5}
    if bad:
        worst = ", ".join(
            f"{_label(k)} (VIF {'infinite' if np.isinf(v) else f'{v:.1f}'})"
            for k, v in sorted(bad.items(), key=lambda kv: -kv[1])[:3]
        )
        warnings.append(
            f"Collinear predictors: {worst}. They carry overlapping information, "
            "so individual coefficients may be unstable and can even flip sign "
            "even when the model as a whole fits well."
        )
    if any(np.isinf(v) for v in vifs.values()):
        warnings.append(
            "At least one predictor is an exact linear combination of the "
            "others; its coefficient and standard error are not identified. "
            "Drop one of them and refit."
        )
    if model.rsquared_adj < 0.05:
        warnings.append(
            f"Adjusted R² is only {model.rsquared_adj:.3f} — these predictors "
            "explain very little of the variation."
        )

    return {
        "outcome": outcome,
        "outcome_label": _label(outcome),
        "lag": lag,
        "n": int(len(data)),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "hac_maxlags": maxlags,
        "terms": terms,
        "warnings": warnings,
        "note": (
            "Standard errors are Newey-West (HAC), which accounts for the "
            "day-to-day autocorrelation in these series. Compare "
            "beta_standardized across predictors; compare coef within one."
        ),
    }


def compare_periods(
    df: pd.DataFrame,
    metrics: Sequence[str],
    a_start: str,
    a_end: str,
    b_start: str,
    b_end: str,
) -> dict[str, Any]:
    """Welch's t-test on each metric between two date windows."""
    A, B = _slice(df, a_start, a_end), _slice(df, b_start, b_end)
    rows = []
    for m in metrics:
        a = _series(A, m).dropna()
        b = _series(B, m).dropna()
        if len(a) < 3 or len(b) < 3:
            rows.append({"metric": m, "error": "not enough data in one window"})
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = math.sqrt((a.var() + b.var()) / 2)
        rows.append(
            {
                "metric": m,
                "label": _label(m),
                "unit": METRICS.get(m, ("", ""))[1],
                "period_a_mean": round(float(a.mean()), 3),
                "period_b_mean": round(float(b.mean()), 3),
                "difference": round(float(b.mean() - a.mean()), 3),
                "cohens_d": round(float((b.mean() - a.mean()) / pooled), 3) if pooled else None,
                "p": _p(float(p)),
                "n_a": len(a),
                "n_b": len(b),
            }
        )

    valid = [r for r in rows if "p" in r]
    for r, q in zip(valid, benjamini_hochberg([r["p"] for r in valid])):
        r["q_fdr"] = None if np.isnan(q) else _p(q)

    return {
        "period_a": f"{a_start} to {a_end}",
        "period_b": f"{b_start} to {b_end}",
        "results": rows,
        "note": "Difference is B minus A. Welch's t-test; days are treated as independent, which mildly overstates significance for strongly autocorrelated metrics.",
    }


def anomalies(
    df: pd.DataFrame,
    metrics: Sequence[str] = (
        "resting_hr",
        "respiratory_rate",
        "skin_temp_celsius",
        "hrv_rmssd_milli",
        "spo2_percentage",
    ),
    baseline_days: int = 30,
    z_threshold: float = 2.0,
    last_n_days: int = 14,
) -> dict[str, Any]:
    """Flag days where vitals departed from their own trailing baseline.

    The baseline excludes the day under test, so a single extreme value cannot
    inflate the very mean it is being compared against.
    """
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        s = _series(df, m)
        if s.notna().sum() < baseline_days // 2:
            continue
        base = s.shift(1).rolling(baseline_days, min_periods=max(7, baseline_days // 3))
        z = (s - base.mean()) / base.std()

        recent = z.tail(last_n_days)
        for date, zval in recent.items():
            if pd.notna(zval) and abs(zval) >= z_threshold:
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "metric": m,
                        "label": _label(m),
                        "value": round(float(s.loc[date]), 3),
                        "baseline_mean": round(float(base.mean().loc[date]), 3),
                        "z": round(float(zval), 2),
                        "direction": "above" if zval > 0 else "below",
                    }
                )

    rows.sort(key=lambda r: (r["date"], -abs(r["z"])), reverse=True)
    by_date: dict[str, int] = {}
    for r in rows:
        by_date[r["date"]] = by_date.get(r["date"], 0) + 1
    multi = [d for d, c in by_date.items() if c >= 2]

    return {
        "window_days": last_n_days,
        "baseline_days": baseline_days,
        "z_threshold": z_threshold,
        "flags": rows,
        "multi_metric_days": sorted(multi, reverse=True),
        "note": (
            "Days where two or more vitals deviate together are the ones worth "
            "attention — elevated resting HR plus elevated respiratory rate is "
            "the classic pre-symptomatic illness pattern. This is a description "
            "of your own data, not a diagnosis."
        ),
    }
