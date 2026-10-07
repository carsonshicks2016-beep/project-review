"""Fable Five — the PPO Nuerburgring training pipeline.

A ground-up redesign of the 787B Nordschleife program aimed at SUPERHUMAN clean
laps. It reuses the proven PPO optimiser, physics, sensors, and viewers, but
replaces everything the old ring pipeline got wrong:

  * a PHYSICS-TRUE speed envelope computed from the actual car model
    (downforce-aware cornering caps, power/traction acceleration, braking
    passes) replaces the fixed 12 m/s^2 lateral-g guess that capped pace at
    road-car levels;
  * the envelope feeds the OBSERVATION (a pace block: correct speed here + at
    6 points down the road), the REWARD (pace measured relative to the local
    envelope), and the EVAL (lap budget + superhuman benchmarks);
  * exploring starts rehearse WEAK SECTORS more (per-env adaptive weights) at
    RACE PACE (envelope-scaled spawn speeds) — repair happens inside the run;
  * the eval line run gets a budget a slow-but-clean lap can actually finish
    in, so the first clean lap can bank the moment it exists;
  * a warm-start TRANSPLANT imports the old ring pipeline's best policy into
    the new observation layout instead of starting from zero.

Five stages: foundation -> flow -> finish -> fast -> frontier.
See FABLE5_PLAN.md for the full design rationale.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .config import PPOSpec, SensorSpec, SimSpec, get_car
from .checkpoint_io import load_torch_checkpoint_safe
from .physics import Controls
from .ppo import PPO, validate_checkpoint_payload
from .ppo_env import SupraEnv, on_track_factor
from .track import named_track


RING_TRACK = "nordschleife"
FABLE_CAR = "mazda787b"
FABLE_PROFILE = "nordschleife-full-20.832km"
FABLE_BEST = "fable5_ring_best.pt"
PIPELINE_MANIFEST = "fable5_ring_pipeline.json"
LATEST_EVAL_JSON = "fable5_ring_eval_latest.json"

def pipeline_manifest_for(car: str) -> str:
    if car == FABLE_CAR:
        return PIPELINE_MANIFEST
    tag = {"porsche_919evo": "919", "porsche_956": "956"}.get(car, car)
    return f"fable5_{tag}_ring_pipeline.json"

def latest_eval_for(car: str) -> str:
    if car == FABLE_CAR:
        return LATEST_EVAL_JSON
    tag = {"porsche_919evo": "919", "porsche_956": "956"}.get(car, car)
    return f"fable5_{tag}_ring_eval_latest.json"

FABLE_REWARD_VERSION = "fable5-v4-steer-smooth"
EVAL_PROTOCOL_VERSION = "fable5-eval-v3-footprint-envelope"
OBS_LAYOUT = "fable-v1"            # 58 sensors + 8 pace + 2 mode = 68 dims
OBS_LAYOUT_HYBRID = "fable-v2"     # + battery SOC + MGU power = 70 dims
DRIVETRAIN_VERSION = "mazda787b-5spd-ring-v1"


def car_has_hybrid_telemetry(car: str) -> bool:
    """True when the car runs a live hybrid energy ledger (919 Evo) — the same
    predicate that arms the ledger in physics gates the fable-v2 obs block."""
    try:
        spec = get_car(car)
    except Exception:
        return False
    return (float(getattr(spec, "hybrid_mgu_power_w", 0.0)) > 0.0
            and float(getattr(spec, "hybrid_battery_kj", 0.0)) > 0.0)


def obs_layout_for(car: str) -> str:
    """Per-car obs layout: hybrid cars append SOC + MGU power (fable-v2, 70
    dims); every other edition keeps the frozen fable-v1 68-dim layout."""
    return OBS_LAYOUT_HYBRID if car_has_hybrid_telemetry(car) else OBS_LAYOUT


def drivetrain_for(car: str) -> str:
    """The drivetrain identity a checkpoint trained as `car` must carry.
    The module constant remains the 787B's for back-compat comparisons."""
    try:
        return getattr(get_car(car), "drivetrain_version", DRIVETRAIN_VERSION)
    except Exception:
        return DRIVETRAIN_VERSION


def champion_path_for(car: str) -> str:
    """Per-edition global champion file. The 787B keeps the historical
    fable5_ring_best.pt; other cars get their own champion so editions can
    never clobber each other's banked laps."""
    if car == FABLE_CAR:
        return FABLE_BEST
    tag = {"porsche_919evo": "919", "porsche_956": "956"}.get(car, car)
    return f"fable5_{tag}_ring_best.pt"
STAGES = ("foundation", "flow", "finish", "fast", "frontier")
LOOKAHEAD = (15.0, 30.0, 55.0, 85.0, 125.0, 180.0)
PACE_DISTANCES = (0.0, 25.0, 50.0, 100.0, 175.0, 275.0, 400.0)

# Reference laps of the real Nordschleife. "Superhuman" = a CLEAN sim lap under
# the fastest lap a human has ever driven in comparable machinery.
HUMAN_BENCHMARKS = {
    "bellof_956_1983_qualifying": 371.13,   # 6:11.13 — the superhuman line
    "porsche_919_evo_2018_record": 319.55,  # 5:19.55 — the outright record
}
SUPERHUMAN_LAP = HUMAN_BENCHMARKS["bellof_956_1983_qualifying"]


def superhuman_lap_for(drivetrain_version: str | None) -> tuple[float, str]:
    """The human benchmark a car is measured against. The 919 Evo chases the
    outright 5:19.55 record; the 956 and 787B chase Bellof's 6:11.13 qualifying
    lap. Keyed on the drivetrain version stamped into the speed envelope so the
    eval, the log line, and the dashboard all agree on the target."""
    dv = drivetrain_version or ""
    if dv.startswith("porsche919"):
        return HUMAN_BENCHMARKS["porsche_919_evo_2018_record"], "porsche_919_evo_2018_record"
    return HUMAN_BENCHMARKS["bellof_956_1983_qualifying"], "bellof_956_1983_qualifying"

# auto-ladder: the recommendation string that gates each stage forward, and
# the share of the total iteration budget each stage gets (leftovers from
# stages that gate early roll into frontier — the lap-time compressor).
GATE_ADVANCE = {"foundation": "advance to flow", "flow": "advance to finish",
                "finish": "advance to fast", "fast": "advance to frontier"}
AUTO_WEIGHTS = {"foundation": 0.15, "flow": 0.15, "finish": 0.20,
                "fast": 0.20, "frontier": 0.30}

# self-correcting ladder: a stage that misses its gate retries ONCE with an
# eased envelope scale, then SOFT-ADVANCES (training the next stage from the
# best it banked beats idling the rest of the night). Frontier runs in
# SEGMENTS with an adaptive scale: mastery pushes the reward target beyond
# the centerline envelope (the racing line is faster than the centerline —
# that's where the superhuman margin lives), losing the lap eases it back.
STAGE_MAX_ATTEMPTS = 2
STAGE_RETRY_SCALE_STEP = 0.05
STAGE_SCALE_FLOOR = 0.60
FRONTIER_SCALE_MIN = 0.90
FRONTIER_SCALE_MAX = 1.15
FRONTIER_SCALE_UP = 0.03
FRONTIER_SCALE_DOWN = 0.04

# fast runs adaptive segments too (KLGUARD, 2026-07-10/11): finish banks its
# lap at ~0.90 and the old attempt branch dropped that brain straight into a
# fixed 0.96-scaled world — ~6.6% more entry speed everywhere at once
# (sectors 10-13 first). Rollouts turn crash-dominated, every update points
# away from the lap, and the pit wall can only roll back until its calm
# ladder freezes the run (18 rollbacks in one hour, live, lr pinned at the
# 2e-5 floor). Rollback can't fix a DATA problem. So fast now starts from the
# scale the incoming brain actually banked and climbs toward its 0.96 target
# only on mastery proven at the current scale — the car EARNS 0.96, it is
# never thrown at it. The floor sits just under finish's 0.90 so a struggling
# fast can ease slightly below its inheritance without re-running finish.
FAST_SCALE_MIN = 0.88

# finish is the make-or-break stage: it runs adaptive segments (like frontier)
# and a LAP is required to climb higher — fast/frontier train speed ON TOP of
# a finisher, so training them on a non-finisher wastes the night. The one
# exception is the QUALITY BAR: a near-lap this strong is worth building on.
FINISH_SCALE_FLOOR = 0.70
FINISH_SCALE_DOWN = 0.04
FINISH_BAR_PROGRESS = 0.98
FINISH_BAR_CLEAN = 15
FINISH_BAR_TERMINAL = 0.125

# closure milestones: one-time bonuses at these fractions of a lap survived
# from spawn — sharpens the gradient toward WHOLE-lap continuity (the failure
# mode of a brain that drives fast fragments but never carries a lap home)
_MILESTONES = (0.50, 0.75, 0.90)
_MILESTONE_W = (0.30, 0.50, 1.00)

# hall of fame: category bests banked beside the single metric best, so the
# pit wall can reseed from the brain that FIXES the current failure mode
HOF_CATEGORIES = ("progress", "clean", "lap")

# RaceBox upshift point, as a fraction of redline. The stock 0.42 shifts for
# ECONOMY: the moment the next-taller gear clears 42% of redline the box
# upshifts, landing at ~50% in the gear it just left — so the R26B never sees
# the top half of its powerband (peak power is ~94% of redline) and the policy,
# to dodge the wheelspin penalty, learned to short-shift on top of that (gear
# offset pinned at +2). 0.63 holds each gear toward redline so upshifts land in
# the meat of the curve. HARD CEILING: the 1->2 ratio step is 1.425 and
# rpm_hi = 0.97*redline, so this MUST stay <= 0.97/1.425 = 0.68 or a speed
# exists where 1st overrevs while 2nd still lugs (a dead-band). 0.63 keeps a
# safe margin; _shift_table() below verifies no dead-band at load-import time in
# the validator. Single source of truth: RaceBox's default AND FableSpec's
# default both read this, and it rides in checkpoint metadata so the viewer
# drives the box the brain was trained against.
RACE_SHIFT_LO_FRAC = 0.63
RACE_SHIFT_UNLOAD_S = 0.08
RACE_SHIFT_REENGAGE_S = 0.16


def _source_fingerprint() -> str:
    """Hash the policy/simulator sources that define checkpoint semantics."""
    root = Path(__file__).resolve().parent.parent
    names = ("supra/fable5.py", "supra/ppo.py", "supra/ppo_env.py",
             "supra/config.py", "supra/physics.py", "supra/sensors.py",
             "supra/track.py", "supra/data/tracks/nordschleife.json")
    h = hashlib.sha256()
    for name in names:
        path = root / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


CODE_FINGERPRINT = _source_fingerprint()


def _finish_bar_met(ev: dict) -> bool:
    """The soft-advance quality bar past finish WITHOUT a banked lap."""
    if not ev:
        return False
    tr = ev.get("terminal_rate")
    return (float(ev.get("progress_frac") or 0.0) >= FINISH_BAR_PROGRESS
            and int(ev.get("clean_sectors") or 0) >= FINISH_BAR_CLEAN
            and float(1.0 if tr is None else tr) <= FINISH_BAR_TERMINAL)


def _checkpoint_score(ev: dict) -> float:
    """Monotonic checkpoint ordering separate from the stage trend metric.

    A gate proof must beat every ungated score; any footprint-valid clean lap
    must beat a lapless fragment; among laps, faster wins. This prevents a gate
    from stopping on the latest candidate while AUTO later reads an older,
    ungated `_best.pt` selected by a loosely aligned composite metric.
    """
    stage = ev.get("stage")
    metric = float(ev.get("metric") or -1e9)
    lap = ev.get("lap_time")
    gate = GATE_ADVANCE.get(stage)
    gated = bool(gate and ev.get("recommendation") == gate)
    if lap:
        return (2_000_000.0 if gated else 1_000_000.0) + 100000.0 / float(lap)
    if gated:
        return 1_000_000.0 + metric
    return metric


def _hof_key(cat: str, ev: dict) -> tuple | None:
    """Ordering key for a hall-of-fame category; None = not eligible."""
    if not ev:
        return None
    if cat == "lap":
        lap = ev.get("lap_time")
        return (-float(lap),) if lap else None
    if cat == "progress":
        return (float(ev.get("progress_frac") or 0.0),
                float(ev.get("metric") or -1e9))
    tr = ev.get("terminal_rate")
    return (int(ev.get("clean_sectors") or 0),
            int(ev.get("clean_chain") or 0),
            -float(1.0 if tr is None else tr),
            float(ev.get("metric") or -1e9))


def _frontier_scale_next(ev: dict, scale: float,
                         lo: float = FRONTIER_SCALE_MIN,
                         hi: float = FRONTIER_SCALE_MAX,
                         proven_scale: float | None = None,
                         progressed: bool = True) -> tuple[float, str]:
    """Decide the next segment's envelope scale from the banked best eval
    (shared by the fast and frontier ladders): push when the current level is
    mastered, ease when the lap is lost, hold while consolidating — plus two
    lessons KLGUARD taught. (1) Mastery only counts if the banked eval was
    EARNED at this scale or above (proven_scale = the scale stored in the
    best checkpoint): a lap banked at an easier target can't justify a push,
    or the ladder climbs on stale evidence until the car dies. (2) A
    lap-capable stage whose whole segment banked NO new best
    (progressed=False) eases instead of holding forever — the banked eval
    keeps its lap for the rest of the run, so the old "hold while
    consolidating" could pin a fragile brain at an unlearnable scale
    indefinitely (18 rollbacks in one hour at a frozen 0.96, live).
    Defaults (proven_scale=None, progressed=True) reproduce the legacy
    policy exactly."""
    lap = ev.get("lap_time")
    tr = float(ev.get("terminal_rate") or 0.0)
    pr = float(ev.get("pace_ratio") or 0.0)
    proven = proven_scale is None or proven_scale >= scale - 1e-9
    if lap and tr <= 0.25 and pr >= 0.88 and proven:
        new = min(hi, round(scale + FRONTIER_SCALE_UP, 3))
        if new > scale:
            return new, (f"mastered scale {scale:.2f} (clean lap, pace "
                         f"{pr:.2f}, terminal {tr:.2f}) — pushing the target")
        return scale, f"holding at the {hi:.2f} scale ceiling"
    if not lap:
        new = max(lo, round(scale - FRONTIER_SCALE_DOWN, 3))
        if new < scale:
            return new, f"no clean lap at scale {scale:.2f} — easing off"
        return scale, f"holding at the {lo:.2f} scale floor"
    if not progressed:
        new = max(lo, round(scale - FRONTIER_SCALE_DOWN, 3))
        if new < scale:
            return new, (f"lap banked but the segment banked no new best at "
                         f"scale {scale:.2f} — easing to consolidate")
        return scale, f"holding at the {lo:.2f} scale floor"
    return scale, (f"holding scale {scale:.2f} (lap banked, pace {pr:.2f} — "
                   f"still consolidating)")


def _stored_scale(path: str) -> float | None:
    """The envelope scale a checkpoint was trained at (for resuming a ladder
    exactly where its curriculum left off)."""
    if not path or not os.path.exists(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    v = d.get("fable_envelope_scale")
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def _checkpoint_updates(path: str | None) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return int(d.get("updates", 0))
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# The speed envelope — a three-pass lap-time solver over the real geometry,
# using the ACTUAL car constants. This is the centerpiece: it tells the policy
# (obs), the reward, and the eval what "the limit" actually is.
# --------------------------------------------------------------------------- #
def _peak_wheel_power(car) -> float:
    """Peak engine power (W) from the torque curve, after drivetrain losses."""
    best = 0.0
    for rpm, tq in car.torque_curve:
        best = max(best, float(tq) * float(rpm) * 2.0 * math.pi / 60.0)
    return best * float(getattr(car, "drivetrain_efficiency", 0.92))


def compute_speed_envelope_geometry(car, ds, curvature, grade, *,
                                    lat_util: float = 0.92,
                                    long_util: float = 0.90,
                                    drive_frac: float = 0.62,
                                    v_floor: float = 6.0,
                                    sweeps: int = 4) -> dict:
    """The envelope core over EXPLICIT geometry (ds, signed curvature, grade
    arrays) — used by both the centerline wrapper below (byte-identical) and
    the racing-line oracle (supra/raceline.py), so a lap on an optimized line
    is judged by exactly the same physics as the centerline reference.
    """
    g = 9.81
    rho = float(getattr(car, "air_density", 1.225))
    m = float(car.mass)
    mu0 = float(car.mu)
    ls = float(getattr(car, "load_sensitivity", 0.0))
    q = 0.5 * rho * float(car.downforce_ClA) / m          # downforce accel / v^2
    cd = 0.5 * rho * float(car.drag_area) / m             # drag decel / v^2
    crr = float(getattr(car, "rolling_resistance", 0.012)) * g
    p_kg = _peak_wheel_power(car) / m                     # W / kg

    ratios = np.asarray(car.gear_ratios, dtype=float) * float(car.final_drive)
    wheel_r = float(car.wheel_radius)
    curve_rpm = np.asarray([p[0] for p in car.torque_curve], dtype=float)
    curve_tq = np.asarray([p[1] for p in car.torque_curve], dtype=float)
    cutoff = float(car.cutoff_rpm)
    # what the brake hardware can actually produce at the wheels (m/s^2)
    brake_auth = (float(getattr(car, "max_brake_torque", 0.0)) / wheel_r / m
                  if getattr(car, "max_brake_torque", 0.0) > 0 else float("inf"))

    def geared_accel(speed: float) -> float:
        """Best available longitudinal acceleration from a real forward gear."""
        wheel_w = max(float(speed), 0.1) / wheel_r
        best_force = 0.0
        for ratio in ratios:
            rpm = wheel_w * ratio * 60.0 / (2.0 * math.pi)
            if rpm < car.idle_rpm * 0.75 or rpm >= cutoff:
                continue
            tq = float(np.interp(rpm, curve_rpm, curve_tq))
            force = tq * ratio * float(car.drivetrain_efficiency) / wheel_r
            best_force = max(best_force, force)

        if getattr(car, "is_ttr_hybrid", False):
            mgu_ratio = getattr(car, "mgu_gear_ratio", 5.5)
            mgu_power = float(getattr(car, "hybrid_mgu_power_w", 0.0))
            if mgu_power > 0.0:
                mgu_omega = max(speed / max(wheel_r, 0.1), 1.0) * mgu_ratio
                # Full deployment requested to hit 369 km/h (MGU doesn't taper off)
                deploy_factor = 1.0
                mgu_tq = (mgu_power / mgu_omega) * deploy_factor
                # Constant-torque below base speed — mirror physics.py's cap so
                # the envelope's low-speed MGU force matches the real car.
                peak_tq = float(getattr(car, "hybrid_mgu_peak_torque_nm", 0.0))
                if peak_tq > 0.0:
                    mgu_tq = min(mgu_tq, peak_tq * deploy_factor)
                mgu_force = mgu_tq * mgu_ratio * float(car.drivetrain_efficiency) / wheel_r
                best_force += mgu_force

        return best_force / m

    ds = np.asarray(ds, dtype=float)
    n = len(ds)
    k = np.abs(np.asarray(curvature, dtype=float))
    # light smoothing: kill single-sample curvature spikes from the resampling
    k = np.maximum(k, 1e-9)
    k = np.minimum.reduce([np.roll(k, -1), k, np.roll(k, 1)]) * 0.0 + (
        (np.roll(k, -1) + 2.0 * k + np.roll(k, 1)) / 4.0)
    grade = np.asarray(grade if grade is not None else np.zeros(n), dtype=float)

    def mu_of(v):
        # slick peak mu falls with the EXTRA per-wheel load from downforce
        extra = m * q * v * v / 4.0
        return mu0 * np.maximum(0.55, 1.0 - ls * extra)

    # Active aero (LMP1 DRS): straights run the low-drag state, corners the
    # high-downforce state. Terminal speed happens on a straight -> low drag;
    # per-sample drag follows the local curvature (radius > 800 m = straight).
    # Cornering grip keeps the HIGH-downforce ClA everywhere (conservative:
    # the car only sheds wing where it isn't cornering).
    lowdrag = (float(getattr(car, "aero_lowdrag_factor", 1.0))
               if getattr(car, "aero_active", False) else 1.0)
    cd_low = cd * lowdrag
    cd_arr = np.where(k < (1.0 / 800.0), cd_low, cd)

    # terminal speed: power balance (cd v^2 + crr) * v = p_kg, monotonic -> bisect
    lo, hi = 10.0, 160.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (cd_low * mid * mid + crr) * mid < p_kg:
            lo = mid
        else:
            hi = mid
    v_max = 0.5 * (lo + hi)

    # 1) cornering caps (fixed-point in v because mu and downforce depend on v)
    v = np.full(n, v_max)
    for _ in range(30):
        mu = lat_util * mu_of(v)
        denom = k - mu * q
        vc = np.where(denom > 1e-9,
                      np.sqrt(np.maximum(mu * g, 0.0) / np.maximum(denom, 1e-9)),
                      v_max)
        v = np.minimum(vc, v_max)

    # 2/3) forward + backward passes around the loop (wrap via repeated sweeps)
    for _ in range(max(1, int(sweeps))):
        for i in range(n):                       # forward: accelerate
            j = (i + 1) % n
            vi = v[i]
            mu = float(mu_of(np.array([vi]))[0])
            cap = mu * (g + q * vi * vi)
            # `cap` is the full tyre limit. lat_util was already applied when
            # constructing the cornering cap; applying it again here erased the
            # deliberately reserved longitudinal grip.
            room = max(0.0, 1.0 - (vi * vi * k[i] / max(cap, 1e-9)) ** 2)
            a_tr = long_util * mu * (g + q * vi * vi) * math.sqrt(room) * drive_frac
            a_pw = min(p_kg / max(vi, 4.0), geared_accel(vi))
            a = min(a_tr, a_pw) - cd_arr[i] * vi * vi - crr - g * grade[i]
            # Reachability applies for BOTH signs. The old `if a > 0` let the
            # next straight sample remain at v_max when drag/grade/combined
            # grip made acceleration negative, creating impossible 3 m jumps
            # as large as ~60 m/s in the real Ring profile.
            vj2 = max(v_floor * v_floor, vi * vi + 2.0 * a * ds[i])
            v[j] = min(v[j], math.sqrt(vj2))
        for i in range(n - 1, -1, -1):           # backward: brake
            j = (i + 1) % n
            vj = v[j]
            mu = float(mu_of(np.array([vj]))[0])
            cap = mu * (g + q * vj * vj)
            room = max(0.0, 1.0 - (vj * vj * k[j] / max(cap, 1e-9)) ** 2)
            # Braking is the LESSER of tyre grip and actual brake authority
            # (max_brake_torque through the wheels). The old tyre-only pass
            # assumed up to ~8 g at speed on downforce cars whose brakes could
            # produce a third of that, so the pace reference demanded braking
            # points the vehicle physically cannot hit. Drag/rolling/grade
            # still slow the car on top of whichever limit binds.
            a_wheel = min(long_util * mu * (g + q * vj * vj) * math.sqrt(room),
                          brake_auth)
            a_br = a_wheel + cd_arr[j] * vj * vj + crr + g * grade[j]
            a_br = max(a_br, 1.0)
            v[i] = min(v[i], math.sqrt(vj * vj + 2.0 * a_br * ds[i]))

    v = np.maximum(v, v_floor)
    lap_time = float(np.sum(ds / v))
    return {
        "v": v.astype(np.float64),
        "lap_time": lap_time,
        "v_max": float(v_max),
        "params": {"lat_util": lat_util, "long_util": long_util,
                   "drive_frac": drive_frac, "sweeps": sweeps,
                   "peak_power_kw": round(p_kg * m / 1000.0, 1),
                   "brake_authority_mps2": (round(brake_auth, 2)
                                            if math.isfinite(brake_auth) else None),
                   "drivetrain_version": getattr(car, "drivetrain_version", "generic-v1"),
                   "gear_ratios": [float(x) for x in car.gear_ratios],
                   "final_drive": float(car.final_drive)},
    }


def compute_speed_envelope(trk, car, **kw) -> dict:
    """Centerline wrapper: pulls (ds, curvature, grade) off the track object
    and defers to the geometry core. Results are identical to the historical
    single-function implementation."""
    n = len(trk.center)
    ds = np.asarray(getattr(trk, "seg_len", None)
                    if getattr(trk, "seg_len", None) is not None
                    else np.diff(trk.arc, append=trk.length), dtype=float)
    if len(ds) != n:
        ds = np.diff(np.append(trk.arc, trk.length))
    grade = getattr(trk, "grade", None)
    return compute_speed_envelope_geometry(car, ds, trk.curvature, grade, **kw)


def attach_envelope(trk, car_name: str = FABLE_CAR, use_raceline: bool = False, **kw):
    """Compute (once) and attach the envelope to the track object. Sensors read
    `trk.fable_vref` (RAW, scale 1.0 — the obs must mean the same thing in
    every stage); the reward applies its own stage scale on top."""
    if getattr(trk, "fable_vref", None) is None:
        car = get_car(car_name)
        if use_raceline:
            import json
            from pathlib import Path
            cache_file = Path(__file__).resolve().parent / f"data/{car_name}_raceline_env.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    env = json.load(f)
            else:
                from .raceline import optimize_raceline
                res = optimize_raceline(trk, car, iters=5)
                env = res["profile"]
                # Convert numpy array to list for JSON serialization
                env["v"] = env["v"].tolist() if isinstance(env["v"], np.ndarray) else env["v"]
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(env, f)
            env["v"] = np.asarray(env["v"], dtype=float)
        else:
            env = compute_speed_envelope(trk, car, **kw)
            
        trk.fable_vref = env["v"]
        trk.fable_envelope = {k2: v2 for k2, v2 in env.items() if k2 != "v"}
    return trk


def vref_at(trk, arc: float, dist: float = 0.0) -> float:
    return float(np.interp((float(arc) + dist) % trk.length, trk.arc,
                           trk.fable_vref, period=trk.length))


# --------------------------------------------------------------------------- #
# Stage specs
# --------------------------------------------------------------------------- #
@dataclass
class FableReward:
    pace: float = 0.10               # v_progress / envelope_target (dominant)
    progress_per_m: float = 0.012    # absolute forward progress
    align: float = 0.0               # heading alignment (early stages only)
    edge: float = 0.0                # near-wall shaping (foundation only)
    slip_guard: float = 0.0          # big-slip shaping (foundation only)
    overspeed: float = 0.03          # beyond envelope*margin (soft, quadratic)
    overspeed_margin: float = 1.0    # fraction of RAW envelope before penalty
    offtrack: float = 0.45
    smooth_steer: float = 0.005      # steering rate (fixes oscillation)
    # The rate term above is weakened per-stage (0.006 -> 0.003) to let the car
    # steer hard when it must, which is what let the fast/frontier brain settle
    # into a jerky, wheel-sawing optimum (diag: 343 reversals/km, steering
    # saturated 14% of the time, only 84% of grip used). These two terms attack
    # that directly and are held CONSTANT across stages (no per-stage override),
    # so the anti-saw pressure does not relax exactly where the sawing appears.
    smooth_steer2: float = 0.015     # steering 2nd difference (jerk): a steady
                                     # +0.5/-0.5 oscillation has near-zero rate
                                     # cost but huge jerk cost, so this is what
                                     # actually punishes sawing vs a smooth turn.
    steer_sat: float = 1.5           # penalty for steering authority spent past
    steer_sat_thresh: float = 0.85   # this deadzone -> stop living at the rail
                                     # and find a line inside the grip envelope.
    smooth_long: float = 0.0015
    lap_bonus: float = 40.0
    lap_pace_bonus: float = 40.0     # scaled by theoretical/actual lap ratio
    milestone_bonus: float = 0.0     # closure milestones at 50/75/90% of a lap
    crash: float = 9.0
    backwards: float = 0.05
    spin: float = 0.03


@dataclass
class FableSpec:
    stage: str = "foundation"
    episode_seconds: float = 240.0
    envelope_scale: float = 0.78     # reward speed target = raw vref * this
    shift_lo_frac: float = RACE_SHIFT_LO_FRAC  # RaceBox upshift point / redline
    start_speed_lo: float = 0.75     # spawn speed range x scaled envelope
    start_speed_hi: float = 1.05
    start_speed_max: float = 34.0
    line_start_prob: float = 0.05    # fraction of episodes starting at the line
    closure_start_prob: float = 0.0  # lap-closure drills: spawn late in the lap
    closure_window: tuple[float, float] = (0.70, 0.98)
    n_start_sectors: int = 24
    sector_fail_gain: float = 1.0    # weakest-sector replay: weight per failure
    sector_fail_decay: float = 0.995
    sector_seed_bias: tuple[float, ...] | None = None  # persisted failure heat
    # terminations
    offtrack_timeout: float = 0.40
    stall_timeout: float = 2.5
    progress_timeout: float = 4.0
    spin_timeout: float = 0.50
    left_track_margin: float = 2.0
    backwards_tol: float = 0.02
    # eval
    eval_starts: int = 16
    eval_sector_seconds: float = 70.0
    eval_lap_budget: float = 0.0     # 0 -> max(900, 2.4 * theoretical)
    eval_every: int = 25
    lookahead_distances: tuple[float, ...] = LOOKAHEAD
    pace_distances: tuple[float, ...] = PACE_DISTANCES
    reward: FableReward = field(default_factory=FableReward)


def _norm_stage(stage: str) -> str:
    stage = (stage or "foundation").strip().lower()
    if stage not in (*STAGES, "auto"):
        raise ValueError(f"unknown fable stage '{stage}' (want one of {STAGES} or auto)")
    return stage


def stage_defaults(stage: str) -> FableSpec:
    stage = _norm_stage(stage)
    if stage == "foundation":
        return FableSpec(
            stage=stage, episode_seconds=240.0, envelope_scale=0.78,
            start_speed_lo=0.70, start_speed_hi=1.00, start_speed_max=34.0,
            line_start_prob=0.05, eval_every=25,
            reward=FableReward(pace=0.055, align=0.040, edge=0.020,
                               slip_guard=0.015, overspeed=0.050,
                               overspeed_margin=0.92, offtrack=0.50,
                               smooth_steer=0.006, lap_bonus=20.0,
                               lap_pace_bonus=20.0, crash=10.0))
    if stage == "flow":
        return FableSpec(
            stage=stage, episode_seconds=360.0, envelope_scale=0.80,
            start_speed_lo=0.75, start_speed_hi=1.05, start_speed_max=40.0,
            line_start_prob=0.10, eval_every=30, offtrack_timeout=0.35,
            reward=FableReward(pace=0.085, align=0.020, overspeed=0.040,
                               overspeed_margin=0.97, offtrack=0.45,
                               edge=0.005, slip_guard=0.008,
                               smooth_steer=0.005, lap_bonus=30.0,
                               lap_pace_bonus=30.0, crash=9.0))
    if stage == "finish":
        # "finish the lap first, then go fast": survival-weighted reward
        # (milestones + big lap bonus), closure drills through the last
        # sectors, and light edge/slip shaping against the two killers
        # (left_track = line discipline, spin = slip control). Speed pressure
        # belongs to fast/frontier.
        return FableSpec(
            stage=stage, episode_seconds=1000.0, envelope_scale=0.90,
            start_speed_lo=0.78, start_speed_hi=1.05, start_speed_max=44.0,
            line_start_prob=0.10, closure_start_prob=0.30,
            eval_every=40, offtrack_timeout=0.35,
            reward=FableReward(pace=0.10, align=0.010, edge=0.008,
                               slip_guard=0.008, overspeed=0.030,
                               overspeed_margin=1.00, offtrack=0.42,
                               smooth_steer=0.004, lap_bonus=60.0,
                               lap_pace_bonus=60.0, milestone_bonus=8.0,
                               crash=8.0))
    if stage == "fast":
        # eval_every 45 -> 30 (2026-07-10): 45 unguarded iterations is a blind
        # window long enough to erode a knife-edge lap even at calm lr/std —
        # observed across 3 live confirmation runs; a shorter window is what
        # lets the pit-wall ROLLBACK catch damage while it is still small.
        return FableSpec(
            stage=stage, episode_seconds=800.0, envelope_scale=0.96,
            start_speed_lo=0.80, start_speed_hi=1.08, start_speed_max=50.0,
            line_start_prob=0.20, closure_start_prob=0.15,
            eval_every=30, offtrack_timeout=0.30,
            reward=FableReward(pace=0.13, overspeed=0.020,
                               overspeed_margin=1.05, offtrack=0.42,
                               smooth_steer=0.004, lap_bonus=80.0,
                               lap_pace_bonus=80.0, milestone_bonus=4.0,
                               crash=8.0))
    return FableSpec(                                        # frontier
        stage=stage, episode_seconds=720.0, envelope_scale=1.00,
        start_speed_lo=0.82, start_speed_hi=1.10, start_speed_max=56.0,
        line_start_prob=0.25, eval_every=32, offtrack_timeout=0.30,
        reward=FableReward(pace=0.16, overspeed=0.008,
                           overspeed_margin=1.10, offtrack=0.40,
                           smooth_steer=0.003, lap_bonus=100.0,
                           lap_pace_bonus=100.0, crash=8.0))


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if hasattr(v, "__dataclass_fields__"):
        return asdict(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _write_json_atomic(path: Path, payload) -> None:
    """Write a manifest/eval JSON via tmp + rename, mirroring PPO.save's
    checkpoint pattern — the dashboard polls these files, and a plain
    write_text truncates first, so a poll mid-write reads torn JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _arc_to_idx(trk, arc: float) -> int:
    return int(np.searchsorted(trk.arc, float(arc) % trk.length) % len(trk.center))


# --------------------------------------------------------------------------- #
# RaceBox — a proper race transmission with AGENT gear authority
# --------------------------------------------------------------------------- #
class RaceBox:
    """Speed-matched race gearbox + agent gear offset.

    The stock AutoBox only downshifts below 1500 rpm — under the 787B's 2000
    idle — so through a braking zone the car stays in top gear and exits slow
    corners off the powerband. This box picks the highest gear that keeps rpm
    in the band at the CURRENT speed (so it walks down through braking zones
    on its own), and the policy's third action shifts that recommendation by
    -2..+2 gears (hold a gear over a crest, short-shift for traction, drop an
    extra one for rotation). A predicted-rpm guard refuses money-shifts: a
    downshift that would spin the R26B past ~96% of cutoff is held until the
    speed comes down. One shift per cooldown; clutch dips like the AutoBox.
    """

    def __init__(self, spec, rpm_lo_frac: float = RACE_SHIFT_LO_FRAC,
                 cooldown: float = 0.28):
        self.spec = spec
        r = float(spec.wheel_radius)
        self.k = [g * float(spec.final_drive) * 60.0 / (2.0 * math.pi * r)
                  for g in spec.gear_ratios]           # rpm per m/s, per gear
        self.rpm_lo = float(spec.redline_rpm) * rpm_lo_frac
        self.rpm_hi = float(spec.redline_rpm) * 0.97
        self.guard = float(getattr(spec, "cutoff_rpm", spec.redline_rpm)) * 0.96
        self.cooldown = cooldown
        self.cool = 0.0
        self.clutch = 0.0
        self.requested_gear = 1
        self.rejected_shift = ""
        self.pending_dir = 0
        self.unload_timer = 0.0
        self.reengage_timer = 0.0

    @property
    def shift_phase(self) -> str:
        if self.pending_dir:
            return "unload"
        if self.reengage_timer > 0.0:
            return "reengage"
        return "engaged"

    def reset(self):
        self.cool = 0.0
        self.clutch = 0.0
        self.requested_gear = 1
        self.rejected_shift = ""
        self.pending_dir = 0
        self.unload_timer = 0.0
        self.reengage_timer = 0.0

    def predicted_rpm(self, speed: float, gear: int) -> float:
        gear = int(np.clip(gear, 1, len(self.k)))
        return max(0.0, float(speed)) * self.k[gear - 1]

    def wheel_force_score(self, speed: float, gear: int) -> float:
        """Full-throttle wheel torque proxy in a candidate gear."""
        rpm = self.predicted_rpm(speed, gear)
        if rpm < self.spec.idle_rpm * 0.75 or rpm >= self.guard:
            return -1e18
        rpms = np.asarray([p[0] for p in self.spec.torque_curve], dtype=float)
        torque = np.asarray([p[1] for p in self.spec.torque_curve], dtype=float)
        tq = float(np.interp(rpm, rpms, torque))
        return tq * float(self.spec.gear_ratios[gear - 1]) * float(self.spec.final_drive)

    def recommend(self, speed: float, throttle: float = 1.0,
                  current_gear: int | None = None) -> int:
        """Fastest safe gear, with coast downshift bias and anti-hunt hysteresis."""
        n = len(self.k)
        current = int(np.clip(current_gear or 1, 1, n))
        valid = [g for g in range(1, n + 1)
                 if self.spec.idle_rpm * 0.75 <= self.predicted_rpm(speed, g) < self.guard]
        if not valid:
            return n if self.predicted_rpm(speed, n) >= self.guard else 1
        if throttle < 0.15:
            # On approach, prepare the lowest safe exit gear without money-shifting.
            useful = [g for g in valid if self.predicted_rpm(speed, g) >= self.rpm_lo]
            return min(useful or valid)
        scores = {g: self.wheel_force_score(speed, g) for g in valid}
        best = max(valid, key=lambda g: scores[g])
        if current in scores and best != current:
            # Require a real acceleration advantage to change gear before redline.
            cur_rpm = self.predicted_rpm(speed, current)
            if cur_rpm < self.rpm_hi and scores[best] < scores[current] * 1.025:
                return current
        return best

    def update(self, veh, throttle: float, dt: float, gear_offset: float = 0.0):
        """Return (clutch, shift_up, shift_down) — AutoBox-compatible."""
        self.cool = max(0.0, self.cool - dt)
        self.unload_timer = max(0.0, self.unload_timer - dt)
        self.reengage_timer = max(0.0, self.reengage_timer - dt)
        up = down = False
        n = len(self.k)
        base = self.recommend(max(veh.speed, 0.1), throttle, veh.gear)
        target = int(np.clip(base
                             + round(float(np.clip(gear_offset, -2.0, 2.0))),
                             1, n))
        self.requested_gear = target
        self.rejected_shift = ""
        if self.pending_dir and self.unload_timer == 0.0:
            up = self.pending_dir > 0
            down = self.pending_dir < 0
            self.pending_dir = 0
            self.reengage_timer = RACE_SHIFT_REENGAGE_S
            self.cool = self.cooldown
        elif not self.pending_dir and self.cool == 0.0 and veh.speed > 1.0:
            if veh.gear < target:
                self.pending_dir = 1
                self.unload_timer = RACE_SHIFT_UNLOAD_S
            elif veh.gear > target:
                pred = veh.speed * self.k[veh.gear - 2]
                if pred < self.guard:            # never money-shift the rotary
                    self.pending_dir = -1
                    self.unload_timer = RACE_SHIFT_UNLOAD_S
                else:
                    self.rejected_shift = f"money_shift:{pred:.0f}rpm"
        # hard-cutoff escape: upshift out of the limiter regardless of target
        if not (up or down) and not self.pending_dir and self.cool == 0.0 \
                and veh.rpm > self.spec.redline_rpm * 0.985 and veh.gear < n:
            self.pending_dir = 1
            self.unload_timer = RACE_SHIFT_UNLOAD_S
        idle_thresh = self.spec.idle_rpm * 1.25
        if up or down or self.pending_dir or self.reengage_timer > 0.0:
            c_target = 0.2
        elif throttle < 0.05 and (veh.speed < 1.5 or veh.rpm < idle_thresh):
            c_target = 0.0
        else:
            c_target = 1.0
        self.clutch += (c_target - self.clutch) * min(1.0, dt * 12.0)
        return self.clutch, up, down


def _shift_table(car: str = FABLE_CAR, rpm_lo_frac: float | None = None,
                 v_max: float = 90.0, dv: float = 0.1) -> dict:
    """Sweep speed through a RaceBox and report the automatic upshift points and
    any dead-band. An upshift's rpm is the engine speed in the gear being LEFT
    at the moment the box changes up (so ~90% of redline is healthy, ~50% is the
    old economy lug). A dead-band is a speed where the recommended gear is
    neither bottom (1st, crawling) nor top (on the limiter) yet its rpm
    still sits outside [rpm_lo, rpm_hi] — i.e. 1st overrevs before 2nd clears
    rpm_lo. Used by the validator (must be empty) and the sanity print."""
    spec = get_car(car)
    box = RaceBox(spec, rpm_lo_frac=(RACE_SHIFT_LO_FRAC if rpm_lo_frac is None
                                     else rpm_lo_frac))
    n = len(box.k)
    redline = float(spec.redline_rpm)
    upshifts: list[dict] = []
    deadband: list[float] = []
    prev = box.recommend(dv)
    v = dv
    while v <= v_max:
        g = box.recommend(v)
        rpm = v * box.k[g - 1]
        if g > prev:                                   # upshift boundary
            rpm_left = v * box.k[prev - 1]             # rpm in the gear left
            upshifts.append({"from": prev, "to": g, "speed": round(v, 1),
                             "rpm": round(rpm_left),
                             "frac": round(rpm_left / redline, 3)})
        if g != prev:
            prev = g
        if 1 < g < n and not (box.rpm_lo <= rpm <= box.rpm_hi):
            deadband.append(round(v, 1))
        v += dv
    return {"car": car, "rpm_lo_frac": round(box.rpm_lo / redline, 4),
            "rpm_lo": round(box.rpm_lo), "rpm_hi": round(box.rpm_hi),
            "redline": redline, "upshifts": upshifts, "deadband": deadband}


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class FableEnv(SupraEnv):
    """Solo Nordschleife env: envelope-based reward, weakest-sector exploring
    starts at race pace, clean-lap bookkeeping (any off-track invalidates)."""

    def __init__(self, *args, fable_spec: FableSpec | dict | None = None, **kwargs):
        if fable_spec is None:
            fable_spec = stage_defaults("foundation")
        elif isinstance(fable_spec, dict):
            r = fable_spec.get("reward")
            if isinstance(r, dict):
                fable_spec = dict(fable_spec)
                fable_spec["reward"] = FableReward(**r)
            fable_spec = FableSpec(**fable_spec)
        self.fable = fable_spec
        kwargs["reward"] = fable_spec.reward
        ft = kwargs.get("fixed_track")
        if ft is not None:
            attach_envelope(ft, kwargs.get("car", FABLE_CAR))
        self._sector_w = np.ones(max(4, int(fable_spec.n_start_sectors)))
        # weak-sector memory: failure heat persisted across restarts seeds the
        # start weights, so a fresh process resumes rehearsing the same zones
        if getattr(fable_spec, "sector_seed_bias", None):
            bias = np.asarray(fable_spec.sector_seed_bias, dtype=float)
            if bias.shape == self._sector_w.shape and np.all(np.isfinite(bias)):
                self._sector_w += np.clip(bias, 0.0, 5.0)
        self._pending_fail_arc = None
        super().__init__(*args, **kwargs)
        # third action = GEAR OFFSET from the RaceBox recommendation (-2..+2).
        # Declared after super().__init__ (mode "race" defaults to 2 actions).
        self.action_dim = 3
        self.rbox = RaceBox(self.spec, rpm_lo_frac=self.fable.shift_lo_frac)
        self.prev_action = np.zeros(self.action_dim)
        self.prev_prev_action = np.zeros(self.action_dim)

    # -- placement --------------------------------------------------------- #
    def reset(self):
        obs = super().reset()          # builds track + zeroes bookkeeping
        if self.trk is not None and getattr(self.trk, "fable_vref", None) is None:
            attach_envelope(self.trk, self.spec.name)
        sp = self.fable
        # bank the previous episode's failure into the sector weights
        if self._pending_fail_arc is not None:
            sec = int(self._pending_fail_arc / self.trk.length * len(self._sector_w)) \
                % len(self._sector_w)
            self._sector_w *= sp.sector_fail_decay
            self._sector_w[sec] += sp.sector_fail_gain
            self._pending_fail_arc = None
        if getattr(self.ppo, "random_start", False):
            roll = self.rng.random()
            if roll < sp.line_start_prob:
                return self._place_at(0, speed=0.0)
            if roll < sp.line_start_prob + sp.closure_start_prob:
                # lap-closure drill: drop in at race pace late in the lap and
                # practice carrying it home through the finish line
                frac = float(self.rng.uniform(*sp.closure_window))
                idx = _arc_to_idx(self.trk, self.trk.length * frac)
                v = vref_at(self.trk, float(self.trk.arc[idx])) * sp.envelope_scale
                v *= float(self.rng.uniform(0.85, 1.02))
                v = float(np.clip(v, 0.0, sp.start_speed_max))
                lat = float(self.rng.uniform(-0.15, 0.15)) * self.trk.half
                return self._place_at(idx, speed=v, lateral=lat,
                                      yaw_jitter=float(self.rng.uniform(-0.04, 0.04)))
            # weakest-sector replay: sample a sector by failure weight
            w = self._sector_w / self._sector_w.sum()
            sec = int(self.rng.choice(len(w), p=w))
            frac = (sec + float(self.rng.random())) / len(w)
            idx = _arc_to_idx(self.trk, self.trk.length * frac)
            v = vref_at(self.trk, float(self.trk.arc[idx])) * sp.envelope_scale
            v *= float(self.rng.uniform(sp.start_speed_lo, sp.start_speed_hi))
            v = float(np.clip(v, 0.0, sp.start_speed_max))
            lat = float(self.rng.uniform(-0.25, 0.25)) * self.trk.half
            return self._place_at(idx, speed=v, lateral=lat,
                                  yaw_jitter=float(self.rng.uniform(-0.06, 0.06)))
        return self._place_at(0, speed=0.0)

    def reset_at(self, index, speed=0.0):
        if self.trk is None:
            self.trk = self.fixed_track or attach_envelope(named_track(RING_TRACK))
        return self._place_at(index, speed=speed)

    def _place_at(self, index: int, speed: float = 0.0, lateral: float = 0.0,
                  yaw_jitter: float = 0.0):
        sx, sy, syaw = self.trk.pose_at(index)
        if lateral:
            nx, ny = self.trk.normal[int(index) % len(self.trk.normal)]
            sx += nx * lateral
            sy += ny * lateral
        self.veh.reset(sx, sy, syaw + yaw_jitter, speed=speed)
        self.box.__init__(self.spec)
        if hasattr(self, "rbox"):
            self.rbox.reset()
        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                          fr["z"], fr["vcurv"])
        self.prev_frac = fr["progress"]
        self.cum = 0.0
        self.max_cum = 0.0
        self.best_cum = 0.0
        self.t = 0.0
        self.total_dist = 0.0
        self.off_t = 0.0
        self.stall_t = 0.0
        self.since_prog = 0.0
        self.spin_t = 0.0
        self.step_count = 0
        self.drift_steps = 0
        self.ep_air_time = 0.0
        self.ep_jumps = 0
        self.ep_max_lg = 0.0
        self._was_air = False
        self.prev_action = np.zeros(self.action_dim)
        self.prev_prev_action = np.zeros(self.action_dim)
        self._milestone_i = 0
        # clean-lap bookkeeping (real Ring rules: any off-track invalidates)
        self.fable_invalid = False
        self.fable_offtrack_total = 0.0
        self.fable_footprint_offtrack_total = 0.0
        self.fable_footprint_invalid = False
        self.fable_terminal_reason = None
        self.fable_progress_m = 0.0
        self.fable_pace_num = 0.0     # sum of v/target -> mean pace ratio
        self.fable_pace_den = 0.0
        return self._obs()

    # -- reward ------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel(), -1.0, 1.0)
        steer = float(a[0])
        long = float(a[1])
        throttle = max(long, 0.0)
        brake = max(-long, 0.0)
        # gear offset: -1..1 -> -2..+2 gears off the RaceBox recommendation.
        # Legacy 2-action policies (diagnostics of old checkpoints) get 0.
        gear_offset = float(a[2]) * 2.0 if a.size > 2 else 0.0
        if a.size < self.action_dim:
            a = np.concatenate([a, np.zeros(self.action_dim - a.size)])

        sp = self.fable
        rw = sp.reward
        dt = self.sim.dt
        reward = 0.0
        reward_parts = {
            "pace": 0.0, "progress_m": 0.0, "align": 0.0, "edge": 0.0,
            "slip_guard": 0.0, "overspeed": 0.0, "offtrack": 0.0,
            "spin": 0.0, "backwards": 0.0, "lap_bonus": 0.0,
            "milestone": 0.0, "smooth": 0.0, "crash": 0.0,
        }
        diag_values = {"vref": 0.0, "target": 0.0, "pace_ratio": 0.0,
                       "heading_error": 0.0}
        terminated = False
        termination_reason = None
        lap_completed_t = None

        footprint_off = False

        def body_offtrack() -> bool:
            try:
                return any(self.trk.frame(float(x), float(y))["off_track"]
                           for x, y in self.veh.get_obb())
            except Exception:
                return True

        for _ in range(self.control_period):
            fr = self.trk.frame(self.veh.x, self.veh.y)
            self.veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
            self.veh.set_road(fr["grade"], fr["bank"], fr["heading"],
                              fr["z"], fr["vcurv"])
            if self.diagnostics and body_offtrack():
                footprint_off = True
                self.fable_invalid = True
                self.fable_footprint_invalid = True
                self.fable_footprint_offtrack_total += dt
                if not fr.get("off_track", False):
                    self.fable_offtrack_total += dt

            frac = fr["progress"]
            d = frac - self.prev_frac
            if d < -0.5:
                d += 1.0
            elif d > 0.5:
                d -= 1.0
            self.prev_frac = frac
            prev_cum = self.cum
            self.cum += d
            self.max_cum = max(self.max_cum, self.cum)
            delta_m = d * self.trk.length
            self.fable_progress_m = max(self.fable_progress_m,
                                        self.max_cum * self.trk.length)

            head_err = (self.veh.yaw - fr["heading"] + np.pi) % (2 * np.pi) - np.pi
            ot = on_track_factor(fr, self.trk, 1.8)
            vref = vref_at(self.trk, fr["arc"])
            target = max(vref * sp.envelope_scale, 6.0)
            # continuous forward speed along the road — NOT delta_m/dt, which is
            # quantized to the 3 m centreline sample spacing (progress jumps a
            # whole sample every ~8 substeps, so a rate built from it saturates
            # the clip and collapses the envelope-relative scaling).
            v_prog = self.veh.speed * float(np.cos(head_err))
            pace_ratio = v_prog / target
            self.fable_pace_num += max(0.0, min(pace_ratio, 1.5))
            self.fable_pace_den += 1.0
            diag_values.update(vref=vref, target=target,
                               pace_ratio=pace_ratio, heading_error=head_err)

            # -- the dominant term: envelope-relative pace ------------------ #
            p_term = rw.pace * float(np.clip(pace_ratio, -0.5, 1.30))
            m_term = rw.progress_per_m * delta_m
            if delta_m < 0.0:
                m_term *= 2.0
            step_reward = (p_term + m_term) * ot
            reward_parts["pace"] += p_term * ot
            reward_parts["progress_m"] += m_term * ot

            if rw.align:
                a_term = rw.align * max(0.0, np.cos(head_err)) * min(
                    1.0, self.veh.speed / 30.0)
                step_reward += a_term * ot
                reward_parts["align"] += a_term * ot

            # -- soft envelope-overspeed shaping (annealed away by frontier) - #
            over = self.veh.speed - vref * rw.overspeed_margin
            if rw.overspeed and over > 0.5:
                pen = rw.overspeed * (over / max(vref, 1.0)) ** 2 * 40.0
                step_reward -= pen
                reward_parts["overspeed"] -= pen

            if fr["off_track"]:
                self.off_t += dt
                self.fable_offtrack_total += dt
                self.fable_invalid = True
                step_reward -= rw.offtrack
                reward_parts["offtrack"] -= rw.offtrack
            else:
                self.off_t = 0.0

            lat_abs = abs(float(fr["lateral"]))
            half_w = float(fr.get("half_width", self.trk.half))
            if rw.edge and lat_abs > 0.85 * half_w and not fr["off_track"]:
                pen = rw.edge * (lat_abs / half_w - 0.85) / 0.15
                step_reward -= pen
                reward_parts["edge"] -= pen

            slip = abs(float(np.degrees(self.veh.slip_angle)))
            if rw.slip_guard and slip > 16.0 and self.veh.speed > 10.0:
                pen = rw.slip_guard * min(2.0, (slip - 16.0) / 24.0)
                step_reward -= pen
                reward_parts["slip_guard"] -= pen

            # spin detection (same shape that worked in the ring pipeline)
            yaw_rate = abs(float(self.veh.r))
            path_yaw_rate = self.veh.speed * abs(float(fr["curvature"]))
            yaw_limit = max(1.15, path_yaw_rate * 1.35 + 0.45)
            spin_slip = slip > 46.0 or (slip > 32.0 and abs(head_err) > 0.35)
            spin_yaw = yaw_rate > yaw_limit and abs(head_err) > 0.35
            spinning = self.veh.speed > 8.0 and (spin_slip or spin_yaw)
            self.spin_t = self.spin_t + dt if spinning else max(0.0, self.spin_t - dt)
            if spinning and rw.spin:
                pen = rw.spin * min(2.0, max(slip / 65.0,
                                             yaw_rate / max(yaw_limit, 1e-6)))
                step_reward -= pen
                reward_parts["spin"] -= pen

            if int(self.cum) > int(prev_cum) and d > 0:
                theo = float(getattr(self.trk, "fable_envelope", {}).get(
                    "lap_time", 0.0) or 0.0)
                bonus = rw.lap_bonus
                if theo > 0 and self.t > 1.0:
                    bonus += rw.lap_pace_bonus * float(np.clip(
                        theo / max(self.t, 1.0), 0.0, 1.2))
                step_reward += bonus
                reward_parts["lap_bonus"] += bonus
                lap_completed_t = self.t

            # closure milestones: one-time bonuses for carrying the run deep —
            # the gradient a near-lap brain needs to close the final sectors
            if rw.milestone_bonus and self._milestone_i < len(_MILESTONES) \
                    and self.max_cum >= _MILESTONES[self._milestone_i]:
                b = rw.milestone_bonus * _MILESTONE_W[self._milestone_i]
                self._milestone_i += 1
                step_reward += b
                reward_parts["milestone"] += b

            crawl = self.t > 1.0 and self.veh.speed < 0.8
            self.stall_t = self.stall_t + dt if crawl else 0.0
            if self.cum > self.best_cum + 0.00025:
                self.best_cum = self.cum
                self.since_prog = 0.0
            else:
                self.since_prog += dt

            if self.off_t > sp.offtrack_timeout:
                terminated = True; termination_reason = "offtrack_timeout"
            elif lat_abs > half_w + sp.left_track_margin:
                terminated = True; termination_reason = "left_track"
            elif self.stall_t > sp.stall_timeout:
                terminated = True; termination_reason = "stall"
            elif self.since_prog > sp.progress_timeout:
                terminated = True; termination_reason = "no_progress"
            elif self.cum < self.max_cum - sp.backwards_tol:
                step_reward -= rw.backwards
                reward_parts["backwards"] -= rw.backwards
                terminated = True; termination_reason = "backwards"
            elif self.spin_t > sp.spin_timeout:
                terminated = True; termination_reason = "spin"
            elif (getattr(self.veh, "engine_damage", 0.0) > 0.75
                  or getattr(self.veh, "drivetrain_broken", False)):
                terminated = True; termination_reason = "severe_damage"

            reward += step_reward
            clutch, up, down = self.rbox.update(self.veh, throttle, dt,
                                                gear_offset=gear_offset)
            self.veh.step(Controls(steer=steer, throttle=throttle, brake=brake,
                                   clutch=clutch, handbrake=0.0,
                                   shift_up=up, shift_down=down))
            self.t += dt
            self.total_dist += self.veh.speed * dt
            if self.veh.airborne:
                self.ep_air_time += dt
                if not self._was_air:
                    self.ep_jumps += 1
            self._was_air = self.veh.airborne
            self.ep_max_lg = max(self.ep_max_lg, self.veh.landing_g)
            if terminated:
                break

        # Certification/evaluation uses the whole body footprint, not merely
        # the centre of mass. This runs only in diagnostics/eval environments,
        # keeping rollout throughput intact while making every "clean" gate
        # honest about the car fitting inside the road boundary.
        # Also sample the state produced by the final physics substep. The loop
        # above certifies every substep start; this closes the last interval.
        if self.diagnostics:
            final_footprint_off = body_offtrack()
            if final_footprint_off:
                self.fable_invalid = True
                self.fable_footprint_invalid = True
                if not footprint_off:
                    self.fable_footprint_offtrack_total += dt
                    if not fr.get("off_track", False):
                        self.fable_offtrack_total += dt
            footprint_off = footprint_off or final_footprint_off

        # steering jerk (2nd difference): d_now vs d_prev. A smooth turn keeps
        # d_now ~ d_prev (low cost); a saw flips it sign-to-sign (high cost).
        d_now = float(a[0] - self.prev_action[0])
        d_prev = float(self.prev_action[0] - self.prev_prev_action[0])
        jerk = -sp.reward.smooth_steer2 * (d_now - d_prev) ** 2
        # steering saturation: only the authority spent past the deadzone.
        sat_excess = max(0.0, abs(float(a[0])) - sp.reward.steer_sat_thresh)
        sat = -sp.reward.steer_sat * sat_excess ** 2
        smooth = (-sp.reward.smooth_steer * float((a[0] - self.prev_action[0]) ** 2)
                  - sp.reward.smooth_long * float((a[1] - self.prev_action[1]) ** 2)
                  + jerk + sat)
        reward += smooth
        reward_parts["smooth"] = smooth
        self.prev_prev_action = self.prev_action
        self.prev_action = a
        if terminated:
            reward -= rw.crash
            reward_parts["crash"] = -rw.crash
            self.fable_terminal_reason = termination_reason
            # feed the weakest-sector replay with WHERE it died
            self._pending_fail_arc = float(self.trk.frame(
                self.veh.x, self.veh.y)["arc"])

        truncated = (not terminated) and (self.t >= self.ppo.episode_seconds)
        if truncated:
            termination_reason = termination_reason or "time_limit"
        pace_mean = (self.fable_pace_num / self.fable_pace_den
                     if self.fable_pace_den else 0.0)
        info = {
            "laps": self.max_cum,
            "speed": self.veh.speed,
            "t": self.t,
            "difficulty": self.difficulty,
            "airtime": self.ep_air_time,
            "jumps": self.ep_jumps,
            "max_landing_g": self.ep_max_lg,
            "fable_stage": sp.stage,
            "fable_progress_m": float(max(0.0, self.max_cum * self.trk.length)),
            "fable_clean": bool(not self.fable_invalid and not terminated),
            "fable_invalid": bool(self.fable_invalid),
            "fable_offtrack_seconds": float(self.fable_offtrack_total),
            "fable_footprint_offtrack_seconds": float(
                self.fable_footprint_offtrack_total),
            "fable_footprint_valid": bool(not self.fable_footprint_invalid),
            "fable_pace_ratio": float(pace_mean),
            "fable_lap_t": lap_completed_t,
            "drivetrain_version": getattr(self.veh.spec, "drivetrain_version",
                                          DRIVETRAIN_VERSION),
            "gear": int(self.veh.gear),
            "requested_gear": int(self.rbox.requested_gear),
            "shift_phase": self.rbox.shift_phase,
            "shift_rejected": self.rbox.rejected_shift,
        }
        if self.diagnostics:
            info["reward_parts"] = {k: float(v) for k, v in reward_parts.items()
                                    if abs(float(v)) > 1e-12}
            info["diagnostics"] = {k: float(v) for k, v in diag_values.items()}
            info["termination_reason"] = termination_reason
        return self._obs(), float(reward), terminated, truncated, info


# --------------------------------------------------------------------------- #
# PitWall — the race-engineer loop above PPO
# --------------------------------------------------------------------------- #
PIT_LOG = "fable5_pit_log.txt"
PIT_LR_FLOOR = 1e-5


class PitWall:
    """Judges every eval against the RUN'S TREND (not the latest snapshot),
    decides continue / reseed-from-best, and leaves a paper trail.

    PPO's own plateau logic only reacts to "no new best for `patience` iters".
    The pit wall reacts much sooner to the classic failure signature: a streak
    of evals that are far below the banked best (exploration wandered off) or
    that keep terminating. Every decision is appended to fable5_pit_log.txt
    and mirrored into the manifest for the dashboard.
    """

    def __init__(self, stage: str, run_name: str = "",
                 regress_streak: int = 3, regress_frac: float = 0.60,
                 min_evals_between_reseeds: int = 4, max_reseeds: int = 6,
                 max_rollbacks: int = 12, collapse_frac: float = 0.25,
                 collapse_floor: float = 60.0,
                 healthy_clean: int = 12, healthy_terminal: float = 0.25,
                 healthy_progress: float = 0.90):
        self.stage = stage
        self.run_name = run_name
        self.history: list[dict] = []
        self.decisions: list[dict] = []
        self.best_metric = -1e18
        self.evals_since_best = 0
        self.reseeds = 0                    # total (stats)
        self.reseeds_since_best = 0         # the actual budget: REFILLS on a
                                            # new best, so progress re-arms
                                            # the pit wall instead of retiring it
        self.consolidations = 0
        self._consolidated_stretch = False  # one consolidation per dry spell
        self.evals_since_reseed = 10 ** 6
        self.regress_streak = regress_streak
        self.regress_frac = regress_frac
        self.min_gap = min_evals_between_reseeds
        self.max_reseeds = max_reseeds
        # ROLLBACK — the collapse lever, distinct from RESEED (the plateau
        # lever). Once this run has banked a LAP-SCALE best (metric >=
        # collapse_floor with a lap: fable's lap metric is 100000/lap_time
        # ~130+, while no-lap evals top out ~20-35, so the floor separates
        # them cleanly), any eval below collapse_frac x best means the last
        # eval-window of updates destroyed the lap. Response: restore the
        # best CALMLY (no noise re-widening, no anneal reset — see
        # PPO.reseed_from_best(calm=True)) and decay the LR, immediately, with
        # no streak/gap wait. Budget refills on a new best, like reseeds.
        self.max_rollbacks = max_rollbacks
        self.collapse_frac = collapse_frac
        self.collapse_floor = collapse_floor
        self.rollbacks = 0                  # total (stats)
        self.rollbacks_since_best = 0
        # HEALTHY thresholds — a lapless eval that still clears all three is a
        # car that DRIVES (mostly-clean sectors, low terminal rate, near-lap
        # progress), so the collapse/streak levers leave it alone to keep
        # refining. 12/16 sectors clean; <=1/4 of sectors ending in a crash;
        # the flying-line run covering >=90% of lap distance. progress_frac
        # can exceed 1.0 (Run-4's healthy "wreck" covered 1.7 laps).
        self.healthy_clean = int(healthy_clean)
        self.healthy_terminal = float(healthy_terminal)
        self.healthy_progress = float(healthy_progress)
        self.best_lap = None                # lap_time of the best-metric eval
        self.fallback_source = None         # rollback source of last resort
                                            # (the resumed checkpoint)
        self.hof_provider = None            # set by the trainer: () -> {cat: rec}

    def seed_from_resume(self, path: str) -> bool:
        """Arm the pit wall from the RESUMED checkpoint's stored eval. Without
        this, a lap-capable resume that degrades within its FIRST eval window
        (up to eval_every iterations of drift) banks the wreck as the run's
        'best' and the rollback lever never arms — seen live on the first
        KLGUARD confirmation run (resumed a 176.56/565s brain, first eval
        0.063, pit called it a new best). Seeded at a 5% discount so an eval
        that merely MATCHES the resumed quality still banks a new best and
        refills the lever budgets. Cross-stage resumes seed the shared
        lap-metric core (100000/lap) since stage bonuses aren't comparable."""
        if not path or not os.path.exists(path):
            return False
        try:
            meta = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        ev = meta.get("fable_eval") or {}
        self.fallback_source = path
        # Never compare unlike stage metrics. Foundation/flow live evidence had
        # ~37-point scores permanently judged against an inherited ~185 lap
        # metric. A baseline destination-stage eval now arms the new stage.
        if ev.get("stage") != self.stage:
            return False
        stored = float(ev.get("metric") or 0.0)
        if not np.isfinite(stored):
            return False
        lap = ev.get("lap_time")
        base = stored
        self.best_metric = max(self.best_metric, 0.95 * base)
        self.best_lap = float(lap) if lap else self.best_lap
        return True

    def restore_state(self, state: dict | None) -> bool:
        """Restore same-run Pit state across adaptive auto segments/restarts."""
        state = state or {}
        if (state.get("stage") != self.stage
                or state.get("run_name") != self.run_name):
            return False
        self.history = list(state.get("history") or [])[-40:]
        self.decisions = list(state.get("decisions") or [])[-40:]
        for name in ("evals_since_best", "reseeds", "reseeds_since_best",
                     "consolidations", "evals_since_reseed", "rollbacks",
                     "rollbacks_since_best"):
            if name in state:
                setattr(self, name, int(state[name]))
        self.best_metric = float(state.get("best_metric", self.best_metric))
        lap = state.get("best_lap")
        self.best_lap = float(lap) if lap else None
        self._consolidated_stretch = bool(state.get("consolidated_stretch", False))
        return True

    # -- trend windows ------------------------------------------------------ #
    def _window(self, n: int) -> list[dict]:
        return self.history[-n:]

    def trends(self) -> dict:
        w5 = self._window(5)
        if not w5:
            return {}
        return {
            "evals": len(self.history),
            "mean_metric_5": float(np.mean([h["metric"] for h in w5])),
            "mean_pace_5": float(np.mean([h.get("pace_ratio", 0.0) for h in w5])),
            "clean_rate_5": float(np.mean([
                h.get("clean_sectors", 0) / max(1, h.get("sector_count", 1))
                for h in w5])),
            "terminal_rate_5": float(np.mean([h.get("terminal_rate", 0.0)
                                              for h in w5])),
            "best_metric": self.best_metric,
            "evals_since_best": self.evals_since_best,
            "reseeds": self.reseeds,
        }

    def _regressed(self, m: float) -> bool:
        # far below best, robust to negative early metrics
        return m < self.best_metric - abs(self.best_metric) * (1 - self.regress_frac) - 1.0

    def _reseed_source(self, ppo, latest: dict) -> tuple[str | None, str]:
        """Pick WHICH banked brain to reseed from by the current failure mode:
        dying a lot -> the cleanest brain; falling far short of the distance
        record -> the farthest brain; otherwise the metric best."""
        best = getattr(ppo, "_best_path", None)
        if not (best and os.path.exists(best)):
            # resumed run that hasn't banked its own _best yet: the resumed
            # checkpoint is still a valid brain to reseed from
            best = (self.fallback_source
                    if self.fallback_source and os.path.exists(self.fallback_source)
                    else None)
        default = (best, "metric best") if best else (None, "")
        try:
            hof = dict(self.hof_provider()) if self.hof_provider else {}
        except Exception:
            hof = {}
        if not hof:
            return default

        def pick(cat: str, why: str):
            p = (hof.get(cat) or {}).get("path")
            if p and os.path.exists(p) and p != default[0]:
                return p, why
            return None

        tr5 = float(self.trends().get("terminal_rate_5") or 0.0)
        if tr5 >= 0.4:
            c = pick("clean", f"terminal rate {tr5:.2f} — the cleanest brain")
            if c:
                return c
        prog_rec = float(((hof.get("progress") or {}).get("eval") or {})
                         .get("progress_frac") or 0.0)
        prog_now = float(latest.get("progress_frac") or 0.0)
        if prog_rec > 0.5 and prog_now < 0.6 * prog_rec:
            c = pick("progress", f"progress collapsed ({prog_now:.2f} vs "
                                 f"record {prog_rec:.2f}) — the farthest brain")
            if c:
                return c
        return default

    def _consolidate(self, ppo) -> float | None:
        """Second lever after reseeds stop working: halve the learning rate and
        entropy so the policy stops churning and polishes the basin it's in.
        cfg.lr is halved too, so later reseeds/anneals adopt the calmer rate."""
        try:
            lr_now = None
            for g in ppo.opt.param_groups:
                g["lr"] = max(PIT_LR_FLOOR, float(g["lr"]) * 0.5)
                lr_now = g["lr"]
            ppo._ent_coef = max(1e-4, float(getattr(ppo, "_ent_coef", 3e-3)) * 0.5)
            if hasattr(ppo, "cfg") and getattr(ppo.cfg, "lr", None):
                ppo.cfg.lr = max(PIT_LR_FLOOR, float(ppo.cfg.lr) * 0.5)
            # halve cfg.ent_coef too: with anneal on (finish/fast/frontier),
            # set_schedule recomputes _ent_coef from cfg.ent_coef EVERY
            # iteration — without this the entropy half of the lever was
            # overwritten one iteration later in exactly the stages that
            # needed it.
            if hasattr(ppo, "cfg") and getattr(ppo.cfg, "ent_coef", None):
                ppo.cfg.ent_coef = max(1e-4, float(ppo.cfg.ent_coef) * 0.5)
            return lr_now
        except Exception:
            return None

    # -- the decision ------------------------------------------------------- #
    def note(self, ppo, latest: dict) -> dict:
        m = float(latest.get("metric", -1e9))
        # HEALTHY = the car still drives well even if the (binary, fragile)
        # flying lap didn't close this eval: mostly-clean sectors, low
        # terminal rate, real progress. Run-4 live data made the case: an
        # eval that covered 35.6 km (1.7 laps of distance) at terminal 0.06
        # was rolled back because the lap wasn't scored — refinement can
        # never accumulate if every lapless eval resets to the checkpoint.
        # The pit retires a BROKEN car, not a car that missed one lap.
        healthy = (int(latest.get("clean_sectors") or 0) >= self.healthy_clean
                   and float(latest.get("terminal_rate") if
                             latest.get("terminal_rate") is not None else 1.0)
                   <= self.healthy_terminal
                   and float(latest.get("progress_frac") or 0.0)
                   >= self.healthy_progress)
        self.history.append({**{k: latest.get(k) for k in
                                ("metric", "pace_ratio", "clean_sectors",
                                 "sector_count", "terminal_rate", "lap_time",
                                 "progress_frac")},
                             "healthy": healthy})
        self.evals_since_reseed += 1
        new_best = m > self.best_metric
        if new_best:
            self.best_metric = m
            self.evals_since_best = 0
            self.reseeds_since_best = 0          # progress refills the budget
            self.rollbacks_since_best = 0        # ...and the rollback budget
            self._consolidated_stretch = False
            if latest.get("lap_time"):
                self.best_lap = float(latest["lap_time"])
        else:
            self.evals_since_best += 1

        # the collapse/calm levers arm only once the run is chasing a BANKED
        # LAP at lap-scale metric (fast/frontier reality; flow's 542s lap at
        # metric ~40 stays below the floor, keeping its plateau levers intact)
        lap_armed = (self.best_lap is not None
                     and self.best_metric >= self.collapse_floor)

        recent = self._window(self.regress_streak)
        bad_streak = (len(recent) >= self.regress_streak
                      and all((self._regressed(h["metric"]) or
                               (h.get("terminal_rate") or 0) >= 0.5)
                              and not (lap_armed and h.get("healthy"))
                              for h in recent)
                      and not new_best)

        # collapse: a lap-scale best is banked and this eval fell off a cliff.
        # Source: the run's _best.pt — but a RESUMED run inherits its keep-best
        # floor, so its own _best may not exist until a NEW record; fall back
        # to the hall-of-fame lap brain (banked from this run's first eval).
        best_path = getattr(ppo, "_best_path", None)
        if not (best_path and os.path.exists(best_path)):
            best_path = None
            try:
                hof = dict(self.hof_provider()) if self.hof_provider else {}
                p = (hof.get("lap") or {}).get("path")
                if p and os.path.exists(p):
                    best_path = p
            except Exception:
                pass
            if best_path is None and self.fallback_source \
                    and os.path.exists(self.fallback_source):
                best_path = self.fallback_source
        collapsed = (not new_best
                     and lap_armed
                     and not healthy
                     and m < self.collapse_frac * self.best_metric
                     and best_path is not None)

        src_path, src_why = (None, "")
        if not new_best and bad_streak:
            src_path, src_why = self._reseed_source(ppo, latest)

        if new_best:
            decision, reason = "continue", (
                f"new best banked (metric {m:.3f}"
                + (f", lap {latest['lap_time']:.2f}s" if latest.get("lap_time")
                   else "") + ")")
        elif collapsed and self.rollbacks_since_best < self.max_rollbacks \
                and ppo.reseed_from_best(best_path, getattr(ppo, "updates", 0),
                                         calm=True):
            self.rollbacks += 1
            self.rollbacks_since_best += 1
            lr_now = None
            try:
                # decay the LR each rollback so repeated falls keep calming the
                # updates (set_schedule recomputes from cfg.lr, so decay THAT);
                # ease the entropy pressure with it — wide noise at race pace is
                # what fills the rollouts with crashes in the first place
                if hasattr(ppo, "cfg") and getattr(ppo.cfg, "lr", None):
                    ppo.cfg.lr = max(PIT_LR_FLOOR, float(ppo.cfg.lr) * 0.85)
                if hasattr(ppo, "cfg") and getattr(ppo.cfg, "ent_coef", None):
                    ppo.cfg.ent_coef = max(1e-4, float(ppo.cfg.ent_coef) * 0.85)
                for g in ppo.opt.param_groups:
                    g["lr"] = max(PIT_LR_FLOOR, float(g["lr"]) * 0.85)
                    lr_now = g["lr"]
            except Exception:
                pass
            try:
                # each restore re-loads the banked (wide) log_std — the very
                # noise that crashes the rollouts at race pace, and entropy
                # decay can't bite because the restore snaps it back (run-2
                # live data: ent pinned at 2.9 across 4 rollbacks, every
                # window lost the lap). Narrow it a notch per rollback so the
                # restored brain trains ever closer to the line it actually
                # drives: refinement, not exploration.
                ls = ppo.net.log_std
                ls.data.sub_(0.10 * self.rollbacks_since_best)
                ppo.net.clamp_log_std_()
            except Exception:
                pass
            decision = "rollback"
            reason = (f"collapse ({m:.3f} vs lap-best {self.best_metric:.3f}) — "
                      f"the last eval-window of updates lost the lap; restored "
                      f"{os.path.basename(best_path)} calmly (log-std "
                      f"-{0.10 * self.rollbacks_since_best:.2f})"
                      + (f", lr -> {lr_now:.2e}" if lr_now else "")
                      + f"; rollback {self.rollbacks_since_best}/"
                        f"{self.max_rollbacks} (budget refills on a new best)")
        elif bad_streak and self.evals_since_reseed >= self.min_gap \
                and self.reseeds_since_best < self.max_reseeds \
                and src_path:
            # while a lap is armed, even the streak reseed stays CALM: noise
            # re-widening is the plateau-escape tool for a brain still HUNTING
            # its first lap — re-injecting it into a lap-capable brain is the
            # exact loop that burned the MANPLEASEWORK night
            calm = self.best_lap is not None
            ok = ppo.reseed_from_best(src_path, getattr(ppo, "updates", 0),
                                      calm=calm)
            if ok:
                self.reseeds += 1
                self.reseeds_since_best += 1
                self.evals_since_reseed = 0
                decision = "reseed"
                reason = (f"last {self.regress_streak} evals far below best "
                          f"({m:.3f} vs {self.best_metric:.3f}) — exploration "
                          f"wandered; {'CALM ' if calm else ''}reseed "
                          f"{self.reseeds_since_best}/"
                          f"{self.max_reseeds} (budget refills on a new best) "
                          f"from {os.path.basename(src_path)} ({src_why})")
            else:
                decision, reason = "continue", "reseed unavailable; continuing"
        elif bad_streak and self.reseeds_since_best >= self.max_reseeds \
                and not self._consolidated_stretch:
            lr_now = self._consolidate(ppo)
            self._consolidated_stretch = True
            if lr_now is not None:
                self.consolidations += 1
                decision = "consolidate"
                reason = (f"reseeds exhausted without a new best — halving "
                          f"lr (now {lr_now:.2e}) + entropy to polish the "
                          f"current basin instead of churning")
            else:
                decision, reason = "continue", "consolidation unavailable"
        elif bad_streak:
            decision, reason = "continue", (
                f"regressing; pit levers spent for this dry spell — a new "
                f"best refills them, else PPO plateau logic ends the stage")
        elif self._regressed(m):
            decision, reason = "continue", (
                f"weak eval ({m:.3f} vs best {self.best_metric:.3f}) — "
                f"watching for a streak")
        else:
            decision, reason = "continue", (
                f"holding near best ({m:.3f} vs {self.best_metric:.3f})")

        entry = {
            "t": time.strftime("%H:%M:%S"),
            "stage": self.stage,
            "metric": round(m, 3),
            "lap": latest.get("lap_time"),
            "decision": decision,
            "reason": reason,
        }
        self.decisions.append(entry)
        self.decisions = self.decisions[-40:]
        lap_s = f" lap={latest['lap_time']:.2f}s" if latest.get("lap_time") else ""
        line = (f"[pit] {entry['t']} {self.run_name or self.stage}: "
                f"metric={m:.3f} best={self.best_metric:.3f}{lap_s} "
                f"-> {decision.upper()} ({reason})")
        print("  " + line, flush=True)
        try:
            with open(PIT_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        return entry

    def state(self) -> dict:
        return {"decisions": self.decisions[-12:], "trends": self.trends(),
                "history": self.history[-20:],
                "best_metric": self.best_metric,
                "best_lap": self.best_lap,
                "evals_since_best": self.evals_since_best,
                "evals_since_reseed": self.evals_since_reseed,
                "reseeds": self.reseeds,
                "reseeds_since_best": self.reseeds_since_best,
                "rollbacks": self.rollbacks,
                "rollbacks_since_best": self.rollbacks_since_best,
                "consolidations": self.consolidations,
                "consolidated_stretch": self._consolidated_stretch,
                "stage": self.stage, "run_name": self.run_name}


# --------------------------------------------------------------------------- #
# Evaluator — sector cleanliness + pace + a line lap with a REACHABLE budget
# --------------------------------------------------------------------------- #
class FableEvaluator:
    def __init__(self, spec: FableSpec, track,
                 manifest_path: str | Path | None = None,
                 stop_on_gate: bool = False, pit: PitWall | None = None,
                 resumed_from: str | None = None, hof_base: str | None = None,
                 checkpoint: str | None = None, car: str = FABLE_CAR):
        self.spec = copy.deepcopy(spec)
        self.track = attach_envelope(track)
        if manifest_path is None:
            manifest_path = pipeline_manifest_for(car)
        self.manifest_path = Path(manifest_path)
        self.latest: dict[str, Any] = {}
        self.theoretical = float(track.fable_envelope["lap_time"])
        self.superhuman_lap, self.superhuman_key = superhuman_lap_for(
            track.fable_envelope.get("params", {}).get("drivetrain_version"))
        # the identity every stamp in this run must carry: the RUN's car and
        # its drivetrain — never the module's 787B constant (that mislabeled
        # the first 919 lineage as Mazda evidence).
        self.car = car
        self.drivetrain_version = (
            track.fable_envelope.get("params", {}).get("drivetrain_version")
            or drivetrain_for(car))
        self.pit = pit
        self.resumed_from = resumed_from
        self.checkpoint = checkpoint
        # auto ladder: once this stage's gate recommendation appears, ask the
        # trainer to stop cleanly (the peak is banked) so the budget rolls on
        self.stop_on_gate = bool(stop_on_gate)
        self.gate_rec = GATE_ADVANCE.get(self.spec.stage)
        # hall of fame: reload existing category bests so a rerun can never
        # clobber a stronger banked record with a weaker one
        self.hof_base = hof_base
        self.hof: dict[str, dict] = {}
        if hof_base:
            for cat in HOF_CATEGORIES:
                p = f"{hof_base}_hof_{cat}.pt"
                ok, _, ev = _best_eval_of(p)
                if (ok and ev
                        and _checkpoint_protocol_current(p, self.spec.stage,
                                                         self.car)):
                    self.hof[cat] = {"path": p, "eval": ev}

    def __call__(self, ppo: PPO) -> dict[str, Any]:
        latest = self.evaluate(ppo)
        latest["checkpoint_score"] = _checkpoint_score(latest)
        latest["evaluated_policy_sha256"] = ppo.policy_hash()
        latest["eval_protocol"] = EVAL_PROTOCOL_VERSION
        latest["code_fingerprint"] = CODE_FINGERPRINT
        latest["drivetrain_version"] = self.drivetrain_version
        latest["car"] = self.car
        self.latest = latest
        self._bank_hall_of_fame(ppo, latest)
        if (self.stop_on_gate and self.gate_rec
                and latest.get("recommendation") == self.gate_rec):
            ppo.request_stop = True
        return latest

    def after_eval(self, ppo: PPO, latest: dict[str, Any]) -> dict[str, Any]:
        """Phase two of eval: mutate only after PPO banked evaluated weights."""
        entry = self.pit.note(ppo, latest) if self.pit is not None else {
            "decision": "continue", "reason": "no pit wall"}
        changed = entry.get("decision") in ("rollback", "reseed", "consolidate")
        entry = dict(entry, policy_changed=changed)
        self._write_manifest(ppo, latest, checkpoint=self.checkpoint)
        self._append_event(ppo, latest, entry)
        return entry

    def metadata(self) -> dict[str, Any]:
        return {
            "fable_pipeline": True,
            "fable_stage": self.spec.stage,
            "fable_reward_version": FABLE_REWARD_VERSION,
            "fable_theoretical_lap": self.theoretical,
            "fable_envelope_scale": float(self.spec.envelope_scale),
            "fable_shift_lo_frac": float(getattr(self.spec, "shift_lo_frac",
                                                 RACE_SHIFT_LO_FRAC)),
            "fable_resumed_from": self.resumed_from,
            "fable_eval": self.latest,
            "fable_eval_protocol": EVAL_PROTOCOL_VERSION,
            "fable_code_fingerprint": CODE_FINGERPRINT,
            "fable_drivetrain_version": self.drivetrain_version,
            "fable_pit_state": self.pit.state() if self.pit is not None else None,
        }

    def _append_event(self, ppo, latest: dict, pit_entry: dict) -> None:
        run = ((self.pit.run_name if self.pit is not None else None)
               or self.spec.stage)
        path = Path("runtime") / "fable5" / run / "events.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            event_eval = dict(latest)
            event_eval.pop("latest_trace", None)
            payload = {"time_unix": time.time(), "run": run,
                       "car": self.car,
                       "drivetrain_version": self.drivetrain_version,
                       "stage": self.spec.stage,
                       "checkpoint": self.checkpoint,
                       "updates": int(getattr(ppo, "updates", 0)),
                       "policy_sha256": latest.get("evaluated_policy_sha256"),
                       "eval_protocol": EVAL_PROTOCOL_VERSION,
                       "code_fingerprint": CODE_FINGERPRINT,
                       "evaluation": event_eval, "pit": pit_entry}
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=_jsonable) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as e:
            print(f"  [health] event log append failed: {e}", flush=True)

    def lap_budget(self) -> float:
        if self.spec.eval_lap_budget and self.spec.eval_lap_budget > 0:
            return float(self.spec.eval_lap_budget)
        return max(900.0, 2.4 * self.theoretical)

    def _bank_hall_of_fame(self, ppo, latest: dict):
        """Category bests beside the single metric best — the farthest brain,
        the cleanest brain, the fastest lap — so the pit wall can reseed from
        whichever one fixes the CURRENT failure mode."""
        if not self.hof_base or not hasattr(ppo, "save"):
            return
        ppo.metric = float(latest.get("metric", 0.0))   # stamp before save
        for cat in HOF_CATEGORIES:
            key = _hof_key(cat, latest)
            if key is None:
                continue
            cur = self.hof.get(cat)
            if cur is not None:
                cur_key = _hof_key(cat, cur.get("eval") or {})
                if cur_key is not None and key <= cur_key:
                    continue
            path = f"{self.hof_base}_hof_{cat}.pt"
            try:
                ppo.save(path)
            except Exception as e:
                print(f"  [hof] save failed ({cat}): {e}", flush=True)
                continue
            self.hof[cat] = {"path": path, "eval": dict(latest)}
            print(f"  [hof] {cat} record -> {os.path.basename(path)}", flush=True)

    def hof_summary(self) -> dict:
        out = {}
        for cat, rec in self.hof.items():
            ev = rec.get("eval") or {}
            out[cat] = {"path": rec.get("path"),
                        "progress_frac": ev.get("progress_frac"),
                        "clean_sectors": ev.get("clean_sectors"),
                        "clean_chain": ev.get("clean_chain"),
                        "terminal_rate": ev.get("terminal_rate"),
                        "lap_time": ev.get("lap_time"),
                        "metric": ev.get("metric")}
        return out

    @torch.no_grad()
    def evaluate(self, ppo: PPO) -> dict[str, Any]:
        cfg = copy.copy(ppo.cfg)
        cfg.random_start = False
        cfg.episode_seconds = max(self.lap_budget(), self.spec.eval_sector_seconds)
        env = FableEnv(mode="race", car=ppo.car, ppo=cfg, sim=ppo.sim_cfg,
                       fixed_track=self.track, fable_spec=self.spec,
                       rng_seed=880, diagnostics=True)
        n = max(4, int(self.spec.eval_starts))
        starts = [int(len(self.track.center) * i / n) for i in range(n)]
        steps_sector = int(round(self.spec.eval_sector_seconds * cfg.control_hz))
        rows = []
        for idx in starts:
            v0 = vref_at(self.track, float(self.track.arc[idx])) \
                * self.spec.envelope_scale * 0.90
            rows.append(self._run_from(ppo, env, idx, min(v0, 60.0), steps_sector))

        # FLYING line lap: Bellof's 6:11 was a flying qualifying lap, so the
        # benchmark run drops in at race pace rather than a standing start —
        # the pipeline can bank a lap the moment the driving exists, without
        # also needing a launch skill the benchmark never measured.
        # Fixed protocol start: lap times remain comparable across stages/scales.
        v0_line = min(60.0, vref_at(self.track, float(self.track.arc[0])) * 0.90)
        line_steps = int(round(self.lap_budget() * cfg.control_hz))
        line = self._run_from(ppo, env, 0, v0_line, line_steps)

        clean_sectors = sum(1 for r in rows if r["clean"])
        clean_chain = self._clean_chain(rows)
        terminal_rate = (sum(1 for r in rows
                             if r["termination_reason"] not in (None, "time_limit"))
                         / max(1, len(rows)))
        pace_ratio = float(np.mean([r["pace_ratio"] for r in rows])) if rows else 0.0
        max_progress_m = float(max([line["progress_m"],
                                    *[r["progress_m"] for r in rows]], default=0.0))
        offtrack_seconds = float(line["offtrack_seconds"]
                                 + sum(r["offtrack_seconds"] for r in rows))
        lap_time = line["lap_time"] if line["clean_lap"] else None
        invalid_laps = int(0 if line["clean_lap"] else (1 if line["laps"] >= 1.0 else 0))
        distance_laps = max_progress_m / max(1.0, self.track.length)
        # Invalid multi-lap distance must not outrank a clean 99% near-lap.
        progress_frac = min(1.0, distance_laps)

        st = self.spec.stage
        if lap_time:
            lap_metric = 100000.0 / lap_time
        else:
            lap_metric = None
        if st == "foundation":
            metric = clean_sectors * 2.0 + pace_ratio * 5.0 \
                - terminal_rate * 6.0 - offtrack_seconds * 0.05
            gate = clean_sectors >= 14 and terminal_rate <= 0.125
            rec = "advance to flow" if gate else "keep training foundation"
        elif st == "flow":
            metric = clean_chain + clean_sectors + pace_ratio * 10.0 \
                - terminal_rate * 5.0 - offtrack_seconds * 0.05
            gate = clean_chain >= n and pace_ratio >= 0.55
            rec = "advance to finish" if gate else "keep training flow"
        elif st == "finish":
            metric = (lap_metric + clean_sectors * 0.5) if lap_metric else \
                (progress_frac * 20.0 + clean_chain * 0.5 - terminal_rate * 4.0)
            rec = "advance to fast" if lap_time else "keep training finish"
        elif st == "fast":
            metric = (lap_metric - terminal_rate * 3.0) if lap_metric else \
                (progress_frac * 10.0 - terminal_rate * 4.0)
            rec = ("advance to frontier" if lap_time and lap_time <= 450.0
                   else "keep training fast")
        else:
            metric = (lap_metric - terminal_rate * 3.0
                      - offtrack_seconds * 0.02) if lap_metric else \
                (progress_frac * 10.0 - terminal_rate * 4.0)
            if lap_time and lap_time < self.superhuman_lap:
                rec = "SUPERHUMAN — keep compressing toward the theoretical lap"
            elif lap_time:
                rec = "keep attacking the benchmark lap"
            else:
                rec = "recover a clean lap first"

        termination_counts: dict[str, int] = {}
        for r in [line, *rows]:
            reason = r["termination_reason"] or "none"
            termination_counts[reason] = termination_counts.get(reason, 0) + 1

        # which eval sectors failed this round (feeds the persisted weak-sector
        # heat; the line lap's death maps to the sector it died in)
        fail_sectors = [min(n - 1, int(float(r.get("terminal_arc") or 0.0)
                                      / max(1.0, self.track.length) * n))
                        for r in rows
                        if r["termination_reason"] not in (None, "time_limit")]
        if not line["clean_lap"] and line["termination_reason"] \
                not in (None, "time_limit"):
            fail_sectors.append(min(n - 1, int(
                line["progress_m"] / max(1.0, self.track.length) * n)))

        # "why slow": rank the sector starts by pace (terminations sort worst)
        ranked = sorted(range(len(rows)),
                        key=lambda k: (rows[k]["clean"],
                                       rows[k]["pace_ratio"]))
        worst_sectors = [{
            "sector": int(k),
            "pace": round(float(rows[k]["pace_ratio"]), 3),
            "clean": bool(rows[k]["clean"]),
            "reason": rows[k]["termination_reason"],
        } for k in ranked[:3]]

        latest = {
            "stage": st,
            "metric": float(metric),
            "clean_sectors": int(clean_sectors),
            "sector_count": int(len(rows)),
            "clean_chain": int(clean_chain),
            "terminal_rate": float(terminal_rate),
            "pace_ratio": float(pace_ratio),
            "max_progress_m": max_progress_m,
            "progress_frac": float(progress_frac),
            "distance_laps": float(distance_laps),
            "offtrack_seconds": offtrack_seconds,
            "lap_time": lap_time,
            "lap_style": "flying",
            "invalid_laps": invalid_laps,
            "mean_speed": float(np.mean([r["mean_speed"] for r in rows])) if rows else 0.0,
            "theoretical_lap": self.theoretical,
            "superhuman": bool(lap_time and lap_time < self.superhuman_lap),
            "vs_bellof": (float(lap_time / self.superhuman_lap) if lap_time else None),
            "vs_theoretical": (float(lap_time / self.theoretical) if lap_time else None),
            "superhuman_target": self.superhuman_lap,
            "superhuman_key": self.superhuman_key,
            "benchmarks": dict(HUMAN_BENCHMARKS),
            "termination_counts": termination_counts,
            "worst_sectors": worst_sectors,
            "fail_sectors": fail_sectors,
            "recommendation": rec,
            "laps": float(line["laps"]),
            "drift": 0.0,
            "latest_trace": line.get("trace", []),
            "log": self._log_line(metric, clean_sectors, len(rows), clean_chain,
                                  pace_ratio, lap_time, terminal_rate,
                                  max_progress_m, offtrack_seconds,
                                  worst_sectors),
            # PPO's generic keep-best line is tagged [best-ring]; give it a
            # Fable tag so overnight logs don't look like the wrong pipeline.
            "best_tag": f"[best-fable/{self.spec.stage}]",
        }
        return latest

    def _run_from(self, ppo: PPO, env: FableEnv, idx: int, speed: float,
                  steps: int) -> dict[str, Any]:
        obs = env.reset_at(idx, speed=speed)
        info: dict[str, Any] = {}
        speeds = []
        trace_sm = []
        lap_time = None
        clean_lap = False
        for i in range(max(1, steps)):
            nobs = ppo._norm(obs[None])[0]
            a = ppo.net.act_mean(ppo._t(nobs).unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(a)
            if i % 10 == 0:
                trace_sm.append(float(info.get("fable_progress_m", 0.0)))
            speeds.append(float(info.get("speed", 0.0)))
            if info.get("laps", 0.0) >= 1.0 and not info.get("fable_invalid"):
                lap_time = float(info.get("t", 0.0))
                clean_lap = True
                break
            if term or trunc:
                break
        reason = info.get("termination_reason")
        terminal_arc = float(env.trk.frame(env.veh.x, env.veh.y)["arc"])
        clean = bool(reason in (None, "time_limit")
                     and not info.get("fable_invalid", False))
        return {
            "clean": clean,
            "clean_lap": clean_lap,
            "lap_time": lap_time,
            "termination_reason": reason,
            "terminal_arc": terminal_arc,
            "progress_m": float(info.get("fable_progress_m", 0.0)),
            "laps": float(info.get("laps", 0.0)),
            "offtrack_seconds": float(info.get("fable_offtrack_seconds", 0.0)),
            "pace_ratio": float(info.get("fable_pace_ratio", 0.0)),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "trace": trace_sm,
        }

    @staticmethod
    def _clean_chain(rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        # A circuit wraps: a clean run spanning sectors 14,15,0,1 is one chain,
        # not two fragments split by the arbitrary start/finish array boundary.
        best = cur = 0
        for row in rows + rows:
            cur = cur + 1 if row["clean"] else 0
            best = min(len(rows), max(best, cur))
        return int(best)

    def _log_line(self, metric, clean_sectors, n, chain, pace, lap_time,
                  terminal_rate, max_progress_m, offtrack_seconds,
                  worst_sectors=None) -> str:
        lap = 0.0 if lap_time is None else float(lap_time)
        worst = ",".join(str(w["sector"]) for w in (worst_sectors or [])[:3])
        return (f"[eval-fable] stage={self.spec.stage} "
                f"clean={clean_sectors}/{n} chain={chain} pace={pace:.2f} "
                f"lap={lap:.2f} theo={self.theoretical:.1f} "
                f"progress={max_progress_m:.0f}m terminal={terminal_rate:.2f} "
                f"offtrack={offtrack_seconds:.2f} worst={worst or 'none'} "
                f"metric={metric:.3f}")

    def _write_manifest(self, ppo, latest: dict[str, Any],
                        checkpoint: str | None = None):
        old = {}
        if self.manifest_path.exists():
            try:
                old = json.loads(self.manifest_path.read_text())
            except Exception:
                old = {}
        stages = dict(old.get("stages") or {})
        stages[self.spec.stage] = latest
        # weak-sector memory: a decayed per-stage failure heatmap that survives
        # restarts (the trainer seeds start weights from it on the next run)
        heat = dict(old.get("sector_heat") or {})
        n = max(1, int(latest.get("sector_count") or 16))
        prev = list(heat.get(self.spec.stage) or [])
        if len(prev) != n:
            prev = [0.0] * n
        prev = [round(float(h) * 0.90, 4) for h in prev]
        for s in latest.get("fail_sectors") or []:
            prev[int(s) % n] += 1.0
        heat[self.spec.stage] = prev
        # The current run's stage-best (e.g. AlleyHouse_fast_best.pt). This can
        # be genuinely stronger than the promoted global champion whenever
        # promotion hasn't run yet — a killed/interrupted run, or a stage still
        # in flight. The dashboard and the watch default prefer this so they
        # surface what you're actually training, not a stale foundation champion.
        active_best = None
        if checkpoint:
            cand = PPO.best_path_for(checkpoint)
            if os.path.exists(cand):
                active_best = os.path.basename(cand)
        active_best = active_best or old.get("active_best_checkpoint")
        champion = champion_path_for(self.car)
        manifest = {
            "pipeline": "fable_five",
            "track": RING_TRACK,
            "car": getattr(ppo, "car", self.car),
            "drivetrain_version": self.drivetrain_version,
            "current_stage": self.spec.stage,
            "active_checkpoint": checkpoint or old.get("active_checkpoint"),
            "active_best_checkpoint": active_best,
            "best_checkpoint": champion if Path(champion).exists()
                               else old.get("best_checkpoint"),
            "latest_eval": latest,
            "stages": stages,
            "stage_gates": {
                "foundation_to_flow": ">=14/16 clean sectors, terminal_rate <= 0.125",
                "flow_to_finish": "16/16 clean chain, pace ratio >= 0.55",
                "finish_to_fast": "a clean lap banked (any time)",
                "fast_to_frontier": "clean lap <= 450 s (7:30)",
                "frontier": (f"beat {self.superhuman_lap:.2f}s "
                             f"({'919 Evo record' if self.superhuman_key.startswith('porsche_919') else 'Bellof 1983'}), "
                             f"then chase the theoretical {self.theoretical:.1f}s"),
            },
            "theoretical_lap": self.theoretical,
            "benchmarks": dict(HUMAN_BENCHMARKS),
            "envelope": {k: v for k, v in
                         getattr(self.track, "fable_envelope", {}).items()},
            "recommended_next_action": latest.get("recommendation"),
            "fable_reward_version": FABLE_REWARD_VERSION,
            "eval_protocol": EVAL_PROTOCOL_VERSION,
            "code_fingerprint": CODE_FINGERPRINT,
            "envelope_scale": float(self.spec.envelope_scale),
            "resumed_from": self.resumed_from,
            "pit": self.pit.state() if self.pit is not None else old.get("pit"),
            "sector_heat": heat,
            "hall_of_fame": (self.hof_summary() if self.hof_base
                             else old.get("hall_of_fame")),
            # the auto ladder stamps this between stages — every eval rewrite
            # must carry it forward or the dashboard loses the ladder state
            "auto": old.get("auto"),
        }
        _write_json_atomic(self.manifest_path, manifest)
        # Keep the standalone latest-eval NEXT TO its manifest. A smoke test
        # (validate_fable5.py) redirects manifest_path to /tmp; writing the
        # global relative path here would clobber the real repo-root
        # fable5_ring_eval_latest.json that the dashboard and humans read.
        _write_json_atomic(self.manifest_path.parent / latest_eval_for(self.car), latest)


# --------------------------------------------------------------------------- #
# Training entry points
# --------------------------------------------------------------------------- #
def checkpoint_for_stage(stage: str, out: str | None = None) -> str:
    if out:
        return out if out.endswith(".pt") else out + ".pt"
    return f"fable5_ring_{_norm_stage(stage)}.pt"


def default_resume_for(stage: str) -> str | None:
    order = {"flow": "foundation", "finish": "flow",
             "fast": "finish", "frontier": "fast"}
    prev = order.get(stage)
    if prev:
        cand = f"fable5_ring_{prev}_best.pt"
        if os.path.exists(cand):
            return cand
        if os.path.exists(FABLE_BEST):
            return FABLE_BEST
        return None
    # foundation: transplant the old ring pipeline's champion if present
    for cand in (FABLE_BEST, "ring_787b_best.pt"):
        if os.path.exists(cand):
            return cand
    return None


def _stage_heat(stage: str, car: str = FABLE_CAR) -> list[float] | None:
    """The persisted eval-failure heatmap for a stage (from the manifest)."""
    try:
        m = json.loads(Path(pipeline_manifest_for(car)).read_text())
    except Exception:
        return None
    h = (m.get("sector_heat") or {}).get(stage)
    if isinstance(h, list) and len(h) >= 4 and max(h, default=0.0) > 0.05:
        return [float(x) for x in h]
    return None


def _seed_weights_from_heat(heat: list[float], n: int) -> tuple[float, ...] | None:
    """Map the persisted eval failure heat onto the n start sectors, weighting
    the failing zone AND the approach just before it (start early, arrive at
    speed, rehearse the actual mistake)."""
    top = max(heat)
    if top <= 0:
        return None
    w = np.zeros(max(4, int(n)))
    nb = len(heat)
    for b, h in enumerate(heat):
        if h <= 0:
            continue
        for off, share in ((-1.0, 0.7), (0.0, 1.0)):
            f = ((b + 0.5 + off) % nb) / nb
            w[int(f * len(w)) % len(w)] += (h / top) * share
    if w.max() <= 0:
        return None
    w *= 3.0 / w.max()          # cap: at most +3 on the base weight of 1.0
    return tuple(float(x) for x in np.round(w, 3))


def _worker_count(n_envs: int, requested: int | None) -> int:
    """Choose a host-safe divisor and reserve two logical cores for PPO/OS."""
    logical = max(2, int(os.cpu_count() or 8))
    cap = max(1, logical - 2)
    want = min(int(requested or min(8, n_envs)), n_envs, cap)
    divisors = [d for d in range(1, want + 1) if n_envs % d == 0]
    return max(divisors or [1])


def _eval_every_for(spec: FableSpec, n_envs: int) -> int:
    """Keep collapse detection roughly transition-based as population grows."""
    if n_envs <= 16:
        return int(spec.eval_every)
    return max(8, int(round(spec.eval_every * 16.0 / n_envs)))


def _configure_ppo(spec: FableSpec, *, car: str = FABLE_CAR, workers=None,
                   pop=None, anneal=None, lr=None, patience=None,
                   max_restarts=None) -> PPOSpec:
    cfg = PPOSpec()
    cfg.episode_seconds = spec.episode_seconds
    cfg.random_start = True
    cfg.sensor_lookahead_distances = spec.lookahead_distances
    cfg.sensor_pace_block = True
    cfg.sensor_pace_distances = spec.pace_distances
    # hybrid cars (919 Evo) train on fable-v2: SOC + MGU power appended to the
    # obs, flowing identically into the PPO and GA learners via this cfg
    cfg.sensor_hybrid_block = car_has_hybrid_telemetry(car)
    cfg.action_bias = (0.0, 0.6, 0.0)   # a[2] = gear offset, neutral 0 (NOT
                                        # the -1 "handbrake off" default)
    cfg.track_profile = FABLE_PROFILE
    cfg.n_envs = int(pop or 8)
    cfg.n_workers = _worker_count(cfg.n_envs, workers)
    st = spec.stage
    stage_i = STAGES.index(st) if st in STAGES else 0
    cfg.anneal = bool(anneal if anneal is not None else stage_i >= 2)
    cfg.ent_coef = (0.008, 0.005, 0.004, 0.0035, 0.0025)[stage_i]
    cfg.init_log_std = (-0.30, -0.45, -0.55, -0.60, -0.70)[stage_i]
    # Gear is a rounded -2..+2 offset, not a smooth actuator. Sharing the
    # steer/throttle std let it sit at std=1.0 and hunt multiple gears randomly.
    gear_cap = (-0.90, -1.00, -1.15, -1.30, -1.45)[stage_i]
    cfg.action_log_std_max = (0.0, 0.0, gear_cap)
    cfg.lr = float(lr) if lr else (3e-4, 3e-4, 2.5e-4, 1.8e-4, 1.2e-4)[stage_i]
    # trust region: loose while the policy is still finding the track (big
    # exploratory updates are the point there), tight once it is refining a
    # knife-edge lap — fast/frontier collapses were single unguarded updates
    # turning a banked 565s-lap brain into one that spins at 20% of the lap.
    cfg.target_kl = (0.08, 0.06, 0.04, 0.02, 0.015)[stage_i]
    cfg.vf_clip = (6.0, 8.0, 10.0, 10.0, 10.0)[stage_i]
    # frontier runs in SHORT SEGMENTS under the auto ladder (the segment loop
    # reseeds/adapts between them), so its in-run patience is tighter than the
    # old 1400 — a dead segment ends early and its budget rolls forward.
    if patience is not None:
        cfg.specialist_patience = int(patience)
    else:
        base_patience = (500, 600, 700, 900, 700)[stage_i]
        transition_scaled = int(math.ceil(base_patience * 8.0 / cfg.n_envs))
        cfg.specialist_patience = max(
            3 * _eval_every_for(spec, cfg.n_envs), transition_scaled)
    cfg.max_restarts = (int(max_restarts) if max_restarts is not None
                        else (4, 4, 5, 6, 3)[stage_i])
    return cfg


def _assert_fable_resume_car_identity(payload: dict, path: str,
                                      run_car: str = FABLE_CAR) -> None:
    """Prevent cross-car policies from entering a run: a 919 run may only
    resume 919 checkpoints, a 787B run only 787B checkpoints."""
    checkpoint_car = payload.get("car")
    if checkpoint_car != run_car:
        raise ValueError(
            f"refusing Fable resume from {path!r}: checkpoint car "
            f"{checkpoint_car!r} does not exactly match this run's car "
            f"{run_car!r}. Cross-car resumes are never valid."
        )


def load_or_transplant(ppo: PPO, path: str, stage: str) -> str:
    """Resume a matching fable checkpoint normally; TRANSPLANT anything older:
    hills-v1 obs (58+2) -> fable-v1 (58+pace+2), and/or 2-action (steer, long)
    -> 3-action (…+ gear offset). Transplanted extras start neutral: pace
    input columns blank, gear-offset head at bias 0 = "trust the RaceBox"."""
    d = load_torch_checkpoint_safe(path)
    validate_checkpoint_payload(d)
    run_car = getattr(ppo, "car", FABLE_CAR)
    _assert_fable_resume_car_identity(d, path, run_car)
    old_obs = int(d.get("obs_dim", 0))
    old_act = int(d.get("act_dim", 0))
    if old_obs == ppo.obs_dim and old_act == ppo.act_dim:
        meta = ppo.load_state(path)
        expected_dv = drivetrain_for(run_car)
        stamped_dv = (meta.get("fable_drivetrain_version")
                      or d.get("fable_drivetrain_version"))
        # Legacy accommodation: the first 919 lineage was mis-stamped with the
        # 787B constant while genuinely training porsche_919evo (car matches).
        # Same lineage, wrong label — do NOT reset its gear head.
        legacy_misstamp = (stamped_dv == DRIVETRAIN_VERSION
                           and d.get("car") == run_car and run_car != FABLE_CAR)
        drivetrain_migration = (stamped_dv != expected_dv
                                and not legacy_misstamp)
        if drivetrain_migration:
            # Preserve driving intelligence but discard six-speed semantics.
            with torch.no_grad():
                ppo.net.mean.weight.data[2].zero_()
                ppo.net.mean.bias.data[2] = 0.0
                ppo.net.log_std.data[2] = min(float(ppo.cfg.init_log_std), -0.90)
                ppo.net.clamp_log_std_()
            gear_obs_i = int(SensorSpec().n_beams) + 11
            if gear_obs_i < len(ppo.norm.mean):
                ppo.norm.mean[gear_obs_i] = 0.0
                ppo.norm.var[gear_obs_i] = 1.0
            ppo.updates = 0
            ppo.resumed_metric = -1e9
        if meta.get("fable_stage") != stage or drivetrain_migration:
            # Cross-stage is a policy/normalizer warm start, NOT an optimiser or
            # schedule continuation. Importing Adam moments and the source
            # stage's broad noise defeated the tighter finish/fast tuning.
            ppo.resumed_metric = -1e9
            ppo.opt = torch.optim.Adam(ppo.net.parameters(), lr=ppo.cfg.lr)
            with torch.no_grad():
                ppo.net.log_std.data.copy_(torch.minimum(
                    ppo.net.log_std.data,
                    torch.full_like(ppo.net.log_std.data, ppo.cfg.init_log_std)))
                ppo.net.clamp_log_std_()
            ppo._resume_lr_cap = None
            ppo._resume_ent_cap = None
            ppo._lr_safety_scale = 1.0
        if drivetrain_migration:
            return (f"MIGRATED {path} to {DRIVETRAIN_VERSION}; retained steer/long, "
                    "reset gear head + gear normalization, recertification required")
        return f"resumed {path} (updates {ppo.updates})"

    old_sdim = int(d.get("sdim", max(old_obs - 2, 0)))
    if (list(d.get("hidden", [])) != list(ppo.cfg.hidden)
            or old_sdim > ppo.sdim or old_act > ppo.act_dim or old_act < 2):
        raise ValueError(
            f"cannot transplant '{path}': arch (obs {old_obs}, act {old_act}, "
            f"hidden {d.get('hidden')}) is incompatible with fable "
            f"(obs {ppo.obs_dim}, act {ppo.act_dim}, hidden {list(ppo.cfg.hidden)})")
    sd = d["state_dict"]
    with torch.no_grad():
        for name, p in ppo.net.named_parameters():
            if name == "log_std":
                # reopen exploration on every dim: the gearbox change alters
                # the dynamics under the old policy too
                p.data.fill_(float(ppo.cfg.init_log_std))
                continue
            if name == "trunk.0.weight":
                p[:, :old_sdim] = sd[name][:, :old_sdim]
                p[:, ppo.sdim:ppo.sdim + 2] = sd[name][:, old_sdim:old_sdim + 2]
                continue
            if name == "mean.weight":
                p[:old_act].copy_(sd[name][:old_act])   # new rows keep tiny init
                continue
            if name == "mean.bias":
                p[:old_act].copy_(sd[name][:old_act])
                if old_act < ppo.act_dim:
                    p[old_act:] = 0.0                   # gear offset: neutral
                continue
            if name in sd and sd[name].shape == p.shape:
                p.copy_(sd[name])
        ppo.net.clamp_log_std_()
    ppo.norm.mean[:old_sdim] = np.asarray(d["norm_mean"])[:old_sdim]
    ppo.norm.var[:old_sdim] = np.asarray(d["norm_var"])[:old_sdim]
    ppo.norm.count = float(d.get("norm_count", 1.0))
    ppo.norm_update_mode = ("raw-v2" if d.get("norm_schema") == "raw-v2"
                            else "legacy-frozen-v1")
    ppo.opt = torch.optim.Adam(ppo.net.parameters(), lr=ppo.cfg.lr)
    ppo.resumed_metric = -1e9              # different arch -> fresh best floor
    return (f"TRANSPLANTED {path} ({old_sdim} sensors, {old_act} actions) into "
            f"{obs_layout_for(run_car)} ({ppo.sdim} sensors, {ppo.act_dim} "
            f"actions); new sensor inputs blank, gear head neutral")


def _champion_key(meta: dict) -> tuple:
    """Ordering for 'who is the fastest driver': a banked clean lap beats no
    lap; among laps, faster wins; otherwise robust distance/cleanliness beats a
    weak later-stage label."""
    ev = (meta or {}).get("fable_eval") or {}
    lap = ev.get("lap_time")
    stage = (meta or {}).get("fable_stage")
    rank = STAGES.index(stage) if stage in STAGES else -1
    tr = ev.get("terminal_rate")
    if lap:
        return (1, -float(lap), int(ev.get("clean_sectors") or 0),
                -float(1.0 if tr is None else tr),
                float(ev.get("pace_ratio") or 0.0), rank,
                float(ev.get("metric", -1e9) or -1e9))
    return (0, float(ev.get("progress_frac") or 0.0),
            int(ev.get("clean_chain") or 0), int(ev.get("clean_sectors") or 0),
            -float(1.0 if tr is None else tr),
            float(ev.get("pace_ratio") or 0.0), rank,
            float(ev.get("metric", -1e9) or -1e9))


def _best_eval_of(path: str) -> tuple[bool, str | None, dict]:
    """(exists, recommendation, fable_eval) of a checkpoint's stored eval."""
    if not os.path.exists(path):
        return False, None, {}
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False, None, {}
    ev = d.get("fable_eval") or {}
    return True, ev.get("recommendation"), ev


def _checkpoint_protocol_current(path: str, expected_stage: str | None = None,
                                 car: str = FABLE_CAR) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    try:
        validate_checkpoint_payload(d, require_hash=True)
    except (TypeError, ValueError):
        return False
    ev_stage = (d.get("fable_eval") or {}).get("stage")
    evaluated_hash = (d.get("fable_eval") or {}).get(
        "evaluated_policy_sha256")
    stage = d.get("fable_stage")
    expected_dv = drivetrain_for(car)
    stamped_dv = d.get("fable_drivetrain_version")
    # Legacy accommodation: the first 919 lineage (2026-07-16) was mis-stamped
    # with the 787B constant while genuinely training porsche_919evo. Those
    # checkpoints carry the correct `car`, so accept the old stamp for them.
    dv_ok = (stamped_dv == expected_dv
             or (stamped_dv == DRIVETRAIN_VERSION and d.get("car") == car
                 and car != FABLE_CAR))
    return (stage in STAGES and ev_stage == stage
            and (expected_stage is None or stage == expected_stage)
            and evaluated_hash == d.get("policy_sha256")
            and d.get("fable_eval_protocol") == EVAL_PROTOCOL_VERSION
            and d.get("fable_code_fingerprint") == CODE_FINGERPRINT
            and dv_ok
            and d.get("obs_layout") == obs_layout_for(car)
            and int(d.get("act_dim", 0)) == 3
            and d.get("car") == car
            and d.get("track") == RING_TRACK)


def _promote_champion(best_path: str, stage: str, car: str = FABLE_CAR):
    """Copy a stage best to the CAR'S global champion (fable5_ring_best.pt for
    the 787B, fable5_919_ring_best.pt for the 919, …) ONLY if it beats the
    current champion — a weak rerun can never clobber a banked fast lap, and
    one edition can never overwrite another's champion."""
    if not os.path.exists(best_path):
        return
    import fcntl
    champion = champion_path_for(car)
    lock_path = champion + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            new = torch.load(best_path, map_location="cpu", weights_only=False)
            validate_checkpoint_payload(new, require_hash=True)
            if not _checkpoint_protocol_current(best_path, stage, car):
                raise ValueError("stale or mismatched Fable protocol")
        except Exception as e:
            print(f"[fable5] promotion skipped (invalid {best_path}: {e})", flush=True)
            return
        old = None
        if os.path.exists(champion):
            try:
                old = torch.load(champion, map_location="cpu", weights_only=False)
                validate_checkpoint_payload(old)
            except Exception:
                old = None
        old_certified = bool(old is not None
                             and _checkpoint_protocol_current(champion, car=car))
        if old is not None and not old_certified:
            print("[fable5] existing global champion is legacy/stale evidence; "
                  "archiving it but requiring current-protocol recertification",
                  flush=True)
        if old_certified and _champion_key(new) < _champion_key(old):
            oev = (old.get("fable_eval") or {})
            print(f"[fable5] {stage} best NOT promoted — champion is stronger "
                  f"(champion lap {oev.get('lap_time')}, metric "
                  f"{oev.get('metric', 0):.3f})", flush=True)
            return

        # Preserve immutable rollback lineage before atomically changing the
        # small global pointer slot.
        archive = Path("runtime") / "fable5" / "champions"
        archive.mkdir(parents=True, exist_ok=True)
        if old is not None:
            old_sha = str(old.get("policy_sha256") or "legacy")[:12]
            prior = archive / f"{time.time_ns()}_{old_sha}.pt"
            shutil.copy2(champion, prior)
        fd, tmp = tempfile.mkstemp(prefix=Path(champion).name + ".candidate.",
                                   dir=Path(champion).parent or Path("."))
        os.close(fd)
        try:
            shutil.copyfile(best_path, tmp)
            check = torch.load(tmp, map_location="cpu", weights_only=False)
            if _champion_key(check) != _champion_key(new):
                raise ValueError("candidate changed during atomic promotion")
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, champion)
            try:
                dfd = os.open(str(Path(champion).resolve().parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        ev = new.get("fable_eval") or {}
        lap = ev.get("lap_time")
        print(f"[fable5] {stage} best promoted -> {champion} "
              f"(lap {'%.2f' % lap if lap else 'none'}, "
              f"metric {ev.get('metric', 0):.3f})", flush=True)


def _train_one_stage(stage: str, iterations: int, *, car=FABLE_CAR, resume=None,
                     out=None, live=False, workers=None, pop=None, anneal=None,
                     lr=None, patience=None, max_restarts=None,
                     envelope_scale=None, promote_best=True,
                     open_best_floor=False, stop_on_gate=False,
                     on_stage_ready=None) -> dict[str, Any]:
    spec = stage_defaults(stage)
    if envelope_scale is not None:
        spec.envelope_scale = float(np.clip(float(envelope_scale), 0.4, 1.3))
        # a target pushed past the raw envelope must not fight the overspeed
        # penalty: keep the penalty margin above the reward target
        spec.reward.overspeed_margin = max(spec.reward.overspeed_margin,
                                           spec.envelope_scale + 0.08)
    trk = attach_envelope(named_track(RING_TRACK), car)
    theo = trk.fable_envelope["lap_time"]
    sh_bar, _sh_key = superhuman_lap_for(trk.fable_envelope["params"].get("drivetrain_version"))
    print(f"[fable5] envelope ready: theoretical lap {theo:.1f}s "
          f"({theo/60:.0f}:{theo%60:05.2f}) | superhuman bar "
          f"{sh_bar:.2f}s | stage={spec.stage} "
          f"scale={spec.envelope_scale:.2f}", flush=True)
    cfg = _configure_ppo(spec, car=car, workers=workers, pop=pop, anneal=anneal,
                         lr=lr, patience=patience, max_restarts=max_restarts)
    if workers is not None and int(workers) != cfg.n_workers:
        print(f"[fable5] worker request {workers} capped to {cfg.n_workers} "
              f"for {os.cpu_count() or '?'} logical cores and {cfg.n_envs} envs",
              flush=True)
    ckpt = checkpoint_for_stage(stage, out)
    resume_path = resume or default_resume_for(stage)
    if resume_path and not os.path.exists(resume_path):
        print(f"[fable5] resume not found: {resume_path}; starting fresh",
              flush=True)
        resume_path = None
    # weak-sector memory: seed the exploring-start weights from the failure
    # heat this stage banked in earlier runs (rehearse where it actually dies)
    heat = _stage_heat(spec.stage)
    if heat:
        bias = _seed_weights_from_heat(heat, spec.n_start_sectors)
        if bias:
            spec.sector_seed_bias = bias
            hot = sorted(range(len(heat)), key=lambda b: -heat[b])[:3]
            print(f"[fable5] weak-sector memory: start bias armed around eval "
                  f"sectors {hot}", flush=True)
    pit = PitWall(stage, run_name=os.path.basename(ckpt)[:-3])
    evaluator = FableEvaluator(spec, trk, Path(pipeline_manifest_for(car)), car=car,
                               stop_on_gate=stop_on_gate, pit=pit,
                               resumed_from=resume_path, hof_base=ckpt[:-3],
                               checkpoint=ckpt)
    pit.hof_provider = lambda: evaluator.hof
    try:
        old_manifest = json.loads(Path(pipeline_manifest_for(car)).read_text())
    except Exception:
        old_manifest = {}
    if (old_manifest.get("eval_protocol") == EVAL_PROTOCOL_VERSION
            and pit.restore_state(old_manifest.get("pit"))):
        print(f"[fable5] restored pit-wall state for {pit.run_name}: "
              f"best {pit.best_metric:.3f}, rollbacks {pit.rollbacks_since_best}, "
              f"reseeds {pit.reseeds_since_best}", flush=True)
    if (resume_path and _checkpoint_protocol_current(resume_path, car=car)
            and pit.seed_from_resume(resume_path)):
        lap_note = (f"lap {pit.best_lap:.2f}s, " if pit.best_lap else "")
        print(f"[fable5] pit wall armed from same-stage resume: {lap_note}"
              f"bar {pit.best_metric:.3f}", flush=True)

    if live:
        from .app_ppo import run as live_run
        live_run(car=car, iterations=iterations, checkpoint=ckpt, mode="race",
                 resume=resume_path, fixed_track=trk, track_name=RING_TRACK,
                 ppo_cfg=cfg, reward=spec.reward,
                 env_cls_override=FableEnv,
                 env_kwargs={"fable_spec": spec},
                 eval_callback=evaluator,
                 extra_metadata=evaluator.metadata)
        latest = evaluator.latest
    else:
        import signal
        ppo = PPO(mode="race", car=car, ppo=cfg, reward=spec.reward,
                  fixed_track=trk, track_name=RING_TRACK,
                  env_cls_override=FableEnv,
                  env_kwargs={"fable_spec": spec},
                  eval_callback=evaluator,
                  extra_metadata=evaluator.metadata)
        stage_signal_handlers = {}

        def _stage_signal(signum, frame):
            print(f"\n[interrupted] saving current {stage} checkpoint -> {ckpt}",
                  flush=True)
            try:
                if all(bool(torch.isfinite(p).all())
                       for p in ppo.net.parameters()):
                    ppo.save(ckpt)
            finally:
                try:
                    ppo.vec.close()
                except Exception:
                    pass
            raise SystemExit(0)

        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                stage_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _stage_signal)
        except (ValueError, OSError):
            stage_signal_handlers = {}
        if resume_path:
            msg = load_or_transplant(ppo, resume_path, stage)
            if open_best_floor:
                ppo.resumed_metric = -1e9
            print(f"[fable5] {msg}", flush=True)
        print(f"Fable Five [{stage}]: {cfg.n_envs} envs x {cfg.rollout} steps, "
              f"episode cap {cfg.episode_seconds:.0f}s -> {ckpt}", flush=True)
        save_every = max(1, min(_eval_every_for(spec, cfg.n_envs),
                                int(iterations)))
        print(f"[fable5] safety cadence: full eval every {save_every} updates "
              f"({save_every * cfg.n_envs * cfg.rollout:,} transitions max)",
              flush=True)
        u0 = ppo.updates
        best_path = PPO.best_path_for(ckpt)
        ppo._best_path = best_path
        # Old protocol evidence is not a valid selection floor. Re-evaluate the
        # incoming brain at update zero under the corrected destination stage,
        # before even one optimiser step can erase its lap.
        ppo.ignore_existing_best = (open_best_floor
                                    or not _checkpoint_protocol_current(best_path,
                                                                        stage, car))
        if ppo.ignore_existing_best:
            # A same-stage legacy resume may carry its old metric through
            # load_state(). That evidence was produced by a different eval
            # protocol, so it cannot remain an invisible selection floor.
            ppo.resumed_metric = -1e9
        ppo.init_best_metric(best_path)
        print(f"[fable5] baseline destination-stage eval before training", flush=True)
        ppo.eval_and_save(ckpt, best_path)
        ppo.ignore_existing_best = False
        if callable(on_stage_ready):
            on_stage_ready(ckpt, u0)
        if not (stop_on_gate and getattr(ppo, "request_stop", False)):
            ppo.train(iterations=iterations, checkpoint=ckpt, log_every=5,
                      save_every=save_every)
        else:
            print(f"  [gate] incoming policy already meets {stage}; no update "
                  f"needed (peak banked -> {best_path})", flush=True)
        # bank the tail: the final weights may be better than the last banked
        # eval (up to eval_every-1 iterations are otherwise never scored).
        # Skip only when the gate already stopped the run — that peak IS banked.
        if (not evaluator.latest
                or (not getattr(ppo, "request_stop", False)
                    and int(getattr(ppo, "_last_eval_updates", -1))
                    != int(ppo.updates))):
            ppo.eval_and_save(ckpt, PPO.best_path_for(ckpt))
        latest = dict(evaluator.latest)
        latest["iters_used"] = int(ppo.updates - u0)
        # PPO.train() closes in its own finally; a baseline that already gates
        # never enters train(), so close here as an idempotent final barrier.
        try:
            ppo.vec.close()
        except Exception:
            pass
        for sig, handler in stage_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    best_path = PPO.best_path_for(ckpt)
    if promote_best:
        candidates = [p for p in (best_path, f"{ckpt[:-3]}_hof_lap.pt")
                      if _checkpoint_protocol_current(p, stage, car)]
        if candidates:
            def key(path):
                try:
                    return _champion_key(torch.load(
                        path, map_location="cpu", weights_only=False))
                except Exception:
                    return (-1,)
            _promote_champion(max(candidates, key=key), stage, car)
    elif os.path.exists(best_path):
        print(f"[fable5] {stage} best kept local at {best_path} "
              f"(promotion disabled)", flush=True)
    # Every successful eval already writes the manifest in after_eval(). Do not
    # write the same final eval twice: failure heat would otherwise be added a
    # second time and the prior heat decayed twice.
    return latest


def _update_manifest_auto(auto_state: dict, car: str = FABLE_CAR):
    """Stamp the auto-ladder progress into the pipeline manifest."""
    manifest = pipeline_manifest_for(car)
    try:
        m = json.loads(Path(manifest).read_text()) \
            if Path(manifest).exists() else {}
    except Exception:
        m = {}
    m["auto"] = auto_state
    _write_json_atomic(Path(manifest), m)


def run_fable(iterations: int, stage: str = "foundation", **kwargs) -> dict[str, Any]:
    requested_car = kwargs.get("car") or FABLE_CAR
    stage = _norm_stage(stage)
    iterations = max(1, int(iterations))
    if stage != "auto":
        return _train_one_stage(stage, iterations, **kwargs)
    if kwargs.get("live"):
        raise ValueError("--fable-stage auto cannot run with --live: the live "
                         "window does not own autonomous stage transitions")

    # ---------------- the overnight auto ladder ---------------- #
    # * --out NAME is a RUN PREFIX: checkpoints <prefix>_<stage>.pt, so
    #   separate auto runs never clobber each other's files;
    # * each stage warm-starts from the previous stage's ★best (this prefix);
    # * gates are judged on the BANKED BEST (not the possibly-collapsed final
    #   eval), and a stage stops early the moment its gate is met — leftover
    #   budget rolls into frontier, the lap-time compressor;
    # * a stage that MISSES its gate retries once at an eased envelope scale,
    #   then SOFT-ADVANCES — the ladder never idles the rest of the night;
    # * EXCEPT finish: it runs adaptive segments starting at the scale FLOW
    #   banked ("finish the lap first, then go fast"), keeps the budget while
    #   the lap is missing (fast/frontier need a finisher), and only
    #   soft-advances on the quality bar (>=98% progress, >=15/16 clean,
    #   terminal <= 0.125);
    # * and EXCEPT fast: adaptive segments seeded from the scale the incoming
    #   brain BANKED (never the 0.96 default — the 0.90 -> 0.96 cliff is what
    #   froze KLGUARD), climbing only on mastery proven at the current scale;
    #   once its share is spent it soft-advances to frontier;
    # * frontier runs in SEGMENTS with an ADAPTIVE scale: mastery (clean lap,
    #   high pace, low terminal rate) pushes the reward target up to 1.15x the
    #   centerline envelope (the racing line is faster than the centerline —
    #   that margin is where superhuman lives), losing the lap eases it back;
    #   each segment resumes from the banked best = built-in rollback;
    # * already-gated stages are SKIPPED on rerun (crash-resumable), and the
    #   adaptive scale RESUMES from the value stored in the stage best;
    # * promotion to fable5_ring_best.pt is metric-compared, so a weak run
    #   can never overwrite the champion.
    prefix = os.path.basename((kwargs.pop("out", None) or "fable5_ring").strip()
                              or "fable5_ring")
    if prefix.endswith(".pt"):
        prefix = prefix[:-3]
    # --resume seeds the FIRST trained stage of the ladder (loudly); after
    # that the ladder chains stage bests as usual. Never silently ignored.
    user_resume = kwargs.pop("resume", None)
    if user_resume and not os.path.exists(user_resume):
        print(f"[fable5] auto: --resume '{user_resume}' not found — "
              f"falling back to the stage chain", flush=True)
        user_resume = None
    total = iterations
    spent = 0
    history: list[dict] = []
    # Crash-resume the advertised budget as well as the stage checkpoints.
    # Older code reset spent/history to zero on every process restart, so a
    # supervisor could unknowingly exceed the requested transition budget.
    car = kwargs.get("car", "mazda787b")
    manifest = pipeline_manifest_for(car)
    try:
        prior_auto = (json.loads(Path(manifest).read_text()).get("auto")
                      or {})
    except Exception:
        prior_auto = {}
    if (prior_auto.get("prefix") == prefix
            and prior_auto.get("eval_protocol") == EVAL_PROTOCOL_VERSION
            and prior_auto.get("code_fingerprint") == CODE_FINGERPRINT
            and int(prior_auto.get("total") or 0) == total):
        spent = max(0, min(total, int(prior_auto.get("spent") or 0)))
        history = list(prior_auto.get("history") or [])
        inflight = prior_auto.get("inflight") or {}
        if inflight:
            delta = max(0, _checkpoint_updates(inflight.get("checkpoint"))
                        - int(inflight.get("start_updates") or 0))
            recovered_spent = min(total,
                                  int(inflight.get("spent_before") or spent) + delta)
            if recovered_spent == total and total > 1:
                # Reserve one final update slot so the restarted ladder enters
                # the interrupted stage, runs its destination baseline, and has
                # a chance to promote/bank the tail. Otherwise a crash after the
                # last checkpoint could print completion without finalization.
                recovered_spent = total - 1
            if recovered_spent > spent:
                spent = recovered_spent
                history.append({"stage": inflight.get("stage", "unknown"),
                                "iters": delta, "recovered_after_crash": True,
                                "segment": inflight.get("segment"),
                                "attempt": inflight.get("attempt")})
            print(f"[fable5] auto: restored budget {spent}/{total} from "
                  f"the prior run state", flush=True)
    latest: dict[str, Any] = {}
    print(f"[fable5] AUTO ladder '{prefix}': {total} iterations across "
          f"{' -> '.join(STAGES)}", flush=True)

    def _chain_resume(idx: int) -> str | None:
        # Prefer this stage's own bank only when it is certified under the
        # CURRENT protocol. Otherwise rank every compatible ancestor/champion
        # by actual driving evidence. This prevents a stale lapless finish file
        # from outranking a clean global champion merely because its filename
        # appears first after a code upgrade.
        own = f"{prefix}_{STAGES[idx]}_best.pt"
        if _checkpoint_protocol_current(own, STAGES[idx], requested_car):
            return own
        if idx > 0:
            previous = f"{prefix}_{STAGES[idx - 1]}_best.pt"
            if _checkpoint_protocol_current(previous, STAGES[idx - 1],
                                            requested_car):
                return previous
        candidates = ([f"{prefix}_{STAGES[j]}_best.pt"
                       for j in range(idx - 1, -1, -1)]
                      + ([FABLE_BEST, "ring_787b_best.pt"]
                         if requested_car == FABLE_CAR
                         else [champion_path_for(requested_car)]))
        ranked: list[tuple[tuple, str]] = []
        for cand in candidates:
            if not os.path.exists(cand):
                continue
            try:
                meta = torch.load(cand, map_location="cpu", weights_only=False)
                if (int(meta.get("obs_dim", 0)) > 0
                        and int(meta.get("act_dim", 0)) in (2, 3)):
                    ranked.append((_champion_key(meta), cand))
            except Exception:
                continue
        return max(ranked, default=((), None))[1]

    scale_override = kwargs.pop("envelope_scale", None)   # --fable-scale seeds
                                                          # the adaptive scale
    def _stamp(extra=None):
        state = {"prefix": prefix, "total": total, "spent": spent,
                 "history": history, "eval_protocol": EVAL_PROTOCOL_VERSION,
                 "code_fingerprint": CODE_FINGERPRINT}
        if extra:
            state.update(extra)
        _update_manifest_auto(state, car=requested_car)

    def _inflight(stage_name: str, **detail):
        spent_before = spent
        def ready(checkpoint, start_updates):
            _stamp({"inflight": {"stage": stage_name,
                                  "checkpoint": checkpoint,
                                  "start_updates": int(start_updates),
                                  "spent_before": int(spent_before),
                                  **detail}})
        return ready

    for i, st in enumerate(STAGES):
        gate = GATE_ADVANCE.get(st)
        ckpt = f"{prefix}_{st}.pt"
        best = ckpt[:-3] + "_best.pt"

        # crash-resume: a stage whose banked best already meets its gate is done
        exists, rec, ev = _best_eval_of(best)
        if (exists and gate and rec == gate
                and _checkpoint_protocol_current(best, st, requested_car)):
            print(f"[fable5] auto: {st} already gated ({best}) — skipping",
                  flush=True)
            history.append({"stage": st, "skipped": True, "iters": 0,
                            "gated": True, "metric": ev.get("metric")})
            _stamp()
            continue

        remaining = total - spent
        if remaining <= 0:
            print(f"[fable5] auto: budget exhausted before {st}", flush=True)
            break

        if st == "finish":
            # ---- finish: adaptive segments; the LAP is the gate ---- #
            # fast/frontier train SPEED on top of a finisher — training them
            # on a non-finisher wastes the night. So while the lap is missing
            # finish keeps the budget (eating fast/frontier's share if it
            # must), easing the envelope target toward the floor. The only
            # way past without a lap is the QUALITY BAR (a near-lap so strong
            # it's worth building speed on). The scale starts where FLOW
            # actually banked its brain — finish the lap first, THEN go fast.
            share = min(max(1, int(round(total * AUTO_WEIGHTS[st]))), remaining)
            default_scale = stage_defaults(st).envelope_scale
            flow_scale = _stored_scale(f"{prefix}_flow_best.pt")
            scale = (float(scale_override) if scale_override is not None
                     else (_stored_scale(best)
                           or min(default_scale, flow_scale or default_scale)))
            scale_override = None
            spent0 = spent
            segment = 0
            gated = bar_met = over_share = False
            ev, rec, prog = {}, None, 0.0
            while spent < total:
                segment += 1
                seg_iters = min(total - spent, max(150, share // 3))
                seed_from = user_resume or _chain_resume(i)
                user_resume = None
                print(f"[fable5] auto: finish segment {segment} for up to "
                      f"{seg_iters} iters at scale {scale:.2f} "
                      f"({total - spent} remaining of {total}) — resuming "
                      f"from {seed_from or 'scratch'}", flush=True)
                latest = _train_one_stage(st, seg_iters, out=ckpt,
                                          resume=seed_from, stop_on_gate=True,
                                          envelope_scale=scale,
                                          on_stage_ready=_inflight(
                                              st, segment=segment),
                                          **dict(kwargs))
                used_raw = latest.get("iters_used")
                used = seg_iters if used_raw is None else int(used_raw)
                spent += used
                exists, rec, ev = _best_eval_of(best)
                gated = bool(exists and rec == gate)
                bar_met = _finish_bar_met(ev)
                prog = float(ev.get("progress_frac") or 0.0)
                new_scale, why = scale, f"progress {prog:.2f} — holding"
                if not gated and not bar_met and prog < 0.93:
                    new_scale = max(FINISH_SCALE_FLOOR,
                                    round(scale - FINISH_SCALE_DOWN, 3))
                    why = (f"best progress {prog:.2f} — easing the target"
                           if new_scale < scale else
                           f"holding at the {FINISH_SCALE_FLOOR:.2f} floor")
                history.append({"stage": st, "skipped": False, "iters": used,
                                "segment": segment, "scale": scale,
                                "next_scale": new_scale, "gated": gated,
                                "soft_advanced": bool(bar_met and not gated),
                                "metric": ev.get("metric"),
                                "lap_time": ev.get("lap_time"),
                                "progress_frac": prog,
                                "recommendation": rec})
                _stamp()
                scale = new_scale
                if gated or bar_met or used == 0:
                    break
                if not over_share and spent - spent0 >= share:
                    over_share = True
                    print(f"[fable5] auto: finish is HOLDING THE LADDER — no "
                          f"lap yet, so it keeps training on the fast/frontier "
                          f"share (a lap is required to climb; scale eases "
                          f"toward {FINISH_SCALE_FLOOR:.2f})", flush=True)
                else:
                    print(f"[fable5] auto: finish segment {segment} done — "
                          f"{why}", flush=True)
            if gated:
                lap = ev.get("lap_time") or 0.0
                print(f"  [gate] finish banked a clean lap ({lap:.2f}s) — "
                      f"advancing to fast", flush=True)
                continue
            if bar_met:
                print(f"[fable5] auto: finish SOFT-ADVANCES on the quality "
                      f"bar (progress {prog:.2f}, clean "
                      f"{ev.get('clean_sectors')}/16, terminal "
                      f"{float(ev.get('terminal_rate') or 0.0):.2f}) — a "
                      f"near-lap this strong is worth building speed on",
                      flush=True)
                continue
            print(f"[fable5] auto: budget exhausted with no finish lap — the "
                  f"ladder stops here (fast/frontier need a finisher; a rerun "
                  f"resumes finish from its banked best)", flush=True)
            break

        if st == "fast":
            # ---- fast: adaptive segments; the car EARNS 0.96 ---- #
            # Seeded from the scale the incoming brain actually BANKED (its
            # own resume first, else what finish banked its lap at), never
            # from the 0.96 stage default — the 0.90 -> 0.96 cliff is what
            # broke KLGUARD (see FAST_SCALE_MIN above). Climb +0.03 only on
            # mastery proven AT the current scale, ease -0.04 when the lap is
            # lost or a whole segment banks nothing. Fresh segments also
            # re-open the lr/entropy the pit wall's calm ladder closed, and
            # each one resumes from the banked best = built-in rollback.
            # Unlike finish, fast never holds the ladder: once its share is
            # spent it soft-advances (frontier's own ladder keeps adapting).
            share = min(max(1, int(round(total * AUTO_WEIGHTS[st]))), remaining)
            target = stage_defaults(st).envelope_scale
            scale = (float(scale_override) if scale_override is not None
                     else (_stored_scale(best)
                           or _stored_scale(f"{prefix}_finish_best.pt")
                           or FAST_SCALE_MIN))
            scale = min(max(float(scale), FAST_SCALE_MIN), target)
            scale_override = None
            spent0 = spent
            segment = 0
            gated = False
            ev, rec = {}, None
            while spent < total:
                segment += 1
                seg_iters = min(total - spent, max(150, share // 3))
                seed_from = user_resume or _chain_resume(i)
                user_resume = None
                print(f"[fable5] auto: fast segment {segment} for up to "
                      f"{seg_iters} iters at scale {scale:.2f} "
                      f"({total - spent} remaining of {total}) — resuming "
                      f"from {seed_from or 'scratch'}", flush=True)
                _, _, ev0 = _best_eval_of(best)
                m_before = float((ev0 or {}).get("metric") or -1e18)
                latest = _train_one_stage(st, seg_iters, out=ckpt,
                                          resume=seed_from, stop_on_gate=True,
                                          envelope_scale=scale,
                                          on_stage_ready=_inflight(
                                              st, segment=segment),
                                          **dict(kwargs))
                used_raw = latest.get("iters_used")
                used = seg_iters if used_raw is None else int(used_raw)
                spent += used
                exists, rec, ev = _best_eval_of(best)
                gated = bool(exists and rec == gate)
                progressed = (float(ev.get("metric") or -1e18)
                              > m_before + 1e-9)
                new_scale, why = _frontier_scale_next(
                    ev, scale, lo=FAST_SCALE_MIN, hi=target,
                    proven_scale=_stored_scale(best), progressed=progressed)
                history.append({"stage": st, "skipped": False, "iters": used,
                                "segment": segment, "scale": scale,
                                "next_scale": new_scale, "gated": gated,
                                "soft_advanced": False,
                                "metric": ev.get("metric"),
                                "lap_time": ev.get("lap_time"),
                                "recommendation": rec})
                _stamp({"fast_scale": new_scale})
                scale = new_scale
                if gated or used == 0 or spent - spent0 >= share:
                    break
                print(f"[fable5] auto: fast segment {segment} done — {why}",
                      flush=True)
            if gated:
                lap = ev.get("lap_time") or 0.0
                print(f"  [gate] fast banked a {lap:.2f}s lap — advancing to "
                      f"frontier", flush=True)
            else:
                if history and history[-1].get("stage") == st:
                    history[-1]["soft_advanced"] = True
                    _stamp({"fast_scale": scale})
                print(f"[fable5] auto: fast share spent without the gate — "
                      f"SOFT-ADVANCING to frontier with its banked best "
                      f"(frontier's own adaptive ladder keeps working the "
                      f"scale)", flush=True)
            continue

        if st != "frontier":
            # ---- gated stage: attempt -> retry eased -> SOFT-ADVANCE ---- #
            stage_budget = min(max(1, int(round(total * AUTO_WEIGHTS[st]))),
                               remaining)
            default_scale = stage_defaults(st).envelope_scale
            scale = (float(scale_override) if scale_override is not None
                     else (_stored_scale(best) or default_scale))
            scale_override = None            # only seeds the first stage
            attempt = 0
            gated = False
            used_total = 0
            while attempt < STAGE_MAX_ATTEMPTS and spent < total:
                attempt += 1
                slice_iters = min(stage_budget if attempt == 1
                                  else max(1, stage_budget // 2),
                                  total - spent)
                seed_from = user_resume or _chain_resume(i)
                user_resume = None           # only seeds the very first run
                print(f"[fable5] auto: {st} attempt {attempt}/"
                      f"{STAGE_MAX_ATTEMPTS} for up to {slice_iters} iters "
                      f"at scale {scale:.2f} ({total - spent} remaining of "
                      f"{total}) — resuming from {seed_from or 'scratch'}",
                      flush=True)
                latest = _train_one_stage(
                    st, slice_iters, out=ckpt, resume=seed_from,
                    stop_on_gate=True,
                    envelope_scale=(scale if abs(scale - default_scale) > 1e-9
                                    else None),
                    on_stage_ready=_inflight(st, attempt=attempt),
                    **dict(kwargs))
                used_raw = latest.get("iters_used")
                used = slice_iters if used_raw is None else int(used_raw)
                used_total += used
                spent += used
                exists, rec, ev = _best_eval_of(best)
                gated = bool(exists and rec == gate)
                if gated or used == 0:
                    break
                if attempt < STAGE_MAX_ATTEMPTS and spent < total:
                    scale = max(STAGE_SCALE_FLOOR,
                                round(scale - STAGE_RETRY_SCALE_STEP, 3))
                    print(f"[fable5] auto: {st} gate not met — retrying at "
                          f"eased scale {scale:.2f}", flush=True)
            history.append({"stage": st, "skipped": False, "iters": used_total,
                            "attempts": attempt, "scale": scale,
                            "gated": gated, "soft_advanced": not gated,
                            "metric": ev.get("metric"),
                            "lap_time": ev.get("lap_time"),
                            "recommendation": rec})
            _stamp()
            if not gated:
                print(f"[fable5] auto: {st} still ungated after {attempt} "
                      f"attempt(s) — SOFT-ADVANCING with its banked best "
                      f"(training the next stage beats idling; rerun later "
                      f"re-attacks any ungated stage)", flush=True)
            continue

        # ---- frontier: adaptive-scale SEGMENT loop (never goes stale) ---- #
        incoming_scale = _stored_scale(f"{prefix}_fast_best.pt")
        scale = (float(scale_override) if scale_override is not None
                 else (_stored_scale(best)
                       or (max(FRONTIER_SCALE_MIN, incoming_scale)
                           if incoming_scale is not None else None)
                       or stage_defaults("frontier").envelope_scale))
        scale = min(FRONTIER_SCALE_MAX,
                    max(FRONTIER_SCALE_MIN, float(scale)))
        scale_override = None
        segment = 0
        while spent < total:
            segment += 1
            remaining = total - spent
            seg_iters = min(remaining, max(150, remaining // 4))
            seed_from = user_resume or _chain_resume(i)
            user_resume = None
            print(f"[fable5] auto: frontier segment {segment} for up to "
                  f"{seg_iters} iters at scale {scale:.2f} ({remaining} "
                  f"remaining of {total}) — resuming from "
                  f"{seed_from or 'scratch'}", flush=True)
            _, _, ev0 = _best_eval_of(best)
            m_before = float((ev0 or {}).get("metric") or -1e18)
            latest = _train_one_stage("frontier", seg_iters, out=ckpt,
                                      resume=seed_from, stop_on_gate=False,
                                      envelope_scale=scale,
                                      on_stage_ready=_inflight(
                                          "frontier", segment=segment),
                                      **dict(kwargs))
            used_raw = latest.get("iters_used")
            used = seg_iters if used_raw is None else int(used_raw)
            spent += used
            exists, rec, ev = _best_eval_of(best)
            lap = ev.get("lap_time")
            sh_bar = ev.get("superhuman_target", SUPERHUMAN_LAP)
            if lap and lap < sh_bar:
                print(f"  [gate] SUPERHUMAN — clean flying lap {lap:.2f}s "
                      f"beats the {sh_bar:.2f}s human bar; compressing on",
                      flush=True)
            new_scale, why = _frontier_scale_next(
                ev, scale, proven_scale=_stored_scale(best),
                progressed=float(ev.get("metric") or -1e18) > m_before + 1e-9)
            print(f"[fable5] auto: frontier segment {segment} done — scale "
                  f"{scale:.2f} -> {new_scale:.2f} ({why})", flush=True)
            history.append({"stage": st, "skipped": False, "iters": used,
                            "segment": segment, "scale": scale,
                            "next_scale": new_scale, "gated": True,
                            "metric": ev.get("metric"), "lap_time": lap,
                            "superhuman": bool(lap and lap < sh_bar),
                            "recommendation": ev.get("recommendation") or rec})
            _stamp({"frontier_scale": new_scale})
            scale = new_scale
            if used == 0:                    # safety: never spin in place
                break

    laps = [h.get("lap_time") for h in history if h.get("lap_time")]
    lap_note = f"best lap {min(laps):.2f}s" if laps else "no clean lap yet"
    print(f"[fable5] AUTO ladder done: {spent}/{total} iters, stages "
          f"{[h['stage'] for h in history]}, {lap_note}. Champion: {FABLE_BEST}",
          flush=True)
    return latest


# --------------------------------------------------------------------------- #
# Watching
# --------------------------------------------------------------------------- #
def _make_gear_agent(net, norm, meta):
    """PolicyAgent whose third action is a GEAR OFFSET, not a handbrake."""
    from .aiviz import PolicyAgent

    class FableGearAgent(PolicyAgent):
        gear_offset = 0.0

        def _to_controls(self, a):
            steer = float(np.clip(a[0], -1, 1))
            lo = float(np.clip(a[1], -1, 1))
            if a.shape[0] > 2:
                self.gear_offset = float(np.clip(a[2], -1, 1)) * 2.0
            return (steer, max(lo, 0.0), max(-lo, 0.0), 0.0)

    return FableGearAgent(net, norm, meta, mode="race")


def watch_fable(checkpoint: str = FABLE_BEST, car: str | None = None,
                seed: int = 7, audio_on: bool = True,
                max_frames: int | None = None):
    from .viewer2 import run as run_app
    from .sensors import SensorSuite

    if not os.path.exists(checkpoint):
        raise SystemExit(f"No Fable Five policy at '{checkpoint}'. "
                         f"Train one with --fable first.")
    net, norm, meta = PPO.load_policy(checkpoint)
    checkpoint_car = str(meta.get("car") or FABLE_CAR)
    if car is not None and car != checkpoint_car:
        raise ValueError(
            f"checkpoint vehicle identity is {checkpoint_car!r}, but playback "
            f"requested {car!r}; cross-car playback is forbidden"
        )
    resolved_car = checkpoint_car
    # Non-787B playback is a first-class edition now: per-car identity is
    # stamped and certified (drivetrain_for/champion_path_for), and the 919's
    # audio is driven by REAL ledger telemetry (boost, signed MGU power, SOC)
    # through audio schema v6 — the former "unsourced turbo/MGU" silence gate
    # no longer has a factual basis.
    agent = _make_gear_agent(net, norm, meta)
    trk = attach_envelope(named_track(RING_TRACK), resolved_car)
    sensor_spec = SensorSpec()
    if meta.get("sensor_lookahead_distances"):
        sensor_spec.lookahead_distances = tuple(meta["sensor_lookahead_distances"])
    if meta.get("sensor_pace_block"):
        sensor_spec.pace_block = True
        if meta.get("sensor_pace_distances"):
            sensor_spec.pace_distances = tuple(meta["sensor_pace_distances"])
    if meta.get("sensor_hybrid_block"):
        sensor_spec.hybrid_block = True        # fable-v2 SOC + MGU telemetry
    agent.sensors = SensorSuite(sensor_spec)   # predicted path uses full obs too
    # 3-action (gear-aware) checkpoints drive the RaceBox with their offset;
    # legacy 2-action checkpoints keep the plain AutoBox they trained with.
    shift = None
    if int(meta.get("act_dim", 2)) >= 3:
        # drive the box the brain trained against (falls back to the current
        # default for pre-shift-frac checkpoints)
        rbox = RaceBox(get_car(resolved_car),
                       rpm_lo_frac=float(meta.get("fable_shift_lo_frac",
                                                  RACE_SHIFT_LO_FRAC)))
        shift = lambda veh, thr, dt: rbox.update(veh, thr, dt,
                                                 gear_offset=agent.gear_offset)
    label = ""
    run_app(car=resolved_car, seed=seed, audio_on=audio_on,
            controller=agent.act, track=trk,
            title=f"Fable Five Watch{label} - {os.path.basename(checkpoint)}",
            agent=agent, sensor_spec=sensor_spec,
            max_frames=max_frames, shift_controller=shift)
